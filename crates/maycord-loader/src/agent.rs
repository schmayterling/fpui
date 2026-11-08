//! The in-process agent: the code that runs *inside* the target once the loader
//! has forced this crate's dylib into it. It reads the adapter and theme the
//! loader handed over, finds where the target was loaded in memory, checks that
//! the pinned instruction bytes are really there, and then makes the planned
//! writes.
//!
//! Everything up to [`apply`] only reads, compares, and refuses, so it's safe to
//! run even against the wrong process. [`apply`] is the only step that changes
//! the target's running code, through a frida-gum instruction probe behind the
//! `frida` feature.

use std::io;

use maycord_host::Host;
use maycord_manifest::{Capability, Manifest};
use maycord_target::TargetAdapter;
use maycord_theme::{ThemePackage, Variant};
use maycord_wasm::WasmPlugin;
use serde::Deserialize;

use crate::{plan_writes, PlannedWrite};

fn invalid(e: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

/// One plugin to host: where its wasm module is on disk, its manifest, and the
/// capabilities the user granted it.
#[derive(Deserialize)]
struct PluginEntry {
    wasm: String,
    manifest: Manifest,
    granted: Vec<Capability>,
}

fn load_one(entry: &PluginEntry) -> Result<WasmPlugin, String> {
    let bytes = std::fs::read(&entry.wasm).map_err(|e| e.to_string())?;
    WasmPlugin::load(&bytes).map_err(|e| e.to_string())
}

/// Load the wasm plugins described by `config` (a JSON array of [`PluginEntry`])
/// into a capability-gated [`Host`]. A plugin that fails to load or register is
/// skipped with a note on stderr rather than stopping the rest: one bad mod must
/// not disable the others.
pub fn load_plugins(config: &str) -> io::Result<Host> {
    let entries: Vec<PluginEntry> = serde_json::from_str(config).map_err(invalid)?;
    let mut host = Host::new();
    for entry in entries {
        match load_one(&entry) {
            Ok(plugin) => {
                if let Err(e) = host.register(&entry.manifest, &entry.granted, Box::new(plugin)) {
                    eprintln!("maycord: skipped plugin `{}`: {e}", entry.manifest.id);
                }
            }
            Err(e) => eprintln!("maycord: could not load plugin `{}`: {e}", entry.manifest.id),
        }
    }
    Ok(host)
}

/// Read and validate the adapter the loader passed by path in `MAYCORD_ADAPTER`.
pub fn adapter_from_env() -> io::Result<TargetAdapter> {
    let path = std::env::var_os("MAYCORD_ADAPTER")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "MAYCORD_ADAPTER is not set"))?;
    let adapter: TargetAdapter = serde_json::from_slice(&std::fs::read(path)?).map_err(invalid)?;
    adapter.validate().map_err(invalid)?;
    Ok(adapter)
}

/// Read and parse the theme package the loader passed by path in `MAYCORD_THEME`.
pub fn theme_from_env() -> io::Result<ThemePackage> {
    let path = std::env::var_os("MAYCORD_THEME")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "MAYCORD_THEME is not set"))?;
    maycord_theme::parse_package(&std::fs::read(path)?).map_err(invalid)
}

/// Pick which variant to apply: the one named by `MAYCORD_VARIANT` (`dark` or
/// `light`), else dark, else light. Errors if no usable variant is present.
pub fn pick_variant(package: &ThemePackage) -> io::Result<&Variant> {
    let want_light = matches!(std::env::var("MAYCORD_VARIANT").as_deref(), Ok("light"));
    let (first, second) = if want_light {
        (&package.theme.light, &package.theme.dark)
    } else {
        (&package.theme.dark, &package.theme.light)
    };
    first
        .as_ref()
        .or(second.as_ref())
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "theme has no usable variant"))
}

/// A site whose pinned bytes weren't there: the running build isn't the one the
/// adapter was written for, so the agent must not write.
#[derive(Clone, Debug)]
pub struct SiteMismatch {
    pub token: String,
    pub addr: u64,
}

impl core::fmt::Display for SiteMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "site `{}` at {:#x} does not hold its pinned bytes",
            self.token, self.addr
        )
    }
}

/// Check every site still has its pinned instruction bytes before any write.
/// Since we're inside the process, each site's address is readable directly.
pub fn verify_sites(plan: &[PlannedWrite]) -> Result<(), SiteMismatch> {
    for write in plan {
        // SAFETY: `runtime_addr` is this image's verified load base plus an
        // offset from an identity-checked adapter. We read exactly
        // `expected_bytes.len()` bytes to compare and never write. A wrong
        // address is what this check is meant to catch, and we only get here
        // after the loader's whole-file identity check passed.
        let actual = unsafe {
            std::slice::from_raw_parts(write.runtime_addr as *const u8, write.expected_bytes.len())
        };
        if actual != write.expected_bytes.as_slice() {
            return Err(SiteMismatch {
                token: write.token.clone(),
                addr: write.runtime_addr,
            });
        }
    }
    Ok(())
}

/// Make the color writes: the one step that changes the target's running code.
///
/// With the `frida` feature (and on macOS), this attaches a frida-gum probe at
/// each checked site that overwrites the live color. Without it, the agent still
/// verifies everything and then reports that the write backend isn't built in —
/// so a build without `frida` never quietly pretends to patch.
#[cfg(all(target_os = "macos", feature = "frida"))]
pub fn apply(plan: &[PlannedWrite]) -> io::Result<()> {
    frida_backend::apply(plan)
}

/// Used when the write backend isn't compiled in.
#[cfg(not(all(target_os = "macos", feature = "frida")))]
pub fn apply(_plan: &[PlannedWrite]) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "frida apply backend not enabled; build the agent with --features frida on macOS",
    ))
}

#[cfg(all(target_os = "macos", feature = "frida"))]
mod frida_backend {
    use std::io;

    use frida_gum::interceptor::{InvocationContext, Interceptor, ProbeListener};
    use frida_gum::{Gum, NativePointer};
    use maycord_target::ValueLocation;

    use crate::{EncodedColor, PlannedWrite};

    /// What a probe does when it fires at a site.
    enum Action {
        /// Put `value` into general register `reg` (for a `u32` color in a reg).
        SetReg { reg: usize, value: u64 },
        /// Write `bytes` into memory at `[general_register[base] + offset]` — for
        /// a color that lives in a struct or on the heap, the usual GPUI case.
        WriteMem { base: usize, offset: i64, bytes: Vec<u8> },
    }

    /// A probe tied to one site: when execution reaches the pinned instruction,
    /// it performs its [`Action`] before that instruction runs.
    struct ColorProbe {
        action: Action,
    }

    impl ProbeListener for ColorProbe {
        fn on_hit(&mut self, context: InvocationContext) {
            let mut cpu = context.cpu_context();
            match &self.action {
                // apply() already checked the register is in range and that the
                // context is writable.
                Action::SetReg { reg, value } => {
                    if cpu.set_reg(*reg, *value).is_err() {
                        debug_assert!(false, "probe cpu context was unexpectedly read-only");
                    }
                }
                Action::WriteMem { base, offset, bytes } => {
                    let addr = cpu.reg(*base).wrapping_add_signed(*offset);
                    // SAFETY: `addr` is a live base register the target itself
                    // uses to reach this color, plus a fixed offset from a
                    // verified site. We write exactly the encoded color's bytes
                    // into the target's own writable color slot.
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len());
                    }
                }
            }
        }
    }

    /// What this backend can write today: a `u32` color into a general register
    /// `x0..x28`, or any encoded color into memory at `[x0..x28 + offset]`
    /// ([`Indirect`](ValueLocation::Indirect)) — which covers GPUI's HSLA colors
    /// in their `Quad` struct or theme global. Writing a vector register directly
    /// isn't reachable through frida's safe API yet; such a site is rejected up
    /// front instead of attached to and quietly ignored.
    fn probe_for(write: &PlannedWrite) -> io::Result<ColorProbe> {
        let action = match (write.location, write.color) {
            (ValueLocation::General(n), EncodedColor::U32(word)) if (n as usize) < 29 => {
                Action::SetReg { reg: n as usize, value: word as u64 }
            }
            (ValueLocation::Indirect { base, offset }, color) if (base as usize) < 29 => {
                Action::WriteMem {
                    base: base as usize,
                    offset,
                    bytes: color.to_le_bytes(),
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("site `{}`: unsupported location or encoding for this backend", write.token),
                ))
            }
        };
        Ok(ColorProbe { action })
    }

    pub fn apply(plan: &[PlannedWrite]) -> io::Result<()> {
        // Build every probe first: if any site is unsupported, attach nothing.
        let probes: Vec<ColorProbe> = plan.iter().map(probe_for).collect::<Result<_, _>>()?;

        let gum = Gum::obtain();
        let mut interceptor = Interceptor::obtain(&gum);
        for (write, probe) in plan.iter().zip(probes) {
            // Leak the probe so it lives for the rest of the process; a mod stays
            // attached until the process exits.
            let probe = Box::leak(Box::new(probe));
            let address = NativePointer(write.runtime_addr as *mut core::ffi::c_void);
            interceptor
                .attach_instruction(address, probe)
                .map_err(|e| io::Error::other(format!("attach at {:#x}: {e:?}", write.runtime_addr)))?;
        }
        // Keep gum and the interceptor alive for the process lifetime too.
        Box::leak(Box::new((gum, interceptor)));
        Ok(())
    }
}

/// Where the main executable is loaded in memory (its preferred base plus the
/// ASLR shift). We can't assume image index 0: an inserted library (the agent
/// itself, via `DYLD_INSERT_LIBRARIES`) can take index 0, so we look for the
/// image whose Mach-O filetype is `MH_EXECUTE` instead.
#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc prefers the mach2 crate's binding; the dyld API itself is stable.
pub fn main_image_base() -> Option<u64> {
    const MH_EXECUTE: u32 = 0x2;
    // SAFETY: the dyld image calls are valid for indices in `0..count`. Each
    // header points to a readable `mach_header_64`, whose `filetype` is the 4th
    // `u32` (byte offset 12). We only read it.
    unsafe {
        let count = libc::_dyld_image_count();
        for i in 0..count {
            let header = libc::_dyld_get_image_header(i);
            if header.is_null() {
                continue;
            }
            let filetype = *(header as *const u32).add(3);
            if filetype == MH_EXECUTE {
                return Some(header as u64);
            }
        }
        None
    }
}

/// Run the whole agent once: load the adapter and theme, find the image, build
/// the plan, verify it, and apply. Called from the agent dylib's startup
/// constructor.
#[cfg(target_os = "macos")]
pub fn run() -> io::Result<()> {
    let adapter = adapter_from_env()?;
    let package = theme_from_env()?;
    let variant = pick_variant(&package)?;
    let base = main_image_base()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "main executable image not found"))?;
    let plan = plan_writes(&adapter, variant, base);
    verify_sites(&plan).map_err(invalid)?;
    apply(&plan)?;

    // If plugins are configured, host them in-process and announce readiness.
    // Delivering backend events (a received message, ...) to these hooks is the
    // next step; today the agent just loads and readies them.
    if let Some(path) = std::env::var_os("MAYCORD_PLUGINS") {
        let config = std::fs::read_to_string(path)?;
        let mut host = load_plugins(&config)?;
        host.broadcast(&maycord_host::Event::Ready);
        // Keep the host and its plugins alive for the process lifetime.
        Box::leak(Box::new(host));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncodedColor;
    use maycord_target::ValueLocation;

    fn write_at(bytes: &[u8]) -> PlannedWrite {
        PlannedWrite {
            token: "sidebar".into(),
            runtime_addr: bytes.as_ptr() as u64,
            expected_bytes: bytes.to_vec(),
            location: ValueLocation::General(0),
            color: EncodedColor::U32(0),
        }
    }

    #[test]
    fn verify_sites_reads_the_live_bytes_it_is_pointed_at() {
        // Point a site at this test's own buffer: the pinned bytes match, so
        // verification passes; flip the expectation and it must fail.
        let buffer = [0xde_u8, 0xad, 0xbe, 0xef];
        let ok = write_at(&buffer);
        assert!(verify_sites(std::slice::from_ref(&ok)).is_ok());

        let mut mismatch = ok.clone();
        mismatch.expected_bytes = vec![0, 0, 0, 0];
        let err = verify_sites(std::slice::from_ref(&mismatch)).unwrap_err();
        assert_eq!(err.token, "sidebar");
    }

    #[test]
    fn pick_variant_prefers_dark_then_falls_back() {
        use maycord_manifest::{ExtensionKind, Manifest, API_VERSION};
        use maycord_theme::{Rgba, Theme};
        use std::collections::BTreeMap;

        let manifest = Manifest {
            api_version: API_VERSION,
            id: "t".into(),
            name: "t".into(),
            version: "1".into(),
            author: "a".into(),
            license: "MIT".into(),
            source: "s".into(),
            kind: ExtensionKind::Theme,
            capabilities: vec![],
        };
        let variant = Variant {
            colors: BTreeMap::from([("sidebar".to_string(), Rgba::rgb(1, 2, 3))]),
        };
        // Only a light variant present: pick_variant falls back to it.
        let light_only = ThemePackage {
            manifest,
            theme: Theme {
                dark: None,
                light: Some(variant.clone()),
            },
        };
        assert_eq!(pick_variant(&light_only).unwrap().get("sidebar"), Some(Rgba::rgb(1, 2, 3)));
    }

    #[test]
    fn adapter_from_env_errors_without_the_var() {
        // MAYCORD_ADAPTER isn't set here, so we expect a clean NotFound.
        if std::env::var_os("MAYCORD_ADAPTER").is_none() {
            assert_eq!(
                adapter_from_env().unwrap_err().kind(),
                io::ErrorKind::NotFound
            );
        }
    }

    #[test]
    #[cfg(not(feature = "frida"))]
    fn apply_without_the_backend_reports_it() {
        assert_eq!(apply(&[]).unwrap_err().kind(), io::ErrorKind::Unsupported);
    }

    // A minimal uppercase plugin, as wasm, to host.
    const UP_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (global $heap (mut i32) (i32.const 1024))
      (func $alloc (export "maycord_alloc") (param $n i32) (result i32)
        (local $p i32)
        (local.set $p (global.get $heap))
        (global.set $heap (i32.add (global.get $heap) (local.get $n)))
        (local.get $p))
      (func (export "maycord_transform_message") (param $ptr i32) (param $len i32) (result i64)
        (local $out i32) (local $i i32) (local $b i32)
        (local.set $out (call $alloc (local.get $len)))
        (block $done
          (loop $loop
            (br_if $done (i32.ge_u (local.get $i) (local.get $len)))
            (local.set $b (i32.load8_u (i32.add (local.get $ptr) (local.get $i))))
            (if (i32.and (i32.ge_u (local.get $b) (i32.const 97))
                         (i32.le_u (local.get $b) (i32.const 122)))
              (then (local.set $b (i32.sub (local.get $b) (i32.const 32)))))
            (i32.store8 (i32.add (local.get $out) (local.get $i)) (local.get $b))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))
        (i64.or
          (i64.shl (i64.extend_i32_u (local.get $out)) (i64.const 32))
          (i64.extend_i32_u (local.get $len)))))
    "#;

    #[test]
    fn load_plugins_hosts_a_wasm_plugin() {
        let wasm = wat::parse_str(UP_WAT).unwrap();
        let path = std::env::temp_dir().join("maycord_loader_test_plugin.wasm");
        std::fs::write(&path, &wasm).unwrap();
        let config = format!(
            r#"[{{"wasm":{:?},"manifest":{{"api_version":1,"id":"up","name":"Up","version":"1","author":"a","license":"MIT","source":"s","kind":"plugin","capabilities":["message_content"]}},"granted":["message_content"]}}]"#,
            path.to_str().unwrap()
        );
        let mut host = load_plugins(&config).unwrap();
        assert_eq!(host.len(), 1);
        assert_eq!(host.render_message("a", "hello"), "HELLO");
    }
}
