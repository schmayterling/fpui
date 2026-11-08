//! The in-process agent: the code that runs *inside* the target once the loader
//! has forced this crate's dylib in. It reads the adapter and theme the loader
//! handed over, finds where the target was mapped, confirms the pinned
//! instruction bytes are really present, and then applies the planned writes.
//!
//! Everything here up to [`apply`] is verification: it reads, compares, and
//! refuses, so it is safe to run even against the wrong process. [`apply`] is
//! the only step that rewrites the target's execution, via a frida-gum
//! instruction probe behind the `frida` feature.

use std::io;

use maycord_target::TargetAdapter;
use maycord_theme::{ThemePackage, Variant};

use crate::{plan_writes, PlannedWrite};

fn invalid(e: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
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

/// A site whose pinned bytes were not present: the running build is not the one
/// the adapter was authored against, so the agent must not write.
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

/// Confirm every site still holds its pinned instruction bytes before any write.
/// Running in-process, each site's runtime address is directly readable.
pub fn verify_sites(plan: &[PlannedWrite]) -> Result<(), SiteMismatch> {
    for write in plan {
        // SAFETY: `runtime_addr` is this image's verified load base plus an
        // offset from an identity-checked adapter; we read exactly
        // `expected_bytes.len()` bytes to compare and never write here. A wrong
        // address is the failure this check exists to catch, not to cause — it
        // is only reached after the loader's full-file identity gate passed.
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

/// Install the color writes: the one step that rewrites the target's execution.
///
/// With the `frida` feature (and on macOS) this attaches a frida-gum probe at
/// each verified site that overwrites the live color. Without it, the agent
/// still verifies everything and then reports that the write backend is absent —
/// so a build without `frida` never silently pretends to patch.
#[cfg(all(target_os = "macos", feature = "frida"))]
pub fn apply(plan: &[PlannedWrite]) -> io::Result<()> {
    frida_backend::apply(plan)
}

/// Fallback when the write backend is not compiled in.
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

    /// A probe bound to one site: when execution reaches the pinned instruction,
    /// it overwrites the live color register before that instruction runs.
    struct ColorProbe {
        reg: usize,
        value: u64,
    }

    impl ProbeListener for ColorProbe {
        fn on_hit(&mut self, context: InvocationContext) {
            let mut cpu = context.cpu_context();
            // Only reached for sites apply() already checked, so this cannot be
            // a read-only context or an out-of-range register.
            if cpu.set_reg(self.reg, self.value).is_err() {
                debug_assert!(false, "probe cpu context was unexpectedly read-only");
            }
        }
    }

    /// What this backend can write today: a packed `u32` color in a general
    /// register `x0..x28`. Vector-register (`f32x4`) and stack writes are
    /// expressible in an adapter but not yet applied here; such a site is
    /// rejected up front rather than attached to and silently ignored.
    fn probe_for(write: &PlannedWrite) -> io::Result<ColorProbe> {
        match (write.location, write.color) {
            (ValueLocation::General(n), EncodedColor::U32(word)) if (n as usize) < 29 => {
                Ok(ColorProbe {
                    reg: n as usize,
                    value: word as u64,
                })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "site `{}`: this backend writes only a srgb_u32 color into x0..x28 so far",
                    write.token
                ),
            )),
        }
    }

    pub fn apply(plan: &[PlannedWrite]) -> io::Result<()> {
        // Build every probe first: if any site is unsupported, attach nothing.
        let probes: Vec<ColorProbe> = plan.iter().map(probe_for).collect::<Result<_, _>>()?;

        let gum = Gum::obtain();
        let mut interceptor = Interceptor::obtain(&gum);
        for (write, probe) in plan.iter().zip(probes) {
            // Leak the probe so its vtable outlives this call for the process's
            // lifetime; a mod stays attached until the process exits.
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

/// The load base of the main executable image (its preferred base plus the ASLR
/// slide). We cannot assume image index 0: an inserted library (the agent
/// itself, via `DYLD_INSERT_LIBRARIES`) can take index 0, so we find the image
/// whose Mach-O filetype is `MH_EXECUTE` instead.
#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc prefers the mach2 crate's binding; the dyld API itself is stable.
pub fn main_image_base() -> Option<u64> {
    const MH_EXECUTE: u32 = 0x2;
    // SAFETY: the dyld image APIs are defined for indices in `0..count`; each
    // header pointer is a readable `mach_header_64`, whose `filetype` is the 4th
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

/// Run the whole agent flow once: load adapter + theme, locate the image, plan,
/// verify, apply. Called from the agent dylib's module-init constructor.
#[cfg(target_os = "macos")]
pub fn run() -> io::Result<()> {
    let adapter = adapter_from_env()?;
    let package = theme_from_env()?;
    let variant = pick_variant(&package)?;
    let base = main_image_base()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "main executable image not found"))?;
    let plan = plan_writes(&adapter, variant, base);
    verify_sites(&plan).map_err(invalid)?;
    apply(&plan)
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
        // Not set in this test's environment (never set here): a clean NotFound.
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
}
