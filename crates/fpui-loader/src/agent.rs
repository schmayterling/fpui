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

use fpui_host::Host;
use fpui_manifest::{Capability, Manifest};
use fpui_target::TargetAdapter;
use fpui_theme::{ThemePackage, Variant};
use fpui_wasm::WasmPlugin;
use serde::Deserialize;

use crate::{plan_writes, resolve_adapter, PlannedWrite};

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
                    eprintln!("fpui: skipped plugin `{}`: {e}", entry.manifest.id);
                }
            }
            Err(e) => eprintln!("fpui: could not load plugin `{}`: {e}", entry.manifest.id),
        }
    }
    Ok(host)
}

/// Own the hosted plugins on a dedicated thread: ready them, then run each
/// detoured message through their message hook and let them observe the arrival.
/// Keeping this off the detour's hot path is why wasm never runs inside a probe.
fn plugin_consumer(config: String, rx: std::sync::mpsc::Receiver<(String, String, Vec<u8>)>) {
    let mut host = match load_plugins(&config) {
        Ok(host) => host,
        Err(e) => {
            eprintln!("fpui: could not load plugins: {e}");
            return;
        }
    };
    // Each detoured event arrives as (hook, capability, payload) chosen by the
    // adapter; hand it to every plugin granted that capability. fpui names no
    // hook itself — the application (e.g. maycord) does.
    for (hook, capability, payload) in rx {
        host.observe(&fpui_manifest::Capability::new(capability), &hook, &payload);
    }
}

/// Read and validate the adapter the loader passed by path in `FPUI_ADAPTER`.
pub fn adapter_from_env() -> io::Result<TargetAdapter> {
    let path = std::env::var_os("FPUI_ADAPTER")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "FPUI_ADAPTER is not set"))?;
    let adapter: TargetAdapter = serde_json::from_slice(&std::fs::read(path)?).map_err(invalid)?;
    adapter.validate().map_err(invalid)?;
    Ok(adapter)
}

/// Read and parse the theme package the loader passed by path in `FPUI_THEME`.
pub fn theme_from_env() -> io::Result<ThemePackage> {
    let path = std::env::var_os("FPUI_THEME")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "FPUI_THEME is not set"))?;
    fpui_theme::parse_package(&std::fs::read(path)?).map_err(invalid)
}

/// Pick which variant to apply: the one named by `FPUI_VARIANT` (`dark` or
/// `light`), else dark, else light. Errors if no usable variant is present.
pub fn pick_variant(package: &ThemePackage) -> io::Result<&Variant> {
    let want_light = matches!(std::env::var("FPUI_VARIANT").as_deref(), Ok("light"));
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

/// Split a plan into the sites that still hold their pinned bytes (safe to
/// write) and the ones a newer build has moved. This is what lets the agent run
/// on any version: it patches every site that still matches and reports the rest
/// instead of refusing to do anything when one site has drifted.
pub fn partition_live(plan: &[PlannedWrite]) -> (Vec<PlannedWrite>, Vec<SiteMismatch>) {
    let mut live = Vec::new();
    let mut stale = Vec::new();
    for write in plan {
        // SAFETY: identical to `verify_sites` — a read-only compare of exactly
        // `expected_bytes.len()` bytes at an identity-checked image offset.
        let actual = unsafe {
            std::slice::from_raw_parts(write.runtime_addr as *const u8, write.expected_bytes.len())
        };
        if actual == write.expected_bytes.as_slice() {
            live.push(write.clone());
        } else {
            stale.push(SiteMismatch {
                token: write.token.clone(),
                addr: write.runtime_addr,
            });
        }
    }
    (live, stale)
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

    use std::sync::mpsc::Sender;

    use frida_gum::interceptor::{InvocationContext, InvocationListener, Interceptor, ProbeListener};
    use frida_gum::{Gum, NativePointer};
    use fpui_target::{
        ArgOp, ReplaceOp, ReturnOp, StringArg, TargetAdapter, ValueLocation,
    };

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

    /// A probe at a detoured function's entry: it reads the adapter's declared
    /// string arguments out of the call and queues them — as a JSON-array payload
    /// under the site's hook — for the plugin thread. No wasm or heavy work runs
    /// here, per the no-blocking-in-callback rule.
    struct EventProbe {
        hook: String,
        capability: String,
        args: Vec<StringArg>,
        tx: Sender<(String, String, Vec<u8>)>,
    }

    impl ProbeListener for EventProbe {
        fn on_hit(&mut self, context: InvocationContext) {
            let mut strings = Vec::with_capacity(self.args.len());
            for arg in &self.args {
                let ptr = context.arg(arg.ptr_arg as u32);
                let len = context.arg(arg.len_arg as u32);
                if ptr == 0 || len > 64 * 1024 {
                    return; // a bad argument drops the whole event, not a partial one
                }
                // SAFETY: the target passed (ptr, len) as a string argument; we
                // read exactly `len` bytes (capped above) and never write.
                let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
                strings.push(String::from_utf8_lossy(bytes).into_owned());
            }
            let payload = serde_json::to_vec(&strings).unwrap_or_default();
            let _ = self.tx.send((self.hook.clone(), self.capability.clone(), payload));
        }
    }

    /// Detour each of the adapter's event sites — checking its pinned bytes first
    /// — and wire hits to `tx` so they reach the plugin thread.
    pub fn attach_events(
        adapter: &TargetAdapter,
        image_base: u64,
        tx: Sender<(String, String, Vec<u8>)>,
    ) -> io::Result<()> {
        let gum = Gum::obtain();
        let mut interceptor = Interceptor::obtain(&gum);
        for site in &adapter.events {
            let addr = site.runtime_addr(image_base);
            // SAFETY: read-only check of the pinned bytes at the resolved entry.
            let actual =
                unsafe { std::slice::from_raw_parts(addr as *const u8, site.expected_bytes.len()) };
            if actual != site.expected_bytes.as_slice() {
                // Same version-resilience as the color sites: a moved event hook
                // is skipped and reported, not treated as a fatal error.
                eprintln!("fpui: skipping event site at {addr:#x}; pinned bytes moved");
                continue;
            }
            let probe = Box::leak(Box::new(EventProbe {
                hook: site.hook.clone(),
                capability: site.capability.clone(),
                args: site.args.clone(),
                tx: tx.clone(),
            }));
            let np = NativePointer(addr as *mut core::ffi::c_void);
            interceptor.attach_instruction(np, probe).map_err(|e| {
                io::Error::other(format!("attach event at {addr:#x}: {e:?}"))
            })?;
        }
        Box::leak(Box::new((gum, interceptor)));
        Ok(())
    }

    /// A listener on a whole function: it lets the body run, then rewrites the
    /// return value as the function leaves. This is fpui's `after` primitive —
    /// the caller sees a different answer, but the function's own work still ran.
    /// Where a force-return byte patch overwrites the entry so the body never
    /// executes, this keeps the side effects and only changes the result.
    struct ReturnProbe {
        op: ReturnOp,
    }

    impl InvocationListener for ReturnProbe {
        fn on_enter(&mut self, _context: InvocationContext) {}

        fn on_leave(&mut self, context: InvocationContext) {
            let real = context.return_value() as u64;
            context.set_return_value(self.op.rewrite(real) as usize);
        }
    }

    /// Detour each return-override site — checking its pinned entry bytes first —
    /// so the real function runs and fpui rewrites its result on the way out. A
    /// moved site is skipped and reported, like the color and event sites, so one
    /// drifted entry never blocks the others.
    pub fn attach_returns(adapter: &TargetAdapter, image_base: u64) -> io::Result<()> {
        let gum = Gum::obtain();
        let mut interceptor = Interceptor::obtain(&gum);
        for site in &adapter.returns {
            let addr = site.runtime_addr(image_base);
            // SAFETY: read-only check of the pinned bytes at the resolved entry.
            let actual = unsafe {
                std::slice::from_raw_parts(addr as *const u8, site.expected_bytes.len())
            };
            if actual != site.expected_bytes.as_slice() {
                eprintln!(
                    "fpui: skipping return site `{}` at {addr:#x}; pinned bytes moved",
                    site.token
                );
                continue;
            }
            let probe = Box::leak(Box::new(ReturnProbe { op: site.op }));
            let np = NativePointer(addr as *mut core::ffi::c_void);
            interceptor
                .attach(np, probe)
                .map_err(|e| io::Error::other(format!("attach return at {addr:#x}: {e:?}")))?;
        }
        Box::leak(Box::new((gum, interceptor)));
        Ok(())
    }

    /// A listener on a function entry that overwrites one argument before the
    /// body runs — fpui's `before` primitive. The function then executes with
    /// the new value; its own logic and side effects are untouched.
    struct ArgProbe {
        arg: u32,
        op: ArgOp,
    }

    impl InvocationListener for ArgProbe {
        fn on_enter(&mut self, context: InvocationContext) {
            let real = context.arg(self.arg) as u64;
            context.set_arg(self.arg, self.op.rewrite(real) as usize);
        }

        fn on_leave(&mut self, _context: InvocationContext) {}
    }

    /// Detour each argument-override site — checking its pinned entry bytes first
    /// — so fpui rewrites the chosen argument before the real function runs. A
    /// moved site is skipped and reported, like the other detours.
    pub fn attach_args(adapter: &TargetAdapter, image_base: u64) -> io::Result<()> {
        let gum = Gum::obtain();
        let mut interceptor = Interceptor::obtain(&gum);
        for site in &adapter.args {
            let addr = site.runtime_addr(image_base);
            // SAFETY: read-only check of the pinned bytes at the resolved entry.
            let actual = unsafe {
                std::slice::from_raw_parts(addr as *const u8, site.expected_bytes.len())
            };
            if actual != site.expected_bytes.as_slice() {
                eprintln!(
                    "fpui: skipping arg site `{}` at {addr:#x}; pinned bytes moved",
                    site.token
                );
                continue;
            }
            let probe = Box::leak(Box::new(ArgProbe { arg: site.arg as u32, op: site.op }));
            let np = NativePointer(addr as *mut core::ffi::c_void);
            interceptor
                .attach(np, probe)
                .map_err(|e| io::Error::other(format!("attach arg at {addr:#x}: {e:?}")))?;
        }
        Box::leak(Box::new((gum, interceptor)));
        Ok(())
    }

    /// The per-site datum frida carries into the bare replacement through
    /// `replacement_data`: one `u64` today, so a single shared replacement can
    /// serve every `ReturnConst` site without a generated thunk per site.
    struct ReplaceCtx {
        value: u64,
    }

    /// The one shared replacement installed for every `ReturnConst` site. It is
    /// declared with no arguments (`ReturnConst` ignores them) and a `usize`
    /// return, which the compiler places in `x0`/`rax` — correct only for a callee
    /// whose result is an integer, pointer or bool in that register (the ABI
    /// contract `ReplaceOp::ReturnConst` documents). Plain `replace` pushes an
    /// invocation context, so `current_invocation()` is valid here and hands back
    /// this site's `ReplaceCtx`.
    unsafe extern "C" fn return_const_replacement() -> usize {
        let mut inv = Interceptor::current_invocation();
        match inv.replacement_data() {
            // SAFETY: the pointer is the `ReplaceCtx` leaked in `attach_replaces`
            // and handed to `replace` as this site's `replacement_data`; it lives
            // for the whole process and is only ever read here.
            Some(p) => unsafe { (*(p.0 as *const ReplaceCtx)).value as usize },
            None => 0,
        }
    }

    /// Replace each `instead` site outright — checking its pinned entry bytes
    /// first, like the detour family — so the real body never runs and the
    /// `ReplaceOp` supplies the result. frida relocates the displaced prologue
    /// into a trampoline (returned as the original, which `ReturnConst` ignores),
    /// so there is no length budget on the entry. A moved site is skipped and
    /// reported, not fatal.
    pub fn attach_replaces(adapter: &TargetAdapter, image_base: u64) -> io::Result<()> {
        let gum = Gum::obtain();
        let mut interceptor = Interceptor::obtain(&gum);
        for site in &adapter.replaces {
            let addr = site.runtime_addr(image_base);
            // SAFETY: read-only check of the pinned bytes at the resolved entry.
            let actual = unsafe {
                std::slice::from_raw_parts(addr as *const u8, site.expected_bytes.len())
            };
            if actual != site.expected_bytes.as_slice() {
                eprintln!(
                    "fpui: skipping replace site `{}` at {addr:#x}; pinned bytes moved",
                    site.token
                );
                continue;
            }
            match site.op {
                ReplaceOp::ReturnConst { value } => {
                    // Leak the per-site ctx for the process lifetime, like every
                    // other detour here; frida hands its pointer back in the hook.
                    let ctx = Box::leak(Box::new(ReplaceCtx { value })) as *mut ReplaceCtx;
                    let repl_fn: unsafe extern "C" fn() -> usize = return_const_replacement;
                    let func = NativePointer(addr as *mut core::ffi::c_void);
                    let repl = NativePointer(repl_fn as *mut core::ffi::c_void);
                    let data = NativePointer(ctx as *mut core::ffi::c_void);
                    // The Ok value is the original trampoline; ReturnConst never
                    // calls it, so it is dropped.
                    interceptor
                        .replace(func, repl, data)
                        .map_err(|e| io::Error::other(format!("replace at {addr:#x}: {e:?}")))?;
                }
                // A future ReplaceOp variant (e.g. call-through) is not
                // handled by this build; skip the site rather than guess at
                // its ABI.
                _ => eprintln!(
                    "fpui: skipping replace site `{}`; unsupported replace op",
                    site.token
                ),
            }
        }
        Box::leak(Box::new((gum, interceptor)));
        Ok(())
    }

    extern "C" {
        /// Flush the CPU instruction cache for `len` bytes at `start`. arm64 has
        /// split instruction/data caches, so a freshly written code byte is not
        /// executed until the stale i-cache line is invalidated. Declared from
        /// libSystem's `<libkern/OSCacheControl.h>`.
        fn sys_icache_invalidate(start: *mut core::ffi::c_void, len: usize);

        /// Set the protection of a range in a task. We can't use `mprotect`:
        /// apple-silicon refuses to make a code-signed `__TEXT` page writable
        /// that way. `VM_PROT_COPY` here forces a private, writable copy of the
        /// page instead, which is the supported way to edit mapped code. From
        /// `<mach/mach_vm.h>`, resolved in libSystem.
        fn mach_vm_protect(
            target_task: libc::vm_map_t,
            address: libc::mach_vm_address_t,
            size: libc::mach_vm_size_t,
            set_maximum: libc::boolean_t,
            new_protection: libc::vm_prot_t,
        ) -> libc::kern_return_t;
    }

    const VM_PROT_READ: libc::vm_prot_t = 0x1;
    const VM_PROT_WRITE: libc::vm_prot_t = 0x2;
    const VM_PROT_EXECUTE: libc::vm_prot_t = 0x4;
    /// OR'd into the new protection, forces a copy-on-write copy of the range so
    /// a code-signed, file-backed page can be made writable.
    const VM_PROT_COPY: libc::vm_prot_t = 0x10;

    /// This host's page size, to align a protection range. Falls back to 16 KiB
    /// (apple-silicon's size) if the query ever fails.
    fn page_size() -> usize {
        // SAFETY: `sysconf` is always callable; `_SC_PAGESIZE` returns the size.
        let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if n > 0 {
            n as usize
        } else {
            16 * 1024
        }
    }

    /// Overwrite the `bytes.len()` bytes at `addr` in the running image. A
    /// BytePatch site sits in the target's own `__TEXT` segment — a drawn string
    /// literal in `__TEXT,__cstring`/`__const`, or a function entry in
    /// `__TEXT,__text` — which the kernel maps read+execute. apple-silicon won't
    /// let a page be writable and executable at once, and refuses `mprotect`
    /// write on a code-signed page at all, so we take the range writable with
    /// `VM_PROT_COPY` (a private read/write copy, no execute), copy the
    /// replacement in, invalidate the i-cache for the range (required before the
    /// CPU runs an overwritten instruction), then restore the pages to the
    /// read+execute they began with. This edits the target's existing pages in
    /// place; it never maps new executable memory.
    #[allow(deprecated)] // libc prefers the mach2 crate's binding; the task-self trap is stable.
    fn write_in_place(addr: u64, bytes: &[u8]) -> io::Result<()> {
        let page = page_size();
        let start = (addr as usize) & !(page - 1);
        let end = (addr as usize + bytes.len() + page - 1) & !(page - 1);
        let span = (end - start) as libc::mach_vm_size_t;
        // SAFETY: `start..end` is the page-aligned span covering `[addr,
        // addr+len)`, which the caller just re-verified still holds the pinned
        // `expected_bytes`. We take those pages writable as a private copy, write
        // exactly `bytes.len()` bytes (the site's own length-preserving
        // replacement), flush the i-cache for that range, then restore r-x. Every
        // pointer is into the target's own mapped image, and `mach_task_self()`
        // is this process's own task port.
        unsafe {
            let task = libc::mach_task_self();
            let kr = mach_vm_protect(
                task,
                start as libc::mach_vm_address_t,
                span,
                0,
                VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY,
            );
            if kr != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("vm_protect writable failed: kr={kr}"),
                ));
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), addr as *mut u8, bytes.len());
            sys_icache_invalidate(addr as *mut libc::c_void, bytes.len());
            let kr = mach_vm_protect(
                task,
                start as libc::mach_vm_address_t,
                span,
                0,
                VM_PROT_READ | VM_PROT_EXECUTE,
            );
            if kr != 0 {
                return Err(io::Error::other(format!("vm_protect restore failed: kr={kr}")));
            }
        }
        Ok(())
    }

    /// Apply every same-length byte overwrite in `adapter` to the live image.
    /// Unlike the colour and detour sites, a byte patch is not a frida probe: it
    /// rewrites the target's own `__TEXT` bytes in place, which is how fpui
    /// injects UI text (overwrite a drawn string) or neutralises a function
    /// (write a force-return over its entry). Each site is re-verified against
    /// its pinned `expected_bytes` at the live address first; on any mismatch the
    /// site is skipped with a note and never written, so a drifted build is
    /// reported, not scribbled on. One failed site does not stop the rest.
    pub fn apply_byte_patches(adapter: &TargetAdapter, image_base: u64) {
        for patch in &adapter.patches {
            if !patch.is_length_preserving() {
                // validate() already rejects this; guard anyway so a write can
                // never change the length and shift the bytes after the site.
                eprintln!(
                    "fpui: skipping byte patch `{}`; replacement length differs",
                    patch.token
                );
                continue;
            }
            let addr = patch.runtime_addr(image_base);
            let len = patch.expected_bytes.len();
            // SAFETY: a read-only compare of exactly `len` bytes at a verified
            // image offset — the gate that refuses to write on a wrong build.
            let live = unsafe { std::slice::from_raw_parts(addr as *const u8, len) };
            if live != patch.expected_bytes.as_slice() {
                eprintln!(
                    "fpui: skipping byte patch `{}` at {addr:#x}; pinned bytes moved",
                    patch.token
                );
                continue;
            }
            match write_in_place(addr, &patch.replacement_bytes) {
                Ok(()) => {
                    eprintln!("fpui: byte patch `{}` applied at {addr:#x}", patch.token)
                }
                Err(e) => eprintln!(
                    "fpui: byte patch `{}` at {addr:#x} failed: {e}",
                    patch.token
                ),
            }
        }
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

/// Where the main executable is loaded, from the dynamic linker's view of the
/// process: the first object `dl_iterate_phdr` reports is the main executable,
/// and its load bias is the base. Returns `None` for a non-PIE binary (bias 0),
/// which this prototype does not yet handle.
#[cfg(target_os = "linux")]
pub fn main_image_base() -> Option<u64> {
    unsafe extern "C" fn first_object(
        info: *mut libc::dl_phdr_info,
        _size: libc::size_t,
        data: *mut libc::c_void,
    ) -> libc::c_int {
        // SAFETY: `data` is the `&mut u64` we passed below; store the first
        // object's load bias and stop the walk by returning non-zero.
        unsafe { *(data as *mut u64) = (*info).dlpi_addr as u64 };
        1
    }
    let mut base: u64 = 0;
    // SAFETY: a standard `dl_iterate_phdr` call with a matching callback and the
    // address of `base` as its data pointer.
    unsafe {
        libc::dl_iterate_phdr(Some(first_object), (&mut base as *mut u64).cast());
    }
    (base != 0).then_some(base)
}

/// Where the main executable is loaded: `GetModuleHandle(NULL)` returns the
/// running process's own image handle, which is its base address.
#[cfg(target_os = "windows")]
pub fn main_image_base() -> Option<u64> {
    extern "system" {
        fn GetModuleHandleW(module_name: *const u16) -> *mut core::ffi::c_void;
    }
    // SAFETY: `GetModuleHandleW(NULL)` returns the base-address handle of the
    // current process's executable and does not fail for it.
    let handle = unsafe { GetModuleHandleW(core::ptr::null()) };
    (!handle.is_null()).then_some(handle as u64)
}

/// Run the whole agent once: load the adapter and theme, find the image, build
/// the plan, verify it, and apply. Called from the agent dylib's startup
/// constructor.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
pub fn run() -> io::Result<()> {
    let adapter = adapter_from_env()?;
    let package = theme_from_env()?;
    let variant = pick_variant(&package)?;
    let base = main_image_base()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "main executable image not found"))?;
    // Resolve each site to where it actually sits in this build before planning,
    // so a drifted version still patches. We read the target's own file from disk
    // (its on-disk layout matches the loaded image) and scan for each site's
    // signature; sites we can't find are dropped with a note. If we can't read
    // our own image, fall back to the adapter's exact pinned offsets.
    // Fail closed on identity: read our own on-disk image and refuse to patch
    // blind if we can't. A non-exact build is NOT rejected (drift survival is the
    // point — anchors/signatures re-resolve and every site is byte-checked live),
    // but it is reported, and we never fall back to un-resolved pinned offsets
    // against an image we couldn't read.
    let image = std::env::current_exe().and_then(std::fs::read).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("cannot read own image for the identity check; refusing to patch blind: {e}"),
        )
    })?;
    if let Err(e) = adapter.identity.verify_file(&image) {
        eprintln!(
            "fpui: target is not the exact pinned build ({e}); relying on anchor/signature \
             re-resolution and the per-site byte check"
        );
    }
    let adapter = {
        let (resolved, dropped) = resolve_adapter(&adapter, &image);
        for label in &dropped {
            eprintln!("fpui: patch `{label}` not found in this build; skipping");
        }
        resolved
    };
    let plan = plan_writes(&adapter, variant, base);
    // Run on any version: check each site live, patch the ones that still match,
    // and report the ones a newer build has moved. A single drifted site no
    // longer blocks the patches that do still apply.
    let (live, stale) = partition_live(&plan);
    for miss in &stale {
        eprintln!("fpui: skipping {miss}");
    }
    // A build without the write backend (e.g. no `frida`) still injects and
    // verifies; it just can't apply. Report that and carry on to plugin hosting
    // rather than aborting.
    if !live.is_empty() {
        if let Err(e) = apply(&live) {
            if e.kind() == io::ErrorKind::Unsupported {
                eprintln!("fpui: {e}");
            } else {
                return Err(e);
            }
        }
    }

    // Byte patches are same-length overwrites of the target's own __TEXT bytes:
    // inject UI text, neutralise a function. They are not detours, so they run on
    // their own right after the colour writes. Each site re-verifies its pinned
    // bytes live before writing, so a drifted site is skipped, not forced.
    #[cfg(all(target_os = "macos", feature = "frida"))]
    if !adapter.patches.is_empty() {
        frida_backend::apply_byte_patches(&adapter, base);
    }

    // Return overrides need no plugins and no event channel: they detour a
    // function and rewrite its result on the way out. Attach them on their own.
    #[cfg(all(target_os = "macos", feature = "frida"))]
    if !adapter.returns.is_empty() {
        if let Err(e) = frida_backend::attach_returns(&adapter, base) {
            eprintln!("fpui: return override failed: {e}");
        }
    }
    #[cfg(all(target_os = "macos", feature = "frida"))]
    if !adapter.args.is_empty() {
        if let Err(e) = frida_backend::attach_args(&adapter, base) {
            eprintln!("fpui: arg override failed: {e}");
        }
    }
    // Function replacements (Vencord `instead`): detour an entry so the body
    // never runs and a `ReplaceOp` supplies the result. Live and reversible,
    // attached on their own like the return/arg overrides.
    #[cfg(all(target_os = "macos", feature = "frida"))]
    if !adapter.replaces.is_empty() {
        if let Err(e) = frida_backend::attach_replaces(&adapter, base) {
            eprintln!("fpui: function replace failed: {e}");
        }
    }

    // If plugins are configured, host them on their own thread and feed them
    // events detoured from the target (a received message, ...). The plugin
    // thread owns the host, so wasm never runs on the detour's hot path.
    if let Some(path) = std::env::var_os("FPUI_PLUGINS") {
        let config = std::fs::read_to_string(path)?;
        let (tx, rx) = std::sync::mpsc::channel::<(String, String, Vec<u8>)>();
        std::thread::spawn(move || plugin_consumer(config, rx));
        #[cfg(all(target_os = "macos", feature = "frida"))]
        if !adapter.events.is_empty() {
            if let Err(e) = frida_backend::attach_events(&adapter, base, tx.clone()) {
                eprintln!("fpui: event detour failed: {e}");
            }
        }
        // The consumer runs for the process lifetime while an event probe holds a
        // sender; with no probes it readies the plugins and then idles.
        drop(tx);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncodedColor;
    use fpui_target::ValueLocation;

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
    fn partition_live_keeps_matching_sites_and_reports_moved_ones() {
        // One site points at live bytes that match; the other expects bytes that
        // aren't there. Only the first is safe to patch on this build.
        let buffer = [0xde_u8, 0xad, 0xbe, 0xef];
        let matches = write_at(&buffer);
        let mut moved = matches.clone();
        moved.expected_bytes = vec![0, 0, 0, 0];

        let (live, stale) = partition_live(&[matches, moved]);
        assert_eq!(live.len(), 1);
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].token, "sidebar");
    }

    #[test]
    fn pick_variant_prefers_dark_then_falls_back() {
        use fpui_manifest::{ExtensionKind, Manifest, API_VERSION};
        use fpui_theme::{Rgba, Theme};
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
        // FPUI_ADAPTER isn't set here, so we expect a clean NotFound.
        if std::env::var_os("FPUI_ADAPTER").is_none() {
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
      (func $alloc (export "fpui_alloc") (param $n i32) (result i32)
        (local $p i32)
        (local.set $p (global.get $heap))
        (global.set $heap (i32.add (global.get $heap) (local.get $n)))
        (local.get $p))
      (func (export "fpui_transform") (param $hptr i32) (param $hlen i32) (param $ptr i32) (param $len i32) (result i64)
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
        let path = std::env::temp_dir().join("fpui_loader_test_plugin.wasm");
        std::fs::write(&path, &wasm).unwrap();
        let config = format!(
            r#"[{{"wasm":{:?},"manifest":{{"api_version":1,"id":"up","name":"Up","version":"1","author":"a","license":"MIT","source":"s","kind":"plugin","capabilities":["message_content"]}},"granted":["message_content"]}}]"#,
            path.to_str().unwrap()
        );
        let mut host = load_plugins(&config).unwrap();
        assert_eq!(host.len(), 1);
        let cap = fpui_manifest::Capability::new("message_content");
        let out = host.transform(&cap, "message", b"hello");
        assert_eq!(String::from_utf8(out).unwrap(), "HELLO");
    }
}
