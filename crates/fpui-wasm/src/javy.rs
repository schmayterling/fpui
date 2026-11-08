//! A JavaScript plugin backend built on [Javy](https://github.com/bytecodealliance/javy):
//! QuickJS compiled *to* wasm.
//!
//! A [`JavyPlugin`] is a Javy-produced module — the QuickJS engine is wasm
//! bytecode that the same wasmi interpreter runs. So JS authoring is added
//! *without* giving up anything the plain-wasm host bought: the engine is wasm
//! data, not native code, so there is still no new executable memory in the
//! target, and a memory-safety bug in QuickJS is contained by the wasm sandbox
//! rather than becoming native code execution.
//!
//! ## Call model
//! Javy modules are **WASI commands**: a module's `_start` export reads its
//! input from stdin and writes its result to stdout, then exits. There is no
//! parameterized export to call. So a transform runs like a one-shot process:
//! seed stdin with the payload, instantiate fresh, call `_start`, and read
//! whatever it wrote to stdout back out as the new payload. Each call gets its
//! own [`Store`] — its own linear memory, fuel, and stdin/stdout — so plugins
//! and successive calls never share state.
//!
//! ## Static vs dynamic linking
//! [`JavyPlugin`] runs a **static** module: the QuickJS engine (~1.3 MiB) is
//! embedded in every plugin. [`JavyRuntime`] runs **dynamic** modules instead:
//! the engine is compiled *once* as a shared plugin module (`javy emit-plugin`)
//! and each JS plugin is a tiny 1-16 KiB module (`javy build -C dynamic`) that
//! imports the shared engine's `memory`, `cabi_realloc`, and `invoke`. N plugins
//! then cost ~1.3 MiB + N×(KBs) of wasm instead of N×1.3 MiB. Isolation is
//! unchanged — a *fresh* engine instance is built per call, sharing only the
//! compiled module, so no QuickJS state leaks between calls or plugins.
//!
//! ## The shim — two pipes and nothing else
//! A Javy module imports a handful of `wasi_snapshot_preview1` functions (nine
//! for a static module; a dynamic plugin also wants `random_get`). This module
//! provides all ten, backed only by two in-memory buffers: stdin (fd 0,
//! read-only, the payload) and stdout (fd 1, write-only, the result). fd 2
//! (stderr) is discarded; `random_get` is zero-filled; every other descriptor
//! and every other capability — no files, no sockets, no clock beyond a zero
//! reading, no environment, no real entropy — is denied. The guest has zero
//! ambient authority beyond those two pipes, by construction.
//!
//! ## The stdin frame — hook name + payload
//! stdin is **framed** so the JS receives both the hook name and the payload,
//! not the payload alone: a little-endian `u32` hook length, the hook name
//! (UTF-8), then the opaque payload bytes. A length prefix (rather than JSON)
//! keeps the payload raw — it may be arbitrary bytes, which JSON would force
//! through base64. `js/fpui.js` is the author-side helper: it parses the frame
//! and dispatches to a per-hook handler, so one plugin can transform the
//! `"message"` hook and pass the others through unchanged. stdout stays
//! unframed: the guest writes only the rewritten payload (the host already knows
//! which hook it called).
//!
//! ## Ceilings (v2)
//! * A module that writes **nothing** to stdout is treated as "no change" (the
//!   original payload stands), so a JS plugin cannot transform a payload to
//!   genuinely empty output. This mirrors the ptr/len ABI's `0 = no change`
//!   sentinel and keeps a silently-failing plugin from blanking content.
//! * [`Plugin::observe`] now runs the module (fuel-bounded, trap-safe) and
//!   discards stdout, but with no host imports yet a JS observer still cannot
//!   *do* anything — it waits on the capability host-import work.
//! * No capability host-imports yet: a JS plugin has zero ambient authority, so
//!   `observe` runs but cannot act, and a transform is a pure function of its
//!   framed input. Wiring the first host import (e.g. storage) into JS globals
//!   through the import chokepoint is the next step.

use std::sync::Arc;

use fpui_host::Plugin;
use wasmi::{Caller, Config, Engine, Error, Extern, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder};

use crate::{WasmError, MAX_MEMORY_BYTES, MAX_OUTPUT};

/// Fuel for one `_start` run. Larger than the ptr/len path's budget because a
/// Javy call pays to initialize the whole QuickJS engine (interpreted) before it
/// runs a line of the plugin's JS — a minimal plugin measures ~6.9M fuel, almost
/// all of it that one-time engine init. This leaves a comfortable multiple for
/// heavier scripts while still trapping a runaway in a brief, finite spin. The
/// host can override it per plugin via [`JavyPlugin::load_with_fuel`].
pub const JAVY_DEFAULT_FUEL: u64 = 100_000_000;

// WASI `errno` values the shim returns. `0` is success.
const ERRNO_SUCCESS: i32 = 0;
const ERRNO_BADF: i32 = 8; // EBADF — bad file descriptor
const ERRNO_SPIPE: i32 = 29; // ESPIPE — seek on a pipe

/// WASI `filetype` for a character device — what stdin/stdout/stderr report, so
/// the guest's libc treats them as unbuffered pipes rather than seekable files.
const FILETYPE_CHARACTER_DEVICE: u8 = 2;

const FD_STDIN: i32 = 0;
const FD_STDOUT: i32 = 1;
const FD_STDERR: i32 = 2;

/// The host state for one Javy call: the memory cap plus the only two channels
/// the module can reach.
struct JavyCtx {
    limits: StoreLimits,
    /// The payload, consumed by `fd_read` on fd 0.
    stdin: Vec<u8>,
    stdin_pos: usize,
    /// What the module wrote to fd 1; becomes the transformed payload.
    stdout: Vec<u8>,
    /// Set if the guest tried to write past [`MAX_OUTPUT`]; the result is then
    /// discarded rather than returned truncated.
    output_overflow: bool,
}

impl JavyCtx {
    fn new(payload: &[u8]) -> Self {
        let limits = StoreLimitsBuilder::new().memory_size(MAX_MEMORY_BYTES).build();
        Self { limits, stdin: payload.to_vec(), stdin_pos: 0, stdout: Vec::new(), output_overflow: false }
    }
}

/// A JavaScript plugin: a compiled Javy module, re-instantiated per call.
///
/// Holds the engine, the compiled module, and a linker carrying the WASI shim —
/// all built once at load. Each transform spins up a fresh [`Store`] against
/// them, so compilation is paid once and only (cheap) instantiation plus the run
/// happen per call.
pub struct JavyPlugin {
    engine: Engine,
    module: Module,
    linker: Linker<JavyCtx>,
    fuel_per_call: u64,
}

impl JavyPlugin {
    /// Load a Javy module with the [`JAVY_DEFAULT_FUEL`] per-call budget.
    pub fn load(wasm: &[u8]) -> Result<Self, WasmError> {
        Self::load_with_fuel(wasm, JAVY_DEFAULT_FUEL)
    }

    /// Load a Javy module with an explicit per-call fuel budget.
    pub fn load_with_fuel(wasm: &[u8], fuel_per_call: u64) -> Result<Self, WasmError> {
        let mut config = Config::default();
        config.consume_fuel(true);
        // QuickJS-wasm uses the SIMD proposal. wasmi gates it behind the `simd`
        // feature (which `js` turns on) and still *interprets* it — no native
        // SIMD codegen, so the pure-safe-Rust, no-exec-memory posture holds.
        config.wasm_simd(true).wasm_relaxed_simd(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, wasm).map_err(|e| WasmError::Compile(e.to_string()))?;
        let mut linker = Linker::new(&engine);
        define_wasi(&mut linker).map_err(|e| WasmError::Instantiate(e.to_string()))?;
        Ok(Self { engine, module, linker, fuel_per_call })
    }

    /// Run the module once for `hook` over `payload`: fresh store seeded with the
    /// framed stdin (`frame(hook, payload)`), call `_start`, return whatever it
    /// wrote to stdout. `None` on any trap, out-of-fuel, nonzero exit, oversize
    /// output, or empty output — in every case the caller keeps the original
    /// payload. Shared by [`JavyPlugin::transform`] (which returns the output)
    /// and [`JavyPlugin::observe`] (which discards it but still runs the module).
    fn run(&mut self, hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
        let mut store = Store::new(&self.engine, JavyCtx::new(&frame(hook, payload)));
        store.limiter(|ctx| &mut ctx.limits);
        // One budget covers instantiation and the whole `_start` run.
        store.set_fuel(self.fuel_per_call).ok()?;

        let instance = self.linker.instantiate_and_start(&mut store, &self.module).ok()?;
        let start = instance.get_typed_func::<(), ()>(&store, "_start").ok()?;

        // A WASI command ends by calling `proc_exit`, which the shim turns into
        // an `i32_exit` error carrying the code; a module that just returns is
        // clean too. Anything else — a real trap, out-of-fuel, or a nonzero
        // exit — discards the transform.
        match start.call(&mut store, ()) {
            Ok(()) => {}
            Err(e) => match e.i32_exit_status() {
                Some(0) => {}
                _ => return None,
            },
        }

        let ctx = store.data_mut();
        if ctx.output_overflow || ctx.stdout.len() > MAX_OUTPUT || ctx.stdout.is_empty() {
            return None;
        }
        Some(core::mem::take(&mut ctx.stdout))
    }
}

impl Plugin for JavyPlugin {
    fn transform(&mut self, hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
        // stdin is framed with the hook name (see the module docs), so the JS can
        // dispatch per hook; stdout is the rewritten payload.
        self.run(hook, payload)
    }

    fn observe(&mut self, hook: &str, payload: &[u8]) {
        // Run the module for its side effects, discarding stdout. With no host
        // imports yet nothing escapes the sandbox, but the fuel-bounded, trap-safe
        // plumbing is in place for when a capability import lands.
        let _ = self.run(hook, payload);
    }
}

/// The shared QuickJS engine for dynamically-linked plugins: the ~1.3 MiB Javy
/// plugin module (`javy emit-plugin`), compiled once and reused to run many tiny
/// dynamic plugin modules. A dynamic module is 1-16 KiB and imports this engine's
/// `memory`, `cabi_realloc`, and `invoke`, so the engine is stored once instead
/// of embedded in every plugin: N JS plugins cost ~1.3 MiB + N×(KBs) of wasm
/// rather than N×1.3 MiB.
///
/// Isolation is preserved. Only the *compiled* engine module is shared; a fresh
/// instance is built per call, so no QuickJS state leaks between calls or between
/// plugins. The win is module footprint, never a shared live instance.
struct SharedEngine {
    engine: Engine,
    /// The shared QuickJS plugin module.
    plugin: Module,
    /// The WASI shim, reusable across every call's [`Store`].
    wasi: Linker<JavyCtx>,
}

/// A runtime hosting one shared QuickJS engine for dynamically-linked plugins.
/// Load the engine once, then [`JavyRuntime::load`] any number of tiny modules.
pub struct JavyRuntime {
    shared: Arc<SharedEngine>,
    fuel_per_call: u64,
}

impl JavyRuntime {
    /// Build a runtime around the shared plugin module emitted by
    /// `javy emit-plugin`, with the [`JAVY_DEFAULT_FUEL`] per-call budget.
    pub fn new(plugin_wasm: &[u8]) -> Result<Self, WasmError> {
        Self::new_with_fuel(plugin_wasm, JAVY_DEFAULT_FUEL)
    }

    /// Build a runtime with an explicit per-call fuel budget.
    pub fn new_with_fuel(plugin_wasm: &[u8], fuel_per_call: u64) -> Result<Self, WasmError> {
        let mut config = Config::default();
        config.consume_fuel(true);
        config.wasm_simd(true).wasm_relaxed_simd(true);
        let engine = Engine::new(&config);
        let plugin = Module::new(&engine, plugin_wasm).map_err(|e| WasmError::Compile(e.to_string()))?;
        let mut wasi = Linker::new(&engine);
        define_wasi(&mut wasi).map_err(|e| WasmError::Instantiate(e.to_string()))?;
        Ok(Self { shared: Arc::new(SharedEngine { engine, plugin, wasi }), fuel_per_call })
    }

    /// Compile a dynamically-linked plugin module (`javy build -C dynamic`)
    /// against this runtime's shared engine.
    pub fn load(&self, module_wasm: &[u8]) -> Result<DynamicJavyPlugin, WasmError> {
        let module = Module::new(&self.shared.engine, module_wasm).map_err(|e| WasmError::Compile(e.to_string()))?;
        // The module's imports all name the shared plugin; capture that namespace
        // from the module itself so the exact string (e.g. the versioned
        // "javy-default-plugin-v4") is never hard-coded here.
        let plugin_ns = module
            .imports()
            .next()
            .ok_or(WasmError::MissingExport("plugin import"))?
            .module()
            .to_string();
        Ok(DynamicJavyPlugin { shared: Arc::clone(&self.shared), module, plugin_ns, fuel_per_call: self.fuel_per_call })
    }
}

/// One dynamically-linked JS plugin: a tiny module that runs on a shared
/// [`JavyRuntime`] engine. A [`Plugin`] exactly like [`JavyPlugin`], but without
/// the embedded engine.
pub struct DynamicJavyPlugin {
    shared: Arc<SharedEngine>,
    module: Module,
    /// The import namespace the dynamic module expects the shared engine under.
    plugin_ns: String,
    fuel_per_call: u64,
}

impl DynamicJavyPlugin {
    /// Run the tiny module once for `hook` over `payload`: a fresh engine
    /// instance is initialized, the module is linked to it, and its `_start`
    /// runs. Same `None`-keeps-the-original contract as [`JavyPlugin::run`].
    fn run(&self, hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
        let mut store = Store::new(&self.shared.engine, JavyCtx::new(&frame(hook, payload)));
        store.limiter(|ctx| &mut ctx.limits);
        // One budget covers engine init, linking, and the `_start` run.
        store.set_fuel(self.fuel_per_call).ok()?;

        // A fresh engine instance for this call, so no QuickJS state carries over.
        // The shared plugin from `javy emit-plugin` is already pre-initialized, so
        // the host does NOT call `initialize-runtime` — that is the plugin's
        // Wizer pre-init entry, and invoking it on an initialized runtime panics.
        let plugin = self.shared.wasi.instantiate_and_start(&mut store, &self.shared.plugin).ok()?;

        // Wire the tiny module's three imports to this engine instance's exports.
        let memory = plugin.get_memory(&store, "memory")?;
        let cabi_realloc = plugin.get_func(&store, "cabi_realloc")?;
        let invoke = plugin.get_func(&store, "invoke")?;
        let mut linker = Linker::new(&self.shared.engine);
        linker.define(&self.plugin_ns, "memory", memory).ok()?;
        linker.define(&self.plugin_ns, "cabi_realloc", cabi_realloc).ok()?;
        linker.define(&self.plugin_ns, "invoke", invoke).ok()?;

        let instance = linker.instantiate_and_start(&mut store, &self.module).ok()?;
        let start = instance.get_typed_func::<(), ()>(&store, "_start").ok()?;
        match start.call(&mut store, ()) {
            Ok(()) => {}
            Err(e) => match e.i32_exit_status() {
                Some(0) => {}
                _ => return None,
            },
        }

        let ctx = store.data_mut();
        if ctx.output_overflow || ctx.stdout.len() > MAX_OUTPUT || ctx.stdout.is_empty() {
            return None;
        }
        Some(core::mem::take(&mut ctx.stdout))
    }
}

impl Plugin for DynamicJavyPlugin {
    fn transform(&mut self, hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
        self.run(hook, payload)
    }

    fn observe(&mut self, hook: &str, payload: &[u8]) {
        let _ = self.run(hook, payload);
    }
}

/// Frame the host→guest stdin: a little-endian `u32` hook length, the hook name
/// (UTF-8), then the opaque payload. The length prefix keeps the payload raw
/// (JSON would force arbitrary bytes through base64). The guest parses this to
/// recover `(hook, payload)`; `js/fpui.js` does so for JavaScript plugins.
fn frame(hook: &str, payload: &[u8]) -> Vec<u8> {
    let hook = hook.as_bytes();
    let mut buf = Vec::with_capacity(4 + hook.len() + payload.len());
    // Hooks are short application-defined names, never anywhere near u32::MAX.
    buf.extend_from_slice(&(hook.len() as u32).to_le_bytes());
    buf.extend_from_slice(hook);
    buf.extend_from_slice(payload);
    buf
}

/// Read a little-endian `u32` from guest memory, or `None` if out of bounds.
fn read_u32(data: &[u8], at: usize) -> Option<u32> {
    let bytes = data.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

/// Write a little-endian `u32` into guest memory, or `None` if out of bounds.
fn write_u32(data: &mut [u8], at: usize, val: u32) -> Option<()> {
    let slot = data.get_mut(at..at.checked_add(4)?)?;
    slot.copy_from_slice(&val.to_le_bytes());
    Some(())
}

/// Write a little-endian `u64` into guest memory, or `None` if out of bounds.
fn write_u64(data: &mut [u8], at: usize, val: u64) -> Option<()> {
    let slot = data.get_mut(at..at.checked_add(8)?)?;
    slot.copy_from_slice(&val.to_le_bytes());
    Some(())
}

/// The guest's exported linear memory, if present.
fn guest_memory(caller: &Caller<JavyCtx>) -> Option<Memory> {
    caller.get_export("memory").and_then(Extern::into_memory)
}

/// Define the ten `wasi_snapshot_preview1` imports a Javy module needs, backed
/// only by the in-memory stdin/stdout pipes. Everything outside those two pipes
/// is denied, so the guest has no ambient authority to reach a real resource.
fn define_wasi(linker: &mut Linker<JavyCtx>) -> Result<(), Error> {
    const WASI: &str = "wasi_snapshot_preview1";

    // proc_exit(code): a WASI command's normal way to finish. Turn it into an
    // exit-carrying error so `run` can tell a clean exit from a trap.
    linker.func_wrap(WASI, "proc_exit", |_c: Caller<JavyCtx>, code: i32| -> Result<(), Error> {
        Err(Error::i32_exit(code))
    })?;

    // fd_read(fd, iovs, iovs_len, nread): only fd 0 (stdin) reads; it drains the
    // payload buffer into the iovecs. Every other fd is denied.
    linker.func_wrap(
        WASI,
        "fd_read",
        |mut caller: Caller<JavyCtx>, fd: i32, iovs: i32, iovs_len: i32, nread: i32| -> Result<i32, Error> {
            if fd != FD_STDIN {
                return Ok(ERRNO_BADF);
            }
            let Some(mem) = guest_memory(&caller) else { return Ok(ERRNO_BADF) };
            let (data, ctx) = mem.data_and_store_mut(&mut caller);
            let mut total: u32 = 0;
            for i in 0..(iovs_len as usize) {
                if ctx.stdin_pos >= ctx.stdin.len() {
                    break; // EOF — stdin drained
                }
                let rec = (iovs as usize).wrapping_add(i.wrapping_mul(8));
                let (Some(ptr), Some(len)) = (read_u32(data, rec), read_u32(data, rec + 4)) else {
                    return Ok(ERRNO_BADF);
                };
                let avail = ctx.stdin.len() - ctx.stdin_pos;
                let n = (len as usize).min(avail);
                let Some(dst) = data.get_mut(ptr as usize..(ptr as usize).saturating_add(n)) else {
                    return Ok(ERRNO_BADF);
                };
                dst.copy_from_slice(&ctx.stdin[ctx.stdin_pos..ctx.stdin_pos + n]);
                ctx.stdin_pos += n;
                total += n as u32;
                if n < len as usize {
                    break; // this iovec wasn't filled, so stdin is drained
                }
            }
            match write_u32(data, nread as usize, total) {
                Some(()) => Ok(ERRNO_SUCCESS),
                None => Ok(ERRNO_BADF),
            }
        },
    )?;

    // fd_write(fd, iovs, iovs_len, nwritten): fd 1 (stdout) appends to the result
    // buffer, capped at MAX_OUTPUT; fd 2 (stderr) is counted but discarded; every
    // other fd is denied.
    linker.func_wrap(
        WASI,
        "fd_write",
        |mut caller: Caller<JavyCtx>, fd: i32, iovs: i32, iovs_len: i32, nwritten: i32| -> Result<i32, Error> {
            if fd != FD_STDOUT && fd != FD_STDERR {
                return Ok(ERRNO_BADF);
            }
            let Some(mem) = guest_memory(&caller) else { return Ok(ERRNO_BADF) };
            let (data, ctx) = mem.data_and_store_mut(&mut caller);
            let mut total: u32 = 0;
            for i in 0..(iovs_len as usize) {
                let rec = (iovs as usize).wrapping_add(i.wrapping_mul(8));
                let (Some(ptr), Some(len)) = (read_u32(data, rec), read_u32(data, rec + 4)) else {
                    return Ok(ERRNO_BADF);
                };
                let (start, end) = (ptr as usize, (ptr as usize).saturating_add(len as usize));
                let Some(chunk) = data.get(start..end) else { return Ok(ERRNO_BADF) };
                if fd == FD_STDOUT {
                    let room = MAX_OUTPUT.saturating_sub(ctx.stdout.len());
                    if chunk.len() > room {
                        ctx.stdout.extend_from_slice(&chunk[..room]);
                        ctx.output_overflow = true;
                    } else {
                        ctx.stdout.extend_from_slice(chunk);
                    }
                }
                total = total.saturating_add(len);
            }
            match write_u32(data, nwritten as usize, total) {
                Some(()) => Ok(ERRNO_SUCCESS),
                None => Ok(ERRNO_BADF),
            }
        },
    )?;

    // fd_seek: stdin/stdout are pipes, so seeking is never valid.
    linker.func_wrap(
        WASI,
        "fd_seek",
        |_c: Caller<JavyCtx>, _fd: i32, _off: i64, _whence: i32, _new: i32| -> Result<i32, Error> {
            Ok(ERRNO_SPIPE)
        },
    )?;

    // fd_close: nothing to close; report success for any fd.
    linker.func_wrap(WASI, "fd_close", |_c: Caller<JavyCtx>, _fd: i32| -> Result<i32, Error> {
        Ok(ERRNO_SUCCESS)
    })?;

    // fd_fdstat_get(fd, buf): report the three standard streams as character
    // devices (24-byte fdstat, filetype in the first byte, the rest zero) so the
    // guest's stdio treats them as unbuffered pipes; deny any other fd.
    linker.func_wrap(
        WASI,
        "fd_fdstat_get",
        |mut caller: Caller<JavyCtx>, fd: i32, buf: i32| -> Result<i32, Error> {
            if fd != FD_STDIN && fd != FD_STDOUT && fd != FD_STDERR {
                return Ok(ERRNO_BADF);
            }
            let Some(mem) = guest_memory(&caller) else { return Ok(ERRNO_BADF) };
            let (data, _ctx) = mem.data_and_store_mut(&mut caller);
            let base = buf as usize;
            let Some(stat) = data.get_mut(base..base.saturating_add(24)) else { return Ok(ERRNO_BADF) };
            stat.fill(0);
            stat[0] = FILETYPE_CHARACTER_DEVICE;
            Ok(ERRNO_SUCCESS)
        },
    )?;

    // environ_sizes_get(count, size): no environment — report zero of each.
    linker.func_wrap(
        WASI,
        "environ_sizes_get",
        |mut caller: Caller<JavyCtx>, count: i32, size: i32| -> Result<i32, Error> {
            let Some(mem) = guest_memory(&caller) else { return Ok(ERRNO_BADF) };
            let (data, _ctx) = mem.data_and_store_mut(&mut caller);
            match (write_u32(data, count as usize, 0), write_u32(data, size as usize, 0)) {
                (Some(()), Some(())) => Ok(ERRNO_SUCCESS),
                _ => Ok(ERRNO_BADF),
            }
        },
    )?;

    // environ_get: no environment, so there is nothing to write.
    linker.func_wrap(
        WASI,
        "environ_get",
        |_c: Caller<JavyCtx>, _environ: i32, _buf: i32| -> Result<i32, Error> { Ok(ERRNO_SUCCESS) },
    )?;

    // clock_time_get(id, precision, time): no ambient clock — read as 0. Keeps
    // the runtime deterministic and gives the guest no real wall-clock signal.
    linker.func_wrap(
        WASI,
        "clock_time_get",
        |mut caller: Caller<JavyCtx>, _id: i32, _precision: i64, time: i32| -> Result<i32, Error> {
            let Some(mem) = guest_memory(&caller) else { return Ok(ERRNO_BADF) };
            let (data, _ctx) = mem.data_and_store_mut(&mut caller);
            match write_u64(data, time as usize, 0) {
                Some(()) => Ok(ERRNO_SUCCESS),
                None => Ok(ERRNO_BADF),
            }
        },
    )?;

    // random_get(buf, len): a dynamic plugin seeds its RNG here. There is no
    // ambient entropy to give — zero-fill the buffer. Deterministic and
    // authority-free, matching the zero clock; it only makes `Math.random()`
    // non-secure, which is harmless in a transform sandbox that holds no secrets.
    linker.func_wrap(
        WASI,
        "random_get",
        |mut caller: Caller<JavyCtx>, buf: i32, len: i32| -> Result<i32, Error> {
            let Some(mem) = guest_memory(&caller) else { return Ok(ERRNO_BADF) };
            let (data, _ctx) = mem.data_and_store_mut(&mut caller);
            let (start, end) = (buf as usize, (buf as usize).saturating_add(len as usize));
            match data.get_mut(start..end) {
                Some(slot) => {
                    slot.fill(0);
                    Ok(ERRNO_SUCCESS)
                }
                None => Ok(ERRNO_BADF),
            }
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // A hand-written WASI command (no Javy needed): read up to 4 KiB from stdin
    // into a buffer, write that many bytes back to stdout, then exit. Exercises
    // fd_read + fd_write + proc_exit and the stdin/stdout plumbing end to end.
    const ECHO_WAT: &str = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
      (memory (export "memory") 1)
      (func (export "_start")
        (local $n i32)
        ;; read iovec at [0]: base=64, len=4096
        (i32.store (i32.const 0) (i32.const 64))
        (i32.store (i32.const 4) (i32.const 4096))
        (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))
        (local.set $n (i32.load (i32.const 8))) ;; bytes read
        ;; write iovec at [16]: base=64, len=n
        (i32.store (i32.const 16) (i32.const 64))
        (i32.store (i32.const 20) (local.get $n))
        (drop (call $fd_write (i32.const 1) (i32.const 16) (i32.const 1) (i32.const 24)))
        (call $proc_exit (i32.const 0))))
    "#;

    #[test]
    fn the_shim_pipes_framed_stdin_to_stdout() {
        let wasm = wat::parse_str(ECHO_WAT).unwrap();
        let mut plugin = JavyPlugin::load(&wasm).unwrap();
        // The guest echoes stdin verbatim, so it sees — and returns — the exact
        // frame the host seeded: `[u32 hook_len][hook]["hello echo"]`.
        let out = plugin.transform("message", b"hello echo").unwrap();
        assert_eq!(out, frame("message", b"hello echo"));
    }

    // A hand-written WASI command (no Javy needed) that proves the stdin FRAME
    // carries the hook: it parses `[u32 hook_len][hook][payload]`, compares the
    // hook to "message", and uppercases the payload only on a match — otherwise
    // it echoes the payload unchanged. This is the WAT mirror of the real-Javy
    // `js/examples/uppercase-message.js` plugin.
    const DISPATCH_WAT: &str = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
      (memory (export "memory") 1)
      (data (i32.const 16) "message") ;; the hook this guest dispatches on
      (func (export "_start")
        (local $n i32) (local $hlen i32) (local $plen i32)
        (local $pstart i32) (local $i i32) (local $match i32) (local $b i32)
        ;; read the whole frame into [1024..] via one iovec {base:1024,len:4096}
        (i32.store (i32.const 0) (i32.const 1024))
        (i32.store (i32.const 4) (i32.const 4096))
        (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))
        (local.set $n (i32.load (i32.const 8)))
        ;; parse: hook_len @ [1024], hook @ [1028..], payload right after it
        (local.set $hlen (i32.load (i32.const 1024)))
        (local.set $pstart (i32.add (i32.const 1028) (local.get $hlen)))
        (local.set $plen (i32.sub (local.get $n) (i32.add (i32.const 4) (local.get $hlen))))
        ;; match = (hook_len == 7) && memcmp(hook, "message") == 0
        (local.set $match (i32.const 0))
        (if (i32.eq (local.get $hlen) (i32.const 7))
          (then
            (local.set $match (i32.const 1))
            (local.set $i (i32.const 0))
            (block $cmpdone
              (loop $cmp
                (br_if $cmpdone (i32.ge_u (local.get $i) (i32.const 7)))
                (if (i32.ne
                      (i32.load8_u (i32.add (i32.const 1028) (local.get $i)))
                      (i32.load8_u (i32.add (i32.const 16) (local.get $i))))
                  (then (local.set $match (i32.const 0)) (br $cmpdone)))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $cmp)))))
        ;; build the output at [8192..]: uppercase iff match, else copy verbatim
        (local.set $i (i32.const 0))
        (block $done
          (loop $loop
            (br_if $done (i32.ge_u (local.get $i) (local.get $plen)))
            (local.set $b (i32.load8_u (i32.add (local.get $pstart) (local.get $i))))
            (if (i32.and (local.get $match)
                  (i32.and (i32.ge_u (local.get $b) (i32.const 97))
                           (i32.le_u (local.get $b) (i32.const 122))))
              (then (local.set $b (i32.sub (local.get $b) (i32.const 32)))))
            (i32.store8 (i32.add (i32.const 8192) (local.get $i)) (local.get $b))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))
        ;; write the transformed payload from [8192..]
        (i32.store (i32.const 0) (i32.const 8192))
        (i32.store (i32.const 4) (local.get $plen))
        (drop (call $fd_write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 8)))
        (call $proc_exit (i32.const 0))))
    "#;

    #[test]
    fn the_frame_delivers_the_hook_so_the_guest_can_dispatch() {
        let wasm = wat::parse_str(DISPATCH_WAT).unwrap();
        let mut plugin = JavyPlugin::load(&wasm).unwrap();
        // Hook "message" → payload uppercased.
        assert_eq!(plugin.transform("message", b"hello").unwrap(), b"HELLO");
        // Any other hook → payload passed through unchanged. The hook crossed the
        // boundary and the guest branched on it; without the frame both would
        // transform identically.
        assert_eq!(plugin.transform("reaction", b"hello").unwrap(), b"hello");
    }

    // Exits non-zero without writing: the transform must be discarded.
    const FAIL_WAT: &str = r#"
    (module
      (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
      (memory (export "memory") 1)
      (func (export "_start") (call $proc_exit (i32.const 1))))
    "#;

    #[test]
    fn a_nonzero_exit_discards_the_transform() {
        let wasm = wat::parse_str(FAIL_WAT).unwrap();
        let mut plugin = JavyPlugin::load(&wasm).unwrap();
        assert!(plugin.transform("message", b"keep me").is_none());
    }

    // Exits cleanly but writes nothing: treated as "no change".
    const SILENT_WAT: &str = r#"
    (module
      (import "wasi_snapshot_preview1" "proc_exit" (func $proc_exit (param i32)))
      (memory (export "memory") 1)
      (func (export "_start") (call $proc_exit (i32.const 0))))
    "#;

    #[test]
    fn writing_nothing_is_no_change() {
        let wasm = wat::parse_str(SILENT_WAT).unwrap();
        let mut plugin = JavyPlugin::load(&wasm).unwrap();
        assert!(plugin.transform("message", b"keep me").is_none());
    }

    // A command that never returns: fuel must trap it and keep the payload.
    const SPIN_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "_start") (loop $l (br $l))))
    "#;

    #[test]
    fn a_runaway_command_is_trapped() {
        let wasm = wat::parse_str(SPIN_WAT).unwrap();
        let mut plugin = JavyPlugin::load_with_fuel(&wasm, 1_000_000).unwrap();
        assert!(plugin.transform("message", b"intact").is_none());
    }

    // `run` is the shared execution path: instantiate, seed the framed stdin, run
    // `_start`, collect stdout. Proving it here pins down that `observe` (a thin
    // `let _ = self.run(..)`) really executes the module, not just that it is safe.
    #[test]
    fn run_executes_the_module_and_collects_stdout() {
        let wasm = wat::parse_str(DISPATCH_WAT).unwrap();
        let mut plugin = JavyPlugin::load(&wasm).unwrap();
        assert_eq!(plugin.run("message", b"hi").unwrap(), b"HI");
    }

    // Observe runs the module like transform (same fuel/trap guarantees) but
    // reads no output. A module that exits non-zero would discard a transform;
    // observe must swallow it and return.
    #[test]
    fn observe_runs_the_module_and_swallows_failure() {
        let wasm = wat::parse_str(FAIL_WAT).unwrap();
        let mut plugin = JavyPlugin::load(&wasm).unwrap();
        plugin.observe("ready", b"anything"); // must not panic
    }

    // A runaway observer is fuel-trapped just like a transform, so a bad JS
    // observer can't hang the host.
    #[test]
    fn observe_is_fuel_bounded() {
        let wasm = wat::parse_str(SPIN_WAT).unwrap();
        let mut plugin = JavyPlugin::load_with_fuel(&wasm, 1_000_000).unwrap();
        plugin.observe("ready", b"x"); // runaway trapped, observe returns
    }

    // A JavyPlugin is just a Plugin: it drops into the capability-gated host and
    // dispatches on the granted hook like any other. The hook reaches the guest
    // through the host, which uppercases only "message".
    #[test]
    fn runs_through_the_capability_gated_host() {
        use fpui_host::Host;
        use fpui_manifest::{Capability, ExtensionKind, Manifest, API_VERSION};

        let wasm = wat::parse_str(DISPATCH_WAT).unwrap();
        let plugin = JavyPlugin::load(&wasm).unwrap();
        let cap = Capability::new("content");
        let manifest = Manifest {
            api_version: API_VERSION,
            id: "js-dispatch".into(),
            name: "JsDispatch".into(),
            version: "1.0.0".into(),
            author: "fpui".into(),
            license: "GPL-3.0-or-later".into(),
            source: "https://example.invalid/js".into(),
            kind: ExtensionKind::Plugin,
            capabilities: vec![cap.clone()],
        };
        let mut host = Host::new();
        host.register(&manifest, std::slice::from_ref(&cap), Box::new(plugin)).unwrap();
        let out = host.transform(&cap, "message", b"through the host");
        assert_eq!(out, b"THROUGH THE HOST");
    }

    // A javy-free stand-in for the shared QuickJS plugin: it exports `memory`,
    // `cabi_realloc`, and `invoke` — the exact surface a real `javy emit-plugin`
    // module exposes to a dynamic module — and imports only the shim's stdio.
    // `invoke` echoes the framed stdin to stdout on this (shared) memory, standing
    // in for "run the embedded JS".
    const FAKE_PLUGIN_WAT: &str = r#"
    (module
      (import "wasi_snapshot_preview1" "fd_read"
        (func $fd_read (param i32 i32 i32 i32) (result i32)))
      (import "wasi_snapshot_preview1" "fd_write"
        (func $fd_write (param i32 i32 i32 i32) (result i32)))
      (memory (export "memory") 1)
      (global $heap (mut i32) (i32.const 2048))
      (func (export "cabi_realloc")
        (param $ptr i32) (param $old i32) (param $align i32) (param $new i32) (result i32)
        (local $p i32)
        (local.set $p (global.get $heap))
        (global.set $heap (i32.add (global.get $heap) (local.get $new)))
        (local.get $p))
      ;; invoke(a,b,c,d,e): read stdin into [64..], write it back to stdout.
      (func (export "invoke") (param i32 i32 i32 i32 i32)
        (local $n i32)
        (i32.store (i32.const 0) (i32.const 64))
        (i32.store (i32.const 4) (i32.const 4096))
        (drop (call $fd_read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 8)))
        (local.set $n (i32.load (i32.const 8)))
        (i32.store (i32.const 16) (i32.const 64))
        (i32.store (i32.const 20) (local.get $n))
        (drop (call $fd_write (i32.const 1) (i32.const 16) (i32.const 1) (i32.const 24)))))
    "#;

    // A javy-free stand-in for a dynamically-linked module: it declares no memory
    // of its own, imports the shared plugin's `memory`/`cabi_realloc`/`invoke`
    // under a NON-default namespace (so the host can't be hard-coding the real
    // "javy-default-plugin-v4"), and its `_start` just drives `invoke`.
    const FAKE_DYN_MODULE_WAT: &str = r#"
    (module
      (import "fpui-test-plugin" "memory" (memory 0))
      (import "fpui-test-plugin" "cabi_realloc"
        (func $realloc (param i32 i32 i32 i32) (result i32)))
      (import "fpui-test-plugin" "invoke"
        (func $invoke (param i32 i32 i32 i32 i32)))
      (func (export "_start")
        (call $invoke (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 0) (i32.const 0))))
    "#;

    #[test]
    fn a_dynamic_module_runs_on_the_shared_engine() {
        let runtime = JavyRuntime::new(&wat::parse_str(FAKE_PLUGIN_WAT).unwrap()).unwrap();
        let mut js = runtime.load(&wat::parse_str(FAKE_DYN_MODULE_WAT).unwrap()).unwrap();
        // The tiny module shares the plugin's memory and calls its `invoke`, which
        // echoes the framed stdin — so the host gets the whole frame back. Proves
        // the shared-engine hosting chain: instantiate the plugin, wire its memory
        // and funcs into the module, run `_start`, read stdout.
        let out = js.transform("message", b"dyn hi").unwrap();
        assert_eq!(out, frame("message", b"dyn hi"));
    }

    #[test]
    fn the_runtime_hosts_many_modules_from_one_shared_plugin() {
        // The footprint win: one compiled plugin, many tiny modules.
        let runtime = JavyRuntime::new(&wat::parse_str(FAKE_PLUGIN_WAT).unwrap()).unwrap();
        let mut a = runtime.load(&wat::parse_str(FAKE_DYN_MODULE_WAT).unwrap()).unwrap();
        let mut b = runtime.load(&wat::parse_str(FAKE_DYN_MODULE_WAT).unwrap()).unwrap();
        assert_eq!(a.transform("x", b"one").unwrap(), frame("x", b"one"));
        assert_eq!(b.transform("y", b"two").unwrap(), frame("y", b"two"));
    }

    #[test]
    fn the_runtime_rejects_a_non_module() {
        let runtime = JavyRuntime::new(&wat::parse_str(FAKE_PLUGIN_WAT).unwrap()).unwrap();
        assert!(matches!(runtime.load(b"not wasm"), Err(WasmError::Compile(_))));
    }
}
