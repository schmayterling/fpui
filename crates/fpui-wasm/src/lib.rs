//! A sandboxed WebAssembly host that runs a wasm module as a
//! [`fpui_host::Plugin`].
//!
//! A native GPUI app has no JavaScript engine to inject into, so fpui plugins
//! are WebAssembly instead: a plugin author compiles to wasm from any language
//! (Rust, AssemblyScript, Zig, …), and the module runs here under wasmi. wasmi
//! is an interpreter, so it adds no new executable memory to the target, and
//! each call gets a fuel budget. If the guest crashes, runs out of fuel, or
//! breaks the protocol, the payload is left as-is instead of taking the client
//! down: a misbehaving mod must never break the app.
//!
//! Each plugin's linear memory is capped as well (see [`MAX_MEMORY_BYTES`]): a
//! module that merely *declares* a huge initial memory is refused at load time,
//! before it can allocate it. Fuel can't guard that reservation, since fuel is
//! only spent once code is running.
//!
//! Capability enforcement is the host's job, not this crate's: [`fpui_host`]
//! only invokes a hook the plugin was granted the matching capability for, so a
//! wasm plugin without that grant is simply never asked to run the hook. When
//! this crate later exposes host *imports* (storage, …), each import will check
//! the grant on every call.
//!
//! ## The guest ABI (v1)
//! A plugin module exports:
//! * `memory` — its linear memory.
//! * `fpui_alloc(len: i32) -> i32` — reserve `len` bytes, return the offset.
//! * `fpui_transform(hook_ptr, hook_len, in_ptr, in_len) -> i64` — given the
//!   application-defined hook name and the opaque payload, return the rewritten
//!   payload packed as `(out_ptr as u64) << 32 | out_len as u64`.
//! * `fpui_observe(hook_ptr, hook_len, in_ptr, in_len)` — observe a payload for a
//!   hook; returns nothing.
//!
//! The host allocates the hook name and the payload with `fpui_alloc`, writes
//! them, calls the hook, and (for a transform) reads the result back out of the
//! guest's memory. Both hooks are optional; a plugin exports only what it
//! implements. fpui never interprets the hook name or the payload — that is the
//! application's job.
//!
//! ## JavaScript plugins (the `js` feature)
//! With `--features js`, [`JavyPlugin`] runs a plugin authored in JavaScript. It
//! is a [Javy](https://github.com/bytecodealliance/javy)-produced module —
//! QuickJS compiled *to* wasm — so the JS engine is wasm data interpreted by the
//! same wasmi host: no new executable memory, the sandbox intact. It speaks
//! Javy's WASI-command ABI (payload on stdin, result on stdout) behind a
//! minimal, in-memory-only shim, rather than the ptr/len ABI above. The feature
//! is off by default so the plain-wasm host stays lean (it pulls in wasmi's
//! `simd` support, which QuickJS-wasm needs).

use fpui_host::Plugin;
use wasmi::{
    Config, Engine, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc,
};

#[cfg(feature = "js")]
mod javy;
#[cfg(feature = "js")]
pub use javy::{DynamicJavyPlugin, JavyPlugin, JavyRuntime};

/// Fuel spent per hook call before the guest is trapped. Generous for a text
/// transform, small enough that a runaway loop dies in milliseconds.
pub const DEFAULT_FUEL: u64 = 50_000_000;

/// The largest output a guest may return from a transform, capping memory a
/// malicious module can make the host allocate.
pub(crate) const MAX_OUTPUT: usize = 16 * 1024 * 1024;

/// The ceiling on one plugin's linear memory. wasmi consults this on the initial
/// reservation *and* every `memory.grow`, so a module that merely declares a
/// huge memory is refused when it loads instead of OOM-ing the target — fuel
/// can't help there, as it is only spent once code runs. Sized well above
/// [`MAX_OUTPUT`]: a transform building up to a 16 MiB result holds it in this
/// same linear memory, alongside its input and working copies. Roomy for real
/// guests (a QuickJS/Javy engine settles around ~1.5 MiB) yet orders of
/// magnitude below the multi-GiB reservation a hostile module would ask for.
pub(crate) const MAX_MEMORY_BYTES: usize = 64 * 1024 * 1024;

/// Why a wasm module could not be loaded as a plugin.
#[derive(Debug)]
pub enum WasmError {
    /// The bytes were not a valid wasm module for this engine.
    Compile(String),
    /// Instantiation (or the module's start function) failed.
    Instantiate(String),
    /// A required export was missing or had the wrong type.
    MissingExport(&'static str),
}

impl core::fmt::Display for WasmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "invalid wasm module: {e}"),
            Self::Instantiate(e) => write!(f, "failed to instantiate plugin: {e}"),
            Self::MissingExport(name) => write!(f, "plugin is missing export `{name}`"),
        }
    }
}

impl std::error::Error for WasmError {}

/// A loaded wasm plugin. Holds its own [`Store`] (and therefore its own linear
/// memory and fuel), so plugins never share state.
pub struct WasmPlugin {
    store: Store<StoreLimits>,
    memory: Memory,
    alloc: Option<TypedFunc<i32, i32>>,
    transform: Option<TypedFunc<(i32, i32, i32, i32), i64>>,
    observe: Option<TypedFunc<(i32, i32, i32, i32), ()>>,
    fuel_per_call: u64,
}

impl WasmPlugin {
    /// Load a plugin from wasm bytes with the [`DEFAULT_FUEL`] budget.
    pub fn load(wasm: &[u8]) -> Result<Self, WasmError> {
        Self::load_with_fuel(wasm, DEFAULT_FUEL)
    }

    /// Load a plugin with an explicit per-call fuel budget.
    pub fn load_with_fuel(wasm: &[u8], fuel_per_call: u64) -> Result<Self, WasmError> {
        let mut config = Config::default();
        config.consume_fuel(true);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, wasm).map_err(|e| WasmError::Compile(e.to_string()))?;
        let limits = StoreLimitsBuilder::new()
            .memory_size(MAX_MEMORY_BYTES)
            .build();
        let mut store = Store::new(&engine, limits);
        // Cap linear memory before instantiation, which is where the module's
        // declared memory is reserved: an over-large request is denied here
        // rather than allocated and left for fuel (which never runs) to catch.
        store.limiter(|limits| limits);
        // An initial budget so the start function (if any) cannot run forever.
        store.set_fuel(fuel_per_call).ok();
        let linker = Linker::new(&engine);
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(|e| WasmError::Instantiate(e.to_string()))?;

        let memory = instance
            .get_memory(&store, "memory")
            .ok_or(WasmError::MissingExport("memory"))?;
        // alloc pairs with either hook. A plugin that implements neither hook is
        // inert but still valid.
        let alloc = instance
            .get_typed_func::<i32, i32>(&store, "fpui_alloc")
            .ok();
        let transform = instance
            .get_typed_func::<(i32, i32, i32, i32), i64>(&store, "fpui_transform")
            .ok();
        let observe = instance
            .get_typed_func::<(i32, i32, i32, i32), ()>(&store, "fpui_observe")
            .ok();

        Ok(Self {
            store,
            memory,
            alloc,
            transform,
            observe,
            fuel_per_call,
        })
    }

    /// Allocate `bytes.len()` in the guest and copy `bytes` in, returning
    /// `(ptr, len)`. `None` on any trap, out-of-fuel, or a bad pointer.
    fn write_bytes(&mut self, alloc: TypedFunc<i32, i32>, bytes: &[u8]) -> Option<(i32, i32)> {
        let len = i32::try_from(bytes.len()).ok()?;
        let ptr = alloc.call(&mut self.store, len).ok()?;
        if ptr < 0 {
            return None;
        }
        self.memory
            .write(&mut self.store, ptr as usize, bytes)
            .ok()?;
        Some((ptr, len))
    }

    /// Run the guest's transform for `hook` over `payload`. Returns `None` on any
    /// trap, out-of-fuel, or protocol error, so the caller keeps the original.
    fn run_transform(
        &mut self,
        func: TypedFunc<(i32, i32, i32, i32), i64>,
        hook: &str,
        payload: &[u8],
    ) -> Option<Vec<u8>> {
        let alloc = self.alloc?;
        let memory = self.memory;

        // Fresh budget for this call, covering the allocs and the transform.
        self.store.set_fuel(self.fuel_per_call).ok()?;

        let (hook_ptr, hook_len) = self.write_bytes(alloc, hook.as_bytes())?;
        let (in_ptr, in_len) = self.write_bytes(alloc, payload)?;

        let packed = func
            .call(&mut self.store, (hook_ptr, hook_len, in_ptr, in_len))
            .ok()? as u64;
        // `0` is the guest's "no change" sentinel (e.g. no handler matched the
        // hook); keep the original rather than reading an empty null buffer.
        if packed == 0 {
            return None;
        }
        let out_ptr = (packed >> 32) as usize;
        let out_len = (packed & 0xffff_ffff) as usize;
        if out_len > MAX_OUTPUT {
            return None;
        }

        let mut buf = vec![0u8; out_len];
        memory.read(&self.store, out_ptr, &mut buf).ok()?;
        Some(buf)
    }

    /// Hand `hook` + `payload` to the guest's observe hook. `None` on any trap,
    /// out-of-fuel, or protocol error; an observation the guest fails is dropped.
    fn run_observe(
        &mut self,
        func: TypedFunc<(i32, i32, i32, i32), ()>,
        hook: &str,
        payload: &[u8],
    ) -> Option<()> {
        let alloc = self.alloc?;
        self.store.set_fuel(self.fuel_per_call).ok()?;
        let (hook_ptr, hook_len) = self.write_bytes(alloc, hook.as_bytes())?;
        let (in_ptr, in_len) = self.write_bytes(alloc, payload)?;
        func.call(&mut self.store, (hook_ptr, hook_len, in_ptr, in_len))
            .ok()?;
        Some(())
    }
}

impl Plugin for WasmPlugin {
    fn transform(&mut self, hook: &str, payload: &[u8]) -> Option<Vec<u8>> {
        let func = self.transform?;
        self.run_transform(func, hook, payload)
        // On None the original payload stands: a failed mod changes nothing.
    }

    fn observe(&mut self, hook: &str, payload: &[u8]) {
        if let Some(func) = self.observe {
            let _ = self.run_observe(func, hook, payload);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fpui_host::Host;
    use fpui_manifest::{Capability, ExtensionKind, Manifest, API_VERSION};

    // A guest that ASCII-uppercases the payload into a fresh buffer and returns
    // (out_ptr << 32 | len). Ignores the hook name. Exercises the whole ABI.
    const UPPERCASE_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (global $heap (mut i32) (i32.const 1024))
      (func $alloc (export "fpui_alloc") (param $n i32) (result i32)
        (local $p i32)
        (local.set $p (global.get $heap))
        (global.set $heap (i32.add (global.get $heap) (local.get $n)))
        (local.get $p))
      (func (export "fpui_transform")
        (param $hptr i32) (param $hlen i32) (param $ptr i32) (param $len i32) (result i64)
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

    // Echoes the hook name back as the output, proving the hook is delivered.
    const ECHO_HOOK_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (global $heap (mut i32) (i32.const 1024))
      (func $alloc (export "fpui_alloc") (param $n i32) (result i32)
        (local $p i32)
        (local.set $p (global.get $heap))
        (global.set $heap (i32.add (global.get $heap) (local.get $n)))
        (local.get $p))
      (func (export "fpui_transform")
        (param $hptr i32) (param $hlen i32) (param $ptr i32) (param $len i32) (result i64)
        (i64.or
          (i64.shl (i64.extend_i32_u (local.get $hptr)) (i64.const 32))
          (i64.extend_i32_u (local.get $hlen)))))
    "#;

    // A guest whose transform never returns: proves fuel keeps the host safe.
    const SPIN_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "fpui_alloc") (param i32) (result i32) (i32.const 1024))
      (func (export "fpui_transform") (param i32 i32 i32 i32) (result i64)
        (loop $l (br $l))
        (i64.const 0)))
    "#;

    // Records the observed payload length at memory[0] so a test can see it ran.
    const OBSERVE_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "fpui_alloc") (param $n i32) (result i32) (i32.const 1024))
      (func (export "fpui_observe")
        (param $hptr i32) (param $hlen i32) (param $ptr i32) (param $len i32)
        (i32.store (i32.const 0) (local.get $len))))
    "#;

    fn content() -> Capability {
        Capability::new("content")
    }
    fn events() -> Capability {
        Capability::new("events")
    }

    fn manifest(caps: &[Capability]) -> Manifest {
        Manifest {
            api_version: API_VERSION,
            id: "wasm-plugin".into(),
            name: "Wasm".into(),
            version: "1.0.0".into(),
            author: "fpui".into(),
            license: "GPL-3.0-or-later".into(),
            source: "https://example.invalid/w".into(),
            kind: ExtensionKind::Plugin,
            capabilities: caps.to_vec(),
        }
    }

    #[test]
    fn transform_rewrites_the_payload() {
        let wasm = wat::parse_str(UPPERCASE_WAT).unwrap();
        let mut plugin = WasmPlugin::load(&wasm).unwrap();
        let out = plugin.transform("message", b"hello Wasm").unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "HELLO WASM");
    }

    #[test]
    fn the_guest_receives_the_hook_name() {
        let wasm = wat::parse_str(ECHO_HOOK_WAT).unwrap();
        let mut plugin = WasmPlugin::load(&wasm).unwrap();
        let out = plugin.transform("pingpong", b"ignored").unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "pingpong");
    }

    #[test]
    fn runs_through_the_capability_gated_host() {
        let wasm = wat::parse_str(UPPERCASE_WAT).unwrap();
        let plugin = WasmPlugin::load(&wasm).unwrap();
        let mut host = Host::new();
        host.register(&manifest(&[content()]), &[content()], Box::new(plugin))
            .unwrap();
        let out = host.transform(&content(), "message", b"ship it");
        assert_eq!(String::from_utf8(out).unwrap(), "SHIP IT");
    }

    #[test]
    fn without_the_capability_the_host_never_calls_the_guest() {
        let wasm = wat::parse_str(UPPERCASE_WAT).unwrap();
        let plugin = WasmPlugin::load(&wasm).unwrap();
        let mut host = Host::new();
        // Granted only `events`, so a transform under `content` never runs.
        host.register(&manifest(&[events()]), &[events()], Box::new(plugin))
            .unwrap();
        let out = host.transform(&content(), "message", b"ship it");
        assert_eq!(String::from_utf8(out).unwrap(), "ship it");
    }

    #[test]
    fn a_runaway_guest_is_trapped_and_the_payload_is_kept() {
        let wasm = wat::parse_str(SPIN_WAT).unwrap();
        let mut plugin = WasmPlugin::load_with_fuel(&wasm, 1_000_000).unwrap();
        // Burns fuel, traps, returns None.
        assert!(plugin.transform("message", b"intact").is_none());
    }

    #[test]
    fn a_non_module_is_rejected() {
        assert!(matches!(
            WasmPlugin::load(b"not wasm"),
            Err(WasmError::Compile(_))
        ));
    }

    // Declares a ~3 GiB initial linear memory (50000 pages × 64 KiB). Valid wasm,
    // but far past MAX_MEMORY_BYTES.
    const HUGE_MEMORY_WAT: &str = r#"
    (module
      (memory (export "memory") 50000)
      (func (export "fpui_alloc") (param i32) (result i32) (i32.const 0)))
    "#;

    #[test]
    fn a_module_demanding_huge_memory_is_rejected_not_allocated() {
        let wasm = wat::parse_str(HUGE_MEMORY_WAT).unwrap();
        // The limiter denies the initial reservation, so instantiation fails
        // instead of the host trying to back ~3 GiB of guest memory.
        assert!(matches!(
            WasmPlugin::load(&wasm),
            Err(WasmError::Instantiate(_))
        ));
    }

    #[test]
    fn observe_delivers_the_payload_to_the_guest() {
        let wasm = wat::parse_str(OBSERVE_WAT).unwrap();
        let mut plugin = WasmPlugin::load(&wasm).unwrap();
        plugin.observe("ready", b"a payload");
        // The guest recorded the payload's byte length at memory[0]; read it back.
        let mut buf = [0u8; 4];
        plugin.memory.read(&plugin.store, 0, &mut buf).unwrap();
        assert_eq!(i32::from_le_bytes(buf) as usize, "a payload".len());
    }

    #[test]
    fn host_observe_reaches_the_wasm_hook() {
        let wasm = wat::parse_str(OBSERVE_WAT).unwrap();
        let plugin = WasmPlugin::load(&wasm).unwrap();
        let mut host = Host::new();
        host.register(&manifest(&[events()]), &[events()], Box::new(plugin))
            .unwrap();
        host.observe(&events(), "ready", b"hi"); // Events-gated path delivers without panic.
        assert_eq!(host.len(), 1);
    }

    // A guest whose transform returns 0 signals "no change".
    const NOCHANGE_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "fpui_alloc") (param i32) (result i32) (i32.const 1024))
      (func (export "fpui_transform") (param i32 i32 i32 i32) (result i64) (i64.const 0)))
    "#;

    #[test]
    fn a_zero_return_means_no_change() {
        let wasm = wat::parse_str(NOCHANGE_WAT).unwrap();
        let mut plugin = WasmPlugin::load(&wasm).unwrap();
        assert!(plugin.transform("message", b"keep me").is_none());
    }
}
