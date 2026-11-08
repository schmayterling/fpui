//! A sandboxed WebAssembly host that runs a wasm module as a
//! [`maycord_host::Plugin`].
//!
//! A native chat client has no JavaScript engine to inject into, so maycord
//! plugins are WebAssembly instead: a plugin author compiles to wasm from any
//! language (Rust, AssemblyScript, Zig, …), and the module runs here under
//! wasmi. wasmi is an interpreter, so it adds no new executable memory to the
//! target, and each call gets a fuel budget. If the guest crashes, runs out of
//! fuel, or breaks the protocol, the message is left as-is instead of taking the
//! client down: a misbehaving mod must never break the app.
//!
//! Capability enforcement is the host's job, not this crate's: [`maycord_host`]
//! only calls a hook the plugin was granted, so a wasm plugin without
//! [`MessageContent`](maycord_host::Capability::MessageContent) is simply never
//! asked to transform a message. When this crate later exposes host *imports*
//! (storage, …), each import will check the grant on every call.
//!
//! ## The guest ABI (v1)
//! A plugin module exports:
//! * `memory` — its linear memory.
//! * `maycord_alloc(len: i32) -> i32` — reserve `len` bytes, return the offset.
//! * `maycord_transform_message(ptr: i32, len: i32) -> i64` — given the UTF-8
//!   message at `ptr..ptr+len`, return the rewritten message packed as
//!   `(out_ptr as u64) << 32 | out_len as u64`.
//!
//! The host allocates the input with `maycord_alloc`, writes it, calls the
//! transform, and reads the result back out of the guest's memory.

use maycord_host::{Message, Plugin};
use wasmi::{Config, Engine, Linker, Memory, Module, Store, TypedFunc};

/// Fuel spent per hook call before the guest is trapped. Generous for a text
/// transform, small enough that a runaway loop dies in milliseconds.
pub const DEFAULT_FUEL: u64 = 50_000_000;

/// The largest output a guest may return from a transform, capping memory a
/// malicious module can make the host allocate.
const MAX_OUTPUT: usize = 16 * 1024 * 1024;

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
    store: Store<()>,
    memory: Memory,
    alloc: Option<TypedFunc<i32, i32>>,
    transform_message: Option<TypedFunc<(i32, i32), i64>>,
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
        let mut store = Store::new(&engine, ());
        // An initial budget so the start function (if any) cannot run forever.
        store.set_fuel(fuel_per_call).ok();
        let linker = Linker::new(&engine);
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .map_err(|e| WasmError::Instantiate(e.to_string()))?;

        let memory = instance
            .get_memory(&store, "memory")
            .ok_or(WasmError::MissingExport("memory"))?;
        // alloc + transform_message are an optional pair: a plugin that only
        // observes events (a later hook) needs neither.
        let alloc = instance.get_typed_func::<i32, i32>(&store, "maycord_alloc").ok();
        let transform_message = instance
            .get_typed_func::<(i32, i32), i64>(&store, "maycord_transform_message")
            .ok();

        Ok(Self {
            store,
            memory,
            alloc,
            transform_message,
            fuel_per_call,
        })
    }

    /// Run the guest transform over `input`. Returns `None` on any trap,
    /// out-of-fuel, or protocol error, so the caller keeps the original text.
    fn run_transform(&mut self, input: &str) -> Option<String> {
        let (alloc, transform) = (self.alloc?, self.transform_message?);
        let memory = self.memory;

        // Fresh budget for this call.
        self.store.set_fuel(self.fuel_per_call).ok()?;

        let bytes = input.as_bytes();
        let len = i32::try_from(bytes.len()).ok()?;

        let in_ptr = alloc.call(&mut self.store, len).ok()?;
        if in_ptr < 0 {
            return None;
        }
        memory.write(&mut self.store, in_ptr as usize, bytes).ok()?;

        let packed = transform.call(&mut self.store, (in_ptr, len)).ok()? as u64;
        let out_ptr = (packed >> 32) as usize;
        let out_len = (packed & 0xffff_ffff) as usize;
        if out_len > MAX_OUTPUT {
            return None;
        }

        let mut buf = vec![0u8; out_len];
        memory.read(&self.store, out_ptr, &mut buf).ok()?;
        String::from_utf8(buf).ok()
    }
}

impl Plugin for WasmPlugin {
    fn transform_message(&mut self, message: &mut Message) {
        if self.transform_message.is_none() {
            return;
        }
        if let Some(rewritten) = self.run_transform(&message.content) {
            message.content = rewritten;
        }
        // On None the original content stands: a failed mod changes nothing.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maycord_host::Host;
    use maycord_manifest::{Capability, ExtensionKind, Manifest, API_VERSION};

    // A guest that allocates an output buffer and ASCII-uppercases the input
    // into it, returning (out_ptr << 32 | len). Exercises the whole ABI.
    const UPPERCASE_WAT: &str = r#"
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

    // A guest whose transform never returns: proves fuel keeps the host safe.
    const SPIN_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "maycord_alloc") (param i32) (result i32) (i32.const 1024))
      (func (export "maycord_transform_message") (param i32 i32) (result i64)
        (loop $l (br $l))
        (i64.const 0)))
    "#;

    fn manifest(caps: &[Capability]) -> Manifest {
        Manifest {
            api_version: API_VERSION,
            id: "wasm-plugin".into(),
            name: "Wasm".into(),
            version: "1.0.0".into(),
            author: "maycord".into(),
            license: "GPL-3.0-or-later".into(),
            source: "https://example.invalid/w".into(),
            kind: ExtensionKind::Plugin,
            capabilities: caps.to_vec(),
        }
    }

    #[test]
    fn transform_rewrites_the_message() {
        let wasm = wat::parse_str(UPPERCASE_WAT).unwrap();
        let mut plugin = WasmPlugin::load(&wasm).unwrap();
        let mut m = Message { author: "a".into(), content: "hello Wasm".into() };
        plugin.transform_message(&mut m);
        assert_eq!(m.content, "HELLO WASM");
    }

    #[test]
    fn runs_through_the_capability_gated_host() {
        let wasm = wat::parse_str(UPPERCASE_WAT).unwrap();
        let plugin = WasmPlugin::load(&wasm).unwrap();
        let mut host = Host::new();
        host.register(
            &manifest(&[Capability::MessageContent]),
            &[Capability::MessageContent],
            Box::new(plugin),
        )
        .unwrap();
        assert_eq!(host.render_message("alice", "ship it"), "SHIP IT");
    }

    #[test]
    fn without_the_capability_the_host_never_calls_the_guest() {
        let wasm = wat::parse_str(UPPERCASE_WAT).unwrap();
        let plugin = WasmPlugin::load(&wasm).unwrap();
        let mut host = Host::new();
        // Granted only Events, so the message-content hook is never invoked.
        host.register(&manifest(&[Capability::Events]), &[Capability::Events], Box::new(plugin)).unwrap();
        assert_eq!(host.render_message("alice", "ship it"), "ship it");
    }

    #[test]
    fn a_runaway_guest_is_trapped_and_the_message_is_kept() {
        let wasm = wat::parse_str(SPIN_WAT).unwrap();
        let mut plugin = WasmPlugin::load_with_fuel(&wasm, 1_000_000).unwrap();
        let mut m = Message { author: "a".into(), content: "intact".into() };
        plugin.transform_message(&mut m); // burns fuel, traps, returns None
        assert_eq!(m.content, "intact");
    }

    #[test]
    fn a_non_module_is_rejected() {
        assert!(matches!(WasmPlugin::load(b"not wasm"), Err(WasmError::Compile(_))));
    }
}
