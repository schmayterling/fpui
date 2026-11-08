//! Write a fpui plugin and compile it to WebAssembly.
//!
//! fpui runs plugins as wasm modules (a native client has no JS engine to
//! inject into). This crate is the guest side: it hides the raw host↔plugin ABI
//! behind a small [`plugin!`] macro, modelled on Vencord's `definePlugin`, so a
//! plugin author only writes the hook they care about:
//!
//! ```ignore
//! fpui_plugin_sdk::plugin! {
//!     transform_message: |content: &str| content.replace("secret", "[redacted]"),
//! }
//! ```
//!
//! Compile that crate for `wasm32-unknown-unknown` as a `cdylib` and the result
//! loads straight into [`fpui_wasm`](https://docs.rs). The macro emits the
//! two exports the host calls — `fpui_alloc` and `fpui_transform_message`
//! — using the helpers below.

/// Reserve `len` bytes in the plugin's own memory and return the offset. The
/// host calls this, writes the input message there, then calls the transform.
///
/// The buffer is handed to the host and reclaimed by [`run_transform`], so it is
/// deliberately leaked here.
pub fn alloc(len: i32) -> i32 {
    let mut buf = vec![0u8; len.max(0) as usize];
    let ptr = buf.as_mut_ptr() as i32;
    core::mem::forget(buf);
    ptr
}

/// The glue behind `plugin!`'s message hook: read the input the host wrote, run
/// the author's `transform`, and return the result packed as
/// `(out_ptr as u64) << 32 | out_len as u64`.
///
/// Ceiling: the output buffer is leaked so the host can read it after this
/// returns, so a plugin's memory grows slowly over very many messages until it
/// hits wasmi's cap, after which transforms no-op. A later ABI revision adds an
/// explicit free; for now a plugin is cheap to reload.
pub fn run_transform<F: FnOnce(&str) -> String>(ptr: i32, len: i32, transform: F) -> i64 {
    let len = len.max(0) as usize;
    // SAFETY: `ptr`/`len` name the buffer the host got from `alloc` and filled
    // with the message before calling us; we take it back to free it.
    let input = unsafe { Vec::from_raw_parts(ptr as *mut u8, len, len) };
    let text = String::from_utf8_lossy(&input).into_owned();
    drop(input);

    let mut out = transform(&text).into_bytes();
    let out_ptr = out.as_mut_ptr() as u32;
    let out_len = out.len() as u32;
    core::mem::forget(out);
    (((out_ptr as u64) << 32) | out_len as u64) as i64
}

/// The glue behind `plugin!`'s `on_event` hook: read the versioned-JSON event
/// the host wrote and hand it to the author's handler. Events are observe-only,
/// so the handler returns nothing.
pub fn run_event<F: FnOnce(&str)>(ptr: i32, len: i32, handler: F) {
    let len = len.max(0) as usize;
    // SAFETY: `ptr`/`len` name the buffer the host got from `alloc` and filled
    // with the event JSON before calling us; we take it back to free it.
    let input = unsafe { Vec::from_raw_parts(ptr as *mut u8, len, len) };
    let json = String::from_utf8_lossy(&input).into_owned();
    drop(input);
    handler(&json);
}

/// Define a fpui plugin. List the hooks you want; the macro emits the wasm
/// exports fpui's host calls. Each hook takes the input `&str` and returns
/// the rewritten `String`.
///
/// Hooks: `transform_message` (an incoming message) and `transform_draft` (the
/// user's outgoing draft). An unknown hook name is a compile error.
///
/// ```ignore
/// fpui_plugin_sdk::plugin! {
///     transform_message: |content| content.to_uppercase(),
///     transform_draft: |draft| format!("{draft} — via fpui"),
/// }
/// ```
#[macro_export]
macro_rules! plugin {
    ( $( $hook:ident : $f:expr ),+ $(,)? ) => {
        /// Reserve guest memory for the host to write a hook's input into.
        #[no_mangle]
        pub extern "C" fn fpui_alloc(len: i32) -> i32 {
            $crate::alloc(len)
        }
        $( $crate::__fpui_hook!($hook, $f); )+
    };
}

/// Emit the wasm export for one named hook. Matching a hook name as a literal
/// token means an unknown name simply has no rule, so it fails to compile.
#[macro_export]
#[doc(hidden)]
macro_rules! __fpui_hook {
    (transform_message, $f:expr) => {
        /// Transform one message: the host wrote the UTF-8 text at `ptr..ptr+len`.
        #[no_mangle]
        pub extern "C" fn fpui_transform_message(ptr: i32, len: i32) -> i64 {
            $crate::run_transform(ptr, len, $f)
        }
    };
    (transform_draft, $f:expr) => {
        /// Transform the outgoing draft: the host wrote the UTF-8 text at `ptr..ptr+len`.
        #[no_mangle]
        pub extern "C" fn fpui_transform_draft(ptr: i32, len: i32) -> i64 {
            $crate::run_transform(ptr, len, $f)
        }
    };
    (on_event, $f:expr) => {
        /// Observe an event: the host wrote the versioned-JSON event at `ptr..ptr+len`.
        /// The handler takes the JSON `&str` and returns nothing.
        #[no_mangle]
        pub extern "C" fn fpui_on_event(ptr: i32, len: i32) {
            $crate::run_event(ptr, len, $f)
        }
    };
}
