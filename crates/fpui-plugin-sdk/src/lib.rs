//! Write a fpui plugin and compile it to WebAssembly.
//!
//! fpui runs plugins as wasm modules (a native app has no JS engine to inject
//! into). This crate is the guest side: it hides the raw host↔plugin ABI behind
//! a small [`plugin!`] macro, modelled on Vencord's `definePlugin`. fpui is
//! general — a *hook* is just an application-defined name (`"message"`,
//! `"draft"`, … defined by the app, e.g. maycord) carrying an opaque byte
//! payload — so a plugin dispatches on the hook name:
//!
//! ```ignore
//! fpui_plugin_sdk::plugin! {
//!     transform: |hook: &str, payload: &[u8]| match hook {
//!         "message" => Some(String::from_utf8_lossy(payload).to_uppercase().into_bytes()),
//!         _ => None,
//!     },
//! }
//! ```
//!
//! Compile for `wasm32-unknown-unknown` as a `cdylib` and the result loads into
//! fpui's wasm host. The macro emits `fpui_alloc`, `fpui_transform`, and/or
//! `fpui_observe` using the helpers below.

/// Reserve `len` bytes in the plugin's own memory and return the offset. The
/// host calls this to write a hook's input (name and payload) before the hook.
/// The buffer is handed to the host and reclaimed by the hook glue, so it is
/// deliberately leaked here.
pub fn alloc(len: i32) -> i32 {
    let mut buf = vec![0u8; len.max(0) as usize];
    let ptr = buf.as_mut_ptr() as i32;
    core::mem::forget(buf);
    ptr
}

/// Take back a buffer the host allocated via [`alloc`] and filled, as an owned
/// `Vec` that frees on drop.
///
/// # Safety
/// `ptr`/`len` must name a buffer previously returned by [`alloc`] and not yet
/// reclaimed — exactly what the host passes to a hook.
unsafe fn take(ptr: i32, len: i32) -> Vec<u8> {
    Vec::from_raw_parts(ptr as *mut u8, len.max(0) as usize, len.max(0) as usize)
}

/// The glue behind `plugin!`'s `transform`: read the hook name + payload the host
/// wrote, run the author's dispatcher, and pack the result as
/// `(out_ptr as u64) << 32 | out_len`. Returns `0` for `None`, which the host
/// reads as "no change" and keeps the original payload.
///
/// Ceiling: the output buffer is leaked so the host can read it after this
/// returns, so a plugin's memory grows slowly over very many calls until wasmi's
/// cap, after which transforms no-op. A later ABI revision adds an explicit free.
pub fn run_transform<F: FnOnce(&str, &[u8]) -> Option<Vec<u8>>>(
    hook_ptr: i32,
    hook_len: i32,
    in_ptr: i32,
    in_len: i32,
    dispatch: F,
) -> i64 {
    // SAFETY: the host alloc'd + filled both buffers before calling us.
    let hook = unsafe { take(hook_ptr, hook_len) };
    let payload = unsafe { take(in_ptr, in_len) };
    let hook = String::from_utf8_lossy(&hook).into_owned();
    match dispatch(&hook, &payload) {
        Some(mut out) => {
            let out_ptr = out.as_mut_ptr() as u32;
            let out_len = out.len() as u32;
            core::mem::forget(out);
            (((out_ptr as u64) << 32) | out_len as u64) as i64
        }
        None => 0,
    }
}

/// The glue behind `plugin!`'s `observe`: read the hook name + payload and hand
/// them to the author's dispatcher. Observe hooks return nothing.
pub fn run_observe<F: FnOnce(&str, &[u8])>(
    hook_ptr: i32,
    hook_len: i32,
    in_ptr: i32,
    in_len: i32,
    dispatch: F,
) {
    // SAFETY: the host alloc'd + filled both buffers before calling us.
    let hook = unsafe { take(hook_ptr, hook_len) };
    let payload = unsafe { take(in_ptr, in_len) };
    let hook = String::from_utf8_lossy(&hook).into_owned();
    dispatch(&hook, &payload);
}

/// Define a fpui plugin. Provide a `transform` dispatcher, an `observe`
/// dispatcher, or both; each takes the hook name and the opaque payload. The
/// macro emits the wasm exports fpui's host calls.
///
/// ```ignore
/// fpui_plugin_sdk::plugin! {
///     transform: |hook, payload| match hook {
///         "message" => Some(payload.to_ascii_uppercase()),
///         _ => None,
///     },
///     observe: |hook, _payload| { let _ = hook; },
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
        $( $crate::__fpui_export!($hook, $f); )+
    };
}

/// Emit the wasm export for one hook kind. Matching the kind as a literal token
/// means an unknown kind simply has no rule, so it fails to compile.
#[macro_export]
#[doc(hidden)]
macro_rules! __fpui_export {
    (transform, $f:expr) => {
        /// Transform dispatch: the host wrote the hook name and payload.
        #[no_mangle]
        pub extern "C" fn fpui_transform(
            hook_ptr: i32,
            hook_len: i32,
            in_ptr: i32,
            in_len: i32,
        ) -> i64 {
            $crate::run_transform(hook_ptr, hook_len, in_ptr, in_len, $f)
        }
    };
    (observe, $f:expr) => {
        /// Observe dispatch: the host wrote the hook name and payload.
        #[no_mangle]
        pub extern "C" fn fpui_observe(hook_ptr: i32, hook_len: i32, in_ptr: i32, in_len: i32) {
            $crate::run_observe(hook_ptr, hook_len, in_ptr, in_len, $f)
        }
    };
}
