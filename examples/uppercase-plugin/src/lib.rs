//! An example fpui plugin: ASCII-uppercases the payload of the `message` hook,
//! ignoring any other hook. Build for wasm:
//!   cargo build --release --target wasm32-unknown-unknown
fpui_plugin_sdk::plugin! {
    transform: |hook: &str, payload: &[u8]| match hook {
        "message" => Some(String::from_utf8_lossy(payload).to_uppercase().into_bytes()),
        _ => None,
    },
}
