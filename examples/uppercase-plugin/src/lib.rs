//! An example maycord plugin: it uppercases incoming messages and tags the
//! user's outgoing draft. Shows both transform hooks.
//!
//! Build it for wasm, then load the result in maycord:
//!   cargo build --release --target wasm32-unknown-unknown
//!   # -> target/wasm32-unknown-unknown/release/uppercase_plugin.wasm
maycord_plugin_sdk::plugin! {
    transform_message: |content: &str| content.to_uppercase(),
    transform_draft: |draft: &str| format!("{draft} — via maycord"),
}
