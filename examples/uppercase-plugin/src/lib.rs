//! An example maycord plugin: it uppercases every message.
//!
//! Build it for wasm, then load the result in maycord:
//!   cargo build --release --target wasm32-unknown-unknown
//!   # -> target/wasm32-unknown-unknown/release/uppercase_plugin.wasm
maycord_plugin_sdk::plugin! {
    transform_message: |content: &str| content.to_uppercase(),
}
