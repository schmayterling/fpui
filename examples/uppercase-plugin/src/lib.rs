//! An example maycord plugin showing all three hooks: it uppercases incoming
//! messages, tags the outgoing draft, and observes lifecycle events.
//!
//! Build it for wasm, then load the result in maycord:
//!   cargo build --release --target wasm32-unknown-unknown
//!   # -> target/wasm32-unknown-unknown/release/uppercase_plugin.wasm
maycord_plugin_sdk::plugin! {
    transform_message: |content: &str| content.to_uppercase(),
    transform_draft: |draft: &str| format!("{draft} — via maycord"),
    // `event` is the versioned JSON, e.g. {"v":1,"type":"ready"}. A real plugin
    // would react here; this example just accepts them.
    on_event: |_event: &str| {},
}
