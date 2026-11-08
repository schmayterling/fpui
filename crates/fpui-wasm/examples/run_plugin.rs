//! Load a wasm plugin and run its message transform once, from the CLI:
//!   cargo run --example run_plugin -- <plugin.wasm> "a message"
use fpui_host::{Message, Plugin};
use fpui_wasm::WasmPlugin;

fn main() {
    let mut args = std::env::args().skip(1);
    let wasm = args.next().expect("usage: run_plugin <plugin.wasm> [message]");
    let message = args.next().unwrap_or_else(|| "hello from fpui".to_string());

    let bytes = std::fs::read(&wasm).expect("read the wasm plugin");
    let mut plugin = WasmPlugin::load(&bytes).expect("load the plugin");

    let mut m = Message { author: "demo".into(), content: message };
    plugin.transform_message(&mut m);
    println!("{}", m.content);
}
