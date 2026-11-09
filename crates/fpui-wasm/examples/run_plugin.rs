//! Load a wasm plugin and run a transform hook once, from the CLI:
//!   cargo run --example run_plugin -- <plugin.wasm> [hook] [payload]
use fpui_host::Plugin;
use fpui_wasm::WasmPlugin;

fn main() {
    let mut args = std::env::args().skip(1);
    let wasm = args
        .next()
        .expect("usage: run_plugin <plugin.wasm> [hook] [payload]");
    let hook = args.next().unwrap_or_else(|| "message".to_string());
    let payload = args.next().unwrap_or_else(|| "hello from fpui".to_string());

    let bytes = std::fs::read(&wasm).expect("read the wasm plugin");
    let mut plugin = WasmPlugin::load(&bytes).expect("load the plugin");

    match plugin.transform(&hook, payload.as_bytes()) {
        Some(out) => println!("{}", String::from_utf8_lossy(&out)),
        None => println!("{payload}"),
    }
}
