//! Load a Javy-compiled JavaScript plugin and run one transform, from the CLI:
//!   cargo run -p fpui-wasm --features js --example run_js -- <plugin.wasm> [payload]
//!
//! Build `<plugin.wasm>` with Javy, e.g. `javy build uppercase.js -o uppercase.wasm`.
//! The payload is sent to the plugin on stdin; its stdout is printed as the
//! result. Prints the original payload if the plugin makes no change.
use fpui_host::Plugin;
use fpui_wasm::JavyPlugin;

fn main() {
    let mut args = std::env::args().skip(1);
    let wasm = args.next().expect("usage: run_js <plugin.wasm> [payload]");
    let payload = args.next().unwrap_or_else(|| "hello".to_string());

    let bytes = std::fs::read(&wasm).expect("read the wasm plugin");
    let mut plugin = JavyPlugin::load(&bytes).expect("load the plugin");

    match plugin.transform("message", payload.as_bytes()) {
        Some(out) => println!("{}", String::from_utf8_lossy(&out)),
        None => println!("{payload}"),
    }
}
