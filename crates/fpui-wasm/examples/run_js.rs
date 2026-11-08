//! Load a Javy-compiled JavaScript plugin and run one transform, from the CLI:
//!   cargo run -p fpui-wasm --features js --example run_js -- <plugin.wasm> [payload] [hook]
//!
//! Build `<plugin.wasm>` with Javy, e.g. `javy build uppercase.js -o uppercase.wasm`.
//! The host frames `[hook][payload]` onto the plugin's stdin (see `javy.rs`); the
//! plugin's stdout is printed as the result. `hook` defaults to "message", so a
//! hook-dispatching plugin can be probed with different hooks to prove it only
//! transforms the ones it handles. Prints the original payload on no change.
use fpui_host::Plugin;
use fpui_wasm::JavyPlugin;

fn main() {
    let mut args = std::env::args().skip(1);
    let wasm = args.next().expect("usage: run_js <plugin.wasm> [payload] [hook]");
    let payload = args.next().unwrap_or_else(|| "hello".to_string());
    let hook = args.next().unwrap_or_else(|| "message".to_string());

    let bytes = std::fs::read(&wasm).expect("read the wasm plugin");
    let mut plugin = JavyPlugin::load(&bytes).expect("load the plugin");

    match plugin.transform(&hook, payload.as_bytes()) {
        Some(out) => println!("{}", String::from_utf8_lossy(&out)),
        None => println!("{payload}"),
    }
}
