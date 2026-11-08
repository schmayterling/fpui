//! Load a DYNAMICALLY-linked Javy JS plugin against a shared engine, from the CLI:
//!   cargo run -p fpui-wasm --features js --example run_js_dyn -- <plugin.wasm> <module.wasm> [payload] [hook]
//!
//! `<plugin.wasm>` is the shared engine from `javy emit-plugin -o plugin.wasm`.
//! `<module.wasm>` is a tiny module from `javy build -C dynamic -C plugin=plugin.wasm in.js`.
//! The engine is hosted once; the module (1-16 KiB) imports it. Same framed
//! stdin and stdout-as-result as `run_js`.
use fpui_host::Plugin;
use fpui_wasm::JavyRuntime;

fn main() {
    let mut args = std::env::args().skip(1);
    let plugin = args.next().expect("usage: run_js_dyn <plugin.wasm> <module.wasm> [payload] [hook]");
    let module = args.next().expect("usage: run_js_dyn <plugin.wasm> <module.wasm> [payload] [hook]");
    let payload = args.next().unwrap_or_else(|| "hello".to_string());
    let hook = args.next().unwrap_or_else(|| "message".to_string());

    let runtime = JavyRuntime::new(&std::fs::read(&plugin).expect("read the shared plugin")).expect("load the plugin");
    let mut js = runtime.load(&std::fs::read(&module).expect("read the module")).expect("load the module");

    match js.transform(&hook, payload.as_bytes()) {
        Some(out) => println!("{}", String::from_utf8_lossy(&out)),
        None => println!("{payload}"),
    }
}
