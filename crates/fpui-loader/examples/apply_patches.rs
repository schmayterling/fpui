//! Apply an adapter's byte patches to a client binary offline, writing a patched
//! copy. This exercises the byte-patch capability (text injection, function
//! neutering) without injecting into a live process, which is handy for testing
//! and for preparing a patched copy to launch.
//!
//!   cargo run -p fpui-loader --example apply_patches -- <adapter.json> <in> <out>

use fpui_loader::apply_byte_patches;
use fpui_target::TargetAdapter;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [adapter_path, in_path, out_path] = args.as_slice() else {
        eprintln!("usage: apply_patches <adapter.json> <in-binary> <out-binary>");
        std::process::exit(2);
    };
    let adapter: TargetAdapter =
        serde_json::from_slice(&std::fs::read(adapter_path).expect("read adapter"))
            .expect("parse adapter");
    let mut image = std::fs::read(in_path).expect("read input binary");
    let mut applied = 0;
    for (token, ok) in apply_byte_patches(&adapter, &mut image) {
        println!("  {token}: {}", if ok { applied += 1; "applied" } else { "SKIPPED (not found)" });
    }
    std::fs::write(out_path, &image).expect("write output binary");
    println!("applied {applied} patch(es); wrote {out_path}");
}
