//! Resolve several string anchors against a binary in a single `__text` pass,
//! and report the timing against resolving them one at a time. The scan is the
//! cost, so batching N anchors into one pass is what makes string anchoring
//! affordable at load time.
//!
//! Usage: `cargo run --release -p maycord-loader --example resolve_anchors -- <binary> <string>...`

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let binary = args.next().expect("usage: resolve_anchors <binary> <string>...");
    let needles: Vec<String> = args.collect();
    assert!(!needles.is_empty(), "give at least one string to anchor");
    let refs: Vec<&str> = needles.iter().map(String::as_str).collect();
    let image = std::fs::read(&binary).expect("read binary");

    let t = Instant::now();
    let batched = maycord_loader::anchor::resolve_anchors(&image, &refs).expect("parse image");
    let batched_ms = t.elapsed().as_millis();

    let t = Instant::now();
    for n in &refs {
        let _ = maycord_loader::anchor::resolve_anchor(&image, n);
    }
    let per_ms = t.elapsed().as_millis();

    for (n, r) in refs.iter().zip(&batched) {
        match r {
            Ok(a) => println!("ok    func {:#010x}  <- {n}", a.func_offset),
            Err(e) => println!("skip  {e:<42}  <- {n}"),
        }
    }
    eprintln!(
        "\n{} anchors: batched {} ms (one scan) vs per-anchor {} ms ({} scans) = {:.1}x",
        refs.len(),
        batched_ms,
        per_ms,
        refs.len(),
        per_ms as f64 / batched_ms.max(1) as f64,
    );
}
