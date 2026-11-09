//! fpui-discover: locate a patch site in a client binary and emit a
//! signature-equipped adapter entry for it.
//!
//! This is the generator side of the signature scheme. Point it at a symbol, on
//! a build that still has them such as a debug GPUI app, or at a raw offset on a
//! stripped one. It extracts the shortest unique byte window around the site so
//! the loader can relocate that site after the build drifts, and prints the
//! entry as JSON you can drop into an adapter's `sites` (or `events`).
//!
//! ```text
//! fpui-discover <binary> --symbol <name> [--token <t>] [--len <n>] [--max-radius <n>]
//! fpui-discover <binary> --offset <n>    [--token <t>] [--len <n>] [--max-radius <n>]
//! ```
//!
//! `--offset` accepts decimal or `0x` hex. `--len` is the site's byte length (one
//! aarch64 instruction is 4; x86 varies). On a symbol, the site is the function
//! entry, which is what a backend/event hook detours; an intra-function color
//! site is better found by `--offset`.

use std::process::ExitCode;

use fpui_loader::{extract_signature_across, force_return_aarch64};
use object::{Object, ObjectSection, ObjectSymbol};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = {
        let mut out = Vec::new();
        let mut skip = false;
        for a in &args {
            if std::mem::take(&mut skip) {
                continue;
            }
            match a.as_str() {
                "--symbol" | "--offset" | "--anchor" | "--token" | "--len" | "--max-radius"
                | "--also" | "--force-return" => skip = true,
                s if s.starts_with("--") => {}
                _ => out.push(a),
            }
        }
        out
    };

    let [binary] = positional.as_slice() else {
        eprintln!(
            "usage: fpui-discover <binary> (--symbol <name> | --offset <n>) \
             [--token <t>] [--len <n>] [--max-radius <n>]"
        );
        return ExitCode::from(2);
    };

    let token = flag(&args, "--token").unwrap_or("token");
    // `--force-return <v>` emits a byte patch that overwrites the function entry
    // with `mov w0, #v; ret`, forcing it to return `v` (fpui's "make it
    // return true/false" patch). That patch is exactly eight bytes.
    let force_return: Option<u16> = flag(&args, "--force-return").and_then(|s| s.parse().ok());
    let len: usize = if force_return.is_some() {
        8
    } else {
        flag(&args, "--len").map_or(Ok(4), str::parse).unwrap_or(4)
    };
    let max_radius: usize = flag(&args, "--max-radius")
        .map_or(Ok(64), str::parse)
        .unwrap_or(64);

    let image = match std::fs::read(binary) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("fpui-discover: read {binary}: {e}");
            return ExitCode::from(2);
        }
    };

    let site = match site_offset(&image, &args) {
        Ok(off) => off,
        Err(e) => {
            eprintln!("fpui-discover: {e}");
            return ExitCode::from(2);
        }
    };

    let start = site as usize;
    let Some(expected) = image.get(start..start.saturating_add(len)) else {
        eprintln!("fpui-discover: site {site:#x}+{len} is past the end of the file");
        return ExitCode::from(2);
    };
    let expected = expected.to_vec();

    // Extra builds the signature must also be unique in, so it survives drift
    // instead of only matching the one build it was taken from.
    let mut images = vec![image];
    for path in flags_all(&args, "--also") {
        match std::fs::read(path) {
            Ok(bytes) => images.push(bytes),
            Err(e) => {
                eprintln!("fpui-discover: read {path}: {e}");
                return ExitCode::from(2);
            }
        }
    }
    let image_refs: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();

    match extract_signature_across(&image_refs, site, len, max_radius) {
        Some(sig) => {
            eprintln!(
                "found site at {site:#x}; unique {}-byte signature (site {} in) across {} build(s)",
                sig.bytes.len(),
                sig.site_offset,
                images.len()
            );
            let replacement = match force_return {
                Some(v) => format!(
                    r#","replacement_bytes":"{}""#,
                    hex(&force_return_aarch64(v))
                ),
                None => String::new(),
            };
            println!(
                "{{\"token\":{token:?},\"image_offset\":{site},\"expected_bytes\":\"{}\"{replacement},\
                 \"signature\":{{\"bytes\":\"{}\",\"site_offset\":{}}}}}",
                hex(&expected),
                hex(&sig.bytes),
                sig.site_offset
            );
            ExitCode::SUCCESS
        }
        None => {
            eprintln!(
                "fpui-discover: no unique signature within {max_radius} bytes of {site:#x}; \
                 the code there is too generic (try a larger --max-radius)"
            );
            ExitCode::FAILURE
        }
    }
}

/// Resolve the site's file offset from either `--offset` or `--symbol`.
fn site_offset(image: &[u8], args: &[String]) -> Result<u64, String> {
    if let Some(raw) = flag(args, "--offset") {
        return parse_u64(raw).ok_or_else(|| format!("bad --offset `{raw}`"));
    }
    // Anchor by a stable string the target function references: find the string,
    // find the one `adrp`+`add` that forms its address, and pin to the function
    // that holds it. This survives recompiles where a byte signature would not,
    // and needs no symbols — the stripped-binary case.
    if let Some(text) = flag(args, "--anchor") {
        let anchor = fpui_loader::anchor::resolve_anchor(image, text)
            .map_err(|e| format!("anchor `{text}`: {e}"))?;
        eprintln!(
            "anchored to function at {:#x} via string {:#x} referenced from {:#x}",
            anchor.func_offset, anchor.string_vmaddr, anchor.xref_offset
        );
        return Ok(anchor.func_offset);
    }
    let name =
        flag(args, "--symbol").ok_or("need --symbol <name>, --offset <n>, or --anchor <s>")?;
    let file = object::File::parse(image).map_err(|e| format!("parse object: {e}"))?;
    symbol_file_offset(&file, name)
        .ok_or_else(|| format!("symbol `{name}` not found (is this build stripped?)"))
}

/// The file offset of a named symbol: find the symbol's address, then the
/// section that holds it, and map the address through that section's file range.
fn symbol_file_offset(file: &object::File, name: &str) -> Option<u64> {
    let sym = file.symbols().chain(file.dynamic_symbols()).find(|s| {
        s.name()
            .is_ok_and(|n| n == name || n.trim_start_matches('_') == name)
    })?;
    let addr = sym.address();
    let section = file.sections().find(|s| {
        let base = s.address();
        s.size() > 0 && addr >= base && addr < base + s.size()
    })?;
    let (file_start, _) = section.file_range()?;
    Some(file_start + (addr - section.address()))
}

/// Value after a `--name` flag, if present.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).map(String::as_str)
}

/// Every value passed after a repeated `--name` flag.
fn flags_all<'a>(args: &'a [String], name: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            if let Some(v) = args.get(i + 1) {
                out.push(v.as_str());
                i += 2;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Parse a decimal or `0x`-prefixed hex integer.
fn parse_u64(s: &str) -> Option<u64> {
    match s.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
