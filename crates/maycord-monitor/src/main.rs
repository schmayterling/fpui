//! The maycord patch monitor.
//!
//! Point it at an adapter and a client binary and it reports, without running
//! anything, whether that binary is the exact build the adapter was pinned to
//! and which of the adapter's patches still hold against it. Because it checks
//! each patch on its own, it works against any version: run it after a client
//! update, or in continuous integration against each new release, to learn the
//! moment a patch stops matching. It exits non-zero when any patch has broken,
//! so a watcher can treat a failing run as the signal to re-examine that patch.
//!
//! ```text
//! maycord-monitor <adapter.json> <client-binary> [--json]
//! ```

use std::process::ExitCode;

use maycord_loader::{check_patches, PatchReport};
use maycord_target::TargetAdapter;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let as_json = args.iter().any(|a| a == "--json");

    let (adapter_path, binary_path) = match positional.as_slice() {
        [a, b] => (a.as_str(), b.as_str()),
        _ => {
            eprintln!("usage: maycord-monitor <adapter.json> <client-binary> [--json]");
            return ExitCode::from(2);
        }
    };

    let report = match run(adapter_path, binary_path) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("maycord-monitor: {e}");
            return ExitCode::from(2);
        }
    };

    if as_json {
        print_json(&report);
    } else {
        print_text(&report);
    }
    if report.all_supported() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run(adapter_path: &str, binary_path: &str) -> Result<PatchReport, String> {
    let adapter: TargetAdapter = serde_json::from_slice(
        &std::fs::read(adapter_path).map_err(|e| format!("read {adapter_path}: {e}"))?,
    )
    .map_err(|e| format!("parse {adapter_path}: {e}"))?;
    let build = std::fs::read(binary_path).map_err(|e| format!("read {binary_path}: {e}"))?;
    Ok(check_patches(&adapter, &build))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn print_text(report: &PatchReport) {
    println!("maycord patch monitor");
    println!("  build sha256:       {}", hex(&report.sha256));
    println!(
        "  exact pinned build: {}",
        if report.identity_matches { "yes" } else { "no (a different version)" }
    );
    println!("  patches:            {}/{} supported", report.supported(), report.patches.len());
    println!();
    for p in &report.patches {
        let tag = if p.supported { " ok  " } else { "BROKEN" };
        let note = if p.supported { "" } else { "   <- no longer matches" };
        println!("  [{tag}]  {:<20} @ {:#010x}{note}", p.label, p.image_offset);
    }
    println!();
    let broken = report.patches.len() - report.supported();
    if broken == 0 {
        println!("all patches supported on this build.");
    } else {
        println!("{broken} patch(es) no longer work on this build.");
    }
}

fn print_json(report: &PatchReport) {
    let patches: Vec<String> = report
        .patches
        .iter()
        .map(|p| {
            format!(
                r#"{{"label":{:?},"offset":{},"supported":{}}}"#,
                p.label, p.image_offset, p.supported
            )
        })
        .collect();
    println!(
        r#"{{"sha256":"{}","identity_matches":{},"supported":{},"total":{},"patches":[{}]}}"#,
        hex(&report.sha256),
        report.identity_matches,
        report.supported(),
        report.patches.len(),
        patches.join(",")
    );
}
