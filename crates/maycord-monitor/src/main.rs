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

/// The command line after parsing: two paths, the output flag, and the optional
/// version label.
struct Cli<'a> {
    adapter: &'a str,
    binary: &'a str,
    as_json: bool,
    label: Option<&'a str>,
}

/// Parse the arguments, or return `None` when the two positional paths aren't
/// both there. `--label <name>` consumes the following value; any other `--`
/// token is a bare flag, and everything else is a positional path.
fn parse(args: &[String]) -> Option<Cli<'_>> {
    let as_json = args.iter().any(|a| a == "--json");
    let label = flag_value(args, "--label");
    let mut positional = Vec::new();
    let mut skip = false;
    for a in args {
        if std::mem::take(&mut skip) {
            continue;
        }
        match a.as_str() {
            "--label" => skip = true,
            s if s.starts_with("--") => {}
            s => positional.push(s),
        }
    }
    match positional.as_slice() {
        [adapter, binary] => Some(Cli { adapter, binary, as_json, label }),
        _ => None,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cli) = parse(&args) else {
        eprintln!("usage: maycord-monitor <adapter.json> <client-binary> [--json] [--label <name>]");
        return ExitCode::from(2);
    };

    let report = match run(cli.adapter, cli.binary) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("maycord-monitor: {e}");
            return ExitCode::from(2);
        }
    };

    if cli.as_json {
        print_json(&report, cli.label);
    } else {
        print_text(&report, cli.label);
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

/// Value following a `--name` flag, if the flag is present with one.
fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).map(String::as_str)
}

/// Seconds since the Unix epoch, so each record says when it was observed.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn print_text(report: &PatchReport, label: Option<&str>) {
    println!("maycord patch monitor");
    if let Some(label) = label {
        println!("  version label:      {label}");
    }
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

fn print_json(report: &PatchReport, label: Option<&str>) {
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
    let label_json = label.map_or_else(|| "null".to_string(), |l| format!("{l:?}"));
    println!(
        r#"{{"label":{},"checked_at":{},"sha256":"{}","identity_matches":{},"supported":{},"total":{},"patches":[{}]}}"#,
        label_json,
        now_secs(),
        hex(&report.sha256),
        report.identity_matches,
        report.supported(),
        report.patches.len(),
        patches.join(",")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_reads_paths_the_json_flag_and_the_label_value() {
        // The label value must not be mistaken for a positional path, and order
        // of flags and paths should not matter.
        let a = args(&["--json", "adapter.json", "--label", "v0.42", "client"]);
        let cli = parse(&a).expect("two paths present");
        assert_eq!(cli.adapter, "adapter.json");
        assert_eq!(cli.binary, "client");
        assert!(cli.as_json);
        assert_eq!(cli.label, Some("v0.42"));
    }

    #[test]
    fn parse_rejects_the_wrong_number_of_paths() {
        assert!(parse(&args(&["only-one.json"])).is_none());
        assert!(parse(&args(&["a", "b", "c"])).is_none());
    }
}
