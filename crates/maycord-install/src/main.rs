//! Wrap a macOS `.app` so it auto-loads the maycord agent on a normal launch,
//! and unwrap it again.
//!
//! The wrap edits the app's `Contents/Info.plist`, adding the environment the
//! agent needs (notably `DYLD_INSERT_LIBRARIES`) under `LSEnvironment`, then
//! re-signs the bundle ad-hoc so macOS will load the change. Re-signing replaces
//! the app's original signature and notarization, so the user approves the app
//! once in Gatekeeper; after that, opening the app normally loads maycord.
//!
//! ```text
//! maycord-install wrap  /Applications/Example.app DYLD_INSERT_LIBRARIES=/path/agent.dylib MAYCORD_ADAPTER=/path/a.json
//! maycord-install unwrap /Applications/Example.app DYLD_INSERT_LIBRARIES MAYCORD_ADAPTER
//! ```

use std::path::Path;
use std::process::Command;

const USAGE: &str =
    "usage: maycord-install wrap <App.app> KEY=VALUE...  |  unwrap <App.app> KEY...";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("wrap") => run_wrap(&args[1..]),
        Some("unwrap") => run_unwrap(&args[1..]),
        _ => Err(USAGE.to_string()),
    };
    if let Err(e) = result {
        eprintln!("maycord-install: {e}");
        std::process::exit(1);
    }
}

fn run_wrap(args: &[String]) -> Result<(), String> {
    let app = Path::new(args.first().ok_or(USAGE)?);
    let mut vars = Vec::new();
    for pair in &args[1..] {
        let (k, v) = pair.split_once('=').ok_or(format!("expected KEY=VALUE, got `{pair}`"))?;
        vars.push((k.to_string(), v.to_string()));
    }
    if vars.is_empty() {
        return Err("wrap needs at least one KEY=VALUE (e.g. DYLD_INSERT_LIBRARIES=...)".into());
    }
    wrap(app, &vars)?;
    eprintln!("wrapped {} (open it once to clear the Gatekeeper prompt)", app.display());
    Ok(())
}

fn run_unwrap(args: &[String]) -> Result<(), String> {
    let app = Path::new(args.first().ok_or(USAGE)?);
    let keys: Vec<String> = args[1..].to_vec();
    if keys.is_empty() {
        return Err("unwrap needs the KEY names to remove".into());
    }
    unwrap_app(app, &keys)?;
    eprintln!("unwrapped {}", app.display());
    Ok(())
}

fn info_plist(app: &Path) -> std::path::PathBuf {
    app.join("Contents/Info.plist")
}

fn wrap(app: &Path, vars: &[(String, String)]) -> Result<(), String> {
    let path = info_plist(app);
    let mut value = plist::Value::from_file(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    add_env(&mut value, vars)?;
    value.to_file_xml(&path).map_err(|e| format!("write {}: {e}", path.display()))?;
    resign(app)
}

fn unwrap_app(app: &Path, keys: &[String]) -> Result<(), String> {
    let path = info_plist(app);
    let mut value = plist::Value::from_file(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    remove_env(&mut value, keys)?;
    value.to_file_xml(&path).map_err(|e| format!("write {}: {e}", path.display()))?;
    resign(app)
}

/// Add each `KEY=VALUE` to the plist's `LSEnvironment` dictionary, creating it
/// if absent.
fn add_env(value: &mut plist::Value, vars: &[(String, String)]) -> Result<(), String> {
    let dict = value.as_dictionary_mut().ok_or("Info.plist is not a dictionary")?;
    if !dict.contains_key("LSEnvironment") {
        dict.insert("LSEnvironment".into(), plist::Value::Dictionary(plist::Dictionary::new()));
    }
    let env = dict
        .get_mut("LSEnvironment")
        .and_then(|v| v.as_dictionary_mut())
        .ok_or("LSEnvironment is not a dictionary")?;
    for (k, v) in vars {
        env.insert(k.clone(), plist::Value::String(v.clone()));
    }
    Ok(())
}

/// Remove each key from `LSEnvironment`, and drop `LSEnvironment` itself if it
/// ends up empty, restoring the plist to its original shape.
fn remove_env(value: &mut plist::Value, keys: &[String]) -> Result<(), String> {
    let dict = value.as_dictionary_mut().ok_or("Info.plist is not a dictionary")?;
    if let Some(env) = dict.get_mut("LSEnvironment").and_then(|v| v.as_dictionary_mut()) {
        for k in keys {
            env.remove(k);
        }
        if env.is_empty() {
            dict.remove("LSEnvironment");
        }
    }
    Ok(())
}

/// Ad-hoc re-sign the bundle so macOS loads the edited Info.plist. This strips
/// the app's hardened runtime, which is what lets `DYLD_INSERT_LIBRARIES` take
/// effect. (Keeping hardened is also possible by re-signing with the
/// `allow-dyld-environment-variables` and `disable-library-validation`
/// entitlements; ad-hoc is the simpler default.)
fn resign(app: &Path) -> Result<(), String> {
    let status = Command::new("codesign")
        .args(["--force", "--sign", "-"])
        .arg(app)
        .status()
        .map_err(|e| format!("run codesign: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("codesign failed ({status})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_plist() -> plist::Value {
        let mut d = plist::Dictionary::new();
        d.insert("CFBundleExecutable".into(), plist::Value::String("App".into()));
        plist::Value::Dictionary(d)
    }

    #[test]
    fn wrap_then_unwrap_restores_the_plist() {
        let mut v = app_plist();
        add_env(
            &mut v,
            &[("DYLD_INSERT_LIBRARIES".into(), "/x/agent.dylib".into())],
        )
        .unwrap();
        let env = v.as_dictionary().unwrap()["LSEnvironment"].as_dictionary().unwrap();
        assert_eq!(env["DYLD_INSERT_LIBRARIES"].as_string().unwrap(), "/x/agent.dylib");

        remove_env(&mut v, &["DYLD_INSERT_LIBRARIES".into()]).unwrap();
        // Emptied LSEnvironment is dropped, so the plist matches where it started.
        assert!(v.as_dictionary().unwrap().get("LSEnvironment").is_none());
        assert!(v.as_dictionary().unwrap().contains_key("CFBundleExecutable"));
    }
}
