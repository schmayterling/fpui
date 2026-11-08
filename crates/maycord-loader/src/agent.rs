//! The in-process agent: the code that runs *inside* the target once the loader
//! has forced this crate's dylib in. It reads the adapter and theme the loader
//! handed over, finds where the target was mapped, confirms the pinned
//! instruction bytes are really present, and then applies the planned writes.
//!
//! Everything here up to [`apply`] is verification: it reads, compares, and
//! refuses, so it is safe to run even against the wrong process. [`apply`] is
//! the only step that rewrites the target's execution; its backend (a frida-gum
//! instruction probe) and the per-build site offsets are the remaining work,
//! tracked in `the maycord research notes`.

use std::io;

use maycord_target::TargetAdapter;
use maycord_theme::{ThemePackage, Variant};

use crate::{plan_writes, PlannedWrite};

fn invalid(e: impl ToString) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

/// Read and validate the adapter the loader passed by path in `MAYCORD_ADAPTER`.
pub fn adapter_from_env() -> io::Result<TargetAdapter> {
    let path = std::env::var_os("MAYCORD_ADAPTER")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "MAYCORD_ADAPTER is not set"))?;
    let adapter: TargetAdapter = serde_json::from_slice(&std::fs::read(path)?).map_err(invalid)?;
    adapter.validate().map_err(invalid)?;
    Ok(adapter)
}

/// Read and parse the theme package the loader passed by path in `MAYCORD_THEME`.
pub fn theme_from_env() -> io::Result<ThemePackage> {
    let path = std::env::var_os("MAYCORD_THEME")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "MAYCORD_THEME is not set"))?;
    maycord_theme::parse_package(&std::fs::read(path)?).map_err(invalid)
}

/// Pick which variant to apply: the one named by `MAYCORD_VARIANT` (`dark` or
/// `light`), else dark, else light. Errors if no usable variant is present.
pub fn pick_variant(package: &ThemePackage) -> io::Result<&Variant> {
    let want_light = matches!(std::env::var("MAYCORD_VARIANT").as_deref(), Ok("light"));
    let (first, second) = if want_light {
        (&package.theme.light, &package.theme.dark)
    } else {
        (&package.theme.dark, &package.theme.light)
    };
    first
        .as_ref()
        .or(second.as_ref())
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "theme has no usable variant"))
}

/// A site whose pinned bytes were not present: the running build is not the one
/// the adapter was authored against, so the agent must not write.
#[derive(Clone, Debug)]
pub struct SiteMismatch {
    pub token: String,
    pub addr: u64,
}

impl core::fmt::Display for SiteMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "site `{}` at {:#x} does not hold its pinned bytes",
            self.token, self.addr
        )
    }
}

/// Confirm every site still holds its pinned instruction bytes before any write.
/// Running in-process, each site's runtime address is directly readable.
pub fn verify_sites(plan: &[PlannedWrite]) -> Result<(), SiteMismatch> {
    for write in plan {
        // SAFETY: `runtime_addr` is this image's verified load base plus an
        // offset from an identity-checked adapter; we read exactly
        // `expected_bytes.len()` bytes to compare and never write here. A wrong
        // address is the failure this check exists to catch, not to cause — it
        // is only reached after the loader's full-file identity gate passed.
        let actual = unsafe {
            std::slice::from_raw_parts(write.runtime_addr as *const u8, write.expected_bytes.len())
        };
        if actual != write.expected_bytes.as_slice() {
            return Err(SiteMismatch {
                token: write.token.clone(),
                addr: write.runtime_addr,
            });
        }
    }
    Ok(())
}

/// Install the color writes. This is the one step that rewrites the target's
/// execution and is not yet wired.
pub fn apply(_plan: &[PlannedWrite]) -> io::Result<()> {
    // ponytail: the ceiling here is the hook backend, not the surrounding flow.
    // The plan, color encoding, identity gate and byte verification are real and
    // tested; what remains is a frida-gum `Interceptor::attach_instruction` at
    // each site that writes the encoded color into its register/stack slot, plus
    // the reverse engineering that fills a real adapter's offsets. Upgrade path
    // and recipe: the maycord research notes.
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "frida apply backend not yet wired",
    ))
}

/// The load base of the main executable image (its preferred base plus the ASLR
/// slide). Image index 0 is always the main executable.
#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc prefers the mach2 crate's binding; the dyld API itself is stable.
pub fn main_image_base() -> u64 {
    // SAFETY: image 0 (the main executable) always exists; the dyld image API
    // returns its mach_header pointer, which sits at the image's load base.
    unsafe { libc::_dyld_get_image_header(0) as u64 }
}

/// Run the whole agent flow once: load adapter + theme, locate the image, plan,
/// verify, apply. Called from the agent dylib's module-init constructor.
#[cfg(target_os = "macos")]
pub fn run() -> io::Result<()> {
    let adapter = adapter_from_env()?;
    let package = theme_from_env()?;
    let variant = pick_variant(&package)?;
    let plan = plan_writes(&adapter, variant, main_image_base());
    verify_sites(&plan).map_err(invalid)?;
    apply(&plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EncodedColor;
    use maycord_target::ValueLocation;

    fn write_at(bytes: &[u8]) -> PlannedWrite {
        PlannedWrite {
            token: "sidebar".into(),
            runtime_addr: bytes.as_ptr() as u64,
            expected_bytes: bytes.to_vec(),
            location: ValueLocation::General(0),
            color: EncodedColor::U32(0),
        }
    }

    #[test]
    fn verify_sites_reads_the_live_bytes_it_is_pointed_at() {
        // Point a site at this test's own buffer: the pinned bytes match, so
        // verification passes; flip the expectation and it must fail.
        let buffer = [0xde_u8, 0xad, 0xbe, 0xef];
        let ok = write_at(&buffer);
        assert!(verify_sites(std::slice::from_ref(&ok)).is_ok());

        let mut mismatch = ok.clone();
        mismatch.expected_bytes = vec![0, 0, 0, 0];
        let err = verify_sites(std::slice::from_ref(&mismatch)).unwrap_err();
        assert_eq!(err.token, "sidebar");
    }

    #[test]
    fn pick_variant_prefers_dark_then_falls_back() {
        use maycord_manifest::{ExtensionKind, Manifest, API_VERSION};
        use maycord_theme::{Rgba, Theme};
        use std::collections::BTreeMap;

        let manifest = Manifest {
            api_version: API_VERSION,
            id: "t".into(),
            name: "t".into(),
            version: "1".into(),
            author: "a".into(),
            license: "MIT".into(),
            source: "s".into(),
            kind: ExtensionKind::Theme,
            capabilities: vec![],
        };
        let variant = Variant {
            colors: BTreeMap::from([("sidebar".to_string(), Rgba::rgb(1, 2, 3))]),
        };
        // Only a light variant present: pick_variant falls back to it.
        let light_only = ThemePackage {
            manifest,
            theme: Theme {
                dark: None,
                light: Some(variant.clone()),
            },
        };
        assert_eq!(pick_variant(&light_only).unwrap().get("sidebar"), Some(Rgba::rgb(1, 2, 3)));
    }

    #[test]
    fn adapter_from_env_errors_without_the_var() {
        // Not set in this test's environment (never set here): a clean NotFound.
        if std::env::var_os("MAYCORD_ADAPTER").is_none() {
            assert_eq!(
                adapter_from_env().unwrap_err().kind(),
                io::ErrorKind::NotFound
            );
        }
    }

    #[test]
    fn apply_is_honestly_unimplemented() {
        assert_eq!(apply(&[]).unwrap_err().kind(), io::ErrorKind::Unsupported);
    }
}
