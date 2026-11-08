//! The maycord theme package: a manifest plus a map of semantic color tokens
//! for dark and/or light mode.
//!
//! A theme carries no code. It maps the client's *semantic* token names
//! (`background`, `accent`, `text`, …) to concrete colors, and the client
//! resolves its own widgets through those tokens. This is what makes a theme
//! renderer-agnostic: the same package re-skins a GPUI client, an egui client,
//! or anything else that resolves its palette through named tokens, because it
//! never names a pixel, a shader, or a struct offset.
//!
//! ```
//! let json = r##"{
//!   "manifest": {
//!     "api_version": 1, "id": "maycord-ocean", "name": "Ocean",
//!     "version": "1.0.0", "author": "maycord", "license": "MIT",
//!     "source": "https://example.invalid/ocean", "kind": "theme"
//!   },
//!   "theme": { "dark": { "colors": { "background": "#101C28", "accent": "#57CEE3" } } }
//! }"##;
//! let pkg = maycord_theme::parse_package(json.as_bytes()).unwrap();
//! assert_eq!(pkg.theme.dark.as_ref().unwrap().get("accent").unwrap().to_string(), "#57cee3");
//! ```

use std::collections::BTreeMap;

use maycord_manifest::{ExtensionKind, Manifest, ManifestError};
use serde::{Deserialize, Serialize};

mod color;
pub use color::{ParseColorError, Rgba};

/// The semantic tokens the reference client resolves. A theme may define more
/// (a plugin might read them) and need not define all of these, but a theme
/// that covers this set will skin the whole reference UI. Kept deliberately
/// small; it grows as the client grows, never per-widget.
pub const CORE_TOKENS: &[&str] = &[
    "background",  // window and primary surface
    "surface",     // raised panels, cards, the message list
    "sidebar",     // channel and guild rails
    "overlay",     // popovers, menus, modals
    "border",      // hairlines and dividers
    "text",        // primary foreground text
    "muted",       // secondary/placeholder text
    "accent",      // brand/active color
    "accent_text", // text drawn on top of `accent`
    "link",        // hyperlinks and mentions
];

/// A complete, parsed theme package: metadata and its color variants.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThemePackage {
    pub manifest: Manifest,
    pub theme: Theme,
}

/// The color variants a package provides. At least one must be present.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Theme {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dark: Option<Variant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light: Option<Variant>,
}

/// One color scheme: a map from semantic token name to color.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub colors: BTreeMap<String, Rgba>,
}

impl Variant {
    /// Resolve one token, if the variant defines it.
    pub fn get(&self, token: &str) -> Option<Rgba> {
        self.colors.get(token).copied()
    }

    /// The [`CORE_TOKENS`] this variant does not define, in declared order.
    /// Empty means the variant covers the whole reference UI.
    pub fn missing_core_tokens(&self) -> Vec<&'static str> {
        CORE_TOKENS
            .iter()
            .copied()
            .filter(|t| !self.colors.contains_key(*t))
            .collect()
    }
}

/// Why a theme package was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThemeError {
    /// The bytes were not valid JSON in the package shape.
    Json(String),
    /// The embedded manifest failed [`Manifest::validate`].
    Manifest(ManifestError),
    /// The manifest `kind` was not `theme`.
    NotATheme,
    /// Neither a dark nor a light variant was present.
    NoVariants,
    /// A present variant defined no colors at all.
    EmptyVariant(&'static str),
}

impl core::fmt::Display for ThemeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "invalid theme package JSON: {e}"),
            Self::Manifest(e) => write!(f, "{e}"),
            Self::NotATheme => write!(f, "manifest kind must be `theme`"),
            Self::NoVariants => write!(f, "theme must define a dark or light variant"),
            Self::EmptyVariant(v) => write!(f, "the {v} variant defines no colors"),
        }
    }
}

impl std::error::Error for ThemeError {}

impl ThemePackage {
    /// Check everything serde cannot: the manifest invariants, that this really
    /// is a theme, and that it carries at least one non-empty variant.
    pub fn validate(&self) -> Result<(), ThemeError> {
        self.manifest.validate().map_err(ThemeError::Manifest)?;
        if self.manifest.kind != ExtensionKind::Theme {
            return Err(ThemeError::NotATheme);
        }
        if self.theme.dark.is_none() && self.theme.light.is_none() {
            return Err(ThemeError::NoVariants);
        }
        for (name, variant) in [("dark", &self.theme.dark), ("light", &self.theme.light)] {
            if let Some(v) = variant {
                if v.colors.is_empty() {
                    return Err(ThemeError::EmptyVariant(name));
                }
            }
        }
        Ok(())
    }
}

/// Parse and fully validate a theme package from its JSON bytes. This is the
/// one entry point a host should use; it never returns an unvalidated package.
pub fn parse_package(bytes: &[u8]) -> Result<ThemePackage, ThemeError> {
    let pkg: ThemePackage =
        serde_json::from_slice(bytes).map_err(|e| ThemeError::Json(e.to_string()))?;
    pkg.validate()?;
    Ok(pkg)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OCEAN: &str = r##"{
        "manifest": {
            "api_version": 1, "id": "maycord-ocean", "name": "Ocean",
            "version": "1.0.0", "author": "maycord contributors", "license": "MIT",
            "source": "https://example.invalid/ocean", "kind": "theme"
        },
        "theme": {
            "dark": { "colors": {
                "background": "#101C28", "surface": "#192C3C", "sidebar": "#142432",
                "overlay": "#20384A", "border": "#3B576B", "text": "#E8F4F8",
                "muted": "#8FA9B8", "accent": "#57CEE3", "accent_text": "#10202B",
                "link": "#8DE6F2"
            } }
        }
    }"##;

    #[test]
    fn parses_and_validates_a_full_theme() {
        let pkg = parse_package(OCEAN.as_bytes()).unwrap();
        let dark = pkg.theme.dark.unwrap();
        assert_eq!(dark.get("accent"), Some(Rgba::rgb(0x57, 0xCE, 0xE3)));
        assert!(dark.missing_core_tokens().is_empty());
        assert!(pkg.theme.light.is_none());
    }

    #[test]
    fn reports_uncovered_core_tokens() {
        let v = Variant {
            colors: BTreeMap::from([("accent".to_string(), Rgba::rgb(1, 2, 3))]),
        };
        assert!(v.missing_core_tokens().contains(&"background"));
        assert!(!v.missing_core_tokens().contains(&"accent"));
    }

    #[test]
    fn rejects_a_package_with_no_variants() {
        // `Theme { dark: None, light: None }` serializes to `"theme": {}`.
        let json = r#"{
            "manifest": {
                "api_version": 1, "id": "x", "name": "x", "version": "1",
                "author": "x", "license": "MIT", "source": "x", "kind": "theme"
            },
            "theme": {}
        }"#;
        assert_eq!(parse_package(json.as_bytes()), Err(ThemeError::NoVariants));
    }

    #[test]
    fn rejects_an_empty_variant() {
        let json = r#"{
            "manifest": {
                "api_version": 1, "id": "x", "name": "x", "version": "1",
                "author": "x", "license": "MIT", "source": "x", "kind": "theme"
            },
            "theme": { "dark": { "colors": {} } }
        }"#;
        assert_eq!(
            parse_package(json.as_bytes()),
            Err(ThemeError::EmptyVariant("dark"))
        );
    }

    #[test]
    fn rejects_a_plugin_kind_in_a_theme_slot() {
        let json = OCEAN.replace("\"kind\": \"theme\"", "\"kind\": \"plugin\"");
        assert_eq!(parse_package(json.as_bytes()), Err(ThemeError::NotATheme));
    }

    #[test]
    fn round_trips_through_json() {
        let pkg = parse_package(OCEAN.as_bytes()).unwrap();
        let out = serde_json::to_vec(&pkg).unwrap();
        assert_eq!(parse_package(&out).unwrap(), pkg);
    }
}
