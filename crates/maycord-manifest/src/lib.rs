//! The manifest that ships with every maycord extension.
//!
//! A manifest is just metadata. It gives the package a name, says whether the
//! package is a [`Theme`](ExtensionKind::Theme) or a
//! [`Plugin`](ExtensionKind::Plugin), and lists the [`Capability`] permissions a
//! plugin asks for. Listing a capability does not grant it: the host still asks
//! the user to approve each one before any code runs. Serde checks that the JSON
//! has the right shape; [`Manifest::validate`] checks the rules serde can't.

use serde::{Deserialize, Serialize};

/// The manifest schema version this crate reads and writes.
pub const API_VERSION: u32 = 1;

/// Package metadata. Hosts read it to load a package; packaging tools write and
/// check it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Schema version of this manifest; must equal [`API_VERSION`].
    pub api_version: u32,
    /// Stable, unique id, like `maycord-ocean`. Never changes between versions.
    pub id: String,
    /// Human-readable name shown in the extension list.
    pub name: String,
    /// Package version, ideally semver. Bump it on any change to the package.
    pub version: String,
    /// Author or maintainer name.
    pub author: String,
    /// SPDX license string for the package contents.
    pub license: String,
    /// Where the package comes from (its canonical URL).
    pub source: String,
    /// Whether this package is a theme or a plugin.
    pub kind: ExtensionKind,
    /// Capabilities a plugin asks for. Themes leave this empty.
    #[serde(default)]
    pub capabilities: Vec<Capability>,
}

/// What a package adds to the client.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtensionKind {
    /// Appearance only: a map of color tokens, no code. Needs no capabilities.
    Theme,
    /// Code, shipped as a sandboxed WebAssembly module.
    Plugin,
}

/// A permission a plugin requests in its manifest and the user approves before
/// the plugin loads. Themes never have capabilities.
///
/// The list is deliberately small and only grows over time. An older host that
/// doesn't recognize a capability rejects the plugin rather than ignoring it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Capability {
    /// Read or change the text of messages the client shows.
    MessageContent,
    /// Read or change the user's draft before it is sent.
    Composer,
    /// Save and read the plugin's own key/value data.
    Storage,
    /// Add colors or styling beyond a plain theme package.
    Appearance,
    /// Watch lifecycle and UI events the host sends out.
    Events,
}

/// Why a [`Manifest`] was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestError {
    /// `api_version` didn't match [`API_VERSION`].
    UnsupportedApiVersion { found: u32, expected: u32 },
    /// A required text field was empty.
    EmptyField(&'static str),
    /// `id` had characters outside `[a-z0-9._-]`, or was the wrong length.
    InvalidId,
    /// A theme listed capabilities, which it isn't allowed to do.
    ThemeWithCapabilities,
    /// The same capability was listed more than once.
    DuplicateCapability(Capability),
}

impl core::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedApiVersion { found, expected } => {
                write!(f, "unsupported api_version {found}, host speaks {expected}")
            }
            Self::EmptyField(name) => write!(f, "manifest field `{name}` must not be empty"),
            Self::InvalidId => {
                write!(f, "manifest id must be 1-64 chars of [a-z0-9._-]")
            }
            Self::ThemeWithCapabilities => {
                write!(f, "a theme package must not declare capabilities")
            }
            Self::DuplicateCapability(cap) => write!(f, "capability {cap:?} listed twice"),
        }
    }
}

impl std::error::Error for ManifestError {}

impl Manifest {
    /// Check the rules serde can't express. Call this before trusting a parsed
    /// manifest; a host that skips it may load a broken package.
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.api_version != API_VERSION {
            return Err(ManifestError::UnsupportedApiVersion {
                found: self.api_version,
                expected: API_VERSION,
            });
        }
        for (name, value) in [
            ("id", &self.id),
            ("name", &self.name),
            ("version", &self.version),
            ("author", &self.author),
            ("license", &self.license),
            ("source", &self.source),
        ] {
            if value.trim().is_empty() {
                return Err(ManifestError::EmptyField(name));
            }
        }
        if !valid_id(&self.id) {
            return Err(ManifestError::InvalidId);
        }
        if self.kind == ExtensionKind::Theme && !self.capabilities.is_empty() {
            return Err(ManifestError::ThemeWithCapabilities);
        }
        for (i, cap) in self.capabilities.iter().enumerate() {
            if self.capabilities[..i].contains(cap) {
                return Err(ManifestError::DuplicateCapability(*cap));
            }
        }
        Ok(())
    }
}

/// An id is 1-64 bytes of lowercase ASCII, digits, and `.`, `-`, `_`.
fn valid_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'-' | b'_')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> Manifest {
        Manifest {
            api_version: API_VERSION,
            id: "maycord-ocean".into(),
            name: "Ocean".into(),
            version: "1.0.0".into(),
            author: "maycord contributors".into(),
            license: "MIT".into(),
            source: "https://example.invalid/ocean".into(),
            kind: ExtensionKind::Theme,
            capabilities: vec![],
        }
    }

    #[test]
    fn accepts_a_well_formed_theme() {
        assert_eq!(theme().validate(), Ok(()));
    }

    #[test]
    fn rejects_future_api_versions() {
        let mut m = theme();
        m.api_version = API_VERSION + 1;
        assert_eq!(
            m.validate(),
            Err(ManifestError::UnsupportedApiVersion {
                found: API_VERSION + 1,
                expected: API_VERSION,
            })
        );
    }

    #[test]
    fn rejects_bad_ids_and_empty_fields() {
        let mut m = theme();
        m.id = "Ocean Theme!".into();
        assert_eq!(m.validate(), Err(ManifestError::InvalidId));

        let mut m = theme();
        m.author = "   ".into();
        assert_eq!(m.validate(), Err(ManifestError::EmptyField("author")));
    }

    #[test]
    fn a_theme_may_not_hold_capabilities() {
        let mut m = theme();
        m.capabilities = vec![Capability::Storage];
        assert_eq!(m.validate(), Err(ManifestError::ThemeWithCapabilities));
    }

    #[test]
    fn catches_duplicate_capabilities() {
        let mut m = theme();
        m.kind = ExtensionKind::Plugin;
        m.capabilities = vec![Capability::Storage, Capability::Storage];
        assert_eq!(
            m.validate(),
            Err(ManifestError::DuplicateCapability(Capability::Storage))
        );
    }

    #[test]
    fn round_trips_through_json() {
        let m = theme();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<Manifest>(&json).unwrap(), m);
    }

    #[test]
    fn rejects_unknown_fields() {
        let json = r#"{
            "api_version": 1, "id": "x", "name": "x", "version": "1",
            "author": "x", "license": "MIT", "source": "x", "kind": "theme",
            "surprise": true
        }"#;
        assert!(serde_json::from_str::<Manifest>(json).is_err());
    }
}
