//! The per-target adapter: the one stable thing maycord writes for each GPUI
//! app it patches.
//!
//! A shipped GPUI client is release machine code — built with fat LTO, one
//! codegen unit, and symbols stripped. It has no stable symbol names and no
//! stable memory layout (Rust doesn't promise one), so there's nothing reliable
//! to link against. The only reliable handle on such a binary is its exact
//! bytes.
//!
//! So maycord pins every patch to one exact executable and to the exact
//! instruction bytes at a fixed offset in it. Adding support for a new GPUI app
//! — including a future GPUI-based client — just means writing a
//! [`TargetAdapter`]: its [`TargetIdentity`] plus the [`PaletteSite`]s where
//! that specific build builds a color. An adapter is plain data, so a new target
//! needs no code change in maycord; it can ship as JSON.
//!
//! The split of responsibility is on purpose. This crate is pure and safe: it
//! describes a target and checks a file against an identity. The unsafe, per-OS
//! work — mapping the process, re-checking the expected bytes at the real
//! (ASLR-shifted) address, and writing the color — lives in the loader, which
//! won't patch until [`TargetIdentity::verify_file`] passes and the live bytes
//! match. An unknown file, or a single wrong byte, means no patch. An adapter
//! only claims the tokens it can really paint; a request for anything else is
//! rejected, not quietly ignored.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The CPU architecture the target was built for. A site's bytes and register
/// locations only make sense for the matching architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Arch {
    Aarch64,
    X86_64,
}

/// A platform's own id for a binary, checked alongside the file hash. Only the
/// macOS variant exists today; Linux build-ids and Windows PDB signatures get
/// added when those loaders land.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageId {
    /// The Mach-O `LC_UUID` load command's 16-byte UUID.
    MachUuid([u8; 16]),
}

/// How the color is stored at a [`PaletteSite`]. The loader has to encode the
/// replacement color the same way the target does, or it writes garbage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ColorEncoding {
    /// Four straight (non-premultiplied) sRGB `f32` channels, `0.0..=1.0`.
    StraightSrgbF32,
    /// Four premultiplied linear `f32` channels, `0.0..=1.0`.
    PremultipliedLinearF32,
    /// A single `0xRRGGBBAA` word.
    SrgbU32,
    /// Four straight-alpha HSLA `f32` channels, `0.0..=1.0` (hue, saturation,
    /// lightness, alpha) — the form GPUI stores a color in. maycord converts a
    /// hex color to this the same way GPUI's `From<Rgba> for Hsla` does.
    StraightHslaF32,
}

/// Where the color sits when execution reaches a [`PaletteSite`]. Register
/// numbers are per-architecture (e.g. `Vector(8)` is `v8` on aarch64).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ValueLocation {
    /// A general-purpose register, by number.
    General(u8),
    /// A vector/SIMD register, by number (where an `f32x4` color lives).
    Vector(u8),
    /// A slot at `offset` bytes from the stack pointer at the site.
    Stack { offset: i64 },
    /// The color sits in memory at `[general_register[base] + offset]`: read the
    /// base register, add the offset, and write the color there. This is how
    /// GPUI colors are reached in practice — inside a struct like `Quad`, or in
    /// the theme global's heap data.
    Indirect { base: u8, offset: i64 },
}

/// One spot in a specific build where a color is built, plus everything the
/// loader needs to check that spot and rewrite that one color.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaletteSite {
    /// The maycord token this site paints (e.g. `"sidebar"`).
    pub token: String,
    /// Offset of the probe instruction within the image (not a file offset). The
    /// loader adds the ASLR shift to get the real runtime address.
    pub image_offset: u64,
    /// The exact instruction bytes expected at `image_offset`. The loader reads
    /// these back from the running image and refuses to patch if they don't
    /// match, so it can't patch the wrong build by mistake.
    #[serde(with = "hex_bytes")]
    pub expected_bytes: Vec<u8>,
    /// How the color is encoded at this site.
    pub encoding: ColorEncoding,
    /// Where the color value is when the probe fires.
    pub location: ValueLocation,
}

impl PaletteSite {
    /// The real runtime address of the probe, given where the OS loaded the
    /// image (its preferred base plus the ASLR shift).
    pub fn runtime_addr(&self, image_base: u64) -> u64 {
        image_base.wrapping_add(self.image_offset)
    }
}

/// The exact identity of one build. If the whole-file SHA-256 matches, the bytes
/// are identical, so the architecture, image id, and every site's expected bytes
/// all hold too. The extra fields let the loader give a precise reason and
/// re-check identity against the running image, where it can't hash the file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetIdentity {
    pub arch: Arch,
    #[serde(with = "hex_array32")]
    pub sha256: [u8; 32],
    pub image: ImageId,
}

impl TargetIdentity {
    /// Check that a candidate executable's bytes are exactly this identity. The
    /// loader runs this before it launches or touches the target.
    pub fn verify_file(&self, exe_bytes: &[u8]) -> Result<(), VerifyError> {
        let found = sha256(exe_bytes);
        if found != self.sha256 {
            return Err(VerifyError::Sha256Mismatch {
                expected: self.sha256,
                found,
            });
        }
        Ok(())
    }
}

/// Why a target failed its check, or an adapter was malformed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// The candidate file's hash did not match the pinned identity.
    Sha256Mismatch { expected: [u8; 32], found: [u8; 32] },
    /// The adapter listed no palette sites, so it can paint nothing.
    NoSites,
    /// Two sites claimed the same token, so which one wins is ambiguous.
    DuplicateToken(String),
}

impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Sha256Mismatch { expected, found } => write!(
                f,
                "target sha256 {} does not match pinned {}",
                hex(found),
                hex(expected)
            ),
            Self::NoSites => write!(f, "adapter declares no palette sites"),
            Self::DuplicateToken(t) => write!(f, "token `{t}` is claimed by two sites"),
        }
    }
}

impl std::error::Error for VerifyError {}

/// A complete, exact description of how to repaint one GPUI build: which
/// executable it is and where that build builds each color. Writing one of these
/// — and nothing else — is how maycord gains support for a new GPUI app.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetAdapter {
    /// A label for logs, e.g. `"example-client 1.2.0 (macos arm64)"`.
    pub name: String,
    /// The exact build this adapter is pinned to.
    pub identity: TargetIdentity,
    /// Every color site this adapter knows how to rewrite.
    pub sites: Vec<PaletteSite>,
}

impl TargetAdapter {
    /// Check the adapter makes sense on its own: it has at least one site and no
    /// two sites claim the same token. Call this once when loading an adapter,
    /// before trusting [`site`](Self::site) or [`supports`](Self::supports).
    pub fn validate(&self) -> Result<(), VerifyError> {
        if self.sites.is_empty() {
            return Err(VerifyError::NoSites);
        }
        for (i, site) in self.sites.iter().enumerate() {
            if self.sites[..i].iter().any(|s| s.token == site.token) {
                return Err(VerifyError::DuplicateToken(site.token.clone()));
            }
        }
        Ok(())
    }

    /// The site that paints `token`, if this adapter claims it.
    pub fn site(&self, token: &str) -> Option<&PaletteSite> {
        self.sites.iter().find(|s| s.token == token)
    }

    /// Whether this adapter can paint `token`. If there's no site for a token,
    /// the caller should reject the request, not quietly ignore it.
    pub fn supports(&self, token: &str) -> bool {
        self.site(token).is_some()
    }

    /// Every token this adapter can paint, in listed order.
    pub fn supported_tokens(&self) -> impl Iterator<Item = &str> {
        self.sites.iter().map(|s| s.token.as_str())
    }
}

/// SHA-256 of `bytes`.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    out
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("hex string has odd length {}", s.len()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

/// Serialize `Vec<u8>` as a hex string instead of a JSON array of numbers, so an
/// adapter's instruction bytes read like a disassembler dump.
mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&super::hex(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        super::unhex(&s).map_err(serde::de::Error::custom)
    }
}

/// The same hex encoding for the fixed 32-byte digest.
mod hex_array32 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&super::hex(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        let v = super::unhex(&s).map_err(serde::de::Error::custom)?;
        v.as_slice()
            .try_into()
            .map_err(|_| serde::de::Error::custom(format!("expected 32 bytes, got {}", v.len())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> TargetAdapter {
        TargetAdapter {
            name: "fixture 1.0 (macos arm64)".into(),
            identity: TargetIdentity {
                arch: Arch::Aarch64,
                sha256: sha256(b"the exact bytes of the target"),
                image: ImageId::MachUuid([7; 16]),
            },
            sites: vec![PaletteSite {
                token: "sidebar".into(),
                image_offset: 0x12_3456,
                expected_bytes: vec![0x1f, 0x20, 0x03, 0xd5], // aarch64 `nop`
                encoding: ColorEncoding::PremultipliedLinearF32,
                location: ValueLocation::Vector(8),
            }],
        }
    }

    #[test]
    fn verify_accepts_the_exact_bytes_and_rejects_anything_else() {
        let a = adapter();
        assert_eq!(a.identity.verify_file(b"the exact bytes of the target"), Ok(()));
        match a.identity.verify_file(b"a different build") {
            Err(VerifyError::Sha256Mismatch { .. }) => {}
            other => panic!("expected mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_flipped_byte_fails_verification() {
        let a = adapter();
        let mut bytes = b"the exact bytes of the target".to_vec();
        bytes[0] ^= 0x01;
        assert!(matches!(
            a.identity.verify_file(&bytes),
            Err(VerifyError::Sha256Mismatch { .. })
        ));
    }

    #[test]
    fn token_support_is_explicit() {
        let a = adapter();
        assert!(a.supports("sidebar"));
        assert!(!a.supports("background"));
        assert_eq!(a.site("sidebar").unwrap().image_offset, 0x12_3456);
        assert!(a.site("background").is_none());
        assert_eq!(a.supported_tokens().collect::<Vec<_>>(), vec!["sidebar"]);
    }

    #[test]
    fn runtime_address_applies_the_load_base() {
        let site = &adapter().sites[0];
        assert_eq!(site.runtime_addr(0x1_0000_0000), 0x1_0012_3456);
    }

    #[test]
    fn validate_rejects_an_empty_or_ambiguous_adapter() {
        let mut a = adapter();
        a.sites.clear();
        assert_eq!(a.validate(), Err(VerifyError::NoSites));

        let mut a = adapter();
        let mut dup = a.sites[0].clone();
        dup.image_offset = 0x999;
        a.sites.push(dup);
        assert_eq!(a.validate(), Err(VerifyError::DuplicateToken("sidebar".into())));
    }

    #[test]
    fn adapter_round_trips_through_json_with_hex_fields() {
        let a = adapter();
        let json = serde_json::to_string_pretty(&a).unwrap();
        // Bytes render as hex, not number arrays, so an adapter reads like a dump.
        assert!(json.contains("\"1f2003d5\""));
        assert!(json.contains(&hex(&a.identity.sha256)));
        assert_eq!(serde_json::from_str::<TargetAdapter>(&json).unwrap(), a);
    }
}
