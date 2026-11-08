//! The per-target adapter: the one stable interface maycord implements for each
//! GPUI application it patches.
//!
//! A shipped GPUI client is release machine code — fat LTO, `codegen-units = 1`,
//! `strip = "symbols"`. It exposes no stable symbol and no stable ABI (Rust's
//! type layout is explicitly not guaranteed), so there is nothing durable to
//! link against. The *only* durable handle on such a binary is its exact bytes.
//!
//! maycord therefore pins every patch to an exact executable identity and to
//! exact instruction bytes at an image-relative address. Supporting a new GPUI
//! app — a future GPUI-based client included — means authoring a
//! [`TargetAdapter`]: its [`TargetIdentity`] plus the [`PaletteSite`]s where
//! that specific build materializes a color. An adapter is *data*, so a new
//! target needs no maycord code change; it ships as, say, JSON.
//!
//! The division of trust is deliberate. This crate is pure and safe: it
//! describes a target and verifies a file against an identity. The unsafe,
//! per-OS work — mapping the process, re-checking the expected bytes at the
//! ASLR-slid runtime address, writing the color — lives in the loader, which
//! must not patch without a [`TargetIdentity::verify_file`] pass and a live
//! byte match first. An unknown image or a single mismatched byte means *no
//! patch*. An adapter only ever claims the tokens it can actually paint; an
//! unsupported request is rejected, never silently faked.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The instruction-set the target image was compiled for. A site's bytes and
/// value locations are only meaningful under the matching architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Arch {
    Aarch64,
    X86_64,
}

/// A platform's native image identity, cross-checked in addition to the file
/// hash. Only the variant for the current proof target (macOS) exists today;
/// Linux build-ids and Windows PDB signatures are added as those loaders land.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageId {
    /// The Mach-O `LC_UUID` load command's 16-byte UUID.
    MachUuid([u8; 16]),
}

/// How the live color is represented at a [`PaletteSite`]. The loader must
/// encode the replacement color the same way the target expects, or it writes
/// garbage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ColorEncoding {
    /// Four straight (non-premultiplied) sRGB `f32` channels, `0.0..=1.0`.
    StraightSrgbF32,
    /// Four premultiplied linear `f32` channels, as GPUI's scene layer uses.
    PremultipliedLinearF32,
    /// A single `0xRRGGBBAA` word.
    SrgbU32,
}

/// Where the live color sits when control reaches a [`PaletteSite`]. Register
/// numbers are architecture-relative (e.g. `Vector(8)` is `v8` on aarch64).
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
}

/// One place in a specific build where a palette color is materialized, and
/// everything the loader needs to verify it and rewrite that one color.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaletteSite {
    /// The maycord semantic token this site paints (e.g. `"sidebar"`).
    pub token: String,
    /// Image-relative address of the probe instruction (not a file offset); the
    /// loader adds the ASLR slide to get the runtime address.
    pub image_offset: u64,
    /// The exact instruction bytes expected at `image_offset`. The loader
    /// re-reads these from the mapped image and refuses to patch on any
    /// mismatch, so a build the adapter was not authored against cannot be
    /// silently mispatched.
    #[serde(with = "hex_bytes")]
    pub expected_bytes: Vec<u8>,
    /// How the color is encoded at this site.
    pub encoding: ColorEncoding,
    /// Where the live color value is when the probe fires.
    pub location: ValueLocation,
}

impl PaletteSite {
    /// The runtime address of the probe given the image's load base (the base
    /// the OS mapped the image at, i.e. preferred base plus ASLR slide).
    pub fn runtime_addr(&self, image_base: u64) -> u64 {
        image_base.wrapping_add(self.image_offset)
    }
}

/// The exact identity of one executable build. A full-file SHA-256 match means
/// the bytes are identical, so architecture, image id, and every site's
/// expected bytes necessarily hold too; the extra fields let the loader report
/// a precise reason and re-check identity against a mapped image, where the
/// file hash is not available.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetIdentity {
    pub arch: Arch,
    #[serde(with = "hex_array32")]
    pub sha256: [u8; 32],
    pub image: ImageId,
}

impl TargetIdentity {
    /// Verify a candidate executable's bytes are exactly this identity. This is
    /// the gate the loader runs before it launches or touches the target.
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

/// Why a target failed verification or an adapter was malformed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// The candidate file's hash did not match the pinned identity.
    Sha256Mismatch { expected: [u8; 32], found: [u8; 32] },
    /// The adapter declared no palette sites, so it can paint nothing.
    NoSites,
    /// Two sites claimed the same token; the mapping would be ambiguous.
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

/// A complete, exact-identity description of how to repaint one GPUI build:
/// which executable it is and where that executable materializes each color.
/// Authoring one of these — and nothing else — is how maycord gains support for
/// a new GPUI application.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetAdapter {
    /// Human label for diagnostics, e.g. `"example-client 1.2.0 (macos arm64)"`.
    pub name: String,
    /// The exact build this adapter is pinned to.
    pub identity: TargetIdentity,
    /// Every color site this adapter knows how to rewrite.
    pub sites: Vec<PaletteSite>,
}

impl TargetAdapter {
    /// Check the adapter is internally coherent: it has at least one site and no
    /// two sites fight over the same token. Call once when an adapter is loaded,
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

    /// Whether this adapter can paint `token`. A request for a token with no
    /// site must be rejected by the caller, not silently ignored.
    pub fn supports(&self, token: &str) -> bool {
        self.site(token).is_some()
    }

    /// Every token this adapter can paint, in declared order.
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

/// Serialize `Vec<u8>` as a hex string instead of a JSON number array, so an
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
