//! The authorized-target loader: it turns a verified [`TargetAdapter`] and a
//! theme into the concrete memory writes that repaint one GPUI build, and
//! launches that build with the in-process agent attached.
//!
//! The loader is split so that almost all of it stays pure and testable, and
//! the genuinely dangerous part is small and isolated:
//!
//! * [`preflight`] gates everything on exact identity — it refuses to proceed
//!   unless the candidate executable's bytes are the ones the adapter was
//!   authored against.
//! * [`plan_writes`] turns an adapter + a theme [`Variant`] + the image's load
//!   base into a list of [`PlannedWrite`]s. It only plans tokens the adapter
//!   actually supports *and* the theme defines; anything else is skipped, never
//!   faked.
//! * [`encode_color`] renders a color into the exact byte form a site expects.
//! * the `agent` module (feature `frida`) is the only code that writes another
//!   process's memory; it consumes the plan above and nothing else.
//!
//! Nothing here casts foreign memory into `gpui` types: a stripped, fat-LTO
//! release binary has no stable Rust layout to borrow, so the loader only ever
//! touches a single verified color slot at a pinned site.

use maycord_target::{ColorEncoding, TargetAdapter, ValueLocation, VerifyError};
use maycord_theme::{Rgba, Variant};

pub mod agent;

/// A color encoded the way a specific [`PaletteSite`](maycord_target::PaletteSite)
/// expects to find it in memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EncodedColor {
    /// Four `f32` channels (16 bytes) — a vector-register or stack `f32x4`.
    F32x4([f32; 4]),
    /// A single packed `0xRRGGBBAA` word (4 bytes).
    U32(u32),
}

impl EncodedColor {
    /// The little-endian bytes to write for a stack/memory slot. (A register
    /// write uses the typed value directly instead.)
    pub fn to_le_bytes(self) -> Vec<u8> {
        match self {
            Self::F32x4(v) => v.iter().flat_map(|c| c.to_le_bytes()).collect(),
            Self::U32(w) => w.to_le_bytes().to_vec(),
        }
    }
}

/// Encode `color` the way `encoding` stores it. This is where a maycord theme
/// color becomes the bytes a particular build actually paints with, so it must
/// match the target's own convention exactly or the pixel comes out wrong.
///
/// Returns `None` for an encoding this loader build does not understand — a
/// newer adapter naming a future encoding is refused at that site rather than
/// written wrong, since [`ColorEncoding`] is non-exhaustive by design.
pub fn encode_color(color: Rgba, encoding: ColorEncoding) -> Option<EncodedColor> {
    Some(match encoding {
        ColorEncoding::StraightSrgbF32 => EncodedColor::F32x4([
            color.r as f32 / 255.0,
            color.g as f32 / 255.0,
            color.b as f32 / 255.0,
            color.a as f32 / 255.0,
        ]),
        ColorEncoding::PremultipliedLinearF32 => {
            // Straight linear channels plus alpha, then premultiply rgb by alpha
            // — the form GPUI's scene layer blends with.
            let [r, g, b, a] = color.to_linear_f32();
            EncodedColor::F32x4([r * a, g * a, b * a, a])
        }
        ColorEncoding::SrgbU32 => EncodedColor::U32(
            (color.r as u32) << 24
                | (color.g as u32) << 16
                | (color.b as u32) << 8
                | color.a as u32,
        ),
        _ => return None,
    })
}

/// One concrete write the agent will perform: at `runtime_addr`, after
/// confirming `expected_bytes` are still there, place `color` at `location`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedWrite {
    pub token: String,
    pub runtime_addr: u64,
    pub expected_bytes: Vec<u8>,
    pub location: ValueLocation,
    pub color: EncodedColor,
}

/// Build the writes that apply `variant` to the target described by `adapter`,
/// given the image's `image_base` (preferred base plus ASLR slide, which the
/// agent reads from the live process).
///
/// A write is planned only when the adapter has a site for the token *and* the
/// variant defines a color for it. Tokens the adapter cannot paint, or the
/// theme leaves unset, are simply absent from the plan — the loader never
/// pretends to apply a color it has nowhere to put.
pub fn plan_writes(adapter: &TargetAdapter, variant: &Variant, image_base: u64) -> Vec<PlannedWrite> {
    adapter
        .sites
        .iter()
        .filter_map(|site| {
            let color = variant.get(&site.token)?;
            let encoded = encode_color(color, site.encoding)?;
            Some(PlannedWrite {
                token: site.token.clone(),
                runtime_addr: site.runtime_addr(image_base),
                expected_bytes: site.expected_bytes.clone(),
                location: site.location,
                color: encoded,
            })
        })
        .collect()
}

/// Gate on exact identity: confirm `exe_bytes` are the build `adapter` was
/// authored against, and that the adapter is internally coherent. The loader
/// calls this before it launches or touches the target; on any error it must
/// not proceed.
pub fn preflight(exe_bytes: &[u8], adapter: &TargetAdapter) -> Result<(), VerifyError> {
    adapter.validate()?;
    adapter.identity.verify_file(exe_bytes)
}

/// macOS launch: verify the target on disk, then start it with the agent dylib
/// forced in via `DYLD_INSERT_LIBRARIES` and the adapter handed over by path.
///
/// This works against an unsigned or ad-hoc-signed copy of the target; a
/// hardened-runtime, library-validated binary ignores `DYLD_INSERT_LIBRARIES`
/// unless it carries the matching entitlement, which is a property of the target
/// we cannot grant it. The loader reports that rather than pretending to inject.
#[cfg(target_os = "macos")]
pub fn launch_macos(
    exe: &std::path::Path,
    agent_dylib: &std::path::Path,
    adapter: &TargetAdapter,
    adapter_path: &std::path::Path,
) -> std::io::Result<std::process::Child> {
    let bytes = std::fs::read(exe)?;
    preflight(&bytes, adapter)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    std::process::Command::new(exe)
        .env("DYLD_INSERT_LIBRARIES", agent_dylib)
        .env("MAYCORD_ADAPTER", adapter_path)
        .spawn()
}

#[cfg(test)]
mod tests {
    use super::*;
    use maycord_target::{Arch, ImageId, PaletteSite, TargetIdentity};

    fn adapter_with(sites: Vec<PaletteSite>, exe: &[u8]) -> TargetAdapter {
        TargetAdapter {
            name: "fixture".into(),
            identity: TargetIdentity {
                arch: Arch::Aarch64,
                sha256: maycord_target::sha256(exe),
                image: ImageId::MachUuid([0; 16]),
            },
            sites,
        }
    }

    fn site(token: &str, offset: u64, enc: ColorEncoding) -> PaletteSite {
        PaletteSite {
            token: token.into(),
            image_offset: offset,
            expected_bytes: vec![0x1f, 0x20, 0x03, 0xd5],
            encoding: enc,
            location: ValueLocation::Vector(8),
        }
    }

    #[test]
    fn opaque_primary_encodes_consistently() {
        let red = Rgba::rgb(255, 0, 0);
        assert_eq!(
            encode_color(red, ColorEncoding::StraightSrgbF32),
            Some(EncodedColor::F32x4([1.0, 0.0, 0.0, 1.0]))
        );
        assert_eq!(
            encode_color(red, ColorEncoding::SrgbU32),
            Some(EncodedColor::U32(0xff0000ff))
        );
        // Fully opaque, so premultiplied linear keeps linear(255)=1.0 on red.
        assert_eq!(
            encode_color(red, ColorEncoding::PremultipliedLinearF32),
            Some(EncodedColor::F32x4([1.0, 0.0, 0.0, 1.0]))
        );
    }

    #[test]
    fn alpha_premultiplies_and_packs() {
        let half_red = Rgba { r: 255, g: 0, b: 0, a: 128 };
        let a = 128.0 / 255.0;
        match encode_color(half_red, ColorEncoding::PremultipliedLinearF32) {
            Some(EncodedColor::F32x4([r, g, b, out_a])) => {
                assert!((r - a).abs() < 1e-6); // linear(255)=1.0, premultiplied by a
                assert_eq!([g, b], [0.0, 0.0]);
                assert!((out_a - a).abs() < 1e-6);
            }
            other => panic!("expected f32x4, got {other:?}"),
        }
        assert_eq!(
            encode_color(half_red, ColorEncoding::SrgbU32),
            Some(EncodedColor::U32(0xff000080))
        );
    }

    #[test]
    fn f32x4_little_endian_bytes_are_sixteen_wide() {
        let bytes = EncodedColor::F32x4([1.0, 0.0, 0.0, 1.0]).to_le_bytes();
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
    }

    #[test]
    fn plan_skips_tokens_the_adapter_or_theme_lacks() {
        let exe = b"exact target bytes";
        let adapter = adapter_with(
            vec![
                site("sidebar", 0x1000, ColorEncoding::PremultipliedLinearF32),
                site("background", 0x2000, ColorEncoding::SrgbU32),
            ],
            exe,
        );
        // Theme defines sidebar and an unrelated token the adapter can't paint.
        let mut colors = std::collections::BTreeMap::new();
        colors.insert("sidebar".to_string(), Rgba::rgb(10, 20, 30));
        colors.insert("accent".to_string(), Rgba::rgb(1, 2, 3));
        let variant = Variant { colors };

        let plan = plan_writes(&adapter, &variant, 0x1_0000_0000);
        // Only sidebar: background has a site but no theme color; accent has a
        // theme color but no site.
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].token, "sidebar");
        assert_eq!(plan[0].runtime_addr, 0x1_0000_1000);
    }

    #[test]
    fn preflight_refuses_the_wrong_binary() {
        let adapter = adapter_with(vec![site("sidebar", 0x10, ColorEncoding::SrgbU32)], b"right");
        assert_eq!(preflight(b"right", &adapter), Ok(()));
        assert!(matches!(
            preflight(b"wrong", &adapter),
            Err(VerifyError::Sha256Mismatch { .. })
        ));
    }

    #[test]
    fn preflight_refuses_an_incoherent_adapter() {
        // No sites: the adapter can paint nothing, so preflight fails even on the
        // right bytes.
        let adapter = adapter_with(vec![], b"right");
        assert_eq!(preflight(b"right", &adapter), Err(VerifyError::NoSites));
    }
}
