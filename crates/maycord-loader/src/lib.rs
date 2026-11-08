//! The loader: it turns a verified [`TargetAdapter`] and a theme into the actual
//! memory writes that repaint one GPUI build, and launches that build with the
//! in-process agent attached.
//!
//! It's split so almost all of it stays plain and testable, and the genuinely
//! dangerous part is small and off to the side:
//!
//! * [`preflight`] checks exact identity first — it won't go on unless the
//!   candidate executable's bytes are the ones the adapter was written for.
//! * [`plan_writes`] turns an adapter + a theme [`Variant`] + the image's load
//!   base into a list of [`PlannedWrite`]s. It only plans tokens the adapter
//!   supports *and* the theme defines; anything else is skipped, not faked.
//! * [`encode_color`] turns a color into the exact bytes a site expects.
//! * the `agent` module (feature `frida`) is the only code that writes another
//!   process's memory; it works only from the plan above.
//!
//! Nothing here reads the target's memory as `gpui` types: a stripped, fat-LTO
//! release binary has no stable Rust layout to rely on, so the loader only ever
//! touches one checked color slot at a fixed spot.

use maycord_target::{ColorEncoding, TargetAdapter, ValueLocation, VerifyError};
use maycord_theme::{Rgba, Variant};

pub mod agent;

/// A color encoded the way a specific [`PaletteSite`](maycord_target::PaletteSite)
/// expects it in memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EncodedColor {
    /// Four `f32` channels (16 bytes) — a vector-register or stack `f32x4`.
    F32x4([f32; 4]),
    /// A single packed `0xRRGGBBAA` word (4 bytes).
    U32(u32),
}

impl EncodedColor {
    /// The little-endian bytes to write into a stack or memory slot. (A register
    /// write uses the typed value directly instead.)
    pub fn to_le_bytes(self) -> Vec<u8> {
        match self {
            Self::F32x4(v) => v.iter().flat_map(|c| c.to_le_bytes()).collect(),
            Self::U32(w) => w.to_le_bytes().to_vec(),
        }
    }
}

/// Encode `color` the way `encoding` stores it. This is where a maycord theme
/// color becomes the bytes a build paints with, so it has to match the target's
/// own format exactly or the pixel comes out wrong.
///
/// Returns `None` for an encoding this loader build doesn't know. A newer adapter
/// that names a future encoding is skipped at that site instead of written
/// wrong, since [`ColorEncoding`] can grow new variants.
pub fn encode_color(color: Rgba, encoding: ColorEncoding) -> Option<EncodedColor> {
    Some(match encoding {
        ColorEncoding::StraightSrgbF32 => EncodedColor::F32x4([
            color.r as f32 / 255.0,
            color.g as f32 / 255.0,
            color.b as f32 / 255.0,
            color.a as f32 / 255.0,
        ]),
        ColorEncoding::PremultipliedLinearF32 => {
            // Linear channels plus alpha, then multiply rgb by alpha (premultiplied).
            let [r, g, b, a] = color.to_linear_f32();
            EncodedColor::F32x4([r * a, g * a, b * a, a])
        }
        ColorEncoding::SrgbU32 => EncodedColor::U32(
            (color.r as u32) << 24
                | (color.g as u32) << 16
                | (color.b as u32) << 8
                | color.a as u32,
        ),
        ColorEncoding::StraightHslaF32 => EncodedColor::F32x4(to_straight_hsla(color)),
        _ => return None,
    })
}

/// Convert a color to GPUI's HSLA form: hue, saturation, lightness, alpha, each
/// `0.0..=1.0`. This mirrors GPUI's own `From<Rgba> for Hsla` exactly (it works
/// on straight sRGB channels, not linear), so a color maycord writes into a GPUI
/// build matches what GPUI would have computed itself.
fn to_straight_hsla(color: Rgba) -> [f32; 4] {
    let r = color.r as f32 / 255.0;
    let g = color.g as f32 / 255.0;
    let b = color.b as f32 / 255.0;
    let a = color.a as f32 / 255.0;

    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let delta = max - min;

    let l = (max + min) / 2.0;
    let s = if l == 0.0 || l == 1.0 {
        0.0
    } else if l < 0.5 {
        delta / (2.0 * l)
    } else {
        delta / (2.0 - 2.0 * l)
    };
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    [h, s, l, a]
}

/// One write the agent will make: at `runtime_addr`, after checking
/// `expected_bytes` are still there, put `color` at `location`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedWrite {
    pub token: String,
    pub runtime_addr: u64,
    pub expected_bytes: Vec<u8>,
    pub location: ValueLocation,
    pub color: EncodedColor,
}

/// Build the writes that apply `variant` to the target the `adapter` describes,
/// given the image's `image_base` (its preferred base plus the ASLR shift, which
/// the agent reads from the running process).
///
/// A write is planned only when the adapter has a site for the token *and* the
/// variant defines a color for it. Tokens the adapter can't paint, or the theme
/// leaves unset, just aren't in the plan — the loader never pretends to apply a
/// color it has nowhere to put.
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

/// Check exact identity: confirm `exe_bytes` are the build the `adapter` was
/// written for, and that the adapter itself makes sense. The loader calls this
/// before it launches or touches the target; on any error it must stop.
pub fn preflight(exe_bytes: &[u8], adapter: &TargetAdapter) -> Result<(), VerifyError> {
    adapter.validate()?;
    adapter.identity.verify_file(exe_bytes)
}

/// macOS launch: check the target on disk, then start it with the agent dylib
/// forced in via `DYLD_INSERT_LIBRARIES` and the adapter passed by path.
///
/// This works on an unsigned or ad-hoc-signed copy of the target. A
/// hardened-runtime, library-validated binary ignores `DYLD_INSERT_LIBRARIES`
/// unless it has the matching entitlement, which is the target's own setting and
/// not something we can grant it. The loader reports that instead of pretending
/// to inject.
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

/// Linux launch (prototype): check the target on disk, then start it with the
/// agent shared object forced in via `LD_PRELOAD` and the adapter passed by path.
///
/// Like the macOS path, this works on an ordinary dynamically-linked binary; a
/// secure-execution binary (setuid, or `AT_SECURE`) ignores `LD_PRELOAD`, which
/// the loader surfaces rather than pretending to inject.
#[cfg(target_os = "linux")]
pub fn launch_linux(
    exe: &std::path::Path,
    agent_so: &std::path::Path,
    adapter: &TargetAdapter,
    adapter_path: &std::path::Path,
) -> std::io::Result<std::process::Child> {
    let bytes = std::fs::read(exe)?;
    preflight(&bytes, adapter)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    std::process::Command::new(exe)
        .env("LD_PRELOAD", agent_so)
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
            events: vec![],
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
    fn straight_hsla_matches_gpuis_formula() {
        // #3366cc, converted the way GPUI's own `From<Rgba> for Hsla` would.
        match encode_color(Rgba::rgb(0x33, 0x66, 0xcc), ColorEncoding::StraightHslaF32) {
            Some(EncodedColor::F32x4([h, s, l, a])) => {
                assert!((h - 0.611_111).abs() < 1e-5, "h = {h}");
                assert!((s - 0.6).abs() < 1e-5, "s = {s}");
                assert!((l - 0.5).abs() < 1e-5, "l = {l}");
                assert_eq!(a, 1.0);
            }
            other => panic!("expected an hsla f32x4, got {other:?}"),
        }
        // A grey has zero saturation and hue.
        assert_eq!(
            encode_color(Rgba::rgb(128, 128, 128), ColorEncoding::StraightHslaF32),
            Some(EncodedColor::F32x4([0.0, 0.0, 128.0 / 255.0, 1.0]))
        );
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
