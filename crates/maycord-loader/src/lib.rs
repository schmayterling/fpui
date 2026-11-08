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

use maycord_target::{
    ColorEncoding, EventKind, Signature, TargetAdapter, ValueLocation, VerifyError,
};
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

/// How one patch (a colour site or an event site) fares against a build: the
/// label it carries, where it points, and whether its pinned bytes are still
/// there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchStatus {
    pub label: String,
    /// The offset the adapter pins (the hint).
    pub image_offset: u64,
    pub supported: bool,
    /// Where the patch actually sits in this build: `Some(image_offset)` when it
    /// matched at the hint, `Some(other)` when a signature recovered it from
    /// drift, and `None` when it could not be found at all.
    pub resolved_offset: Option<u64>,
}

impl PatchStatus {
    /// How far the patch moved from its pinned offset in this build, if found.
    pub fn drift(&self) -> i64 {
        match self.resolved_offset {
            Some(off) => off as i64 - self.image_offset as i64,
            None => 0,
        }
    }
}

/// How a whole adapter fares against one build. This is the heart of running on
/// any version: rather than demanding the exact pinned build, maycord checks
/// each patch on its own, so it can apply the ones that still match and report
/// the ones that have broken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchReport {
    /// SHA-256 of the build that was checked.
    pub sha256: [u8; 32],
    /// Whether this is the exact build the adapter was pinned to. When it is
    /// false but patches still match, maycord is running on a *different*
    /// version that the adapter nonetheless still fits in part.
    pub identity_matches: bool,
    pub patches: Vec<PatchStatus>,
}

impl PatchReport {
    /// How many patches still verify.
    pub fn supported(&self) -> usize {
        self.patches.iter().filter(|p| p.supported).count()
    }

    /// Whether every patch still verifies.
    pub fn all_supported(&self) -> bool {
        self.patches.iter().all(|p| p.supported)
    }

    /// The patches that no longer verify, i.e. have broken on this build.
    pub fn broken(&self) -> impl Iterator<Item = &PatchStatus> {
        self.patches.iter().filter(|p| !p.supported)
    }
}

fn bytes_match(build: &[u8], offset: u64, expected: &[u8]) -> bool {
    let start = offset as usize;
    build.get(start..start.saturating_add(expected.len())) == Some(expected)
}

/// First index at or after `from` where `needle` occurs in `haystack`.
fn find_from(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

/// Where a site was located in an image.
enum Located {
    /// Found at this offset (equal to the hint when it didn't drift).
    At(u64),
    /// Not at the hint, and either no signature was given or it did not match
    /// exactly once.
    Missing,
}

/// Find a site's real offset in `image`. The fast path is the hint still holding
/// its expected bytes, which is the common case on the exact pinned build. When
/// it has moved, and a signature is given, scan for the signature's single
/// occurrence and recover the drifted offset. Zero or several signature matches
/// both count as missing, because maycord must never guess which copy to patch.
fn locate(image: &[u8], hint: u64, expected: &[u8], signature: Option<&Signature>) -> Located {
    if bytes_match(image, hint, expected) {
        return Located::At(hint);
    }
    let Some(sig) = signature else {
        return Located::Missing;
    };
    let mut found = None;
    let mut from = 0;
    while let Some(at) = find_from(image, &sig.bytes, from) {
        if found.is_some() {
            return Located::Missing; // ambiguous: the signature is not unique
        }
        found = Some(at);
        from = at + 1;
    }
    match found {
        // The recovered spot must still carry the pinned bytes, so a signature
        // that happens to land next to different code is rejected. `checked_add`
        // stops a hostile `site_offset` from wrapping to a small, valid-looking
        // offset that then feeds a memory write.
        Some(at) => match (at as u64).checked_add(sig.site_offset) {
            Some(offset) if bytes_match(image, offset, expected) => Located::At(offset),
            _ => Located::Missing,
        },
        None => Located::Missing,
    }
}

/// Count how many times `needle` occurs in `haystack`. Used by tests to assert
/// exact multiplicity; the fast path uses [`occurs_once`].
#[cfg(test)]
fn occurrences(haystack: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut from = 0;
    while let Some(at) = find_from(haystack, needle, from) {
        count += 1;
        from = at + 1;
    }
    count
}

/// Whether `needle` occurs exactly once in `haystack`. Stops at the second hit,
/// so a common window (thousands of hits early in the search) is rejected after
/// two scans instead of counting them all. This is what keeps signature
/// extraction fast on a large binary.
fn occurs_once(haystack: &[u8], needle: &[u8]) -> bool {
    let Some(first) = find_from(haystack, needle, 0) else {
        return false;
    };
    find_from(haystack, needle, first + 1).is_none()
}

/// Build the shortest byte window around a site that occurs exactly once in
/// `image`, so the site can be relocated after drift. The window grows outward
/// from the site a byte at a time on each side until it is unique, up to
/// `max_radius`. This is the generator half of the signature scheme: it produces
/// what [`locate`] later consumes. Returns `None` if the site is out of range or
/// no unique window fits within `max_radius` (code so generic it never settles).
///
/// A signature from a single build is only guaranteed unique in that build; a
/// later version can add code that collides with it. Use
/// [`extract_signature_across`] with several builds for one that is far likelier
/// to stay unique.
pub fn extract_signature(
    image: &[u8],
    site_offset: u64,
    site_len: usize,
    max_radius: usize,
) -> Option<Signature> {
    extract_signature_across(&[image], site_offset, site_len, max_radius)
}

/// Like [`extract_signature`], but the window must occur exactly once in *every*
/// image in `images`, with the site located in `images[0]`. Validating a
/// signature against several builds at once is what makes it robust: a window
/// that is already unique across past versions is much more likely to stay
/// unique in the next one, which no single-build signature can promise. Returns
/// `None` if no window up to `max_radius` is unique in all of them.
pub fn extract_signature_across(
    images: &[&[u8]],
    site_offset: u64,
    site_len: usize,
    max_radius: usize,
) -> Option<Signature> {
    let first = *images.first()?;
    let site = usize::try_from(site_offset).ok()?;
    let site_end = site.checked_add(site_len)?;
    if site_end > first.len() || site_len == 0 {
        return None;
    }
    for radius in 0..=max_radius {
        let start = site.saturating_sub(radius);
        let end = site_end.saturating_add(radius).min(first.len());
        let window = &first[start..end];
        if images.iter().all(|image| occurs_once(image, window)) {
            return Some(Signature {
                bytes: window.to_vec(),
                site_offset: (site - start) as u64,
            });
        }
    }
    None
}

/// Encode `mov w0, #value; ret` for aarch64: eight bytes that make a function
/// return `value` immediately and ignore its body. This is maycord's form of the
/// most common Vencord patch, forcing a check to a constant (e.g. make a guard
/// return true or false). It is applied as a byte patch over the function entry;
/// because it never touches the stack pointer, returning straight away leaves the
/// caller's frame intact.
pub fn force_return_aarch64(value: u16) -> [u8; 8] {
    // MOVZ w0, #value  =>  0x52800000 | (value << 5)
    let movz: u32 = 0x5280_0000 | ((value as u32) << 5);
    const RET: u32 = 0xd65f_03c0;
    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(&movz.to_le_bytes());
    out[4..8].copy_from_slice(&RET.to_le_bytes());
    out
}

fn event_label(event: &EventKind) -> String {
    match event {
        EventKind::MessageReceived { .. } => "message_received".to_string(),
        _ => "event".to_string(),
    }
}

/// Rewrite every site and event offset to where it actually sits in `image`,
/// using each one's signature to recover from drift. Sites that can't be located
/// are dropped, so planning only ever sees patches that are really present in
/// this build, and the dropped labels are returned for reporting. This is what
/// lets the agent apply a drifted adapter instead of just detecting the drift.
pub fn resolve_adapter(adapter: &TargetAdapter, image: &[u8]) -> (TargetAdapter, Vec<String>) {
    let mut resolved = adapter.clone();
    let mut dropped = Vec::new();
    resolved.sites.retain_mut(|s| {
        match locate(image, s.image_offset, &s.expected_bytes, s.signature.as_ref()) {
            Located::At(off) => {
                s.image_offset = off;
                true
            }
            Located::Missing => {
                dropped.push(s.token.clone());
                false
            }
        }
    });
    resolved.events.retain_mut(|e| {
        match locate(image, e.image_offset, &e.expected_bytes, e.signature.as_ref()) {
            Located::At(off) => {
                e.image_offset = off;
                true
            }
            Located::Missing => {
                dropped.push(event_label(&e.event));
                false
            }
        }
    });
    (resolved, dropped)
}

fn status(label: String, hint: u64, located: Located) -> PatchStatus {
    match located {
        Located::At(offset) => PatchStatus {
            label,
            image_offset: hint,
            supported: true,
            resolved_offset: Some(offset),
        },
        Located::Missing => PatchStatus {
            label,
            image_offset: hint,
            supported: false,
            resolved_offset: None,
        },
    }
}

/// Check every patch in `adapter` against a candidate build's bytes, without
/// running it. This reads the build straight from disk, so it works against any
/// version: it reports whether this is the exact pinned build and which patches
/// still match. It is the basis of both the patch monitor and best-effort
/// application.
pub fn check_patches(adapter: &TargetAdapter, build: &[u8]) -> PatchReport {
    let sha256 = maycord_target::sha256(build);
    let mut patches = Vec::with_capacity(adapter.sites.len() + adapter.events.len());
    for site in &adapter.sites {
        let loc = locate(build, site.image_offset, &site.expected_bytes, site.signature.as_ref());
        patches.push(status(site.token.clone(), site.image_offset, loc));
    }
    for event in &adapter.events {
        let loc = locate(
            build,
            event.image_offset,
            &event.expected_bytes,
            event.signature.as_ref(),
        );
        patches.push(status(event_label(&event.event), event.image_offset, loc));
    }
    for patch in &adapter.patches {
        let loc = locate(
            build,
            patch.image_offset,
            &patch.expected_bytes,
            patch.signature.as_ref(),
        );
        patches.push(status(patch.token.clone(), patch.image_offset, loc));
    }
    PatchReport {
        identity_matches: sha256 == adapter.identity.sha256,
        sha256,
        patches,
    }
}

/// Apply every byte patch in `adapter` to an image in place, overwriting each
/// site's bytes with its replacement. Each patch is located by its offset or
/// signature and its expected bytes are verified first, so a patch that doesn't
/// fit this build is skipped rather than written to the wrong place. Returns one
/// `(token, applied)` per patch. This is a static, offline apply used for
/// testing and tooling; the agent applies the same patches in live memory.
pub fn apply_byte_patches(adapter: &TargetAdapter, image: &mut [u8]) -> Vec<(String, bool)> {
    adapter
        .patches
        .iter()
        .map(|patch| {
            let fits = patch.replacement_bytes.len() == patch.expected_bytes.len();
            let applied = fits
                && match locate(image, patch.image_offset, &patch.expected_bytes, patch.signature.as_ref()) {
                    Located::At(off) => {
                        let start = off as usize;
                        image[start..start + patch.replacement_bytes.len()]
                            .copy_from_slice(&patch.replacement_bytes);
                        true
                    }
                    Located::Missing => false,
                };
            (patch.token.clone(), applied)
        })
        .collect()
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
    use maycord_target::{Arch, BytePatchSite, ImageId, PaletteSite, TargetIdentity};

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
            returns: vec![],
            patches: vec![],
        }
    }

    fn site(token: &str, offset: u64, enc: ColorEncoding) -> PaletteSite {
        PaletteSite {
            token: token.into(),
            image_offset: offset,
            expected_bytes: vec![0x1f, 0x20, 0x03, 0xd5],
            encoding: enc,
            location: ValueLocation::Vector(8),
            signature: None,
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

    #[test]
    fn check_patches_reports_each_patch_and_version_fit() {
        // A build with the site's pinned bytes at offset 8.
        let mut build = vec![0u8; 64];
        build[8..12].copy_from_slice(&[0x1f, 0x20, 0x03, 0xd5]);
        let adapter = adapter_with(vec![site("sidebar", 8, ColorEncoding::SrgbU32)], &build);

        // Exact build: it is the pinned version and the patch verifies.
        let report = check_patches(&adapter, &build);
        assert!(report.identity_matches);
        assert!(report.all_supported());
        assert_eq!(report.supported(), 1);

        // A different build where that spot changed: no longer the pinned
        // version, and the patch is reported broken rather than applied.
        let mut other = build.clone();
        other[8] ^= 0x01;
        let report = check_patches(&adapter, &other);
        assert!(!report.identity_matches);
        assert!(!report.all_supported());
        assert_eq!(report.broken().next().unwrap().label, "sidebar");
    }

    fn at(located: Located) -> Option<u64> {
        match located {
            Located::At(offset) => Some(offset),
            Located::Missing => None,
        }
    }

    const EXPECT: &[u8] = &[0xaa, 0xbb, 0xcc, 0xdd];

    #[test]
    fn locate_takes_the_hint_when_its_bytes_are_there() {
        let mut build = vec![0u8; 64];
        build[8..12].copy_from_slice(EXPECT);
        // No signature needed: the hint still holds.
        assert_eq!(at(locate(&build, 8, EXPECT, None)), Some(8));
    }

    #[test]
    fn locate_recovers_a_drifted_site_via_its_signature() {
        // The hint (offset 8) is wrong, but the signature window sits at 20 with
        // the pinned bytes two bytes into it, so the real site is 22.
        let mut build = vec![0u8; 64];
        build[20..26].copy_from_slice(&[0x11, 0x22, 0xaa, 0xbb, 0xcc, 0xdd]);
        let sig = Signature { bytes: vec![0x11, 0x22, 0xaa, 0xbb, 0xcc, 0xdd], site_offset: 2 };
        assert_eq!(at(locate(&build, 8, EXPECT, Some(&sig))), Some(22));
    }

    #[test]
    fn locate_refuses_an_ambiguous_signature() {
        // The same window appears twice: maycord must not guess which to patch.
        let mut build = vec![0u8; 64];
        let win = [0x11, 0x22, 0xaa, 0xbb, 0xcc, 0xdd];
        build[20..26].copy_from_slice(&win);
        build[40..46].copy_from_slice(&win);
        let sig = Signature { bytes: win.to_vec(), site_offset: 2 };
        assert_eq!(at(locate(&build, 8, EXPECT, Some(&sig))), None);
    }

    #[test]
    fn locate_is_missing_when_the_hint_fails_and_there_is_no_signature() {
        let build = vec![0u8; 64];
        assert_eq!(at(locate(&build, 8, EXPECT, None)), None);
    }

    #[test]
    fn extract_signature_round_trips_through_locate_after_drift() {
        // A build where the 4-byte site repeats, so a bare pin is ambiguous but a
        // wider window around it is unique.
        let mut build = vec![0u8; 96];
        build[10..14].copy_from_slice(EXPECT); // a decoy copy of the site bytes
        build[40..44].copy_from_slice(EXPECT); // the real site
        build[38..40].copy_from_slice(&[0x5a, 0x5b]); // unique lead-in before the real one
        let sig = extract_signature(&build, 40, EXPECT.len(), 32).expect("unique window exists");
        // The window had to widen past the bare 4 bytes to become unique.
        assert!(sig.bytes.len() > EXPECT.len());

        // Now drift the whole thing forward by 16 bytes and confirm locate, given
        // only that signature, recovers the moved site.
        let mut drifted = vec![0u8; 16];
        drifted.extend_from_slice(&build);
        assert_eq!(at(locate(&drifted, 40, EXPECT, Some(&sig))), Some(40 + 16));
    }

    #[test]
    fn extract_signature_across_demands_uniqueness_in_every_build() {
        // In build A the site bytes are unique; build B adds a second copy of
        // them, so a cross-build signature must widen until unique in both.
        let mut a = vec![0u8; 64];
        a[28..30].copy_from_slice(&[0x5a, 0x5b]); // a unique lead-in, only here
        a[30..34].copy_from_slice(EXPECT);
        let mut b = a.clone();
        b[50..54].copy_from_slice(EXPECT); // a colliding copy, with no lead-in

        // Single-build extraction on A is happy with the bare bytes, which then
        // collide in B.
        let solo = extract_signature(&a, 30, EXPECT.len(), 16).unwrap();
        assert_eq!(solo.bytes, EXPECT);
        assert!(occurrences(&b, &solo.bytes) > 1);

        // Cross-build extraction widens until the window is unique in both.
        let cross = extract_signature_across(&[&a, &b], 30, EXPECT.len(), 16).unwrap();
        assert_eq!(occurrences(&a, &cross.bytes), 1);
        assert_eq!(occurrences(&b, &cross.bytes), 1);
        assert!(cross.bytes.len() > EXPECT.len());
    }

    #[test]
    fn force_return_encodes_movz_then_ret() {
        // mov w0, #1 ; ret
        assert_eq!(
            force_return_aarch64(1),
            [0x20, 0x00, 0x80, 0x52, 0xc0, 0x03, 0x5f, 0xd6]
        );
        // mov w0, #0 ; ret
        assert_eq!(
            force_return_aarch64(0),
            [0x00, 0x00, 0x80, 0x52, 0xc0, 0x03, 0x5f, 0xd6]
        );
    }

    #[test]
    fn apply_byte_patches_overwrites_verified_sites_in_place() {
        // "General" at offset 4; patch it to the same-length "maycord".
        let mut image = b"xxxxGeneralxxxx".to_vec();
        let mut adapter = adapter_with(vec![], &image);
        adapter.patches = vec![BytePatchSite {
            token: "label".into(),
            image_offset: 4,
            expected_bytes: b"General".to_vec(),
            replacement_bytes: b"maycord".to_vec(),
            signature: None,
        }];
        let result = apply_byte_patches(&adapter, &mut image);
        assert_eq!(result, vec![("label".to_string(), true)]);
        assert_eq!(&image, b"xxxxmaycordxxxx");

        // A patch whose expected bytes aren't there is reported unapplied and
        // changes nothing.
        let mut other = b"xxxxGeneralxxxx".to_vec();
        adapter.patches[0].expected_bytes = b"Special".to_vec();
        adapter.patches[0].replacement_bytes = b"maycord".to_vec();
        assert_eq!(apply_byte_patches(&adapter, &mut other), vec![("label".to_string(), false)]);
        assert_eq!(&other, b"xxxxGeneralxxxx");
    }

    #[test]
    fn resolve_adapter_rewrites_offsets_and_drops_the_unfound() {
        // sidebar drifts but has a signature to recover; background has no
        // signature and its hint fails, so it is dropped.
        let mut build = vec![0u8; 128];
        build[60..64].copy_from_slice(EXPECT);
        let mut sidebar = site("sidebar", 8, ColorEncoding::SrgbU32);
        sidebar.expected_bytes = EXPECT.to_vec();
        sidebar.signature = Some(Signature { bytes: EXPECT.to_vec(), site_offset: 0 });
        let mut background = site("background", 100, ColorEncoding::SrgbU32);
        background.expected_bytes = EXPECT.to_vec();
        let adapter = adapter_with(vec![sidebar, background], &build);

        let (resolved, dropped) = resolve_adapter(&adapter, &build);
        assert_eq!(resolved.sites.len(), 1);
        assert_eq!(resolved.sites[0].token, "sidebar");
        assert_eq!(resolved.sites[0].image_offset, 60); // recovered from 8
        assert_eq!(dropped, vec!["background".to_string()]);
    }
}
