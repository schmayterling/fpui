//! Locate a patch site by the stable string it references, instead of by a raw
//! offset or a window of machine code.
//!
//! A shipped Rust binary keeps its panic-location and format strings
//! (`crates/gpui/src/app.rs`, ...) long after symbols are stripped and
//! instruction bytes churn, because those are *source paths*, not code: they move
//! only when a file is renamed, where a byte signature moves on any recompile. So
//! a far more drift-resilient anchor is to find such a string, find the single
//! `adrp`+`add` pair that forms its address, and map that reference back to the
//! function that contains it.
//!
//! This augments byte-matching rather than replacing it: an anchor resolves
//! *where* a function is on a new build, and the caller still verifies the pinned
//! bytes there before writing. aarch64 + Mach-O only — maycord's target.

use std::collections::{BTreeSet, HashMap};

use object::read::macho::{LoadCommandVariant, MachOFile64};
use object::{Endianness, Object, ObjectSection, ObjectSegment};

/// A parsed 64-bit little/big-endian Mach-O over a borrowed image. The alias ties
/// the file's data lifetime to the backing slice so helpers can borrow it.
type Macho<'a> = MachOFile64<'a, Endianness, &'a [u8]>;

/// Where a string anchor resolved to in a build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Anchor {
    /// File offset of the enclosing function's entry — what to pin a detour to.
    pub func_offset: u64,
    /// File offset of the `adrp` that begins the reference, for diagnostics and
    /// for intra-function sites that want the reference itself.
    pub xref_offset: u64,
    /// The VM address the string resolved to.
    pub string_vmaddr: u64,
}

/// Why an anchor could not be resolved. Every ambiguous case is a refusal, not a
/// guess: maycord must never pin a site to the wrong function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnchorError {
    /// The image did not parse as a 64-bit Mach-O.
    Parse(String),
    /// The string was not found in any read-only data section.
    StringMissing,
    /// The string occurs more than once, so its address is ambiguous.
    StringAmbiguous(usize),
    /// No `adrp`+`add` pair forms the string's address, so nothing references it.
    NoReference,
    /// The string is referenced from more than one function; pick a more specific
    /// string that only the target function mentions.
    ReferenceAmbiguous(usize),
    /// The build has no `LC_FUNCTION_STARTS`, so function bounds are unknown.
    NoFunctionStarts,
    /// The build has no `__text` section to scan.
    NoText,
}

impl core::fmt::Display for AnchorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "not a 64-bit Mach-O: {e}"),
            Self::StringMissing => write!(f, "string not found in any read-only section"),
            Self::StringAmbiguous(n) => write!(f, "string occurs {n} times; its address is ambiguous"),
            Self::NoReference => write!(f, "no adrp+add pair forms the string's address"),
            Self::ReferenceAmbiguous(n) => {
                write!(f, "string is referenced from {n} functions; use a more specific string")
            }
            Self::NoFunctionStarts => write!(f, "build has no LC_FUNCTION_STARTS"),
            Self::NoText => write!(f, "build has no __text section"),
        }
    }
}

impl std::error::Error for AnchorError {}

/// Resolve `needle` to the function that references it in `image`. The string
/// must occur exactly once and be referenced from exactly one function. A thin
/// wrapper over [`resolve_anchors`] for the single-string case.
pub fn resolve_anchor(image: &[u8], needle: &str) -> Result<Anchor, AnchorError> {
    resolve_anchors(image, &[needle])?.pop().unwrap()
}

/// Resolve several string anchors in a SINGLE pass over `__text`.
///
/// The `__text` scan is the expensive part (~0.6 s on a 177 MB image); parsing
/// and the string search are cheap (~0.1 s). Checking every `adrp`+`add` against
/// a *set* of target addresses means N anchors cost one scan, not N — which is
/// what makes resolving a whole adapter's sites affordable at load time. The
/// returned vector is aligned with `needles`: one string's `Err` (missing,
/// ambiguous, not referenced) never sinks the others. The outer `Err` is only for
/// a malformed image with no `__text`/function starts, where nothing can resolve.
pub fn resolve_anchors(
    image: &[u8],
    needles: &[&str],
) -> Result<Vec<Result<Anchor, AnchorError>>, AnchorError> {
    let macho = Macho::parse(image).map_err(|e| AnchorError::Parse(e.to_string()))?;
    let endian = macho.endian();

    // The __TEXT segment's vmaddr is the image base that function starts and the
    // `adrp` page math are both relative to.
    let text_seg_vmaddr = macho
        .segments()
        .find(|s| matches!(s.name(), Ok(Some("__TEXT"))))
        .map(|s| s.address())
        .ok_or(AnchorError::NoText)?;
    let starts = function_starts(&macho, endian, text_seg_vmaddr)?;
    let text = macho
        .sections()
        .find(|s| matches!(s.name(), Ok("__text")))
        .ok_or(AnchorError::NoText)?;
    let data = text.data().map_err(|e| AnchorError::Parse(e.to_string()))?;
    let text_vmaddr = text.address();
    let (text_file_start, _) = text.file_range().ok_or(AnchorError::NoText)?;

    // Resolve each string to its unique address up front, in ONE pass over the
    // read-only data (~28 MB), not one pass per needle. A missing or ambiguous
    // string takes its error now and is left out of the scan; the rest become
    // scan targets, keyed by address back to the slot that wants them.
    let (counts, firsts) = find_strings(&macho, needles);
    let mut results: Vec<Result<Anchor, AnchorError>> = Vec::with_capacity(needles.len());
    let mut slot_vmaddr: Vec<Option<u64>> = vec![None; needles.len()];
    let mut targets: HashMap<u64, usize> = HashMap::new();
    for i in 0..needles.len() {
        match (counts[i], firsts[i]) {
            (0, _) => results.push(Err(AnchorError::StringMissing)),
            (1, Some(vmaddr)) => {
                results.push(Err(AnchorError::NoReference)); // filled in after the scan
                slot_vmaddr[i] = Some(vmaddr);
                targets.insert(vmaddr, i);
            }
            (n, _) => results.push(Err(AnchorError::StringAmbiguous(n))),
        }
    }

    // One pass over __text: each adjacent `adrp`+`add` that forms a target address
    // records the function it sits in, per anchor slot.
    let mut funcs: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); needles.len()];
    let mut first_xref: Vec<Option<u64>> = vec![None; needles.len()];
    if !targets.is_empty() {
        let mut prev: Option<(usize, u32)> = None;
        for (idx, chunk) in data.chunks_exact(4).enumerate() {
            let word = u32::from_le_bytes(chunk.try_into().unwrap());
            if let Some((adrp_idx, adrp_word)) = prev {
                let pc = text_vmaddr + (adrp_idx as u64) * 4;
                if let (Some((rd, page)), Some((add_rd, add_rn, val))) =
                    (decode_adrp(adrp_word, pc), decode_add_imm(word))
                {
                    if add_rn == rd && add_rd == rd {
                        if let Some(&slot) = targets.get(&page.wrapping_add(val)) {
                            if let Some(func) = enclosing(&starts, pc) {
                                funcs[slot].insert(func);
                                first_xref[slot].get_or_insert(pc);
                            }
                        }
                    }
                }
            }
            prev = Some((idx, word));
        }
    }

    // Decide each scanned slot: exactly one enclosing function is a hit.
    for (i, vmaddr) in slot_vmaddr.into_iter().enumerate() {
        let Some(string_vmaddr) = vmaddr else { continue };
        results[i] = match funcs[i].len() {
            0 => Err(AnchorError::NoReference),
            1 => {
                let func = *funcs[i].iter().next().unwrap();
                let xref_pc = first_xref[i].unwrap_or(func);
                Ok(Anchor {
                    func_offset: text_file_start + (func - text_vmaddr),
                    xref_offset: text_file_start + (xref_pc - text_vmaddr),
                    string_vmaddr,
                })
            }
            n => Err(AnchorError::ReferenceAmbiguous(n)),
        };
    }
    Ok(results)
}

/// For every needle, its occurrence count and first VM address across the
/// read-only data sections, found in a single pass per section. Needles are
/// bucketed by first byte, so the scan cost is independent of the needle count —
/// one rodata pass resolves a whole adapter's strings. Rust string slices are not
/// NUL-terminated, so this matches exact bytes, not a C string. An empty needle
/// has no first byte, so it is never bucketed and reports zero occurrences.
fn find_strings(macho: &Macho<'_>, needles: &[&str]) -> (Vec<usize>, Vec<Option<u64>>) {
    let mut counts = vec![0usize; needles.len()];
    let mut first: Vec<Option<u64>> = vec![None; needles.len()];
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 256];
    for (i, n) in needles.iter().enumerate() {
        if let Some(&b0) = n.as_bytes().first() {
            buckets[b0 as usize].push(i);
        }
    }
    for sec in macho.sections() {
        // Strings live in read-only data; __text is code and would only match by
        // coincidence, so it is excluded to keep the anchor honest.
        if !matches!(sec.name(), Ok("__cstring" | "__const" | "__oslogstring")) {
            continue;
        }
        let Ok(data) = sec.data() else { continue };
        let base = sec.address();
        for pos in 0..data.len() {
            for &ni in &buckets[data[pos] as usize] {
                let nb = needles[ni].as_bytes();
                if data.len() - pos >= nb.len() && &data[pos..pos + nb.len()] == nb {
                    counts[ni] += 1;
                    first[ni].get_or_insert(base + pos as u64);
                }
            }
        }
    }
    (counts, first)
}

/// Decode `LC_FUNCTION_STARTS` into the sorted list of function entry VM
/// addresses. The command holds ULEB128 deltas: the first from the image base,
/// each next from the previous entry, terminated by a zero delta.
fn function_starts(
    macho: &Macho<'_>,
    endian: Endianness,
    image_base: u64,
) -> Result<Vec<u64>, AnchorError> {
    let data = macho.data();
    let mut blob: Option<&[u8]> = None;
    let commands = macho
        .macho_load_commands()
        .map_err(|e| AnchorError::Parse(e.to_string()))?;
    for command in commands {
        let command = command.map_err(|e| AnchorError::Parse(e.to_string()))?;
        if command.cmd() != object::macho::LC_FUNCTION_STARTS {
            continue;
        }
        if let Ok(LoadCommandVariant::LinkeditData(d)) = command.variant() {
            let off = d.dataoff.get(endian) as usize;
            let size = d.datasize.get(endian) as usize;
            blob = data.get(off..off.saturating_add(size));
        }
    }
    let blob = blob.ok_or(AnchorError::NoFunctionStarts)?;

    let mut starts = Vec::new();
    let mut addr = image_base;
    let mut i = 0;
    while i < blob.len() {
        let (delta, used) = uleb128(&blob[i..]);
        if used == 0 {
            break;
        }
        i += used;
        if delta == 0 {
            break;
        }
        addr = addr.wrapping_add(delta);
        starts.push(addr);
    }
    if starts.is_empty() {
        return Err(AnchorError::NoFunctionStarts);
    }
    Ok(starts)
}

/// The largest function start at or before `addr` (the function that contains
/// it). `starts` is ascending.
fn enclosing(starts: &[u64], addr: u64) -> Option<u64> {
    match starts.binary_search(&addr) {
        Ok(i) => Some(starts[i]),
        Err(0) => None,
        Err(i) => Some(starts[i - 1]),
    }
}

/// Decode an `adrp Xd, imm` into `(Xd, page_base)`, the 4 KiB page its result
/// points at. Returns `None` if the word is not an `adrp`.
fn decode_adrp(word: u32, pc: u64) -> Option<(u32, u64)> {
    if word & 0x9F00_0000 != 0x9000_0000 {
        return None;
    }
    let rd = word & 0x1F;
    let immlo = (word >> 29) & 0x3;
    let immhi = (word >> 5) & 0x7_FFFF;
    // 21-bit signed page count, sign-extended from bit 20.
    let imm21 = (((immhi << 2) | immlo) as i64) << 43 >> 43;
    let page = (pc & !0xFFF).wrapping_add((imm21 << 12) as u64);
    Some((rd, page))
}

/// Decode an `add Xd, Xn, #imm` (64-bit, immediate) into `(Xd, Xn, value)`.
/// Returns `None` if the word is not such an `add`.
fn decode_add_imm(word: u32) -> Option<(u32, u32, u64)> {
    if word & 0xFF80_0000 != 0x9100_0000 {
        return None;
    }
    let sh = (word >> 22) & 1;
    let imm12 = (word >> 10) & 0xFFF;
    let rn = (word >> 5) & 0x1F;
    let rd = word & 0x1F;
    let value = if sh == 1 { (imm12 as u64) << 12 } else { imm12 as u64 };
    Some((rd, rn, value))
}

/// Read one ULEB128 value, returning it and how many bytes it used (0 if the
/// slice ended mid-value).
fn uleb128(bytes: &[u8]) -> (u64, usize) {
    let mut result = 0u64;
    let mut shift = 0u32;
    for (i, &byte) in bytes.iter().enumerate() {
        result |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return (result, i + 1);
        }
        shift += 7;
        if shift >= 64 {
            break;
        }
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_adrp_add_matches_a_real_zed_pair() {
        // From /Applications/Zed.app: at 0x1_0000_1ec8 `adrp x2, 0x1_09fa_e000`
        // then `add x2, x2, #0x5a0`, forming 0x1_09fa_e5a0 (verified with otool).
        let (rd, page) = decode_adrp(0xb004_fd62, 0x1_0000_1ec8).expect("is adrp");
        assert_eq!(rd, 2);
        assert_eq!(page, 0x1_09fa_e000);
        let (add_rd, add_rn, val) = decode_add_imm(0x9116_8042).expect("is add");
        assert_eq!((add_rd, add_rn, val), (2, 2, 0x5a0));
        assert_eq!(page + val, 0x1_09fa_e5a0);
    }

    #[test]
    fn decode_rejects_other_instructions() {
        assert!(decode_adrp(0xd503_201f, 0).is_none()); // nop
        assert!(decode_add_imm(0xd503_201f).is_none());
        // An adrp is not an add and vice versa.
        assert!(decode_add_imm(0xb004_fd62).is_none());
        assert!(decode_adrp(0x9116_8042, 0).is_none());
    }

    #[test]
    fn adrp_sign_extends_a_negative_page_offset() {
        // `adrp x0, #-4096`: immlo=0b11, immhi=0x7ffff (all ones) => -1 page.
        let word = 0x9000_0000 | (0b11 << 29) | (0x7_FFFF << 5);
        let (rd, page) = decode_adrp(word, 0x1_0000_0000).expect("is adrp");
        assert_eq!(rd, 0);
        assert_eq!(page, 0x0_FFFF_F000);
    }

    #[test]
    fn uleb128_decodes_multibyte_values() {
        assert_eq!(uleb128(&[0x00]), (0, 1));
        assert_eq!(uleb128(&[0x7f]), (0x7f, 1));
        assert_eq!(uleb128(&[0x80, 0x01]), (128, 2));
        assert_eq!(uleb128(&[0xe5, 0x8e, 0x26]), (624485, 3));
        assert_eq!(uleb128(&[0x80]), (0, 0)); // unterminated
    }

    #[test]
    fn enclosing_finds_the_containing_function() {
        let starts = [0x100, 0x200, 0x340];
        assert_eq!(enclosing(&starts, 0x100), Some(0x100));
        assert_eq!(enclosing(&starts, 0x1ff), Some(0x100));
        assert_eq!(enclosing(&starts, 0x200), Some(0x200));
        assert_eq!(enclosing(&starts, 0x400), Some(0x340));
        assert_eq!(enclosing(&starts, 0x0ff), None);
    }
}
