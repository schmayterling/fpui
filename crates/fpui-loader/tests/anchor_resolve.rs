//! Proof that `resolve_adapter` re-resolves an anchored function entry from a
//! string in a real (stripped) Mach-O and drops a site whose anchor can't be
//! resolved. `fixtures/anchor_victim` is a tiny arm64 Mach-O whose `anchor_target`
//! function references a unique sentinel string; a second, un-anchored function
//! and `main` keep the binary realistic (function starts, a `__text` scan).

use fpui_loader::{anchor, resolve_adapter};
use fpui_target::{
    Arch, ImageId, ReplaceOp, ReplaceSite, ReturnOp, ReturnSite, StringAnchor, TargetAdapter,
    TargetIdentity,
};

/// The string only `anchor_target` references.
const SENTINEL: &str = "maycord::anchor::unit_fixture::unique_sentinel_9f3a";

/// `anchor_target`'s file offset in the fixture, read independently of the anchor
/// code: `nm` puts `_anchor_target` at vmaddr 0x1_0000_0460 and the __TEXT base is
/// 0x1_0000_0000, so its entry is file offset 0x460. Rebuild the fixture and this
/// moves — refresh it from `nm` if you do.
const TARGET_FILE_OFFSET: u64 = 0x460;

const IMAGE: &[u8] = include_bytes!("fixtures/anchor_victim");

fn ret(token: &str, image_offset: u64, anchor: Option<&str>) -> ReturnSite {
    ReturnSite {
        token: token.into(),
        image_offset,
        // Not the gate here: returns aren't byte-checked inside resolve_adapter.
        expected_bytes: vec![0x00, 0x00, 0x00, 0x90],
        op: ReturnOp::InvertBool,
        signature: None,
        anchor: anchor.map(|text| StringAnchor { text: text.into() }),
    }
}

fn adapter(returns: Vec<ReturnSite>) -> TargetAdapter {
    TargetAdapter {
        name: "anchor fixture".into(),
        identity: TargetIdentity {
            arch: Arch::Aarch64,
            sha256: fpui_target::sha256(IMAGE),
            image: ImageId::MachUuid([0; 16]),
        },
        sites: vec![],
        events: vec![],
        returns,
        args: vec![],
        replaces: vec![],
        patches: vec![],
        appends: vec![],
        paints: vec![],
        elements: vec![],
    }
}

#[test]
fn resolve_adapter_rewrites_an_anchored_replace_site() {
    // A `replace` site threads through the same anchor machinery as returns:
    // its image_offset is a lie the sentinel anchor must correct to the real
    // `anchor_target` entry at 0x460.
    let mut a = adapter(vec![]);
    a.replaces = vec![ReplaceSite {
        token: "guard".into(),
        image_offset: 0xBEEF,
        expected_bytes: vec![0x00, 0x00, 0x00, 0x90],
        op: ReplaceOp::ReturnConst { value: 0 },
        signature: None,
        anchor: Some(StringAnchor {
            text: SENTINEL.into(),
        }),
    }];

    let (resolved, dropped) = resolve_adapter(&a, IMAGE);

    assert!(dropped.is_empty());
    assert_eq!(resolved.replaces.len(), 1);
    assert_eq!(resolved.replaces[0].image_offset, TARGET_FILE_OFFSET);
}

#[test]
fn resolve_adapter_rewrites_an_anchored_entry_and_drops_an_unresolvable_one() {
    // Three return sites: one anchored to the sentinel (its image_offset is a lie
    // the anchor must correct), one anchored to a string that isn't in the image
    // (must be dropped), one with no anchor (must be left exactly as-is).
    let a = adapter(vec![
        ret("good", 0xDEAD, Some(SENTINEL)),
        ret(
            "missing",
            0x1000,
            Some("string_that_is_not_in_this_binary_zzz"),
        ),
        ret("plain", 0x2000, None),
    ]);

    let (resolved, dropped) = resolve_adapter(&a, IMAGE);

    // The unresolvable anchor dropped its site and reported it by token.
    assert_eq!(dropped, vec!["missing".to_string()]);

    // The good and plain sites survive, in their original order.
    let tokens: Vec<&str> = resolved.returns.iter().map(|r| r.token.as_str()).collect();
    assert_eq!(tokens, vec!["good", "plain"]);

    // The anchored site was re-pointed from its bogus hint to the real entry,
    // which `nm`/`otool` and the anchor API independently agree sits at 0x460.
    let good = &resolved.returns[0];
    assert_ne!(
        good.image_offset, 0xDEAD,
        "the bogus hint must be overwritten"
    );
    assert_eq!(good.image_offset, TARGET_FILE_OFFSET);
    assert_eq!(
        good.image_offset,
        anchor::resolve_anchor(IMAGE, SENTINEL).unwrap().func_offset,
    );

    // The un-anchored site is untouched: resolve_adapter never relocates a return
    // that has no anchor.
    assert_eq!(resolved.returns[1].image_offset, 0x2000);
}
