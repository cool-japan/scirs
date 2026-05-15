//! Serde round-trip integration tests for EML types.
//!
//! Verifies JSON and oxicode binary round-trip preserves structural identity
//! and semantic equality.
//!
//! Background:
//! - `EmlNode`/`EmlTree`/`LoweredOp`/`OxiOp`/`Interval` all opt into
//!   `serde::Serialize`/`Deserialize` under the `serde` feature.
//! - `EmlNode::Eml` holds `Arc<EmlNode>` children, which require serde's
//!   `rc` feature to derive — enabled via crate-level `features = ["rc"]`
//!   on the optional `serde` dependency.
//! - oxicode's serde bridge lives in `oxicode::serde` (not at the crate
//!   root) — `encode_serde` / `decode_serde` are the equivalents of
//!   `serde_json::to_string` / `from_str` for oxicode binary format.

#![cfg(feature = "serde")]

use scirs2_symbolic::eml::{Canonical, EmlTree, Interval, LoweredOp, OxiOp};

#[test]
fn json_round_trip_emltree_one() {
    let t = EmlTree::one();
    let s = serde_json::to_string(&t).expect("serialize");
    let recovered: EmlTree = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(t, recovered);
    assert_eq!(t.structural_hash(), recovered.structural_hash());
}

#[test]
fn json_round_trip_emltree_eml() {
    let t = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
    let s = serde_json::to_string(&t).expect("serialize");
    let recovered: EmlTree = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(t, recovered);
    assert_eq!(t.num_vars(), recovered.num_vars());
}

#[test]
fn json_round_trip_canonical_sin() {
    let x = EmlTree::var(0);
    let t = Canonical::sin(&x);
    let s = serde_json::to_string(&t).expect("serialize");
    let recovered: EmlTree = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(t, recovered);
    assert_eq!(t.structural_hash(), recovered.structural_hash());
    assert_eq!(t.depth(), recovered.depth());
    assert_eq!(t.size(), recovered.size());
}

#[test]
fn json_round_trip_lowered_op() {
    let op = LoweredOp::Add(
        Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0)))),
        Box::new(LoweredOp::Const(1.0)),
    );
    let s = serde_json::to_string(&op).expect("serialize");
    let recovered: LoweredOp = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(op, recovered);
    assert_eq!(op.structural_hash(), recovered.structural_hash());
}

#[test]
fn json_round_trip_oxi_op_tape() {
    // Round-trip a flat OxiOp tape — the variant set is the same as
    // LoweredOp's, but without recursive Box children.
    let tape = vec![OxiOp::Var(0), OxiOp::Const(1.0), OxiOp::Add, OxiOp::Sin];
    let s = serde_json::to_string(&tape).expect("serialize");
    let recovered: Vec<OxiOp> = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(tape, recovered);
}

#[test]
fn json_round_trip_interval() {
    let i = Interval::new(1.0, 2.5);
    let s = serde_json::to_string(&i).expect("serialize");
    let recovered: Interval = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(i, recovered);
}

#[test]
fn oxicode_round_trip_emltree() {
    let x = EmlTree::var(0);
    let t = Canonical::add(&x, &EmlTree::one());
    let bytes = oxicode::serde::encode_serde(&t).expect("oxicode encode");
    let recovered: EmlTree = oxicode::serde::decode_serde(&bytes).expect("oxicode decode");
    assert_eq!(t, recovered);
    assert_eq!(t.structural_hash(), recovered.structural_hash());
}

#[test]
fn oxicode_round_trip_lowered_op_complex_formula() {
    let op = LoweredOp::Mul(
        Box::new(LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        )),
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0)))),
            Box::new(LoweredOp::Cos(Box::new(LoweredOp::Var(1)))),
        )),
    );
    let bytes = oxicode::serde::encode_serde(&op).expect("oxicode encode");
    let recovered: LoweredOp = oxicode::serde::decode_serde(&bytes).expect("oxicode decode");
    assert_eq!(op, recovered);
    assert_eq!(op.structural_hash(), recovered.structural_hash());
}

#[test]
fn oxicode_round_trip_interval() {
    let i = Interval::new(-2.0, 3.5);
    let bytes = oxicode::serde::encode_serde(&i).expect("oxicode encode");
    let recovered: Interval = oxicode::serde::decode_serde(&bytes).expect("oxicode decode");
    assert_eq!(i, recovered);
}

#[test]
fn cross_format_consistency() {
    // Verify JSON and oxicode produce semantically equivalent results.
    let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0)));
    let json_str = serde_json::to_string(&op).expect("json");
    let oxi_bytes = oxicode::serde::encode_serde(&op).expect("oxicode");

    let from_json: LoweredOp = serde_json::from_str(&json_str).expect("json decode");
    let from_oxi: LoweredOp = oxicode::serde::decode_serde(&oxi_bytes).expect("oxicode decode");

    assert_eq!(from_json, from_oxi);
    assert_eq!(from_json.structural_hash(), from_oxi.structural_hash());
}

#[test]
fn json_round_trip_deep_emltree() {
    // Round-trip a moderately deep tree (16 levels) to exercise the
    // Arc<EmlNode> serialization path. We don't push to 10k-deep here —
    // that would exercise recursion in serde itself, which is a Rust-stack
    // concern unrelated to our derives.
    let mut t = EmlTree::var(0);
    for _ in 0..16 {
        t = EmlTree::eml(&t, &EmlTree::one());
    }
    let s = serde_json::to_string(&t).expect("serialize");
    let recovered: EmlTree = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(t, recovered);
    assert_eq!(t.depth(), recovered.depth());
    assert_eq!(t.size(), recovered.size());
    assert_eq!(t.structural_hash(), recovered.structural_hash());
}
