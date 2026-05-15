//! Parser: text → [`EmlTree`].
//!
//! ## Grammar
//!
//! ```text
//! expr     ::= "1" | var | eml
//! var      ::= ("x" | "X") digit+
//! eml      ::= ("eml" | "E") "(" expr "," expr ")"
//! ```
//!
//! Whitespace is tolerated between tokens. Both `eml(x, y)` and `E(x, y)` are
//! accepted (case-sensitive — `e` only in lowercase `eml`, `E` for the short
//! form).
//!
//! # Phase 0 caveat
//!
//! [`parse`] uses recursive descent. User-typed input is rarely deeper than a
//! handful of nodes, so recursion is acceptable in Phase 0; the canonical
//! `sin` tree (543 nodes deep) is built via [`crate::eml::canonical`]
//! constructors, not parsed. Should the need arise, the descent can be
//! converted to an explicit shift-reduce loop without changing the public
//! API.
//!
//! # Examples
//!
//! ```
//! use scirs2_symbolic::eml::parser::parse;
//!
//! let tree = parse("eml(x0, 1)").expect("parse should succeed");
//! assert_eq!(tree.depth(), 1);
//! ```
//!
//! # Adapted from oxieml v0.1.0, `src/parser.rs`
//!
//! Single-pass recursive-descent shape preserved; integrated with
//! scirs2-symbolic's [`EmlError::ParseError`] error vocabulary and
//! [`EmlTree`] hash-consed constructors. The `to_compact_string` formatter
//! is reimplemented iteratively to avoid OS-stack overflow on deep trees
//! produced by lowering / `Canonical::*`.

use crate::eml::tree::{EmlNode, EmlTree};
use crate::error::EmlError;

/// Parse a string into an [`EmlTree`].
///
/// Accepts:
///
/// - `1` — the EML constant
/// - `xN` or `XN` (where `N` is a non-negative integer) — variable at index N
/// - `eml(expr, expr)` or `E(expr, expr)` — the binary EML operator
///
/// Whitespace between tokens is tolerated. The comma may have surrounding
/// spaces.
///
/// # Errors
///
/// Returns [`EmlError::ParseError`] with a byte position and human-readable
/// message on syntactic failure.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::eml::parser::parse;
/// use scirs2_symbolic::eml::tree::EmlTree;
///
/// let t = parse("eml(x0, 1)").expect("valid syntax");
/// assert_eq!(t, EmlTree::eml(&EmlTree::var(0), &EmlTree::one()));
/// ```
pub fn parse(input: &str) -> Result<EmlTree, EmlError> {
    let bytes = input.as_bytes();
    let mut pos = 0usize;
    skip_ws(bytes, &mut pos);
    let tree = parse_expr(bytes, &mut pos)?;
    skip_ws(bytes, &mut pos);
    if pos != bytes.len() {
        return Err(EmlError::ParseError {
            position: pos,
            message: format!("unexpected trailing input at position {pos}"),
        });
    }
    Ok(tree)
}

/// Format an [`EmlTree`] as a compact text expression — inverse of [`parse`].
///
/// Uses the canonical lowercase form `eml(left, right)`. The round-trip
/// identity `parse(to_compact_string(&t))? == t` holds.
///
/// The implementation is iterative and uses the post-order
/// [`crate::eml::tree::PostOrderIter`] indirectly via an explicit
/// continuation stack — safe for trees hundreds of nodes deep.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::eml::parser::{parse, to_compact_string};
/// use scirs2_symbolic::eml::tree::EmlTree;
///
/// let t = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
/// let s = to_compact_string(&t);
/// assert_eq!(s, "eml(x0, 1)");
/// assert_eq!(parse(&s).expect("round-trip"), t);
/// ```
pub fn to_compact_string(tree: &EmlTree) -> String {
    // Iterative emitter using a continuation stack.
    //
    // Each work entry is a `Frame` describing what needs to happen next:
    // - `Visit(node)` — about to process `node`. For Eml, push the closing
    //   continuation and the right child + separator + left child, then
    //   emit the opening "eml(".
    // - `Emit(static_str)` — append the literal string to the output.
    //
    // The total number of frames is O(tree.size()), and the work stack
    // never recurses on the OS stack.
    enum Frame<'a> {
        Visit(&'a EmlNode),
        Emit(&'static str),
    }

    let mut output = String::with_capacity(tree.size().saturating_mul(8));
    let mut work: Vec<Frame<'_>> = Vec::with_capacity(tree.size().saturating_mul(2));
    work.push(Frame::Visit(&tree.root));

    while let Some(frame) = work.pop() {
        match frame {
            Frame::Emit(s) => output.push_str(s),
            Frame::Visit(node) => match node {
                EmlNode::One => output.push('1'),
                EmlNode::Var(i) => {
                    output.push('x');
                    // Inline integer formatting via std's Display — small alloc
                    // is acceptable; usize has no const-time encoding helper.
                    output.push_str(&i.to_string());
                }
                EmlNode::Eml { left, right } => {
                    // Emit "eml(" now, then schedule the rest in reverse order
                    // (LIFO) so the actual output sequence is:
                    //   "eml(" + left + ", " + right + ")"
                    output.push_str("eml(");
                    work.push(Frame::Emit(")"));
                    work.push(Frame::Visit(right));
                    work.push(Frame::Emit(", "));
                    work.push(Frame::Visit(left));
                }
            },
        }
    }

    output
}

// ============================================================================
// Internal: recursive-descent parser
// ============================================================================

/// Parse a single expression starting at `*pos`.
///
/// Recursive on the grammar's recursive shape (`eml(expr, expr)`); see the
/// module-level Phase 0 caveat for why this is acceptable.
fn parse_expr(bytes: &[u8], pos: &mut usize) -> Result<EmlTree, EmlError> {
    skip_ws(bytes, pos);
    if *pos >= bytes.len() {
        return Err(EmlError::ParseError {
            position: *pos,
            message: "unexpected end of input".to_string(),
        });
    }

    let b = bytes[*pos];

    // Constant `1` — but make sure it isn't the prefix of something else
    // (e.g. `12` would be a multi-digit literal which we do NOT support).
    if b == b'1' {
        let next = bytes.get(*pos + 1).copied();
        if next.is_none_or(|c| !c.is_ascii_digit()) {
            *pos += 1;
            return Ok(EmlTree::one());
        }
        return Err(EmlError::ParseError {
            position: *pos,
            message:
                "multi-digit numeric literals are not supported (only `1` is a valid EML constant)"
                    .to_string(),
        });
    }

    // Variable `xN` or `XN`
    if b == b'x' || b == b'X' {
        return parse_var(bytes, pos);
    }

    // EML call — `eml(...)` or `E(...)`
    if b == b'e' && bytes[*pos..].starts_with(b"eml(") {
        *pos += 4;
        return parse_eml_body(bytes, pos);
    }
    if b == b'E' {
        // Could be `E(` (short form) or the start of `Eml(` (NOT in the grammar
        // — we only accept lowercase `eml`).
        if bytes[*pos..].starts_with(b"E(") {
            *pos += 2;
            return parse_eml_body(bytes, pos);
        }
        return Err(EmlError::ParseError {
            position: *pos,
            message: "expected `E(` (uppercase short form) or `eml(` (lowercase long form)"
                .to_string(),
        });
    }

    Err(EmlError::ParseError {
        position: *pos,
        message: format!("unexpected character {:?} (byte 0x{:02x})", b as char, b),
    })
}

/// After `eml(` or `E(` has been consumed, parse `expr , expr )`.
fn parse_eml_body(bytes: &[u8], pos: &mut usize) -> Result<EmlTree, EmlError> {
    let left = parse_expr(bytes, pos)?;
    skip_ws(bytes, pos);
    expect(bytes, pos, b',')?;
    let right = parse_expr(bytes, pos)?;
    skip_ws(bytes, pos);
    expect(bytes, pos, b')')?;
    Ok(EmlTree::eml(&left, &right))
}

/// Parse a `xN` / `XN` variable token starting at `*pos`.
fn parse_var(bytes: &[u8], pos: &mut usize) -> Result<EmlTree, EmlError> {
    *pos += 1; // skip the leading 'x' / 'X'
    let start = *pos;
    while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
        *pos += 1;
    }
    if *pos == start {
        return Err(EmlError::ParseError {
            position: start,
            message: "expected at least one decimal digit after 'x' or 'X'".to_string(),
        });
    }
    let idx_str = match std::str::from_utf8(&bytes[start..*pos]) {
        Ok(s) => s,
        Err(_) => {
            return Err(EmlError::ParseError {
                position: start,
                message: "non-utf8 bytes in variable index".to_string(),
            });
        }
    };
    let idx: usize = match idx_str.parse() {
        Ok(n) => n,
        Err(_) => {
            return Err(EmlError::ParseError {
                position: start,
                message: format!("invalid variable index '{idx_str}' (out of range for usize)"),
            });
        }
    };
    Ok(EmlTree::var(idx))
}

/// Consume a single expected byte; returns a [`EmlError::ParseError`] otherwise.
fn expect(bytes: &[u8], pos: &mut usize, ch: u8) -> Result<(), EmlError> {
    if *pos < bytes.len() && bytes[*pos] == ch {
        *pos += 1;
        Ok(())
    } else {
        let found = if *pos < bytes.len() {
            format!("{:?}", bytes[*pos] as char)
        } else {
            "end of input".to_string()
        };
        Err(EmlError::ParseError {
            position: *pos,
            message: format!("expected {:?}, found {}", ch as char, found),
        })
    }
}

/// Advance `*pos` past any ASCII whitespace bytes.
fn skip_ws(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && bytes[*pos].is_ascii_whitespace() {
        *pos += 1;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ----------------------------------------------------------------
    // parse — happy paths
    // ----------------------------------------------------------------

    #[test]
    fn parse_one() {
        let t = parse("1").expect("parse should succeed");
        assert_eq!(t, EmlTree::one());
    }

    #[test]
    fn parse_var_x0() {
        assert_eq!(parse("x0").expect("parse"), EmlTree::var(0));
    }

    #[test]
    fn parse_var_x_uppercase() {
        // Variables accept both cases.
        assert_eq!(parse("X7").expect("parse"), EmlTree::var(7));
    }

    #[test]
    fn parse_var_multi_digit() {
        assert_eq!(parse("x123").expect("parse"), EmlTree::var(123));
    }

    #[test]
    fn parse_eml_lowercase() {
        let t = parse("eml(x0, 1)").expect("parse");
        assert_eq!(t, EmlTree::eml(&EmlTree::var(0), &EmlTree::one()));
    }

    #[test]
    fn parse_eml_uppercase_short() {
        let t = parse("E(x0, 1)").expect("parse");
        assert_eq!(t, EmlTree::eml(&EmlTree::var(0), &EmlTree::one()));
    }

    #[test]
    fn parse_whitespace_tolerant() {
        let t = parse("  eml(  x0 ,  1  )  ").expect("parse");
        assert_eq!(t, EmlTree::eml(&EmlTree::var(0), &EmlTree::one()));
    }

    #[test]
    fn parse_no_whitespace() {
        let t = parse("E(E(1,E(1,1)),1)").expect("parse");
        assert_eq!(t.depth(), 3);
    }

    #[test]
    fn parse_nested() {
        let t = parse("eml(eml(x0, 1), x1)").expect("parse");
        assert_eq!(t.depth(), 2);
        assert_eq!(t.num_vars(), 2);
    }

    #[test]
    fn parse_mixed_case() {
        // E and eml are interchangeable.
        let a = parse("E(eml(x0, 1), 1)").expect("parse mixed");
        let b = parse("eml(E(x0, 1), 1)").expect("parse mixed");
        assert_eq!(a, b);
    }

    // ----------------------------------------------------------------
    // parse — error paths
    // ----------------------------------------------------------------

    #[test]
    fn parse_error_empty() {
        let r = parse("");
        assert!(matches!(r, Err(EmlError::ParseError { .. })));
    }

    #[test]
    fn parse_error_unterminated() {
        let r = parse("eml(x0, 1");
        assert!(matches!(r, Err(EmlError::ParseError { .. })));
    }

    #[test]
    fn parse_error_invalid_var() {
        // `xfoo` — 'x' must be followed by digits.
        let r = parse("xfoo");
        assert!(matches!(r, Err(EmlError::ParseError { .. })));
    }

    #[test]
    fn parse_error_trailing_input() {
        let r = parse("1 2");
        assert!(matches!(r, Err(EmlError::ParseError { .. })));
    }

    #[test]
    fn parse_error_unknown_byte() {
        let r = parse("zzz");
        assert!(matches!(r, Err(EmlError::ParseError { .. })));
    }

    #[test]
    fn parse_error_missing_comma() {
        let r = parse("eml(x0 1)");
        assert!(matches!(r, Err(EmlError::ParseError { .. })));
    }

    #[test]
    fn parse_error_multi_digit_constant() {
        // `12` is not a valid EML constant — only `1` is.
        let r = parse("12");
        assert!(matches!(r, Err(EmlError::ParseError { .. })));
    }

    // ----------------------------------------------------------------
    // to_compact_string + round-trip
    // ----------------------------------------------------------------

    #[test]
    fn compact_one() {
        assert_eq!(to_compact_string(&EmlTree::one()), "1");
    }

    #[test]
    fn compact_var() {
        assert_eq!(to_compact_string(&EmlTree::var(42)), "x42");
    }

    #[test]
    fn compact_eml() {
        let t = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        assert_eq!(to_compact_string(&t), "eml(x0, 1)");
    }

    #[test]
    fn compact_nested() {
        let t = EmlTree::eml(
            &EmlTree::eml(&EmlTree::var(0), &EmlTree::one()),
            &EmlTree::var(1),
        );
        assert_eq!(to_compact_string(&t), "eml(eml(x0, 1), x1)");
    }

    #[test]
    fn round_trip_simple() {
        let original = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        let s = to_compact_string(&original);
        let parsed = parse(&s).expect("round-trip parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn round_trip_nested() {
        let original = EmlTree::eml(
            &EmlTree::var(0),
            &EmlTree::eml(&EmlTree::one(), &EmlTree::var(1)),
        );
        let s = to_compact_string(&original);
        let parsed = parse(&s).expect("round-trip parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn compact_deep_no_overflow() {
        // Build a deep tree and ensure to_compact_string doesn't blow the stack.
        let one = EmlTree::one();
        let mut t = one.clone();
        for _ in 0..1000 {
            t = EmlTree::eml(&t, &one);
        }
        let s = to_compact_string(&t);
        // Don't compare the (huge) string; just confirm it's non-empty and ends with `)`.
        assert!(s.ends_with(')'));
    }

    #[test]
    fn parse_deep_round_trip_no_overflow() {
        // Parse-side stack-safety: build a 500-deep tree, serialise it,
        // then parse the result. If `parse` is too aggressive recursively
        // (or the recursion frame bloats in a future refactor), the OS
        // stack overflows here.
        //
        // Empirically validates the Phase 0 caveat in the module docs:
        // recursive descent is acceptable for ≤ 500-deep input.
        let one = EmlTree::one();
        let mut t = one.clone();
        for _ in 0..500 {
            t = EmlTree::eml(&t, &one);
        }
        let s = to_compact_string(&t);
        let parsed = parse(&s).expect("deep round-trip");
        assert_eq!(parsed, t);
    }

    // ----------------------------------------------------------------
    // Property: round-trip identity for arbitrary EmlTree shapes
    // ----------------------------------------------------------------

    /// Build a small random `EmlTree` controlled by `depth` to keep the search
    /// space tractable. The `seed` is consumed bit-by-bit to choose between
    /// `One`, `Var(idx)`, and `Eml(left, right)`.
    fn build_tree(depth: u32, seed: &mut u64) -> EmlTree {
        if depth == 0 {
            // Leaf: pick One or Var
            let bit = *seed & 1;
            *seed >>= 1;
            if bit == 0 {
                EmlTree::one()
            } else {
                let idx = (*seed & 0xff) as usize;
                *seed >>= 8;
                EmlTree::var(idx)
            }
        } else {
            // Branch: with 25% probability emit a leaf early (to vary shape).
            let b = *seed & 0b11;
            *seed >>= 2;
            if b == 0 {
                EmlTree::one()
            } else {
                let l = build_tree(depth - 1, seed);
                let r = build_tree(depth - 1, seed);
                EmlTree::eml(&l, &r)
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Round-trip identity: every `EmlTree` we can build should serialise
        /// to a string that re-parses to the same tree.
        #[test]
        fn round_trip_arbitrary(depth in 0u32..5, seed in any::<u64>()) {
            let mut s = seed;
            let original = build_tree(depth, &mut s);
            let text = to_compact_string(&original);
            let parsed = parse(&text)
                .map_err(|e| TestCaseError::fail(format!("re-parse failed for {text:?}: {e}")))?;
            prop_assert_eq!(parsed, original);
        }
    }
}
