//! Einstein summation engine for arbitrary-rank tensor contractions.
//!
//! This module provides `einsum` and `einsum_grad` — general-purpose Einstein
//! summation functions that support traces, diagonals, outer products, matrix
//! multiplications, batched matmul, and arbitrary contractions over named
//! indices.
//!
//! ## Notation
//!
//! The equation string follows NumPy's einsum convention:
//! * `"ij,jk->ik"` — matrix multiplication
//! * `"ii->"` — trace (scalar result)
//! * `"ii->i"` — diagonal extraction
//! * `"ij->ji"` — transpose
//! * `"i,j->ij"` — outer product
//! * `"ij,ij->"` — Frobenius inner product
//! * `"bij,bjk->bik"` — batched matrix multiplication (explicit batch index)
//! * `"...ij,...jk->...ik"` — batched matrix multiplication with ellipsis
//!
//! ## Gradient Contract
//!
//! `einsum_grad` panics if the equation string fails to parse or shapes are
//! mismatched. This is intentional: callers must validate the equation and
//! operand shapes before requesting gradients (typically done implicitly when
//! `einsum` itself is called first). The infallible signature (`Vec<ArrayD<f64>>`)
//! is required by the upstream autograd interface.

use scirs2_core::ndarray::{Array, ArrayD, ArrayViewD, IxDyn};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can arise when parsing or evaluating an einsum expression.
#[derive(Debug, Clone, PartialEq)]
pub enum EinsumError {
    /// The equation string could not be parsed.
    ParseError(String),
    /// Two operands (or an operand and the output) disagree on the size of a
    /// shared index label.
    ShapeMismatch(String),
    /// An index label appears in the output that never appeared in any input.
    UnknownIndex(String),
    /// Ellipsis (`...`) notation is present but the operands have incompatible
    /// batch-dimension counts.
    /// Aligned ellipsis (all operands share the same number of batch dims) is
    /// supported; broadcasting across different batch ranks is not.
    EllipsisNotSupported(String),
}

impl fmt::Display for EinsumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EinsumError::ParseError(msg) => write!(f, "einsum parse error: {msg}"),
            EinsumError::ShapeMismatch(msg) => write!(f, "einsum shape mismatch: {msg}"),
            EinsumError::UnknownIndex(msg) => write!(f, "einsum unknown index: {msg}"),
            EinsumError::EllipsisNotSupported(msg) => {
                write!(f, "einsum ellipsis not supported: {msg}")
            }
        }
    }
}

impl std::error::Error for EinsumError {}

// ── Internal representation ───────────────────────────────────────────────────

/// A single index in a subscript (either a named label or an ellipsis group).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum IndexSpec {
    Label(char),
    Ellipsis,
}

/// Parsed representation of an einsum equation.
#[derive(Debug, Clone)]
struct ParsedEq {
    /// Subscripts for each input operand.
    input_specs: Vec<Vec<IndexSpec>>,
    /// Subscript for the output.
    output_spec: Vec<IndexSpec>,
    /// How many leading "batch" dimensions the ellipsis expands to (resolved
    /// after seeing the operand shapes).
    ellipsis_rank: Option<usize>,
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Parse one subscript token sequence (e.g. `"bij"` → `[B,i,j]` or
/// `"...ij"` → `[Ellipsis, i, j]`).
fn parse_subscript(s: &str) -> Result<Vec<IndexSpec>, EinsumError> {
    let mut specs = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '.' {
            // Expect exactly two more dots.
            let d1 = chars.next();
            let d2 = chars.next();
            if d1 != Some('.') || d2 != Some('.') {
                return Err(EinsumError::ParseError(format!(
                    "malformed ellipsis in subscript '{s}'"
                )));
            }
            if specs.contains(&IndexSpec::Ellipsis) {
                return Err(EinsumError::ParseError(format!(
                    "multiple ellipses in subscript '{s}'"
                )));
            }
            specs.push(IndexSpec::Ellipsis);
        } else if c.is_ascii_alphabetic() {
            specs.push(IndexSpec::Label(c));
        } else {
            return Err(EinsumError::ParseError(format!(
                "unexpected character '{c}' in subscript '{s}'"
            )));
        }
    }
    Ok(specs)
}

/// Parse a full einsum equation string.
///
/// Supports both `"ij,jk->ik"` (explicit output) and `"ij,jk"` (implicit
/// output: all labels that appear exactly once, in alphabetical order).
fn parse_einsum(eq: &str) -> Result<ParsedEq, EinsumError> {
    let (inputs_str, output_str_opt) = if let Some(arrow) = eq.find("->") {
        (&eq[..arrow], Some(&eq[arrow + 2..]))
    } else {
        (eq, None)
    };

    let input_parts: Vec<&str> = inputs_str.split(',').collect();
    if input_parts.is_empty() {
        return Err(EinsumError::ParseError(
            "equation has no input subscripts".to_owned(),
        ));
    }

    let mut input_specs: Vec<Vec<IndexSpec>> = Vec::with_capacity(input_parts.len());
    for part in &input_parts {
        input_specs.push(parse_subscript(part.trim())?);
    }

    let output_spec = if let Some(out) = output_str_opt {
        parse_subscript(out.trim())?
    } else {
        // Implicit output: labels that appear exactly once across all inputs,
        // sorted alphabetically.
        let mut counts: HashMap<char, usize> = HashMap::new();
        for specs in &input_specs {
            for s in specs {
                if let IndexSpec::Label(c) = s {
                    *counts.entry(*c).or_insert(0) += 1;
                }
            }
        }
        let mut singles: Vec<char> = counts
            .iter()
            .filter(|(_, &v)| v == 1)
            .map(|(&k, _)| k)
            .collect();
        singles.sort_unstable();
        singles.into_iter().map(IndexSpec::Label).collect()
    };

    Ok(ParsedEq {
        input_specs,
        output_spec,
        ellipsis_rank: None,
    })
}

// ── Index-size resolution ─────────────────────────────────────────────────────

/// Given parsed specs and actual operand shapes, build a map from index label
/// to its size.  Also resolves the ellipsis rank if present.
fn resolve_sizes(
    parsed: &mut ParsedEq,
    ops: &[ArrayViewD<f64>],
) -> Result<HashMap<char, usize>, EinsumError> {
    if parsed.input_specs.len() != ops.len() {
        return Err(EinsumError::ParseError(format!(
            "equation has {} input subscripts but {} operands were supplied",
            parsed.input_specs.len(),
            ops.len()
        )));
    }

    // First pass: determine ellipsis rank.
    let mut ell_rank: Option<usize> = None;
    for (i, specs) in parsed.input_specs.iter().enumerate() {
        let explicit_count = specs.iter().filter(|s| **s != IndexSpec::Ellipsis).count();
        if specs.contains(&IndexSpec::Ellipsis) {
            let op_rank = ops[i].ndim();
            if op_rank < explicit_count {
                return Err(EinsumError::ShapeMismatch(format!(
                    "operand {i} has rank {op_rank} but subscript has {explicit_count} explicit indices"
                )));
            }
            let this_ell = op_rank - explicit_count;
            match ell_rank {
                None => ell_rank = Some(this_ell),
                Some(prev) if prev != this_ell => {
                    return Err(EinsumError::EllipsisNotSupported(format!(
                        "operand {i} gives ellipsis rank {this_ell} but earlier operand gave {prev}; \
                         broadcasting ellipsis across different batch ranks is not supported"
                    )));
                }
                _ => {}
            }
        }
    }
    parsed.ellipsis_rank = ell_rank;

    // Second pass: build label→size map and validate.
    let mut sizes: HashMap<char, usize> = HashMap::new();

    // Expand specs with ellipsis into concrete label specs using synthetic
    // characters that cannot clash with user labels (we use the private-use
    // area: U+E000+).
    // We keep this expansion local to the resolution step.
    let ell_rank_val = ell_rank.unwrap_or(0);
    let ell_chars: Vec<char> = (0..ell_rank_val)
        .map(|k| char::from_u32(0xE000 + k as u32).unwrap_or('_'))
        .collect();

    for (i, specs) in parsed.input_specs.iter().enumerate() {
        let expanded = expand_ellipsis(specs, &ell_chars);
        let op_shape = ops[i].shape();
        if expanded.len() != op_shape.len() {
            return Err(EinsumError::ShapeMismatch(format!(
                "operand {i}: subscript has {} indices after expansion but operand has rank {}",
                expanded.len(),
                op_shape.len()
            )));
        }
        for (j, label) in expanded.iter().enumerate() {
            let dim_size = op_shape[j];
            match sizes.entry(*label) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    if *e.get() != dim_size {
                        return Err(EinsumError::ShapeMismatch(format!(
                            "index '{label}' has size {} from one operand but {} from operand {i}",
                            e.get(),
                            dim_size
                        )));
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(dim_size);
                }
            }
        }
    }

    // Validate output labels.
    let expanded_out = expand_ellipsis(&parsed.output_spec, &ell_chars);
    for label in &expanded_out {
        if !sizes.contains_key(label) {
            return Err(EinsumError::UnknownIndex(format!(
                "output index '{label}' does not appear in any input"
            )));
        }
    }

    Ok(sizes)
}

/// Expand a subscript spec by replacing `Ellipsis` with the provided concrete
/// label chars, returning a plain `Vec<char>`.
fn expand_ellipsis(specs: &[IndexSpec], ell_chars: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(specs.len() + ell_chars.len().saturating_sub(1));
    for s in specs {
        match s {
            IndexSpec::Label(c) => out.push(*c),
            IndexSpec::Ellipsis => out.extend_from_slice(ell_chars),
        }
    }
    out
}

// ── General evaluation ────────────────────────────────────────────────────────

/// Core general-case einsum: iterates over all combinations of output + sum
/// indices and accumulates the product of elements.
fn einsum_general(
    input_expanded: &[Vec<char>],
    output_expanded: &[char],
    sizes: &HashMap<char, usize>,
    ops: &[ArrayViewD<f64>],
) -> Result<ArrayD<f64>, EinsumError> {
    // All index labels that appear anywhere.
    let all_labels: Vec<char> = {
        let mut set: HashSet<char> = HashSet::new();
        for spec in input_expanded {
            set.extend(spec);
        }
        set.extend(output_expanded);
        let mut v: Vec<char> = set.into_iter().collect();
        v.sort_unstable();
        v
    };

    // Labels that are summed over (appear in inputs but not in output).
    let output_set: HashSet<char> = output_expanded.iter().copied().collect();
    let sum_labels: Vec<char> = all_labels
        .iter()
        .filter(|c| !output_set.contains(c))
        .copied()
        .collect();

    // Shape of the output tensor.
    let out_shape: Vec<usize> = output_expanded
        .iter()
        .map(|c| *sizes.get(c).unwrap_or(&1))
        .collect();

    // Assign a canonical position to each label in the iteration order.
    // We iterate over output indices in their declared order, then sum indices.
    let iter_labels: Vec<char> = output_expanded
        .iter()
        .copied()
        .chain(sum_labels.iter().copied())
        .collect();

    let iter_sizes: Vec<usize> = iter_labels
        .iter()
        .map(|c| *sizes.get(c).unwrap_or(&1))
        .collect();

    // Build label → position map.
    let label_pos: HashMap<char, usize> = iter_labels
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // For each operand, map its subscript indices to positions in iter_labels.
    let op_pos_maps: Vec<Vec<usize>> = input_expanded
        .iter()
        .map(|spec| {
            spec.iter()
                .map(|c| *label_pos.get(c).unwrap_or(&0))
                .collect()
        })
        .collect();

    // Total number of output elements.
    let out_len: usize = out_shape.iter().product();

    let mut result = Array::zeros(IxDyn(&out_shape));

    if out_len == 0 {
        return Ok(result);
    }

    // Multi-index iteration using a counter array.
    let total_iters: usize = iter_sizes.iter().product();
    let n_dims = iter_sizes.len();

    // Pre-compute stride for converting flat index → multi-index.
    let mut strides = vec![1usize; n_dims];
    for k in (0..n_dims.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * iter_sizes[k + 1];
    }

    for flat in 0..total_iters {
        // Decompose flat → multi-index.
        let mut multi = vec![0usize; n_dims];
        let mut rem = flat;
        for k in 0..n_dims {
            multi[k] = rem / strides[k];
            rem %= strides[k];
        }

        // Compute the product for this index combination.
        let mut prod = 1.0_f64;
        for (op_idx, op) in ops.iter().enumerate() {
            let op_index: Vec<usize> = op_pos_maps[op_idx].iter().map(|&p| multi[p]).collect();
            prod *= op[IxDyn(&op_index)];
        }

        // Accumulate into output.
        let out_index: Vec<usize> = (0..output_expanded.len()).map(|k| multi[k]).collect();
        result[IxDyn(&out_index)] += prod;
    }

    Ok(result)
}

// ── Shortcut paths ────────────────────────────────────────────────────────────

/// Attempt a fast shortcut for common 1-operand patterns.
/// Returns `None` if no shortcut applies (caller should fall back to general).
fn shortcut_single(
    input: &[char],
    output: &[char],
    sizes: &HashMap<char, usize>,
    op: &ArrayViewD<f64>,
) -> Option<ArrayD<f64>> {
    // Transpose / identity: same labels, different order.
    if input.len() == output.len() && {
        let in_set: HashSet<char> = input.iter().copied().collect();
        let out_set: HashSet<char> = output.iter().copied().collect();
        in_set == out_set
    } {
        // Build permutation.
        let perm: Vec<usize> = output
            .iter()
            .map(|c| input.iter().position(|x| x == c).unwrap_or(0))
            .collect();
        let axes: Vec<usize> = perm;
        // Check it's actually a permutation of axes.
        let view = op.view();
        // ndarray permuted_axes requires the same rank.
        if axes.len() == view.ndim() {
            let transposed = view.permuted_axes(IxDyn(&axes));
            return Some(transposed.to_owned());
        }
    }

    // Trace: "ii->" or similar where all inputs collapse.
    if output.is_empty() && input.len() == 2 && input[0] == input[1] {
        let n = *sizes.get(&input[0]).unwrap_or(&0);
        let mut acc = 0.0_f64;
        for i in 0..n {
            acc += op[[i, i].as_ref()];
        }
        return Some(Array::from_elem(IxDyn(&[]), acc));
    }

    // Diagonal: "ii->i"
    if output.len() == 1 && input.len() == 2 && input[0] == input[1] && output[0] == input[0] {
        let n = *sizes.get(&input[0]).unwrap_or(&0);
        let diag: Vec<f64> = (0..n).map(|i| op[[i, i].as_ref()]).collect();
        return Some(
            Array::from_shape_vec(IxDyn(&[n]), diag).unwrap_or_else(|_| Array::zeros(IxDyn(&[n]))),
        );
    }

    None
}

/// Attempt a fast shortcut for the common 2-operand case.
fn shortcut_double(
    in0: &[char],
    in1: &[char],
    output: &[char],
    sizes: &HashMap<char, usize>,
    a: &ArrayViewD<f64>,
    b: &ArrayViewD<f64>,
) -> Option<ArrayD<f64>> {
    // Matmul: "ij,jk->ik"
    if in0.len() == 2 && in1.len() == 2 && output.len() == 2 {
        let (ai, aj) = (in0[0], in0[1]);
        let (bj, bk) = (in1[0], in1[1]);
        let (oi, ok) = (output[0], output[1]);
        if aj == bj && ai == oi && bk == ok {
            let m = *sizes.get(&ai).unwrap_or(&0);
            let k = *sizes.get(&aj).unwrap_or(&0);
            let n = *sizes.get(&bk).unwrap_or(&0);
            if a.shape() == [m, k] && b.shape() == [k, n] {
                let a2 = a
                    .view()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .ok()?;
                let b2 = b
                    .view()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .ok()?;
                let c = a2.dot(&b2);
                return Some(c.into_dyn());
            }
        }
    }

    // Inner product / element-wise sum: "ij,ij->" or "i,i->"
    if in0 == in1 && output.is_empty() {
        let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        return Some(Array::from_elem(IxDyn(&[]), sum));
    }

    // Outer product: "i,j->ij"
    if in0.len() == 1 && in1.len() == 1 && output.len() == 2 {
        let (li, lj) = (in0[0], in1[0]);
        if li != lj && output[0] == li && output[1] == lj {
            let ni = *sizes.get(&li).unwrap_or(&0);
            let nj = *sizes.get(&lj).unwrap_or(&0);
            let mut c = Array::zeros(IxDyn(&[ni, nj]));
            for i in 0..ni {
                for j in 0..nj {
                    c[[i, j]] = a[[i]] * b[[j]];
                }
            }
            return Some(c);
        }
    }

    None
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Evaluate an Einstein summation expression.
///
/// # Arguments
///
/// * `eq`  — einsum equation string (e.g. `"ij,jk->ik"`)
/// * `ops` — slice of operand views
///
/// # Errors
///
/// Returns `EinsumError` if the equation cannot be parsed, if index sizes are
/// inconsistent, or if the output refers to unknown indices.
pub fn einsum(eq: &str, ops: &[ArrayViewD<f64>]) -> Result<ArrayD<f64>, EinsumError> {
    let mut parsed = parse_einsum(eq)?;
    let sizes = resolve_sizes(&mut parsed, ops)?;

    let ell_rank = parsed.ellipsis_rank.unwrap_or(0);
    let ell_chars: Vec<char> = (0..ell_rank)
        .map(|k| char::from_u32(0xE000 + k as u32).unwrap_or('_'))
        .collect();

    // Expand all specs to plain char vectors.
    let input_expanded: Vec<Vec<char>> = parsed
        .input_specs
        .iter()
        .map(|s| expand_ellipsis(s, &ell_chars))
        .collect();
    let output_expanded: Vec<char> = expand_ellipsis(&parsed.output_spec, &ell_chars);

    // Try shortcuts first.
    if ops.len() == 1 {
        if let Some(r) = shortcut_single(&input_expanded[0], &output_expanded, &sizes, &ops[0]) {
            return Ok(r);
        }
    }
    if ops.len() == 2 {
        if let Some(r) = shortcut_double(
            &input_expanded[0],
            &input_expanded[1],
            &output_expanded,
            &sizes,
            &ops[0],
            &ops[1],
        ) {
            return Ok(r);
        }
    }

    // Multi-operand case: reduce pairwise in order using smallest-intermediate
    // heuristic (simplified: just left-to-right for now).
    if ops.len() == 1 || ops.len() == 2 {
        // Already handled shortcuts; fall through to general.
        einsum_general(&input_expanded, &output_expanded, &sizes, ops)
    } else {
        einsum_multi(&input_expanded, &output_expanded, &sizes, ops, &ell_chars)
    }
}

/// Multi-operand einsum: reduce left-to-right via pairwise contractions.
fn einsum_multi(
    input_expanded: &[Vec<char>],
    output_expanded: &[char],
    sizes: &HashMap<char, usize>,
    ops: &[ArrayViewD<f64>],
    _ell_chars: &[char],
) -> Result<ArrayD<f64>, EinsumError> {
    // Identify which labels are "needed" later: output labels + labels that
    // appear in ops[2..].  Labels needed only in the first pair can be summed
    // immediately.
    let needed_later: HashSet<char> = {
        let mut set: HashSet<char> = output_expanded.iter().copied().collect();
        for spec in input_expanded.iter().skip(2) {
            set.extend(spec.iter().copied());
        }
        set
    };

    // First intermediate: contract ops[0] and ops[1].
    let first_out_labels: Vec<char> = {
        let both: HashSet<char> = input_expanded[0]
            .iter()
            .chain(input_expanded[1].iter())
            .copied()
            .collect();
        let mut v: Vec<char> = both
            .into_iter()
            .filter(|c| needed_later.contains(c))
            .collect();
        v.sort_unstable();
        v
    };

    let mut acc = einsum_general(
        &[input_expanded[0].clone(), input_expanded[1].clone()],
        &first_out_labels,
        sizes,
        &[ops[0].view(), ops[1].view()],
    )?;

    let mut current_labels = first_out_labels;

    for step in 2..ops.len() {
        let is_last = step == ops.len() - 1;
        let next_out_labels: Vec<char> = if is_last {
            output_expanded.to_vec()
        } else {
            let needed: HashSet<char> = {
                let mut s: HashSet<char> = output_expanded.iter().copied().collect();
                for spec in input_expanded.iter().skip(step + 1) {
                    s.extend(spec.iter().copied());
                }
                s
            };
            let both: HashSet<char> = current_labels
                .iter()
                .chain(input_expanded[step].iter())
                .copied()
                .collect();
            let mut v: Vec<char> = both.into_iter().filter(|c| needed.contains(c)).collect();
            v.sort_unstable();
            v
        };

        acc = einsum_general(
            &[current_labels.clone(), input_expanded[step].clone()],
            &next_out_labels,
            sizes,
            &[acc.view(), ops[step].view()],
        )?;
        current_labels = next_out_labels;
    }

    Ok(acc)
}

// ── Gradient ──────────────────────────────────────────────────────────────────

/// Compute the gradient of an einsum with respect to each operand.
///
/// For each operand `k`, the gradient is computed by re-running einsum with:
/// * The output grad in place of operand `k`
/// * The remaining operands unchanged
/// * The output spec set to the subscript of operand `k`
///
/// # Panics
///
/// Panics if `eq` fails to parse or shapes are inconsistent.  Callers should
/// ensure they have already called `einsum` successfully with the same
/// arguments (which validates both).
pub fn einsum_grad(
    eq: &str,
    grad_out: ArrayViewD<f64>,
    ops: &[ArrayViewD<f64>],
) -> Vec<ArrayD<f64>> {
    let mut parsed = parse_einsum(eq).expect("einsum_grad: failed to parse equation");

    // Resolve sizes from original operands so we can reconstruct shapes.
    let sizes = resolve_sizes(&mut parsed, ops)
        .expect("einsum_grad: failed to resolve sizes from operands");

    let ell_rank = parsed.ellipsis_rank.unwrap_or(0);
    let ell_chars: Vec<char> = (0..ell_rank)
        .map(|k| char::from_u32(0xE000 + k as u32).unwrap_or('_'))
        .collect();

    let input_expanded: Vec<Vec<char>> = parsed
        .input_specs
        .iter()
        .map(|s| expand_ellipsis(s, &ell_chars))
        .collect();
    let output_expanded: Vec<char> = expand_ellipsis(&parsed.output_spec, &ell_chars);

    let mut grads = Vec::with_capacity(ops.len());

    for k in 0..ops.len() {
        // Gradient equation for operand k:
        //   inputs  = [output_spec] + [input_spec[i] for i != k]
        //   output  = input_spec[k]
        let grad_input_specs: Vec<Vec<char>> = std::iter::once(output_expanded.clone())
            .chain(
                input_expanded
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != k)
                    .map(|(_, s)| s.clone()),
            )
            .collect();

        let grad_output_spec: Vec<char> = input_expanded[k].clone();

        // Assemble operand views: grad_out first, then all others.
        let grad_ops: Vec<ArrayViewD<f64>> = std::iter::once(grad_out.view())
            .chain(
                ops.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != k)
                    .map(|(_, op)| op.view()),
            )
            .collect();

        let gk = einsum_general(&grad_input_specs, &grad_output_spec, &sizes, &grad_ops)
            .expect("einsum_grad: failed to compute gradient");

        grads.push(gk);
    }

    grads
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2, Array3};

    fn approx_eq(a: &ArrayD<f64>, b: &ArrayD<f64>, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    // ── 1. Matrix multiplication ──────────────────────────────────────────

    #[test]
    fn test_matmul() {
        let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let b: Array2<f64> = array![[5.0, 6.0], [7.0, 8.0]];
        let c = einsum("ij,jk->ik", &[a.view().into_dyn(), b.view().into_dyn()]).unwrap();
        let expected: Array2<f64> = array![[19.0, 22.0], [43.0, 50.0]];
        assert!(approx_eq(&c, &expected.into_dyn(), 1e-10));
    }

    // ── 2. Trace ──────────────────────────────────────────────────────────

    #[test]
    fn test_trace() {
        let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let result = einsum("ii->", &[a.view().into_dyn()]).unwrap();
        // trace = 1+4 = 5
        assert!((result[[]] - 5.0).abs() < 1e-10);
    }

    // ── 3. Diagonal ───────────────────────────────────────────────────────

    #[test]
    fn test_diagonal() {
        let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let d = einsum("ii->i", &[a.view().into_dyn()]).unwrap();
        let expected: Array1<f64> = array![1.0, 5.0, 9.0];
        assert!(approx_eq(&d, &expected.into_dyn(), 1e-10));
    }

    // ── 4. Transpose ──────────────────────────────────────────────────────

    #[test]
    fn test_transpose() {
        let a: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let t = einsum("ij->ji", &[a.view().into_dyn()]).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert!((t[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((t[[0, 1]] - 4.0).abs() < 1e-10);
        assert!((t[[1, 0]] - 2.0).abs() < 1e-10);
        assert!((t[[2, 1]] - 6.0).abs() < 1e-10);
    }

    // ── 5. Outer product ──────────────────────────────────────────────────

    #[test]
    fn test_outer_product() {
        let u: Array1<f64> = array![1.0, 2.0, 3.0];
        let v: Array1<f64> = array![4.0, 5.0];
        let outer = einsum("i,j->ij", &[u.view().into_dyn(), v.view().into_dyn()]).unwrap();
        assert_eq!(outer.shape(), &[3, 2]);
        assert!((outer[[0, 0]] - 4.0).abs() < 1e-10);
        assert!((outer[[2, 1]] - 15.0).abs() < 1e-10);
    }

    // ── 6. Inner product ──────────────────────────────────────────────────

    #[test]
    fn test_inner_product() {
        let u: Array1<f64> = array![1.0, 2.0, 3.0];
        let v: Array1<f64> = array![4.0, 5.0, 6.0];
        let s = einsum("i,i->", &[u.view().into_dyn(), v.view().into_dyn()]).unwrap();
        // 1*4 + 2*5 + 3*6 = 4+10+18 = 32
        assert!((s[[]] - 32.0).abs() < 1e-10);
    }

    // ── 7. Frobenius inner product ────────────────────────────────────────

    #[test]
    fn test_frobenius() {
        let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let b: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]];
        let s = einsum("ij,ij->", &[a.view().into_dyn(), b.view().into_dyn()]).unwrap();
        // 1*1 + 2*0 + 3*0 + 4*1 = 5
        assert!((s[[]] - 5.0).abs() < 1e-10);
    }

    // ── 8. Batched matrix multiplication (explicit batch index) ───────────

    #[test]
    fn test_batched_matmul_explicit() {
        // batch of 2 matrix multiplications, each 2×2 @ 2×3 = 2×3
        let a: Array3<f64> =
            Array3::from_shape_fn((2, 2, 2), |(b, i, j)| ((b * 4 + i * 2 + j) as f64) + 1.0);
        let b: Array3<f64> =
            Array3::from_shape_fn((2, 2, 3), |(bb, j, k)| ((bb * 6 + j * 3 + k) as f64) + 1.0);
        let c = einsum("bij,bjk->bik", &[a.view().into_dyn(), b.view().into_dyn()]).unwrap();
        assert_eq!(c.shape(), &[2, 2, 3]);
        // Verify first batch element matches manual matmul
        let a0 = a
            .slice(scirs2_core::ndarray::s![0, .., ..])
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        let b0 = b
            .slice(scirs2_core::ndarray::s![0, .., ..])
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        let expected0: Array2<f64> = a0.dot(&b0);
        for i in 0..2 {
            for k in 0..3 {
                assert!(
                    (c[[0, i, k]] - expected0[[i, k]]).abs() < 1e-8,
                    "mismatch at [0,{i},{k}]: {} vs {}",
                    c[[0, i, k]],
                    expected0[[i, k]]
                );
            }
        }
    }

    // ── 9. Ellipsis batched matmul ────────────────────────────────────────

    #[test]
    fn test_ellipsis_batched_matmul() {
        let a: Array3<f64> =
            Array3::from_shape_fn((3, 2, 4), |(b, i, j)| (b * 8 + i * 4 + j) as f64);
        let b: Array3<f64> =
            Array3::from_shape_fn((3, 4, 5), |(bb, j, k)| (bb * 20 + j * 5 + k) as f64);
        let c = einsum(
            "...ij,...jk->...ik",
            &[a.view().into_dyn(), b.view().into_dyn()],
        )
        .unwrap();
        assert_eq!(c.shape(), &[3, 2, 5]);
        // Spot-check one batch slice against direct dot.
        let a1 = a
            .slice(scirs2_core::ndarray::s![1, .., ..])
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        let b1 = b
            .slice(scirs2_core::ndarray::s![1, .., ..])
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .unwrap();
        let expected1: Array2<f64> = a1.dot(&b1);
        for i in 0..2 {
            for k in 0..5 {
                assert!(
                    (c[[1, i, k]] - expected1[[i, k]]).abs() < 1e-8,
                    "mismatch at [1,{i},{k}]"
                );
            }
        }
    }

    // ── 10. Gradient of matmul (numerical check) ──────────────────────────

    #[test]
    fn test_matmul_gradient() {
        let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let b: Array2<f64> = array![[5.0, 6.0], [7.0, 8.0]];
        // Upstream gradient: all ones.
        let grad_out: Array2<f64> = Array2::ones((2, 2));

        let grads = einsum_grad(
            "ij,jk->ik",
            grad_out.view().into_dyn(),
            &[a.view().into_dyn(), b.view().into_dyn()],
        );

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 2]); // grad w.r.t. A
        assert_eq!(grads[1].shape(), &[2, 2]); // grad w.r.t. B

        // dL/dA = grad_out @ B^T
        // With grad_out = ones(2×2), B = [[5,6],[7,8]]:
        // dA[0,0] = 1*5 + 1*7 = 12 (wrong below -- formula: grad@B^T)
        // Actually dA = grad_out · B^T = [[1,1],[1,1]] · [[5,7],[6,8]] = [[11,15],[11,15]]
        let expected_da: Array2<f64> = array![[11.0, 15.0], [11.0, 15.0]];
        assert!(
            approx_eq(&grads[0], &expected_da.into_dyn(), 1e-8),
            "dA mismatch: {:?}",
            grads[0]
        );

        // dL/dB = A^T @ grad_out = [[1,3],[2,4]] · [[1,1],[1,1]] = [[4,4],[6,6]]
        let expected_db: Array2<f64> = array![[4.0, 4.0], [6.0, 6.0]];
        assert!(
            approx_eq(&grads[1], &expected_db.into_dyn(), 1e-8),
            "dB mismatch: {:?}",
            grads[1]
        );
    }

    // ── 11. Error: shape mismatch ─────────────────────────────────────────

    #[test]
    fn test_shape_mismatch_error() {
        let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]]; // 2×2
        let b: Array2<f64> = array![[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]; // 2×3
                                                                        // "ij,ik->jk" requires a.ncols == b.ncols, but we'll try "ij,kj->ik" with wrong shapes.
                                                                        // Specifically, try matmul where inner dim doesn't match:
        let result = einsum("ij,jk->ik", &[b.view().into_dyn(), a.view().into_dyn()]);
        // b is 2×3, a is 2×2: j=3 in b but j=2 in a → shape mismatch
        assert!(matches!(result, Err(EinsumError::ShapeMismatch(_))));
    }

    // ── 12. Error: parse error ────────────────────────────────────────────

    #[test]
    fn test_parse_error() {
        let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let result = einsum("ij!k->ik", &[a.view().into_dyn()]);
        assert!(matches!(result, Err(EinsumError::ParseError(_))));
    }

    // ── 13. Three-operand contraction ─────────────────────────────────────

    #[test]
    fn test_three_operand() {
        // A[i,j] B[j,k] C[k,l] -> D[i,l]
        let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let b: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]]; // identity
        let c: Array2<f64> = array![[2.0, 0.0], [0.0, 2.0]]; // 2*identity
        let result = einsum(
            "ij,jk,kl->il",
            &[
                a.view().into_dyn(),
                b.view().into_dyn(),
                c.view().into_dyn(),
            ],
        )
        .unwrap();
        // A @ I @ 2I = 2A
        let expected: Array2<f64> = array![[2.0, 4.0], [6.0, 8.0]];
        assert!(
            approx_eq(&result, &expected.into_dyn(), 1e-8),
            "3-op result: {:?}",
            result
        );
    }
}
