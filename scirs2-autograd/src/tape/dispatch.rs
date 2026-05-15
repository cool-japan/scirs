//! Provenance detection and symbolic Jacobian dispatch.
//!
//! This module provides utilities to inspect autograd tensors for EML
//! symbolic provenance and to construct higher-order operations when all
//! outputs are EML-backed.
//!
//! # Design
//!
//! [`is_eml_backed`] and [`extract_lowered_op`] operate on the *existing*
//! scalar [`crate::symbolic_backend::EmlOp`] type; they do not inspect the
//! new [`super::eml_tape::EmlJacobianOp`] or [`super::eml_tape::EmlHessianOp`]
//! types (which are leaves, not scalar ops).
//!
//! [`try_build_symbolic_jacobian`] is the main dispatch entry point: given a
//! slice of EML-backed scalar tensors (one per output function) and a slice of
//! input scalar tensors, it extracts each output's `LoweredOp` and delegates
//! to [`super::eml_tape::eml_jacobian`].

use crate::symbolic_backend::EmlOp;
use crate::tape::eml_tape::eml_jacobian;
use crate::tensor::Tensor;
use crate::{Context, Float};
use scirs2_symbolic::eml::LoweredOp;
use std::sync::Arc;

/// Returns `true` if `t` was produced by a scalar [`EmlOp`].
///
/// Uses `Op::as_any()` + downcast; returns `false` for any other op type,
/// including non-symbolic ops and the new `EmlJacobianOp` / `EmlHessianOp`.
pub fn is_eml_backed<F: Float>(t: &Tensor<F>) -> bool {
    let inner = t.inner();
    inner
        .get_op()
        .as_any()
        .and_then(|any| any.downcast_ref::<EmlOp>())
        .is_some()
}

/// Extract the `Arc<LoweredOp>` from an EML-backed scalar tensor, if any.
///
/// Returns `None` when the tensor was not produced by an [`EmlOp`].
pub fn extract_lowered_op<F: Float>(t: &Tensor<F>) -> Option<Arc<LoweredOp>> {
    let inner = t.inner();
    inner
        .get_op()
        .as_any()
        .and_then(|any| any.downcast_ref::<EmlOp>())
        .map(|eml| Arc::clone(&eml.op))
}

/// Attempt to build a symbolic Jacobian when all outputs are EML-backed.
///
/// For each tensor in `outputs`, this function tries to extract its underlying
/// `LoweredOp` via [`extract_lowered_op`]. If any output is not EML-backed
/// the function returns `None` immediately (no partial construction).
///
/// On success it calls [`eml_jacobian`] with the extracted ops and `inputs`,
/// returning a 2-D tensor of shape `[outputs.len(), inputs.len()]`.
pub fn try_build_symbolic_jacobian<'g, F: Float>(
    outputs: &[Tensor<'g, F>],
    inputs: &[Tensor<'g, F>],
    g: &'g Context<F>,
) -> Option<Tensor<'g, F>> {
    let mut ops: Vec<Arc<LoweredOp>> = Vec::with_capacity(outputs.len());

    for t in outputs {
        match extract_lowered_op(t) {
            Some(op) => ops.push(op),
            None => return None,
        }
    }

    Some(eml_jacobian(ops, inputs, g))
}
