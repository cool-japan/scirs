# scirs2-autograd TODO

## Status: v0.6.5 (released 2026-07-31; last reviewed 2026-07-31)

**The headline fix of the 0.6.5 release cycle**, surfaced by a workspace-wide `#[ignore]`-legitimacy
audit followed to ground. The live backward-pass dispatcher (`gradient.rs`'s `compute_override_grads`
/ `compute_grads_via_op`, previously a ~670-line if/else chain keyed on `Op::name()` strings) covered
only 58 of 281 differentiable op implementations; everything it didn't recognize silently fell
through to an identity gradient (`Some(gy)`), with no error, warning, or `debug_assert`. Dispatch now
consults each op's own `Op::grad()` implementation (`op.grad(&mut ctx)` in `compute_grads_via_op`) for
anything the override table doesn't special-case, fixing roughly 223 wrong-or-absent gradients:
every elementary math function (`sqrt`/`exp`/`ln`/`sin`/`cos`/`tan`/`asin`/`acos`/`atan`/`sinh`/`cosh`/
`asinh`/`acosh`/`atanh`/`log2`/`log10`/`exp2`/`exp10`/`abs`) and every activation (`softplus`/`elu`/
`swish`/`gelu`/`mish`) had a correct `Op::grad()` implementation that was simply never called.

Also fixed independently: `reduce_sum`/`transpose`/`gather` were only correct under an all-ones
cotangent (masking the bug from every existing test); `reduce_mean` lost its `1/N` factor;
`sigmoid_cross_entropy` was caught by a `contains("Sigmoid")` substring match and produced
sign-flipped gradients; `BatchMatMul` was caught by `ends_with("MatMul")` and applied a 2-D transpose
rule to 3-D batched tensors; `concat`/`einsum`/`tensordot` panicked during backprop instead of
returning a gradient. `SymmetricEigenOp`'s forward pass now genuinely diagonalizes via a real
cyclic-Jacobi algorithm (`tensor_ops::matrix_calculus::symmetric_eigen`), shared by one code path for
every matrix size instead of separate `n==1`/`n==2`/general-case special cases with their own
eigenvector sign/ordering conventions.

The public custom-gradient API (`custom_op`, `scale_gradient`, `selective_stop_gradient`, `detach` —
all in `custom_gradient.rs`) was a complete no-op: every one of them returned the identity gradient
regardless of the user-supplied backward closure, silently disabling gradient-reversal layers and
detach-based graph pruning. Now routes through the same fixed dispatch above and is genuinely
functional.

New `tests/gradient_fd_harness.rs` (49 tests) + `tests/gradient_fd_harness_matrix.rs` (31 tests): a
finite-difference regression harness using a **non-uniform** cotangent — a uniform all-ones cotangent
is what let `transpose`/`gather`/`reduce_sum` ship broken for this long without a failing test. See
`CHANGELOG.md` `[0.6.5]` for full detail.

scirs2-autograd's own test suite (last independently run 2026-07-15, pre-fix): 1260 tests pass, 0 failed, 18 skipped (default features); 1345 tests pass, 0 failed, 18 skipped (`--all-features`). Not re-run for this docs update; the 80 new finite-difference harness tests above are additional to this baseline.

## v0.3.3 Completed

### Core Automatic Differentiation
- Reverse-mode AD (VJP / backpropagation) via tape-based gradient accumulation
- Forward-mode AD (JVP / Jacobian-vector products)
- Dynamic computation graph construction
- Lazy evaluation with graph-level optimizations (constant folding, CSE, loop fusion)
- Higher-order derivatives: Hessian, Hessian-vector products
- Second-order optimization support

### Gradient Utilities
- Finite difference numerical differentiation (forward, central, backward)
- Richardson extrapolation for higher-order accuracy
- Gradient checking / numerical verification
- `numerical_diff.rs` module for standalone finite differences

### Memory Optimization
- Gradient checkpointing (recompute-based; `checkpoint`, `adaptive_checkpoint`)
- Checkpoint groups for multi-output operations (`CheckpointGroup`)
- Checkpoint profiler (`CheckpointProfiler` with memory-saved tracking)
- Memory pooling and in-place operations

### Functional Transforms
- `grad` - scalar gradient computation
- `jacobian` - full Jacobian
- `hessian` - second-order derivatives
- `functional_transforms.rs`: vmap-like batching, compose, grad transform

### Implicit Differentiation
- Implicit function theorem-based gradients (`implicit_diff.rs`)
- Fixed-point iteration gradients
- Support for bi-level optimization

### JVP / VJP
- Explicit `jvp` (Jacobian-vector product, forward-mode)
- Explicit `vjp` (vector-Jacobian product, reverse-mode)
- `jvp_vjp.rs` module with composable interfaces

### Differentiable Operations
- Complete arithmetic with broadcasting (add, sub, mul, div, pow)
- Linear algebra with gradients: matmul, inverse, determinant
- Matrix decompositions with gradients: QR, SVD, Cholesky, LU
- Matrix functions: exp, log, sqrt, power, matrix exponential
- Activation functions: ReLU, Sigmoid, Tanh, Softmax, GELU, Swish, Mish
- Loss functions: MSE, cross-entropy, sparse categorical cross-entropy
- Convolution: Conv2D, transposed conv, max/avg pooling
- Tensor manipulation: reshape, slice, concat, pad, advanced indexing
- Reductions: sum, mean, max, min, variance

### Mixed Precision
- FP16 / FP32 mixed precision gradient computation (`mixed_precision.rs`)
- Loss scaling for numeric stability

### Lazy Evaluation
- Deferred execution model (`lazy_eval.rs`)
- JIT-like element-wise operation fusion (`jit_fusion.rs`)

### Optimizers
- SGD (with momentum and Nesterov)
- Adam, AdamW
- AdaGrad, RMSprop
- Plain optimizers API (`plain_optimizers.rs`)
- Learning rate schedulers: step, exponential, cosine annealing
- Gradient clipping (norm-based and value-based)
- Namespace-based variable management

### Higher-Order AD
- `higher_order_new.rs` and `higher_order_advanced.rs`
- Efficient Hessian computation
- Hessian-vector products for Newton-CG and trust-region methods

### Debugging and Visualization
- Computation graph visualization via DOT format (`graph_viz.rs`)
- Gradient tape inspection (`tape/`)
- NaN/Inf detection hooks (`debugging.rs`)

### Custom Gradients
- User-defined gradient rules (`custom_grad.rs`, `custom_grad_advanced.rs`)
- `diff_rules.rs` for registering custom derivative rules

### Distributed Gradients
- Gradient aggregation across workers (`distributed_grad.rs`)
- All-reduce primitives

## v0.4.0 Roadmap

### Source-to-Source Transformation
- Source code transformation for AD (compile-time differentiation)
- Operator overloading with compile-time graph construction

### XLA-Like Compilation
- Computation graph lowering to an IR
- XLA-style device placement and fusion

### Symbolic Differentiation — Implemented in v0.4.0
- [x] CAS-style symbolic derivative rules
- [x] Simplification of symbolic expressions before evaluation

### Improved JIT
- Cross-operation fusion across different op types
- Profile-guided optimization for hot paths

### Sparse Gradients — Implemented in v0.4.0
- [x] Sparse tensor representation in the gradient tape
- [x] Efficient sparse-dense gradient accumulation

## Wave 73 — Gradient correctness repair (2026-05-07)

- [x] **extract_diag deduplication + ScalarMulOp higher-order + jit_fusion expansion** (completed 2026-05-07)
  - Deleted orphaned `linalg_ops_fixed.rs` (was never imported; had compile errors including wrong variant `OpError::InvalidShape` vs `IncompatibleShape`, reference to `_matrix` instead of `matrix`, and wrong `fn grad` signature)
  - Fixed `ScalarMulOp::grad` to propagate symbolically (no `.eval()` collapse); added `as_any()` impl so `gradient.rs` can downcast and retrieve the scalar
  - Added `ScalarMulOp` case to `gradient.rs` name-dispatch (the actual gradient engine — `Op::grad` trait is not called by the engine)
  - Extended `jit_fusion::can_fuse`-equivalent: added `detect_matmul_epilogue` (matmul→elementwise chain, up to 4 ops) and `detect_batched_matmul_reduction` (MatMul/BatchedMatMul→ReduceSum/ReduceMean); added `BatchedMatMul` to `JitOp`, `MatmulEpilogue`/`BatchedMatmulReduction` to `FusionKindJit`
  - Published `jit_fusion` module in `lib.rs` (was orphaned file)
  - Fixed 3 pre-existing clippy warnings in `jit_fusion.rs` (assign_op_pattern, map_or→is_some_and)
  - Files: `src/jit_fusion.rs`, `src/gradient.rs`, `src/tensor_ops/scalar_ops.rs`, `src/tensor_ops/mod.rs`, `src/lib.rs`; deleted `src/tensor_ops/linalg_ops_fixed.rs`
  - Tests: 8 in `tests/gradient_correctness_repair_tests.rs`; all 1320 tests pass

## Known Issues / Technical Debt

- Some gradient implementations for exotic matrix functions use approximate gradients; exact gradients tracked in issue backlog (the 2-norm condition number now has an exact analytic gradient via `CondOp::two_norm_gradient`; the 1-norm/∞-norm/Frobenius condition-number variants and matrix rank remain honestly non-differentiable rather than approximated)
- `graph_viz.rs` DOT output works best for small graphs; large graph layout needs truncation heuristics
