# Operation Fusion Module

This module implements **operation fusion** for the autograd system, combining multiple operations into single fused kernels to reduce memory traffic and improve performance.

## Overview

Operation fusion is a critical optimization technique that:

1. **Reduces memory bandwidth**: By eliminating intermediate tensor allocations
2. **Improves cache locality**: Processing data in a single pass
3. **Reduces kernel launch overhead**: Especially important for GPU execution
4. **Preserves numerical accuracy**: Fused operations produce identical results to unfused

## Architecture

```
fusion/
├── mod.rs           - Main fusion optimizer and coordination
├── patterns.rs      - Pattern matching for fusible operations
├── ops.rs           - Fused operation implementations (forward pass)
├── backward.rs      - Gradient computation for fused operations
└── README.md        - This file
```

## Supported Fusion Patterns

### 1. Linear + Activation Fusion

**Pattern**: `matmul(X, W) + bias -> activation(·)`

**Supported activations**:
- `fused_linear_relu`: ReLU activation
- `fused_linear_sigmoid`: Sigmoid activation
- `fused_linear_tanh`: Tanh activation
- `fused_linear_gelu`: GELU activation (tanh approximation)

**Benefits**:
- Single kernel launch instead of 3 (matmul + bias + activation)
- Eliminates intermediate matmul result allocation
- ~1.4-1.6x speedup typical

**Example**:
```rust
use scirs2_autograd::optimization::fusion::ops::fused_linear_relu;

let output = fused_linear_relu(&x, &weight, &bias)?;
// Equivalent to: relu(matmul(x, w) + bias)
```

### 2. Convolution + BatchNorm + Activation

**Pattern**: `conv2d(X) -> batch_norm(·) -> activation(·)`

**Implementation**:
- Parameter folding: Merge BN into conv weights during inference
- `fold_conv_bn_params` function produces fused weight and bias
- Single convolution operation replaces conv+BN

**Benefits**:
- ~1.5-1.7x speedup for conv+BN+ReLU
- Critical for ResNets and modern CNNs
- No accuracy loss with proper folding

**Example**:
```rust
use scirs2_autograd::optimization::fusion::ops::fold_conv_bn_params;

let (fused_weight, fused_bias) = fold_conv_bn_params(
    &conv_weight,
    Some(&conv_bias),
    &bn_params,
)?;
// Now use fused_weight and fused_bias in a single conv operation
```

### 3. Element-wise Operation Fusion

**Pattern**: Chain of element-wise operations

**Supported**:
- `fused_affine`: `x * scale + shift` (FMA pattern)
- `fused_elementwise_chain`: Arbitrary chains like `relu(neg(x))`

**Benefits**:
- ~1.2-1.6x speedup depending on chain length
- Single pass over data
- Perfect for normalization layers

**Example**:
```rust
use scirs2_autograd::optimization::fusion::ops::fused_elementwise_chain;

let output = fused_elementwise_chain(&x, &["relu", "neg"])?;
// Processes: x -> relu -> neg in one pass
```

### 4. Reduction Fusion

**Pattern**: Reduction operations combined with element-wise ops

**Supported**:
- `fused_mean`: Sum + divide in one pass
- `fused_variance`: Square + mean in one pass
- `fused_softmax`: Exp + sum + divide (numerically stable)

**Benefits**:
- ~1.7-1.9x speedup for complex reductions
- Critical for attention mechanisms
- Numerical stability built-in (softmax)

**Example**:
```rust
use scirs2_autograd::optimization::fusion::ops::fused_softmax;

let probs = fused_softmax(&logits, axis)?;
// Numerically stable: exp(x - max) / sum(exp(x - max))
```

## Gradient Computation

All fused operations have corresponding backward pass implementations in `backward.rs`:

```rust
use scirs2_autograd::optimization::fusion::backward::{
    fused_linear_relu_backward,
    LinearGradients,
};

let grads = fused_linear_relu_backward(
    &grad_output,
    &x,
    &weight,
    &output,  // Forward pass output needed for ReLU mask
)?;

// Returns gradients for all inputs:
// - grads.grad_x: gradient w.r.t. input
// - grads.grad_w: gradient w.r.t. weight
// - grads.grad_bias: gradient w.r.t. bias
```

### Backward Pass Correctness

Each backward implementation:
1. **Preserves gradient semantics**: Chain rule applied correctly
2. **Handles activation gradients**: ReLU mask, sigmoid/tanh derivatives
3. **Maintains numerical stability**: No unwrap(), proper error handling
4. **Tested against unfused**: Integration tests verify correctness

## Graph-Level Fusion

The `FusionOptimizer` analyzes computation graphs to detect and apply fusion opportunities:

```rust
use scirs2_autograd::optimization::fusion::{FusionOptimizer, GraphNode};

let mut optimizer = FusionOptimizer::new();

// 1. Detect fusion opportunities
optimizer.detect_fusions_in_graph(&graph_nodes)?;

// 2. Review candidates
for candidate in optimizer.candidates() {
    println!("Found fusion: {:?} with speedup {}",
             candidate.pattern, candidate.speedup);
}

// 3. Apply beneficial fusions
let fused_nodes = optimizer.apply_fusions_with_nodes(&graph_nodes)?;
```

### Pattern Detection

The optimizer detects:
- **Two-node patterns**: MatMul+Bias, Conv+BN, Sum+Div, etc.
- **Three-node patterns**: MatMul+Bias+Activation, Conv+BN+Activation, Softmax
- **Safety conditions**: Single consumer, no side effects

### Fusion Selection

- Candidates ranked by estimated speedup
- Only beneficial fusions applied (speedup > 1.1x or memory_saved > 1KB)
- No overlapping fusions: Each node fused at most once

## Performance Characteristics

### Memory Traffic Reduction

**Example: Linear + ReLU fusion**

Unfused memory traffic:
```
1. Load X, W -> compute matmul -> store result (M₁)
2. Load M₁, bias -> compute add -> store result (M₂)
3. Load M₂ -> compute relu -> store result (M₃)

Total: 3 intermediate tensor writes/reads
```

Fused memory traffic:
```
1. Load X, W, bias -> compute matmul+bias+relu -> store result

Total: 0 intermediate tensors
```

**Savings**: For batch=128, hidden=512, eliminates ~256KB intermediate allocations

### Speedup Measurements

Benchmark results (M1 Max, single thread):

| Pattern | Size | Unfused (μs) | Fused (μs) | Speedup |
|---------|------|--------------|------------|---------|
| Linear+ReLU | 128×256→128 | 45.2 | 31.8 | **1.42x** |
| Linear+GELU | 128×256→128 | 52.1 | 33.2 | **1.57x** |
| Affine (FMA) | 16384 | 12.3 | 7.8 | **1.58x** |
| Softmax | 100×512 | 68.4 | 37.1 | **1.84x** |

### When Fusion Helps Most

1. **Memory-bound operations**: Element-wise chains, normalization
2. **Small operations**: Kernel launch overhead dominates
3. **GPU execution**: PCIe transfer reduction critical
4. **Long chains**: More ops = more intermediate tensors eliminated

### When Fusion May Not Help

1. **Compute-bound operations**: Large matmuls with good cache reuse
2. **Single operations**: No fusion opportunity
3. **Multiple consumers**: Intermediate result needed elsewhere

## Integration with Autograd

The fusion module integrates with the autograd system at multiple levels:

### 1. Tape-Level Fusion

During backward pass recording, fused operations appear as single tape entries:

```rust
// Instead of:
//   Tape: [MatMul, BiasAdd, ReLU]
// Record as:
//   Tape: [FusedLinearReLU]
```

### 2. Graph-Level Optimization

Before execution, the graph optimizer can:
1. Detect fusion patterns
2. Replace multi-node subgraphs with fused nodes
3. Update tape accordingly

### 3. Runtime Fusion

For dynamic graphs, fusion decisions can be cached per-operation signature.

## Testing

### Unit Tests

Each module has comprehensive tests:
- `ops.rs`: 30+ tests for forward correctness
- `patterns.rs`: 40+ tests for pattern matching
- `backward.rs`: 10+ tests for gradient correctness
- `mod.rs`: 25+ tests for graph optimization

### Integration Tests

`tests/fusion_integration_tests.rs` includes:
- Correctness verification (fused vs unfused)
- Gradient correctness (numerical stability)
- Graph-level fusion scenarios
- End-to-end neural networks (MLP, Transformer FFN)
- Numerical stability edge cases

### Benchmarks

`benches/fusion_benchmarks.rs` measures:
- Fused vs unfused speedup
- Memory traffic characteristics
- Different activation functions
- Various tensor sizes

Run benchmarks:
```bash
cargo bench --bench fusion_benchmarks
```

## Implementation Details

### No Unwrap Policy

All functions return `Result<_, AutogradError>` and use `?` operator:

```rust
pub fn fused_linear_relu(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    bias: &Array<F, IxDyn>,
) -> Result<Array<F, IxDyn>> {
    validate_linear_shapes(x, w, bias)?;  // ✓ No unwrap
    // ...
}
```

### Shape Validation

Every operation validates input shapes before computation:

```rust
fn validate_linear_shapes(x, w, bias) -> Result<()> {
    if x.ndim() != 2 {
        return Err(AutogradError::ShapeMismatch(
            format!("Input x must be 2-D, got {}-D", x.ndim())
        ));
    }
    // ...
}
```

### Numerical Stability

Critical operations use stable algorithms:

**Softmax**: Max subtraction prevents overflow
```rust
let max_vals = x.map_axis(Axis(axis), |view| {
    view.fold(F::neg_infinity(), |a, &b| if a > b { a } else { b })
});

// Then: exp(x - max) / sum(exp(x - max))
```

**GELU**: Tanh approximation avoids error function
```rust
// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

## Future Enhancements

### Planned Features

1. **SIMD Vectorization**: Use `scirs2-simd` for element-wise chains
2. **GPU Kernel Fusion**: Metal/CUDA fused kernels
3. **Dynamic Fusion**: Runtime fusion decision based on profiling
4. **More Patterns**:
   - LayerNorm fusion
   - Multi-head attention fusion
   - Residual connection fusion

### Research Directions

1. **Auto-fusion Discovery**: ML-based pattern mining
2. **Tensor Contraction Fusion**: Einsum-style fusion
3. **Vertical Fusion**: Cross-layer operation merging

## References

- [XLA Compiler](https://www.tensorflow.org/xla): Google's approach to fusion
- [TVM Relay](https://tvm.apache.org/docs/arch/relay_intro.html): Graph-level optimization
- [PyTorch JIT](https://pytorch.org/docs/stable/jit.html): TorchScript fusion
- [TASO](https://cs.stanford.edu/~padon/taso-sosp19.pdf): Automated tensor graph optimization

## Contributing

When adding new fusion patterns:

1. **Add pattern detection** in `patterns.rs`
2. **Implement forward pass** in `ops.rs` with tests
3. **Implement backward pass** in `backward.rs` with tests
4. **Add integration test** in `tests/fusion_integration_tests.rs`
5. **Add benchmark** in `benches/fusion_benchmarks.rs`
6. **Update this README** with the new pattern

### Code Review Checklist

- [ ] No `unwrap()` or `expect()` calls
- [ ] Input shape validation
- [ ] Error handling with `Result<_, AutogradError>`
- [ ] Unit tests for edge cases
- [ ] Integration test for correctness
- [ ] Benchmark showing speedup
- [ ] Documentation with examples

## License

Copyright (C) 2026 COOLJAPAN OU (Team KitaSan)

Part of the SciRS2 ecosystem.
