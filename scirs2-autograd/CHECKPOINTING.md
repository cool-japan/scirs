# Gradient Checkpointing

## Overview

Gradient checkpointing is a memory optimization technique that trades computation time for reduced memory usage during backpropagation. In deep neural networks or complex computation graphs, storing all intermediate activations can consume a significant amount of memory. Gradient checkpointing addresses this by selectively discarding some intermediate results during the forward pass and recomputing them during the backward pass.

## Principles

The core idea of gradient checkpointing is:
- During the forward pass, only save input activations at strategic points (checkpoints)
- During the backward pass, recompute intermediate activations as needed for gradient calculations
- This reduces peak memory usage at the cost of additional computation

## When to Use Checkpointing

Use gradient checkpointing when:
- Training deep models that don't fit in memory
- Working with limited GPU/memory resources
- Processing large inputs or high-resolution data
- Performing computation with large intermediate tensors

## Basic Usage

### Single Tensor Checkpointing

To checkpoint a single tensor, use the `checkpoint` function:

```rust
use ag::tensor_ops as T;

// Normal computation
let x = T::ones(&[128, 128], ctx);
let y = T::matmul(&x, &x);
let z = T::relu(&y);

// With checkpointing
let x = T::ones(&[128, 128], ctx);
let y = T::matmul(&x, &x);
let z = T::checkpoint(&T::relu(&y)); // Checkpoint the activation
```

### Adaptive Checkpointing

To automatically decide whether to checkpoint based on tensor size:

```rust
// Define a memory threshold (in bytes)
let threshold = 10 * 1024; // 10KB

// Apply checkpointing only if the tensor is large
let result = T::adaptive_checkpoint(&large_tensor, threshold);
```

### Profiling Checkpointing

To measure memory savings from checkpointing:

```rust
// Start tracking checkpoint statistics
T::CheckpointProfiler::start_tracking();

// Perform operations with checkpointing
// ...

// Get statistics
let memory_saved = T::CheckpointProfiler::memory_saved();
let num_checkpoints = T::CheckpointProfiler::checkpoint_count();

// Stop tracking
T::CheckpointProfiler::stop_tracking();
```

## Advanced Usage

### Checkpointing Segments

For checkpointing a segment of computation:

```rust
// Classic segment checkpoint with fixed size array
let result = T::checkpoint_segment(
    ctx,
    [&input1, &input2], // Fixed-size array
    |inputs| {
        // Computation using inputs[0] and inputs[1]
        // ...
        output_tensor
    }
);

// More flexible segment checkpoint with slice
let result = T::checkpoint_segment_flex(
    ctx,
    &[&input1, &input2, &input3], // Slice of inputs
    |inputs| {
        // Computation using inputs
        // ...
        output_tensor
    }
);
```

### Using CheckpointGroup for Multiple Outputs

Create a checkpoint group for operations that produce multiple outputs:

```rust
// Create a checkpoint group
let ckpt_group = T::CheckpointGroup::new(ctx);

// Classic method with generics
let (output1, output2) = ckpt_group.checkpoint_fn(
    (&input1, &input2),
    |inputs| {
        let a = inputs.0;
        let b = inputs.1;
        // ...
        (result1, result2)
    }
);

// More flexible method with slices
let result = ckpt_group.checkpoint_fn_flex(
    &[&input1, &input2, &input3],
    |inputs| {
        // ...
        output_tensor
    }
);

// For two output tensors
let (result1, result2) = ckpt_group.checkpoint_fn_flex2(
    &[&input1, &input2],
    |inputs| {
        // ...
        (output1, output2)
    }
);

// For three output tensors
let (result1, result2, result3) = ckpt_group.checkpoint_fn_flex3(
    &[&input1, &input2],
    |inputs| {
        // ...
        (output1, output2, output3)
    }
);
```

## Implementation Details

Checkpointing works by:
1. Detaching intermediate tensors from the computational graph
2. Recomputing these intermediates during the backward pass
3. Ensuring that gradients flow correctly through the recomputed path

Internally, it uses a custom operation called `CheckpointOp` that handles the mechanics of preserving gradient flow while avoiding storage of intermediate activations.

## Examples

Check the provided examples for practical demonstrations:
- `examples/simple_checkpointing.rs`: Basic checkpointing concepts
- `examples/gradient_checkpointing.rs`: Practical applications in a deep network
- `examples/enhanced_gradient_checkpointing.rs`: Advanced techniques with profiling
- `examples/flexible_checkpointing.rs`: Using the new flexible checkpointing APIs

## Performance Considerations

- Checkpointing typically increases computation time by 20-30%
- Memory savings can be 50-80% depending on checkpoint placement
- Strategic placement of checkpoints is key for optimal performance
- For best results, checkpoint operations with large output tensors but relatively cheap computation

## Troubleshooting

If you encounter issues with gradient calculations when using checkpointing:
- Make sure the checkpoint is placed directly on the tensor in the computation graph
- Be careful with operations that have side effects 
- When using checkpoint_segment or CheckpointGroup, ensure inputs are properly accessible in the closure
- If encountering lifetime issues, try the flexible versions of the checkpointing APIs