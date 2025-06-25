# CUDA Integration Guide for scirs2-fft

This guide provides instructions for integrating and using the CUDA-accelerated sparse FFT functionality in the scirs2-fft crate.

## Prerequisites

Before using CUDA acceleration, ensure you have the following:

1. NVIDIA CUDA Toolkit (version 11.0 or later)
2. Compatible NVIDIA GPU with compute capability 5.0 or higher
3. Proper GPU drivers installed

## Installation

To use CUDA acceleration, enable the `cuda` feature in your `Cargo.toml`:

```toml
[dependencies]
scirs2-fft = { version = "0.1.0-alpha.6", features = ["cuda"] }
```

## System Requirements

- Linux, Windows, or macOS (macOS support is limited to older CUDA versions)
- Sufficient system memory for your workload
- Recommended: NVIDIA GPU with at least 4GB of VRAM for large-scale signal processing

## Basic Usage

Here's a simple example of using CUDA-accelerated sparse FFT:

```rust
use scirs2_fft::{
    cuda_sparse_fft, CUDADeviceInfo, is_cuda_available,
    sparse_fft::{SparseFFTAlgorithm, WindowFunction}
};

fn main() {
    // Check if CUDA is available
    if !is_cuda_available() {
        println!("CUDA is not available on this system");
        return;
    }
    
    // Get available CUDA devices
    let devices = get_cuda_devices().unwrap();
    println!("Found {} CUDA device(s)", devices.len());
    
    for device in &devices {
        println!("Device {}: {} ({:.1} GB)", 
            device.device_id, 
            device.name, 
            device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    
    // Create a test signal
    let n = 1024;
    let mut signal = vec![0.0; n];
    
    // Fill with a simple signal (sine waves at different frequencies)
    for i in 0..n {
        let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        signal[i] = (3.0 * t).sin() + 0.5 * (7.0 * t).sin() + 0.25 * (15.0 * t).sin();
    }
    
    // Perform CUDA-accelerated sparse FFT
    let result = cuda_sparse_fft(
        &signal,
        6,                              // Expected sparsity
        0,                              // Use first CUDA device
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann)
    ).unwrap();
    
    // Print the results
    println!("Found {} significant frequency components", result.values.len());
    println!("Computation time: {:?}", result.computation_time);
    
    // Print top 3 components by magnitude
    let mut components: Vec<_> = result.indices.iter()
        .zip(result.values.iter())
        .collect();
    
    components.sort_by(|(_, a), (_, b)| 
        b.norm().partial_cmp(&a.norm()).unwrap_or(std::cmp::Ordering::Equal)
    );
    
    for (i, (&idx, val)) in components.iter().take(3).enumerate() {
        println!("{}. Frequency {}: magnitude = {:.3}", i + 1, idx, val.norm());
    }
}
```

## Advanced Features

### Batch Processing

For processing multiple signals efficiently:

```rust
let signals = vec![
    vec![/* signal 1 data */],
    vec![/* signal 2 data */],
    vec![/* signal 3 data */],
];

let results = cuda_batch_sparse_fft(
    &signals,
    6,                              // Expected sparsity
    0,                              // Use first CUDA device
    Some(SparseFFTAlgorithm::Sublinear),
    Some(WindowFunction::Hann)
).unwrap();
```

### Memory Management

For fine-grained control over GPU memory usage:

```rust
use scirs2_fft::{
    CUDAContext, BufferLocation, BufferType, AllocationStrategy,
    init_global_memory_manager
};

// Initialize memory manager with a limit
init_global_memory_manager(
    GPUBackend::CUDA,
    0,                    // Device ID
    AllocationStrategy::CacheBySize,
    1024 * 1024 * 1024    // 1 GB limit
).unwrap();

// Create CUDA context
let context = CUDAContext::new(0).unwrap();

// Allocate buffer
let buffer = context.allocate(1024 * 1024).unwrap(); // 1 MB buffer

// Use buffer for computation...

// Free buffer when done
context.free(buffer).unwrap();
```

### Stream-Based Processing

For concurrent execution and asynchronous operations:

```rust
// Create a CUDA context
let context = CUDAContext::new(0).unwrap();

// Get the default stream
let stream = context.stream();

// For more advanced use cases, create a custom stream
let custom_stream = CUDAStream::new(0).unwrap();

// When operations are complete, synchronize
custom_stream.synchronize().unwrap();
```

## Troubleshooting

### Common Issues

1. **CUDA not detected**:
   - Ensure NVIDIA drivers are installed and up-to-date
   - Check that CUDA toolkit is properly installed
   - Verify GPU compatibility

2. **Out of memory errors**:
   - Reduce batch size or signal size
   - Use memory-efficient methods for large signals
   - Consider upgrading GPU memory

3. **Performance issues**:
   - Check for CPU-GPU bottlenecks (data transfer)
   - Use batch processing for small signals
   - Ensure proper stream synchronization

### Debugging

When encountering issues, enable debug logging to get more information:

```rust
// Set environment variable for debug logging
std::env::set_var("SCIRS_FFT_LOG_LEVEL", "debug");

// Initialize CUDA
let devices = get_cuda_devices().unwrap();
```

## Performance Optimization

For best performance with CUDA-accelerated sparse FFT:

1. **Batch processing**: Process multiple signals at once to maximize GPU utilization
2. **Minimize transfers**: Keep data on the GPU as much as possible
3. **Choose appropriate algorithm**: Different algorithms perform better for different signal types
4. **Use pinned memory**: For faster host-device transfers
5. **Optimize signal size**: Powers of two often perform better

## Known Limitations

- Very large signals (>1GB) may require special handling
- Small signals (<64 samples) may not benefit from GPU acceleration
- Mixed-precision computation may affect accuracy
- Some window functions may be computationally intensive on GPU

## References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/index.html)
- [GPU-Accelerated FFT Performance Guide](https://developer.nvidia.com/blog/gpu-accelerated-fft-performance/)