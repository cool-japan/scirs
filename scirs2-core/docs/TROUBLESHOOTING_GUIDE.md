# Troubleshooting Guide for scirs2-core

## Overview

This guide helps diagnose and resolve common issues when using scirs2-core. It covers error messages, performance problems, compatibility issues, and debugging techniques.

## Table of Contents

1. [Common Error Messages](#common-error-messages)
2. [Build and Compilation Issues](#build-and-compilation-issues)
3. [Runtime Errors](#runtime-errors)
4. [Performance Problems](#performance-problems)
5. [Memory Issues](#memory-issues)
6. [GPU Problems](#gpu-problems)
7. [Platform-Specific Issues](#platform-specific-issues)
8. [Debugging Techniques](#debugging-techniques)
9. [Getting Help](#getting-help)

## Common Error Messages

### "Array shape mismatch"

**Error:**
```
Error: ShapeError { expected: [100, 100], actual: [100, 101] }
```

**Cause:** Arrays have incompatible dimensions for the operation.

**Solution:**
```rust
use scirs2_core::validation::check_shape_compatibility;

// Verify shapes before operations
check_shape_compatibility(&arr1.shape(), &arr2.shape())?;

// Or use broadcasting
let result = arr1.broadcast(&arr2)?;
```

### "Out of memory"

**Error:**
```
Error: AllocationError { requested: 8589934592, available: 4294967296 }
```

**Cause:** Insufficient memory for array allocation.

**Solution:**
```rust
use scirs2_core::memory_efficient::{ChunkedArray, OutOfCoreArray};

// Use chunked arrays for large data
let array = ChunkedArray::from_file("large_data.bin", 1024)?;

// Or use out-of-core arrays
let array = OutOfCoreArray::new(shape, "data.mmap")?;
```

### "Feature not enabled"

**Error:**
```
Error: FeatureNotEnabled { feature: "gpu" }
```

**Cause:** Required feature flag not enabled during compilation.

**Solution:**
```toml
# Cargo.toml
[dependencies]
scirs2-core = { version = "0.1.0-alpha.6", features = ["gpu", "parallel"] }
```

## Build and Compilation Issues

### OpenBLAS Not Found

**Error:**
```
error: failed to run custom build command for `openblas-src v0.10.11`
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev pkg-config

# macOS
brew install openblas

# Windows
vcpkg install openblas
set PKG_CONFIG_PATH=%VCPKG_ROOT%\installed\x64-windows\lib\pkgconfig
```

### CUDA Libraries Not Found

**Error:**
```
error: could not find CUDA installation
```

**Solution:**
```bash
# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
```

### Compilation Timeout

**Error:**
```
error: could not compile `scirs2-core` (lib) due to previous error
```

**Solution:**
```bash
# Increase codegen units for faster compilation
CARGO_PROFILE_DEV_CODEGEN_UNITS=256 cargo build

# Or use faster linker
# Linux
RUSTFLAGS="-C link-arg=-fuse-ld=lld" cargo build

# macOS
RUSTFLAGS="-C link-arg=-fuse-ld=/usr/local/opt/llvm/bin/ld64.lld" cargo build
```

## Runtime Errors

### Segmentation Fault

**Symptoms:** Program crashes with "Segmentation fault" or "signal 11"

**Common Causes:**
1. Stack overflow in recursive algorithms
2. Incorrect FFI usage
3. Race conditions in parallel code

**Debugging:**
```bash
# Enable core dumps
ulimit -c unlimited

# Run with debugging symbols
RUST_BACKTRACE=full cargo run

# Use valgrind for memory debugging
valgrind --leak-check=full --track-origins=yes ./target/debug/myapp

# Use address sanitizer
RUSTFLAGS="-Z sanitizer=address" cargo +nightly run
```

### Thread Panic

**Error:**
```
thread 'main' panicked at 'index out of bounds: the len is 100 but the index is 100'
```

**Solution:**
```rust
use std::panic;

// Set panic hook for better error messages
panic::set_hook(Box::new(|info| {
    eprintln!("Panic occurred: {}", info);
    eprintln!("Backtrace: {:?}", backtrace::Backtrace::new());
}));

// Use safe indexing
if let Some(value) = array.get(index) {
    // Process value
}
```

### Deadlock

**Symptoms:** Program hangs indefinitely

**Debugging:**
```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

// Add timeout to locks
let result = mutex.try_lock_for(Duration::from_secs(5));
if result.is_err() {
    eprintln!("Potential deadlock detected!");
}

// Use deadlock detection
#[cfg(debug_assertions)]
{
    parking_lot::deadlock::check_deadlock();
}
```

## Performance Problems

### Slow Array Operations

**Symptoms:** Operations taking longer than expected

**Diagnosis:**
```rust
use scirs2_core::profiling::{Profiler, Timer};

// Profile operations
let _timer = Timer::new("array_multiply");
let result = arr1 * arr2;

// Check if SIMD is enabled
if !scirs2_core::simd::is_enabled() {
    eprintln!("Warning: SIMD not enabled!");
}
```

**Solutions:**
1. Enable SIMD features:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

2. Use parallel operations:
```rust
use scirs2_core::parallel::ParallelExt;

// Parallel array operations
let result = arr1.par_mul(&arr2)?;
```

3. Optimize memory layout:
```rust
// Use column-major for better cache performance
let array = Array2::from_shape_vec((rows, cols).f(), data)?;
```

### High Memory Usage

**Diagnosis:**
```rust
use scirs2_core::monitoring::MemoryMonitor;

let monitor = MemoryMonitor::new();
monitor.start();

// Your operations here

let stats = monitor.get_stats();
println!("Peak memory: {} MB", stats.peak_usage_mb);
```

**Solutions:**
```rust
// Use lazy evaluation
use scirs2_core::memory_efficient::LazyArray;

let lazy = LazyArray::new(|| expensive_computation());

// Free memory explicitly
drop(large_array);

// Use memory pools
use scirs2_core::memory::MemoryPool;

let pool = MemoryPool::new(1024 * 1024 * 1024); // 1GB pool
```

## Memory Issues

### Memory Leak Detection

**Tools:**
```bash
# Using heaptrack
heaptrack ./target/release/myapp
heaptrack --analyze heaptrack.myapp.12345.gz

# Using valgrind
valgrind --leak-check=full --show-leak-kinds=all ./target/release/myapp
```

**Common Causes:**
1. Circular references with Rc/Arc
2. Forgotten resources (files, GPU memory)
3. Unbounded caches

**Solutions:**
```rust
// Use weak references to break cycles
use std::rc::{Rc, Weak};

struct Node {
    parent: Weak<Node>,
    children: Vec<Rc<Node>>,
}

// Implement Drop for cleanup
impl Drop for MyResource {
    fn drop(&mut self) {
        // Clean up resources
    }
}

// Bound cache sizes
use scirs2_core::cache::TTLSizedCache;

let cache = TTLSizedCache::new(1000, Duration::from_secs(3600));
```

### Stack Overflow

**Error:**
```
thread 'main' has overflowed its stack
```

**Solutions:**
```rust
// Increase stack size
std::thread::Builder::new()
    .stack_size(8 * 1024 * 1024) // 8MB
    .spawn(|| {
        // Your code here
    })?;

// Convert recursion to iteration
fn iterative_factorial(n: u64) -> u64 {
    (1..=n).product()
}
```

## GPU Problems

### CUDA Out of Memory

**Error:**
```
CudaError: out of memory (error code 2)
```

**Solutions:**
```rust
use scirs2_core::gpu::{GpuMemoryPool, GpuDevice};

// Monitor GPU memory
let device = GpuDevice::current()?;
println!("Available GPU memory: {} MB", device.available_memory_mb());

// Use memory pool
let pool = GpuMemoryPool::new(0.8)?; // Use 80% of GPU memory

// Clear GPU cache
device.clear_cache()?;
```

### GPU Not Detected

**Diagnosis:**
```bash
# Check NVIDIA GPU
nvidia-smi

# Check OpenCL devices
clinfo

# Check environment
echo $CUDA_VISIBLE_DEVICES
```

**Solutions:**
```rust
// Fallback to CPU
use scirs2_core::gpu::{GpuRuntime, CpuFallback};

let runtime = GpuRuntime::new()
    .with_fallback(CpuFallback::Auto)
    .build()?;
```

## Platform-Specific Issues

### macOS: Metal Performance Shaders

**Issue:** Metal backend not working

**Solution:**
```rust
// Check Metal availability
#[cfg(target_os = "macos")]
{
    use scirs2_core::gpu::metal::MetalDevice;
    
    if !MetalDevice::is_available() {
        eprintln!("Metal not available on this system");
    }
}
```

### Windows: Visual C++ Runtime

**Error:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution:**
```powershell
# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools

# Or install full Visual Studio
winget install Microsoft.VisualStudio.2022.Community
```

### Linux: glibc Version

**Error:**
```
/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
```

**Solution:**
```bash
# Check glibc version
ldd --version

# Build with older glibc target
RUSTFLAGS="-C target-feature=-crt-static" cargo build
```

## Debugging Techniques

### Enable Debug Logging

```rust
use env_logger;

// Set log level
std::env::set_var("RUST_LOG", "debug");
env_logger::init();

// Or use scirs2 logging
use scirs2_core::logging::Logger;

Logger::init_with_level("debug")?;
```

### Performance Profiling

```rust
use scirs2_core::profiling::{Profiler, FlameGraph};

// Start profiling
let profiler = Profiler::new();
profiler.start();

// Your code here

// Generate flame graph
profiler.stop();
profiler.generate_flamegraph("profile.svg")?;
```

### Memory Profiling

```rust
use scirs2_core::profiling::MemoryProfiler;

let profiler = MemoryProfiler::new();
profiler.start();

// Your code here

let report = profiler.generate_report();
println!("{}", report);
```

### Debug Assertions

```rust
// Add debug assertions
debug_assert!(array.len() > 0, "Array should not be empty");
debug_assert_eq!(result.shape(), expected_shape);

// Conditional debugging
#[cfg(debug_assertions)]
{
    validate_intermediate_results(&data)?;
}
```

## Getting Help

### Diagnostic Information

When reporting issues, include:

```bash
# System information
uname -a
rustc --version
cargo --version

# Package versions
cargo tree | grep scirs2

# Build configuration
cargo build --verbose 2>&1 | head -20

# Environment
env | grep -E '(RUST|CARGO|SCIRS2)'
```

### Creating Minimal Reproduction

```rust
// minimal_repro.rs
use scirs2_core::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimal code that reproduces the issue
    let array = Array2::zeros((100, 100));
    // Issue occurs here
    
    Ok(())
}
```

### Community Resources

- **GitHub Issues**: https://github.com/scirs2/scirs2-core/issues
- **Discussions**: https://github.com/scirs2/scirs2-core/discussions
- **Discord**: https://discord.gg/scirs2
- **Stack Overflow**: Tag with `scirs2`

### Filing Bug Reports

Include:
1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Rust version, scirs2 version
6. **Logs**: Error messages and stack traces

Example:
```markdown
## Bug Report

**Description:** Array multiplication panics with dimension mismatch

**Steps to Reproduce:**
```rust
let a = Array2::ones((10, 20));
let b = Array2::ones((20, 30));
let c = a.dot(&b); // Panics here
```

**Expected:** Should compute matrix multiplication

**Actual:** Panics with "dimension mismatch"

**Environment:**
- OS: Ubuntu 22.04
- Rust: 1.70.0
- scirs2-core: 0.1.0-alpha.6
```

---

*Last Updated: 2025-06-22 | Version: 0.1.0-alpha.6*