# Troubleshooting Common Issues

This guide helps you diagnose and resolve common problems when using `scirs2-integrate`. Each issue includes symptoms, causes, and step-by-step solutions.

## 🚨 Quick Diagnostic Checklist

Before diving into specific issues, run this quick diagnostic:

```rust
use scirs2_integrate::*;

fn quick_diagnostic() {
    println!("=== scirs2-integrate Diagnostic ===");
    
    // Check basic functionality
    let simple_integral = quad::quad(|x: f64| x, 0.0, 1.0, None);
    println!("Basic integration: {:?}", simple_integral.is_ok());
    
    // Check ODE solver
    let simple_ode = ode::solve_ivp(
        |_t, y| ndarray::array![y[0]],
        [0.0, 0.1],
        ndarray::array![1.0],
        None
    );
    println!("Basic ODE solving: {:?}", simple_ode.is_ok());
    
    // Check features
    println!("SIMD support: {}", cfg!(target_feature = "avx2"));
    println!("Parallel support: {}", cfg!(feature = "parallel"));
    
    // Memory check
    let available_memory = get_available_memory_mb();
    println!("Available memory: {} MB", available_memory);
}
```

## 📊 Integration Problems

### Problem: Integration Fails to Converge

**Symptoms:**
- `IntegrateError::MaxIterationsExceeded`
- Very slow convergence
- Large error estimates

**Causes & Solutions:**

#### Cause 1: Function has singularities
```rust
// ❌ Problem: Function has endpoint singularities
let result = quad::quad(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, None);

// ✅ Solution: Use tanh-sinh quadrature
use scirs2_integrate::tanhsinh::*;
let result = tanhsinh(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, None)?;
```

#### Cause 2: Highly oscillatory function
```rust
// ❌ Problem: High-frequency oscillations
let result = quad::quad(|x: f64| (100.0 * x).sin(), 0.0, 1.0, None);

// ✅ Solution: Use higher-order quadrature or increase evaluations
let options = QuadOptions::new().max_evals(10000);
let result = quad::quad(|x: f64| (100.0 * x).sin(), 0.0, 1.0, Some(options))?;

// Or use Gaussian quadrature
let result = gaussian::gauss_legendre(|x: f64| (100.0 * x).sin(), 0.0, 1.0, 50)?;
```

#### Cause 3: Discontinuous function
```rust
// ❌ Problem: Function has discontinuities
let step_fn = |x: f64| if x < 0.5 { 1.0 } else { 2.0 };
let result = quad::quad(step_fn, 0.0, 1.0, None);

// ✅ Solution: Split at discontinuities
let result1 = quad::quad(|_| 1.0, 0.0, 0.5, None)?;
let result2 = quad::quad(|_| 2.0, 0.5, 1.0, None)?;
let total = result1.value + result2.value;

// Or use Monte Carlo for robustness
let mc_result = monte_carlo::monte_carlo(
    |x| if x[0] < 0.5 { 1.0 } else { 2.0 },
    &[(0.0, 1.0)],
    Some(MonteCarloOptions::new().n_samples(100000))
)?;
```

### Problem: Multi-dimensional Integration Too Slow

**Symptoms:**
- Long computation times
- High memory usage
- Poor accuracy with reasonable time

**Solutions:**

```rust
// Strategy 1: Use appropriate method for dimensionality
fn choose_method_by_dimension(dims: usize) -> IntegrationStrategy {
    match dims {
        1 => IntegrationStrategy::Adaptive1D,
        2..=3 => IntegrationStrategy::AdaptiveCubature,
        4..=10 => IntegrationStrategy::QuasiMonteCarlo,
        _ => IntegrationStrategy::ParallelMonteCarlo,
    }
}

// Strategy 2: Parallel processing for high dimensions
use scirs2_integrate::monte_carlo_parallel::*;
let result = parallel_monte_carlo(
    your_function,
    bounds,
    Some(ParallelMonteCarloOptions::new()
        .workers(num_cpus::get())
        .samples_per_worker(50000))
)?;

// Strategy 3: Variance reduction techniques
let mc_options = MonteCarloOptions::new()
    .variance_reduction(VarianceReduction::AntitheticVariates)
    .importance_sampling(true);
```

## 🔧 ODE Solver Issues

### Problem: ODE Integration Fails or is Very Slow

**Symptoms:**
- `ODEError::StepSizeTooSmall`
- Excessive function evaluations (`nfev` > 100,000)
- Integration stops prematurely

**Diagnostic Steps:**

```rust
fn diagnose_ode_problem<F>(rhs: F, t_span: [f64; 2], y0: ndarray::Array1<f64>) 
where F: Fn(f64, ndarray::ArrayView1<f64>) -> ndarray::Array1<f64>
{
    println!("=== ODE Diagnostic ===");
    
    // Test 1: Check if system is stiff
    let explicit_result = ode::solve_ivp(
        &rhs, t_span, y0.clone(),
        Some(ODEOptions::new().method(ODEMethod::RK23).rtol(1e-3))
    );
    
    match explicit_result {
        Ok(sol) => {
            println!("Explicit method succeeded: {} evaluations", sol.nfev);
            if sol.nfev > 10000 {
                println!("Warning: High evaluation count suggests stiffness");
            }
        }
        Err(e) => {
            println!("Explicit method failed: {:?}", e);
            println!("System is likely stiff - try implicit methods");
        }
    }
    
    // Test 2: Check for rapid changes
    let step_size = (t_span[1] - t_span[0]) / 100.0;
    let dy_dt = rhs(t_span[0], y0.view());
    let time_scale = y0.iter().zip(dy_dt.iter())
        .map(|(&y, &dy)| if dy.abs() > 1e-12 { y.abs() / dy.abs() } else { f64::INFINITY })
        .fold(f64::INFINITY, f64::min);
    
    println!("Estimated time scale: {:.2e}", time_scale);
    if time_scale < step_size / 10.0 {
        println!("Warning: System has fast time scales");
    }
}
```

#### Solution 1: Switch to Appropriate Method

```rust
// For stiff systems
let stiff_options = ODEOptions::new()
    .method(ODEMethod::BDF)  // or Radau, LSODA
    .rtol(1e-6)
    .atol(1e-9);

// For non-stiff but sensitive systems
let sensitive_options = ODEOptions::new()
    .method(ODEMethod::DOP853)  // High-order method
    .rtol(1e-10)
    .atol(1e-13);

// For automatic stiffness detection
let auto_options = ODEOptions::new()
    .method(ODEMethod::LSODA);  // Switches automatically
```

#### Solution 2: Adjust Tolerances

```rust
// Diagnosis: Check if tolerances are too tight
fn find_optimal_tolerances<F>(rhs: F, t_span: [f64; 2], y0: ndarray::Array1<f64>) -> (f64, f64)
where F: Fn(f64, ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> + Clone
{
    let tolerances = [
        (1e-3, 1e-6), (1e-4, 1e-7), (1e-5, 1e-8), 
        (1e-6, 1e-9), (1e-7, 1e-10)
    ];
    
    for (rtol, atol) in tolerances {
        let start = std::time::Instant::now();
        let result = ode::solve_ivp(
            rhs.clone(), t_span, y0.clone(),
            Some(ODEOptions::new().rtol(rtol).atol(atol))
        );
        
        match result {
            Ok(sol) => {
                let time = start.elapsed().as_secs_f64();
                println!("rtol={:.0e}, atol={:.0e}: {:.3}s, {} evals", 
                         rtol, atol, time, sol.nfev);
                         
                if sol.nfev < 10000 && time < 1.0 {
                    return (rtol, atol);
                }
            }
            Err(e) => println!("Failed with rtol={:.0e}, atol={:.0e}: {:?}", rtol, atol, e),
        }
    }
    
    (1e-6, 1e-9) // Default fallback
}
```

### Problem: ODE Solution is Inaccurate

**Symptoms:**
- Solution drifts from expected behavior
- Conservation laws violated
- Instability or blow-up

**Solutions:**

#### Solution 1: Verify Problem Formulation
```rust
// Check that your RHS function is correct
fn verify_rhs_function() {
    let t = 0.0;
    let y = ndarray::array![1.0, 0.0]; // Example state
    let dy_dt = your_rhs_function(t, y.view());
    
    // Manual verification
    println!("At t={}, y={:?}, dy/dt={:?}", t, y, dy_dt);
    
    // Check dimensions
    assert_eq!(y.len(), dy_dt.len(), "RHS dimension mismatch");
    
    // Check for NaN/Inf
    assert!(dy_dt.iter().all(|&x| x.is_finite()), "RHS produces non-finite values");
}
```

#### Solution 2: Use Conservation-Preserving Methods
```rust
// For Hamiltonian systems
use scirs2_integrate::symplectic::*;

let hamiltonian = SeparableHamiltonian {
    kinetic: |p| 0.5 * p.iter().map(|&pi| pi * pi).sum::<f64>(),
    potential: |q| 0.5 * q.iter().map(|&qi| qi * qi).sum::<f64>(),
};

let result = velocity_verlet(hamiltonian, t_span, q0, p0)?;

// Verify energy conservation
let initial_energy = hamiltonian.total_energy(q0.view(), p0.view());
let final_energy = hamiltonian.total_energy(
    result.q.last().unwrap().view(),
    result.p.last().unwrap().view()
);
println!("Energy drift: {:.2e}", (final_energy - initial_energy).abs());
```

#### Solution 3: Implement Invariant Monitoring
```rust
fn solve_with_invariant_monitoring<F, G>(
    rhs: F,
    invariant: G,
    t_span: [f64; 2],
    y0: ndarray::Array1<f64>
) -> Result<ODEResult, ODEError>
where 
    F: Fn(f64, ndarray::ArrayView1<f64>) -> ndarray::Array1<f64>,
    G: Fn(ndarray::ArrayView1<f64>) -> f64,
{
    let initial_invariant = invariant(y0.view());
    
    let result = ode::solve_ivp(rhs, t_span, y0, None)?;
    
    // Check invariant preservation
    for (i, y) in result.y.iter().enumerate() {
        let current_invariant = invariant(y.view());
        let drift = (current_invariant - initial_invariant).abs();
        
        if drift > 1e-6 {
            println!("Warning: Invariant drift {:.2e} at step {}", drift, i);
        }
    }
    
    Ok(result)
}
```

## 💾 Memory Issues

### Problem: Out of Memory Errors

**Symptoms:**
- `std::alloc::alloc` panics
- Sudden termination
- Very slow performance due to swapping

**Solutions:**

#### Solution 1: Enable Memory Pooling
```rust
use scirs2_integrate::memory::*;

// Create a memory pool for reuse
let pool = MemoryPool::new(10 * 1024 * 1024); // 10 MB pool

// Use pooled memory in your computations
fn solve_with_memory_pool(pool: &MemoryPool) -> Result<Vec<ODEResult>, Error> {
    let mut results = Vec::new();
    
    for problem in problem_set {
        let buffer = pool.get_buffer(required_size)?;
        let result = solve_with_preallocated_memory(problem, buffer)?;
        results.push(result);
        // Buffer automatically returned to pool when dropped
    }
    
    Ok(results)
}
```

#### Solution 2: Reduce Memory Footprint
```rust
// Use dense output sparingly
let minimal_output = ODEOptions::new()
    .dense_output(false)  // Don't store intermediate steps
    .max_store_steps(1000); // Limit stored steps

// For very long integrations
let streaming_options = ODEOptions::new()
    .output_callback(|t, y| {
        // Process results immediately instead of storing
        process_result(t, y);
    });
```

#### Solution 3: Chunked Processing
```rust
fn solve_large_system_chunked(
    problem: LargeProblem,
    chunk_size: usize
) -> Result<Vec<f64>, Error> {
    let mut results = Vec::new();
    
    for chunk in problem.chunks(chunk_size) {
        let chunk_result = solve_chunk(chunk)?;
        results.extend(chunk_result);
        
        // Force garbage collection between chunks
        std::hint::black_box(&results);
    }
    
    Ok(results)
}
```

### Problem: Memory Leaks

**Symptoms:**
- Memory usage grows over time
- Performance degrades with repeated calls
- System becomes unresponsive

**Diagnostic Tools:**

```rust
// Memory usage monitoring
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct MemoryTracker;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for MemoryTracker {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static GLOBAL: MemoryTracker = MemoryTracker;

fn check_memory_usage() {
    println!("Current memory usage: {} bytes", ALLOCATED.load(Ordering::SeqCst));
}
```

## ⚠️ Compilation Issues

### Problem: Feature-related Compilation Errors

**Symptoms:**
- Missing function errors
- Feature gate errors
- Linking errors

**Solutions:**

#### Check Feature Flags
```toml
# Ensure all required features are enabled
[dependencies]
scirs2-integrate = { 
    version = "0.1.0-alpha.6", 
    features = ["parallel", "simd", "autotuning", "plotting"] 
}
```

#### Platform-Specific Issues
```rust
// Check platform capabilities
fn check_platform_support() {
    #[cfg(not(feature = "parallel"))]
    println!("Warning: Parallel feature not enabled");
    
    #[cfg(not(target_feature = "avx2"))]
    println!("Warning: AVX2 not available, SIMD acceleration limited");
    
    #[cfg(target_os = "windows")]
    println!("Note: Some features may have limited support on Windows");
}
```

## 🔍 Debugging Techniques

### Enable Debug Logging

```rust
// Enable detailed logging
env_logger::init();

// Use debug builds for better error messages
let options = ODEOptions::new()
    .debug_mode(true)
    .verbose_errors(true);
```

### Step-by-Step Debugging

```rust
fn debug_ode_step_by_step<F>(
    rhs: F,
    t_span: [f64; 2],
    y0: ndarray::Array1<f64>
) where F: Fn(f64, ndarray::ArrayView1<f64>) -> ndarray::Array1<f64>
{
    let mut t = t_span[0];
    let mut y = y0;
    let dt = 0.01; // Small fixed step
    
    while t < t_span[1] {
        let dy_dt = rhs(t, y.view());
        println!("t={:.3}, y={:?}, dy/dt={:?}", t, y, dy_dt);
        
        // Check for problems
        if dy_dt.iter().any(|&x| !x.is_finite()) {
            println!("ERROR: Non-finite derivative at t={}", t);
            break;
        }
        
        // Simple Euler step
        y = &y + &(&dy_dt * dt);
        t += dt;
        
        if t > t_span[0] + 0.1 { break; } // Limit debug output
    }
}
```

### Automated Problem Detection

```rust
use scirs2_integrate::verification::*;

fn comprehensive_problem_analysis<F>(
    rhs: F,
    t_span: [f64; 2],
    y0: ndarray::Array1<f64>
) -> ProblemAnalysis
where F: Fn(f64, ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> + Clone
{
    let mut analysis = ProblemAnalysis::new();
    
    // Test stiffness
    analysis.stiffness = detect_stiffness(rhs.clone(), y0.view(), t_span);
    
    // Test smoothness
    analysis.smoothness = estimate_smoothness(rhs.clone(), y0.view(), t_span);
    
    // Test scaling
    analysis.scaling_issues = detect_scaling_problems(rhs.clone(), y0.view());
    
    // Provide recommendations
    analysis.recommendations = generate_recommendations(&analysis);
    
    analysis
}
```

## 📞 Getting Help

### Information to Include in Bug Reports

```rust
fn generate_bug_report() {
    println!("=== Bug Report Information ===");
    println!("scirs2-integrate version: {}", env!("CARGO_PKG_VERSION"));
    println!("Rust version: {}", env!("RUSTC_VERSION"));
    println!("Target: {}", env!("TARGET"));
    println!("Features enabled: {:?}", get_enabled_features());
    println!("Platform: {} {}", env::consts::OS, env::consts::ARCH);
    
    // Include problem characteristics
    println!("Problem type: {:?}", your_problem_type);
    println!("System size: {}", system_dimension);
    println!("Time span: {:?}", time_span);
    println!("Tolerances: rtol={:.0e}, atol={:.0e}", rtol, atol);
}
```

### Community Resources

- **GitHub Issues**: [Report bugs and request features](https://github.com/cool-japan/scirs/issues)
- **Documentation**: Comprehensive API docs and examples
- **Benchmarks**: Compare performance with SciPy
- **Examples**: 100+ working examples in the `examples/` directory

### Self-Help Checklist

Before reporting an issue:

- [ ] Check this troubleshooting guide
- [ ] Review the method selection guide
- [ ] Try the quick diagnostic function
- [ ] Test with simpler problem first
- [ ] Check if issue reproduces with minimal example
- [ ] Verify all required features are enabled
- [ ] Test with latest version

Remember: **Most integration problems are due to incorrect method selection or inappropriate tolerances rather than bugs in the library.**