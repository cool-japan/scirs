# API Design Guide for SciRS2 Interpolate

This document defines consistent API patterns for the scirs2-interpolate library.

## Constructor Patterns

### 1. Simple Types (0-3 parameters)
Use direct `new()` constructors:
```rust
impl<T> SimpleType<T> {
    pub fn new(param1: Type1, param2: Type2) -> InterpolateResult<Self>
}
```

### 2. Complex Types (4+ parameters or many optional parameters)
Use builder pattern:
```rust
impl<T> ComplexTypeBuilder<T> {
    pub fn new() -> Self
    pub fn with_param1(mut self, value: Type1) -> Self
    pub fn with_param2(mut self, value: Type2) -> Self
    pub fn build(self) -> InterpolateResult<ComplexType<T>>
}
```

### 3. Convenience Functions
Use `make_*` functions for common use cases:
```rust
pub fn make_interpolator_name<T>(
    common_params: CommonType,
) -> InterpolateResult<InterpolatorType<T>>
```

## Naming Conventions

### Types
- Use descriptive names: `LinearInterpolator`, `CubicSpline`, `RBFInterpolator`
- Avoid abbreviations except for well-known terms: `RBF`, `GPU`, `SIMD`
- Use consistent suffixes:
  - `Interpolator` for objects that interpolate
  - `Spline` for spline-based methods
  - `Builder` for builder pattern
  - `Config` for configuration objects

### Functions
- Use verb-based names: `evaluate`, `interpolate`, `construct`, `optimize`
- Use consistent prefixes:
  - `make_*` for convenience constructors
  - `build_*` for complex construction (prefer builders)
  - `compute_*` for calculation functions
  - `validate_*` for validation functions

### Parameters
- Use consistent parameter ordering:
  1. Data (points, values)
  2. Method configuration (kernel, degree, etc.)
  3. Optional parameters (tolerance, iterations, etc.)

## Error Handling

### Error Types
Use specific error types:
```rust
#[derive(Error, Debug)]
pub enum InterpolateError {
    #[error("Invalid input data: {message}")]
    InvalidInput { message: String },
    
    #[error("Domain error: point {point:?} is outside domain [{min}, {max}]")]
    OutOfDomain { point: f64, min: f64, max: f64 },
    
    #[error("Computation failed: {reason}")]
    ComputationFailed { reason: String },
    
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },
}
```

### Error Messages
- Be specific: "smoothing parameter must be positive, got -0.5"
- Include context: "in RBF interpolation with Gaussian kernel"
- Suggest fixes: "try increasing epsilon or using a different kernel"

## Documentation Standards

### Function Documentation
```rust
/// Brief one-line description.
///
/// Longer description explaining the mathematical method,
/// when to use it, and any important considerations.
///
/// # Arguments
///
/// * `points` - Input data points as (n_points, n_dims) array
/// * `values` - Function values at input points
/// * `kernel` - RBF kernel type (Gaussian, Multiquadric, etc.)
/// * `epsilon` - Shape parameter (larger = smoother, smaller = more accurate)
///
/// # Returns
///
/// An interpolator that can evaluate the function at new points.
///
/// # Errors
///
/// * `InvalidInput` - If points and values have incompatible shapes
/// * `ComputationFailed` - If the interpolation matrix is singular
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_interpolate::advanced::rbf::{RBFInterpolator, RBFKernel};
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let values = array![0.0, 1.0, 1.0];
/// 
/// let interp = RBFInterpolator::new(&points.view(), &values.view(), 
///                                   RBFKernel::Gaussian, 1.0)?;
/// 
/// let result = interp.interpolate(&array![[0.5, 0.5]].view())?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Performance
///
/// - Time complexity: O(n³) for construction, O(n) for evaluation
/// - Memory complexity: O(n²)
/// - For large datasets (n > 1000), consider FastRBFInterpolator
///
/// # Mathematical Details
///
/// Uses radial basis functions of the form:
/// f(x) = Σᵢ wᵢ φ(||x - xᵢ||)
/// 
/// where φ is the chosen kernel function and weights wᵢ are computed
/// by solving the linear system Aw = y.
pub fn new(...) -> InterpolateResult<Self>
```

## Implementation Standards

### Method Signatures
```rust
// Evaluation methods - consistent naming and signatures
impl Interpolator<T> {
    // Single point evaluation
    pub fn evaluate(&self, point: &ArrayView1<T>) -> InterpolateResult<T>
    
    // Multiple point evaluation  
    pub fn evaluate_batch(&self, points: &ArrayView2<T>) -> InterpolateResult<Array1<T>>
    
    // With derivatives (if supported)
    pub fn evaluate_with_derivatives(&self, point: &ArrayView1<T>) -> 
        InterpolateResult<(T, Array1<T>)>
}
```

### Configuration Objects
```rust
#[derive(Debug, Clone)]
pub struct InterpolatorConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub parallel: bool,
    pub cache_size: Option<usize>,
}

impl Default for InterpolatorConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 1000,
            parallel: true,
            cache_size: Some(1024),
        }
    }
}
```

This guide ensures consistency across the entire library while providing flexibility for different use cases.