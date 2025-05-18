# Least Squares Module

This module provides a comprehensive suite of least squares optimization algorithms for different problem types.

## Available Methods

### Standard Least Squares
- `least_squares()`: Basic nonlinear least squares solver
  - Levenberg-Marquardt algorithm
  - Trust Region Reflective algorithm
  - Suitable for well-conditioned problems with clean data

### Robust Least Squares
- `robust_least_squares()`: M-estimators for outlier-resistant regression
  - **Huber Loss**: Quadratic for small residuals, linear for large ones
  - **Bisquare (Tukey) Loss**: Completely rejects extreme outliers
  - **Cauchy Loss**: Very strong outlier protection with slowly decreasing influence
  - Uses Iteratively Reweighted Least Squares (IRLS) algorithm

### Weighted Least Squares
- `weighted_least_squares()`: Handles heteroscedastic data (varying variance)
  - Allows different importance weights for each data point
  - Essential when measurement errors vary across observations
  - Uses modified Gauss-Newton algorithm

### Bounded Least Squares
- `bounded_least_squares()`: Box constraints on parameters
  - Enforces upper and lower bounds on variables
  - Trust region reflective algorithm adapted for bounds
  - Useful when parameters have physical constraints

### Separable Least Squares
- `separable_least_squares()`: Variable projection for partially linear models
  - Optimizes models of the form: f(x, α, β) = Σ αᵢ φᵢ(x, β)
  - Reduces dimensionality by eliminating linear parameters
  - Uses VARPRO algorithm for efficient optimization
  - Useful for exponential models, Gaussian mixtures, etc.

### Total Least Squares
- `total_least_squares()`: Errors-in-variables regression
  - Handles measurement errors in both x and y variables
  - Minimizes orthogonal distances to the fitted line
  - Supports weighted TLS with known error variances
  - Useful for instrument calibration and measurement analysis

## When to Use Each Method

1. **Standard Least Squares**: 
   - Data is clean (no outliers)
   - Measurement errors have constant variance
   - No constraints on parameters

2. **Robust Least Squares**:
   - Data contains outliers
   - Need to minimize influence of anomalous measurements
   - Choose loss function based on outlier severity:
     - Huber: Moderate outliers
     - Bisquare: Strong outliers
     - Cauchy: Very strong outliers

3. **Weighted Least Squares**:
   - Measurement errors vary across observations
   - Some data points are more reliable than others
   - Known variance structure in the data

4. **Bounded Least Squares**:
   - Parameters must stay within physical limits
   - Need to enforce box constraints
   - Regularization through bounds

5. **Separable Least Squares**:
   - Model is linear in some parameters
   - Want to reduce optimization complexity
   - Common in curve fitting (exponentials, Gaussians)

6. **Total Least Squares**:
   - Both x and y have measurement errors
   - Need to minimize perpendicular distances
   - Instrument calibration problems
   - Error variances are known (for weighted version)

## Example Usage

```rust
use ndarray::{array, Array1};
use scirs2_optimize::least_squares::{
    least_squares, robust_least_squares, weighted_least_squares, bounded_least_squares,
    Method, HuberLoss
};

// Define residual function
fn residual(params: &[f64], data: &[f64]) -> Array1<f64> {
    // Your model here
    todo!()
}

// Standard least squares
let result = least_squares(residual, &x0, Method::LevenbergMarquardt, None, &data, None)?;

// Robust least squares with Huber loss
let loss = HuberLoss::new(1.0);
let result = robust_least_squares(residual, &x0, loss, None, &data, None)?;

// Weighted least squares
let weights = array![1.0, 0.5, 2.0]; // Different weights for each residual
let result = weighted_least_squares(residual, &x0, &weights, None, &data, None)?;

// Bounded least squares
let bounds = Bounds::new(&[(Some(0.0), None), (None, Some(1.0))]);
let result = bounded_least_squares(residual, &x0, Some(bounds), None, &data, None)?;
```

## Implementation Details

All methods support:
- Optional analytical Jacobian (uses finite differences if not provided)
- Configurable convergence criteria
- Maximum iteration limits
- Function evaluation counting

The algorithms are implemented in pure Rust for:
- Memory safety
- Performance
- Cross-platform compatibility