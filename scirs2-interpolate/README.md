# SciRS2 Interpolation Module

[![crates.io](https://img.shields.io/crates/v/scirs2-interpolate.svg)](https://crates.io/crates/scirs2-interpolate)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-interpolate)](https://docs.rs/scirs2-interpolate)

The `scirs2-interpolate` crate provides a comprehensive set of interpolation methods for scientific computing in Rust. It aims to provide functionality similar to SciPy's interpolation module while leveraging Rust's performance and safety features.

## Features

- **1D Interpolation**
  - Linear interpolation
  - Nearest neighbor interpolation
  - Cubic interpolation

- **Spline Interpolation**
  - Natural cubic splines
  - Not-a-knot cubic splines
  - Akima splines (robust to outliers)
  - PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for shape preservation

- **Multi-dimensional Interpolation**
  - Regular grid interpolation
  - Scattered data interpolation
  - Map coordinates (similar to SciPy's `map_coordinates`)

- **Advanced Interpolation Methods**
  - Radial Basis Function (RBF) interpolation with multiple kernel types
  - Kriging (Gaussian process regression) with uncertainty quantification
  - Barycentric interpolation for stable polynomial fitting

- **Grid Transformation and Resampling**
  - Resample scattered data onto regular grids
  - Convert between grids of different resolutions
  - Map grid data to arbitrary points

- **Tensor Product Interpolation**
  - Efficient high-dimensional interpolation on structured grids
  - Higher-order interpolation using Lagrange polynomials

- **Utility Functions**
  - Error estimation for interpolation methods
  - Parameter optimization
  - Differentiation of interpolated functions
  - Integration of interpolated functions

## Usage Examples

### 1D Interpolation

```rust
use ndarray::array;
use scirs2_interpolate::{Interp1d, InterpolationMethod};

// Create sample data
let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

// Create a 1D interpolator with linear interpolation
let interp = Interp1d::new(&x.view(), &y.view(), InterpolationMethod::Linear).unwrap();

// Evaluate at a specific point
let y_interp = interp.evaluate(2.5).unwrap();
println!("Interpolated value at x=2.5: {}", y_interp);

// Evaluate at multiple points
let x_new = array![1.5, 2.5, 3.5];
let y_new = interp.evaluate_array(&x_new.view()).unwrap();
println!("Interpolated values: {:?}", y_new);
```

### Cubic Spline Interpolation

```rust
use ndarray::array;
use scirs2_interpolate::{CubicSpline, make_interp_spline};

// Create sample data
let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

// Create a cubic spline
let spline = make_interp_spline(&x.view(), &y.view()).unwrap();

// Evaluate at a specific point
let y_interp = spline.evaluate(2.5).unwrap();
println!("Spline interpolated value at x=2.5: {}", y_interp);

// Compute the derivative
let y_prime = spline.derivative(2.5).unwrap();
println!("Spline derivative at x=2.5: {}", y_prime);
```

### Multidimensional Regular Grid Interpolation

```rust
use ndarray::{array, Array2};
use scirs2_interpolate::{make_interp_nd, InterpolationMethod};

// Create sample grid coordinates
let x = array![0.0, 1.0, 2.0];
let y = array![0.0, 1.0, 2.0];

// Create 2D grid values (z = x^2 + y^2)
let mut grid_values = Array2::zeros((3, 3));
for i in 0..3 {
    for j in 0..3 {
        grid_values[[i, j]] = x[i].powi(2) + y[j].powi(2);
    }
}

// Create interpolator
let interp = make_interp_nd(
    &[x, y],
    &grid_values.view(),
    InterpolationMethod::Linear
).unwrap();

// Points to evaluate
let points = Array2::from_shape_vec((2, 2), vec![
    0.5, 0.5,
    1.5, 1.5
]).unwrap();

// Interpolate
let results = interp.evaluate(&points.view()).unwrap();
println!("Interpolated values: {:?}", results);
```

### Radial Basis Function Interpolation

```rust
use ndarray::{array, Array2};
use scirs2_interpolate::{RBFInterpolator, RBFKernel};

// Create scattered data points
let points = Array2::from_shape_vec((5, 2), vec![
    0.0, 0.0, 
    1.0, 0.0, 
    0.0, 1.0, 
    1.0, 1.0, 
    0.5, 0.5
]).unwrap();

// Values at those points (z = x^2 + y^2)
let values = array![0.0, 1.0, 1.0, 2.0, 0.5];

// Create RBF interpolator with Gaussian kernel
let interp = RBFInterpolator::new(
    &points.view(),
    &values.view(),
    RBFKernel::Gaussian,
    1.0  // epsilon parameter
).unwrap();

// Interpolate at new points
let test_points = Array2::from_shape_vec((2, 2), vec![
    0.25, 0.25,
    0.75, 0.75
]).unwrap();

let results = interp.interpolate(&test_points.view()).unwrap();
println!("RBF interpolated values: {:?}", results);
```

### Kriging Interpolation with Uncertainty

```rust
use ndarray::{array, Array2};
use scirs2_interpolate::{KrigingInterpolator, CovarianceFunction};

// Create scattered data points
let points = Array2::from_shape_vec((5, 2), vec![
    0.0, 0.0, 
    1.0, 0.0, 
    0.0, 1.0, 
    1.0, 1.0, 
    0.5, 0.5
]).unwrap();

// Values at those points (z = x^2 + y^2)
let values = array![0.0, 1.0, 1.0, 2.0, 0.5];

// Create Kriging interpolator
let interp = KrigingInterpolator::new(
    &points.view(),
    &values.view(),
    CovarianceFunction::SquaredExponential,
    1.0,  // signal variance
    0.5,  // length scale
    1e-10, // nugget
    1.0   // alpha (for RationalQuadratic)
).unwrap();

// Interpolate with uncertainty estimates
let test_points = Array2::from_shape_vec((1, 2), vec![0.25, 0.25]).unwrap();
let result = interp.predict(&test_points.view()).unwrap();

println!("Kriging prediction at (0.25, 0.25): {}", result.value[0]);
println!("Prediction uncertainty: {}", result.variance[0]);
```

### Grid Resampling

```rust
use ndarray::{array, Array2};
use scirs2_interpolate::{resample_to_grid, GridTransformMethod};

// Create scattered data points
let points = Array2::from_shape_vec((5, 2), vec![
    0.0, 0.0, 
    1.0, 0.0, 
    0.0, 1.0, 
    1.0, 1.0, 
    0.5, 0.5
]).unwrap();

// Values at those points (z = x^2 + y^2)
let values = array![0.0, 1.0, 1.0, 2.0, 0.5];

// Resample to a 10x10 grid
let (grid_coords, grid_values) = resample_to_grid(
    &points.view(),
    &values.view(),
    &[10, 10],
    &[(0.0, 1.0), (0.0, 1.0)],
    GridTransformMethod::Linear,
    0.0
).unwrap();

println!("Resampled grid size: {:?}", grid_values.shape());
```

## Advanced Features

### Akima Spline (Robust to Outliers)

```rust
use ndarray::array;
use scirs2_interpolate::{AkimaSpline, make_akima_spline};

// Data with an outlier at x=3
let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let y = array![0.0, 1.0, 4.0, 20.0, 16.0, 25.0];

// Create Akima spline which handles outliers better than cubic spline
let spline = make_akima_spline(&x.view(), &y.view()).unwrap();

// Evaluate at some test points
for x_val in [1.5, 2.5, 3.5, 4.5].iter() {
    println!("Akima spline at x={}: {}", x_val, spline.evaluate(*x_val).unwrap());
}
```

### PCHIP Interpolation (Shape Preserving)

```rust
use ndarray::array;
use scirs2_interpolate::{pchip_interpolate, PchipInterpolator};

// Monotonically increasing data
let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

// Using the convenience function
let x_new = array![0.5, 1.5, 2.5, 3.5];
let y_interp = pchip_interpolate(&x.view(), &y.view(), &x_new.view()).unwrap();
println!("PCHIP interpolated values: {:?}", y_interp);

// Or create an interpolator object for more control
let interp = PchipInterpolator::new(&x.view(), &y.view()).unwrap();
let y_at_point = interp.evaluate(2.5).unwrap();
println!("PCHIP value at x=2.5: {}", y_at_point);

// PCHIP preserves monotonicity of the data, unlike cubic spline which may introduce oscillations
```

### Tensor Product Interpolation

```rust
use ndarray::{array, Array2};
use scirs2_interpolate::{tensor_product_interpolate, InterpolationMethod};

// Create coordinates for each dimension
let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
let y = array![0.0, 1.0, 2.0, 3.0];

// Create values on the grid (z = sin(x) * cos(y))
let mut values = ndarray::ArrayD::zeros(vec![5, 4]);
for i in 0..5 {
    for j in 0..4 {
        values[[i, j]] = (x[i]).sin() * (y[j]).cos();
    }
}

// Points to interpolate
let points = Array2::from_shape_vec((2, 2), vec![
    2.5, 1.5,
    1.5, 2.5
]).unwrap();

// Interpolate using tensor product method
let results = tensor_product_interpolate(
    &[x, y],
    &values,
    &points.view(),
    InterpolationMethod::Linear
).unwrap();

println!("Tensor product interpolation results: {:?}", results);
```

### Error Estimation and Differentiation

```rust
use ndarray::array;
use scirs2_interpolate::{utils, interp1d::linear_interpolate};

// Create sample data
let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

// Estimate error of linear interpolation
let error = utils::error_estimate(&x.view(), &y.view(), linear_interpolate).unwrap();
println!("Linear interpolation error estimate: {}", error);

// Create a function that evaluates the interpolation
let eval_fn = |x_val| {
    linear_interpolate(&x.view(), &y.view(), &array![x_val].view())
        .map(|arr| arr[0])
};

// Compute the derivative at x=2.5
let derivative = utils::differentiate(2.5, 0.001, eval_fn).unwrap();
println!("Numerical derivative at x=2.5: {}", derivative);

// Compute the integral from x=1 to x=3
let integral = utils::integrate(1.0, 3.0, 100, eval_fn).unwrap();
println!("Numerical integral from x=1 to x=3: {}", integral);
```

## Error Handling

The module uses the `InterpolateResult` and `InterpolateError` types for error handling:

```rust
pub enum InterpolateError {
    /// Computation error (generic error)
    ComputationError(String),

    /// Domain error (input outside valid domain)
    DomainError(String),

    /// Value error (invalid value)
    ValueError(String),

    /// Not implemented error
    NotImplementedError(String),
}

pub type InterpolateResult<T> = Result<T, InterpolateError>;
```

## Performance Considerations

The module is designed with performance in mind:

- Efficient memory usage with `ndarray`
- Specialized algorithms for different interpolation needs
- Carefully optimized numerical computations
- Future support for parallel processing with Rayon

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
