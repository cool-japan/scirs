# scirs2-interpolate

[![crates.io](https://img.shields.io/crates/v/scirs2-interpolate.svg)](https://crates.io/crates/scirs2-interpolate)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-interpolate)](https://docs.rs/scirs2-interpolate)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()

**Advanced interpolation and approximation for the SciRS2 scientific computing library (v0.6.1).**

`scirs2-interpolate` provides comprehensive interpolation methods for 1D, 2D, and N-dimensional data. It covers standard spline families (cubic, Akima, PCHIP, B-splines, NURBS), scattered-data methods (RBF, Kriging, Moving Least Squares, Natural Neighbor, Barycentric Rational, Thin-Plate Splines, Shepard's method), and advanced features including adaptive error-controlled refinement, meshless methods, spherical harmonic interpolation, and B-spline surface fitting — all as pure Rust.

Tested: 1143/1143 tests passing with default features (13 skipped), 1173/1173 with `--all-features` (13 skipped) (verified 2026-07-15).

## Features (v0.6.1)

### 1D Interpolation
- **Linear / nearest-neighbor**: Basic 1D interpolation with boundary handling
- **Cubic spline**: Natural, not-a-knot, clamped, and periodic boundary conditions
- **Akima spline**: Outlier-robust local spline; resists oscillation from rogue data points
- **PCHIP**: Piecewise Cubic Hermite Interpolating Polynomial; shape- and monotonicity-preserving
- **B-splines**: Arbitrary-order B-spline basis with de Boor evaluation; knot insertion and removal
- **NURBS**: Non-Uniform Rational B-Splines for exact conic sections and free-form curves
- **Bezier curves**: Rational and polynomial Bezier; de Casteljau evaluation
- **Tension splines**: Splines under tension for feature-preserving interpolation
- **Penalized splines (P-splines)**: Regularized B-spline fitting for noisy data
- **Monotone splines**: Constrained splines preserving monotonicity
- **Hermite splines**: Specified derivative values at knots
- **Floater-Hormann barycentric rational**: Stable barycentric rational interpolation of arbitrary order

### Scattered Data Interpolation
- **RBF (Radial Basis Function)**: Multiquadric, thin-plate spline, Gaussian, inverse multiquadric, linear; parameter optimization included
- **Kriging**: Ordinary Kriging, Universal Kriging, Indicator Kriging; variogram fitting; Bayesian uncertainty quantification
- **Moving Least Squares (MLS)**: Weighted polynomial fitting for scattered point clouds
- **Natural Neighbor (Sibson)**: Voronoi-based area-steal interpolation; C1 continuity
- **Thin-Plate Spline (TPS)**: Global scattered-data interpolant; bending energy minimization
- **Shepard's method**: Inverse distance weighting; modified Shepard for reduced flat-spot artifacts
- **Scattered 2D interpolation**: Delaunay triangulation based linear/cubic interpolation

### Spherical and Parametric Interpolation
- **Spherical harmonic interpolation**: Expand scattered data on the sphere in terms of real spherical harmonics
- **Parametric curve interpolation**: Arc-length parameterized fitting of 2D/3D point sequences
- **Barycentric coordinates on manifolds**: Interpolation in generalized barycentric coordinates

### Multidimensional Grid Interpolation
- **Regular grid (N-D)**: `RegularGridInterpolator` for arbitrary-dimensional rectilinear grids; linear and cubic
- **Tensor product interpolation**: Kronecker product construction for separable grids
- **Bivariate splines**: Smoothing and interpolating splines on 2D rectangular grids
- **B-spline surface fitting**: NURBS surface interpolation of 3D point clouds

### Adaptive Interpolation
- **Error-controlled refinement**: Iterative subdivision until local error tolerance is met
- **Hierarchical adaptive interpolation**: Multi-level sparse-grid construction
- **Meshless methods**: Partition-of-unity and reproducing-kernel methods for complex domains

### Performance and Accuracy
- **SIMD-accelerated B-spline evaluation**: Vectorized de Boor algorithm (2-4x speedup)
- **Parallel interpolation**: Multi-threaded batch evaluation for large point clouds
- **Fast Kriging**: O(k^3) local Kriging; fixed-rank approximation; sparse tapering; HODLR
- **Spatial data structures**: K-d trees and ball trees for O(log n) neighbor queries
- **Cache-aware memory access**: Minimized cache misses in hot evaluation paths

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-interpolate = "0.6.1"
```

With optional performance features:

```toml
[dependencies]
scirs2-interpolate = { version = "0.6.1", features = ["simd", "linalg"] }
```

### Cubic spline interpolation

```rust
use scirs2_core::ndarray::array;
use scirs2_interpolate::spline::{CubicSpline, SplineBoundaryCondition};

let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
let y = array![0.0f64, 1.0, 4.0, 9.0, 16.0];

let spline = CubicSpline::new(
    x.view(),
    y.view(),
    SplineBoundaryCondition::Natural,
    SplineBoundaryCondition::Natural,
)?;

let y_interp = spline.evaluate(2.5)?;
println!("spline(2.5) = {}", y_interp);
```

### PCHIP (shape-preserving)

```rust
use scirs2_core::ndarray::array;
use scirs2_interpolate::pchip::{PchipInterpolator, pchip_interpolate};

let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
let y = array![0.0f64, 1.0, 4.0, 9.0, 16.0];

// Convenience function
let x_new = array![0.5f64, 1.5, 2.5, 3.5];
let y_new = pchip_interpolate(&x.view(), &y.view(), &x_new.view())?;

// Or create an object for repeated evaluation
let interp = PchipInterpolator::new(&x.view(), &y.view())?;
println!("PCHIP(2.5) = {}", interp.evaluate(2.5)?);
```

### RBF interpolation of scattered 2D data

```rust
use scirs2_core::ndarray::{array, Array2};
use scirs2_interpolate::rbf::{RBFInterpolator, RBFKernel};

// Scattered points (x,y) in 2D
let pts = Array2::from_shape_vec((5, 2), vec![
    0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,  0.5, 0.5,
])?;
let vals = array![0.0f64, 1.0, 1.0, 2.0, 0.5];

let interp = RBFInterpolator::new(&pts.view(), &vals.view(), RBFKernel::ThinPlateSpline, 0.0)?;

let query = Array2::from_shape_vec((1, 2), vec![0.25, 0.75])?;
let result = interp.interpolate(&query.view())?;
println!("RBF(0.25, 0.75) = {}", result[0]);
```

### Kriging with uncertainty

```rust
use scirs2_core::ndarray::{array, Array2};
use scirs2_interpolate::kriging::{OrdinaryKriging, CovarianceFunction};

let pts = Array2::from_shape_vec((5, 2), vec![
    0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,  0.5, 0.5,
])?;
let vals = array![0.0f64, 1.0, 1.0, 2.0, 0.5];

let kriging = OrdinaryKriging::fit(
    &pts.view(),
    &vals.view(),
    CovarianceFunction::SquaredExponential { variance: 1.0, length_scale: 0.5 },
)?;

let query = Array2::from_shape_vec((1, 2), vec![0.3, 0.7])?;
let (pred, variance) = kriging.predict(&query.view())?;
println!("Kriging pred = {}, std = {}", pred[0], variance[0].sqrt());
```

### Natural Neighbor interpolation

```rust
use scirs2_core::ndarray::{array, Array2};
use scirs2_interpolate::natural_neighbor::NaturalNeighborInterpolator;

let pts = Array2::from_shape_vec((6, 2), vec![
    0.0, 0.0,  2.0, 0.0,  1.0, 1.5,
    0.0, 3.0,  2.0, 3.0,  1.0, 1.0,
])?;
let vals = array![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];

let interp = NaturalNeighborInterpolator::new(&pts.view(), &vals.view())?;
println!("NN(1.0, 1.0) = {}", interp.evaluate(&[1.0, 1.0])?);
```

### Barycentric rational interpolation (Floater-Hormann)

```rust
use scirs2_core::ndarray::array;
use scirs2_interpolate::barycentric::{FloaterHormann, fh_interpolate};

let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
let y = array![0.0f64, 0.841, 0.909, 0.141, -0.757];

// d=3 blending parameter
let interp = FloaterHormann::new(&x.view(), &y.view(), 3)?;
println!("FH(1.5) = {}", interp.evaluate(1.5)?);
```

### Adaptive error-controlled refinement

```rust
use scirs2_core::ndarray::array;
use scirs2_interpolate::adaptive_interpolation::{AdaptiveInterpolator, AdaptiveConfig};

let cfg = AdaptiveConfig { tol: 1e-6, max_levels: 12, ..Default::default() };
let interp = AdaptiveInterpolator::build(|x: f64| x.sin(), 0.0, 2.0 * std::f64::consts::PI, cfg)?;
println!("adaptive sin(pi) = {}", interp.evaluate(std::f64::consts::PI)?);
```

## API Overview

| Module | Description |
|--------|-------------|
| `interp1d` | 1D interpolation: linear, nearest, cubic, PCHIP |
| `spline` | Cubic splines with multiple boundary conditions |
| `bspline` | B-spline basis and fitting |
| `bspline_curves` | B-spline curves: knot insertion, fitting |
| `bspline_surface` | B-spline surface fitting of 3D clouds |
| `nurbs` | NURBS curves and surfaces |
| `bezier` | Rational and polynomial Bezier |
| `pchip` | PCHIP shape-preserving interpolation |
| `barycentric` | Floater-Hormann barycentric rational interpolation |
| `rbf` | Radial Basis Function interpolation |
| `rbf_compact` | Compactly supported RBF kernels |
| `kriging` | Ordinary, Universal, Indicator Kriging |
| `mls` / `moving_least_squares` | Moving Least Squares |
| `natural_neighbor` | Natural Neighbor (Sibson) interpolation |
| `thin_plate_spline` | Thin-Plate Spline scattered data |
| `shepard` | Shepard inverse-distance weighting |
| `scattered_2d` | Delaunay-based 2D scattered interpolation |
| `spherical` | Spherical harmonic interpolation |
| `parametric` | Parametric arc-length curve interpolation |
| `tensor_product` | Tensor product N-D grid interpolation |
| `interpnd` | Regular grid N-D interpolation |
| `polynomial_interpolation` | Lagrange, Newton polynomial interpolation |
| `adaptive_interpolation` | Error-controlled adaptive refinement |
| `meshless` | Partition-of-unity and reproducing-kernel methods |
| `utils` | Differentiation, integration, error estimation helpers |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `simd` | SIMD-accelerated B-spline and distance computations (`scirs2-core/simd`) |
| `linalg` | Advanced linear algebra via OxiBLAS (pure Rust) |
| `parallel` | Multi-threaded batch evaluation via Rayon (`scirs2-core/parallel`) |
| `scipy-comparison` | SciPy comparison benchmarks, requires Python with scipy/numpy (`pyo3`) |

`default = []` — no features are enabled by default; core interpolation methods are always available regardless.

## Documentation

Full API documentation is available at [docs.rs/scirs2-interpolate](https://docs.rs/scirs2-interpolate).

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
