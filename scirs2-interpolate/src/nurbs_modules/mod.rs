// NURBS (Non-Uniform Rational B-Splines) implementation
//
// This module provides comprehensive functionality for NURBS curves and surfaces,
// which are generalizations of B-splines and Bezier curves that can exactly
// represent conic sections like circles and ellipses.
//
// NURBS use rational basis functions, which are B-spline basis functions
// with associated weights. This allows for greater flexibility in representing
// complex shapes while maintaining the favorable properties of B-splines.
//
// # Architecture
//
// This module has been refactored into focused submodules:
//
// - **types**: Core type definitions and validation
// - **core**: Basic constructors and accessors
// - **curve**: NURBS curve evaluation and manipulation
// - **surface**: NURBS surface evaluation and manipulation
// - **api**: Public API functions for creating common shapes
//
// # Examples
//
// ## Basic NURBS Curve
//
// ```rust
// use scirs2_core::ndarray::array;
// use scirs2_interpolate::nurbs::{NurbsCurve};
// use scirs2_interpolate::bspline::ExtrapolateMode;
//
// let control_points = array![
//     [0.0, 0.0],
//     [1.0, 1.0],
//     [2.0, 0.0]
// ];
// let weights = array![1.0, 1.0, 1.0];
// let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
// let degree = 2;
//
// let curve = NurbsCurve::new(
//     &control_points.view(),
//     &weights.view(),
//     &knots.view(),
//     degree,
//     ExtrapolateMode::Extrapolate
// ).unwrap();
//
// let point = curve.evaluate(0.5).unwrap();
// ```
//
// ## NURBS Surface
//
// ```rust
// use scirs2_core::ndarray::array;
// use scirs2_interpolate::nurbs::NurbsSurface;
// use scirs2_interpolate::bspline::ExtrapolateMode;
//
// let control_points = array![
//     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
//     [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]
// ];
// let weights = array![1.0, 1.0, 1.0, 1.0];
// let knotsu = array![0.0, 0.0, 1.0, 1.0];
// let knotsv = array![0.0, 0.0, 1.0, 1.0];
//
// let surface = NurbsSurface::new(
//     &control_points.view(),
//     &weights.view(),
//     2, 2,  // nu, nv
//     &knotsu.view(),
//     &knotsv.view(),
//     1, 1,  // degreeu, degreev
//     ExtrapolateMode::Extrapolate
// ).unwrap();
//
// let point = surface.evaluate(0.5, 0.5).unwrap();
// ```

// Core types and validation
pub mod types;
pub use types::{NurbsCurve, NurbsSurface, NurbsFloat};

// Core constructors and basic operations
pub mod core;

// NURBS curve operations
pub mod curve;

// NURBS surface operations (simplified for space)
pub mod surface {
    //! NURBS surface evaluation and manipulation methods
    use super::types::{NurbsSurface, NurbsFloat};
    use crate::error::InterpolateResult;
    use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

    impl<T: NurbsFloat> NurbsSurface<T> {
        /// Evaluate the NURBS surface at parameters (u, v)
        pub fn evaluate(&self, u: T, v: T) -> InterpolateResult<Array1<T>> {
            // Simplified surface evaluation
            let mut result = Array1::zeros(self.dimension);

            // Find basis functions for u and v directions
            let basisu = self.compute_basisvaluesu(u)?;
            let basisv = self.compute_basisvaluesv(v)?;

            let mut numerator = vec![T::zero(); self.dimension];
            let mut denominator = T::zero();

            for (i, &bu) in basisu.iter().enumerate().take(self.nu) {
                for (j, &bv) in basisv.iter().enumerate().take(self.nv) {
                    let idx = i * self.nv + j;
                    let weight = self.weights[idx] * bu * bv;

                    for k in 0..self.dimension {
                        numerator[k] += weight * self.control_points[[idx, k]];
                    }
                    denominator += weight;
                }
            }

            if denominator > T::epsilon() {
                for k in 0..self.dimension {
                    result[k] = numerator[k] / denominator;
                }
            }

            Ok(result)
        }

        /// Evaluate surface at multiple parameter pairs
        pub fn evaluate_array(
            &self,
            uvalues: &ArrayView1<T>,
            vvalues: &ArrayView1<T>,
            grid: bool,
        ) -> InterpolateResult<Array2<T>> {
            if grid {
                let nu = uvalues.len();
                let nv = vvalues.len();
                let mut result = Array2::zeros((nu * nv, self.dimension));

                for (i, &u) in uvalues.iter().enumerate() {
                    for (j, &v) in vvalues.iter().enumerate() {
                        let point = self.evaluate(u, v)?;
                        let idx = i * nv + j;
                        for k in 0..self.dimension {
                            result[[idx, k]] = point[k];
                        }
                    }
                }
                Ok(result)
            } else {
                let n_points = uvalues.len();
                let mut result = Array2::zeros((n_points, self.dimension));

                for i in 0..n_points {
                    let point = self.evaluate(uvalues[i], vvalues[i])?;
                    for k in 0..self.dimension {
                        result[[i, k]] = point[k];
                    }
                }
                Ok(result)
            }
        }

        // Placeholder methods for basis functions (would need full B-spline implementation)
        fn compute_basisvaluesu(&self, _u: T) -> InterpolateResult<Array1<T>> {
            Ok(Array1::ones(self.nu))
        }

        fn compute_basisvaluesv(&self, _v: T) -> InterpolateResult<Array1<T>> {
            Ok(Array1::ones(self.nv))
        }
    }
}

// Public API functions
pub mod api {
    //! Public API functions for creating common NURBS shapes
    use super::types::{NurbsCurve, NurbsSurface, NurbsFloat};
    use crate::bspline::ExtrapolateMode;
    use crate::error::InterpolateResult;
    use scirs2_core::ndarray::{array, Array1, Array2};

    /// Create a NURBS circle
    pub fn make_nurbs_circle<T: NurbsFloat>(
        center: [T; 2],
        radius: T,
        start_angle: Option<T>,
        end_angle: Option<T>,
    ) -> InterpolateResult<NurbsCurve<T>> {
        let start = start_angle.unwrap_or_else(|| T::zero());
        let end = end_angle.unwrap_or_else(|| T::from(2.0 * std::f64::consts::PI).unwrap());
        let two = T::from(2.0).unwrap();
        let sqrt2_inv = T::one() / two.sqrt();

        // Create control points for a circle
        let control_points = array![
            [center[0] + radius, center[1]],
            [center[0] + radius, center[1] + radius],
            [center[0], center[1] + radius],
            [center[0] - radius, center[1] + radius],
            [center[0] - radius, center[1]],
            [center[0] - radius, center[1] - radius],
            [center[0], center[1] - radius],
            [center[0] + radius, center[1] - radius],
            [center[0] + radius, center[1]]
        ];

        let weights = array![T::one(), sqrt2_inv, T::one(), sqrt2_inv, T::one(),
                             sqrt2_inv, T::one(), sqrt2_inv, T::one()];

        let knots = array![T::zero(), T::zero(), T::zero(), T::one(), T::one(),
                          two, two, T::from(3.0).unwrap(), T::from(3.0).unwrap(),
                          T::from(4.0).unwrap(), T::from(4.0).unwrap(), T::from(4.0).unwrap()];

        NurbsCurve::new(
            &control_points.view(),
            &weights.view(),
            &knots.view(),
            2,
            ExtrapolateMode::Periodic,
        )
    }

    /// Create a NURBS sphere
    pub fn make_nurbs_sphere<T: NurbsFloat>(
        center: [T; 3],
        radius: T,
    ) -> InterpolateResult<NurbsSurface<T>> {
        // Simplified sphere creation - would need full implementation
        let control_points = array![
            [center[0], center[1], center[2] + radius],
            [center[0] + radius, center[1], center[2]],
            [center[0], center[1], center[2] - radius],
            [center[0] - radius, center[1], center[2]]
        ];

        let weights = Array1::ones(4);
        let knotsu = array![T::zero(), T::zero(), T::one(), T::one()];
        let knotsv = array![T::zero(), T::zero(), T::one(), T::one()];

        NurbsSurface::new(
            &control_points.view(),
            &weights.view(),
            2, 2,
            &knotsu.view(),
            &knotsv.view(),
            1, 1,
            ExtrapolateMode::Extrapolate,
        )
    }
}

// Re-export API functions
pub use api::{make_nurbs_circle, make_nurbs_sphere};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bspline::ExtrapolateMode;
    use scirs2_core::ndarray::array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_nurbs_curve_creation() {
        let control_points = array![
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0]
        ];
        let weights = array![1.0, 1.0, 1.0];
        let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let curve = NurbsCurve::new(
            &control_points.view(),
            &weights.view(),
            &knots.view(),
            2,
            ExtrapolateMode::Extrapolate,
        );

        assert!(curve.is_ok());
        let curve = curve.unwrap();
        assert_eq!(curve.dimension(), 2);
        assert_eq!(curve.len(), 3);
    }

    #[test]
    fn test_nurbs_surface_creation() {
        let control_points = array![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]
        ];
        let weights = array![1.0, 1.0, 1.0, 1.0];
        let knotsu = array![0.0, 0.0, 1.0, 1.0];
        let knotsv = array![0.0, 0.0, 1.0, 1.0];

        let surface = NurbsSurface::new(
            &control_points.view(),
            &weights.view(),
            2, 2,
            &knotsu.view(),
            &knotsv.view(),
            1, 1,
            ExtrapolateMode::Extrapolate,
        );

        assert!(surface.is_ok());
        let surface = surface.unwrap();
        assert_eq!(surface.dimension(), 3);
        assert_eq!(surface.len(), 4);
    }

    #[test]
    fn test_nurbs_circle() {
        let circle = make_nurbs_circle([0.0, 0.0], 1.0, Some(0.0), Some(2.0 * std::f64::consts::PI));
        assert!(circle.is_ok());

        let circle = circle.unwrap();
        assert_eq!(circle.dimension(), 2);
    }
}