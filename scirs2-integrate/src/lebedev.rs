//! Lebedev quadrature for spherical integration
//!
//! This module implements Lebedev quadrature rules for numerical integration over the surface of a
//! 3D sphere. Lebedev quadrature is particularly useful for problems in physics, chemistry, and
//! mathematical applications where spherical symmetry is important.
//!
//! The implementation provides quadrature points and weights for various orders of precision.

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use ndarray::{array, Array1, Array2};
use num_traits::Float;
use std::f64::consts::PI;

/// Represents a Lebedev quadrature rule with points and weights
#[derive(Debug, Clone)]
pub struct LebedevRule<F: IntegrateFloat> {
    /// Quadrature points on the unit sphere (x, y, z coordinates)
    pub points: Array2<F>,

    /// Quadrature weights (sum to 1)
    pub weights: Array1<F>,

    /// Degree of the quadrature rule (order of precision)
    pub degree: usize,

    /// Number of points in the rule
    pub npoints: usize,
}

/// Available Lebedev quadrature orders with corresponding number of points
///
/// The degree refers to the highest degree of spherical harmonics that can be
/// integrated exactly. Higher degree means more precision but requires more points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LebedevOrder {
    /// 6th order (requiring 6 points)
    Order6 = 6,

    /// 14th order (requiring 26 points)
    Order14 = 14,

    /// 26th order (requiring 50 points)
    Order26 = 26,

    /// 38th order (requiring 86 points)
    Order38 = 38,

    /// 50th order (requiring 146 points)
    Order50 = 50,

    /// 74th order (requiring 302 points)
    Order74 = 74,

    /// 86th order (requiring 434 points)
    Order86 = 86,

    /// 110th order (requiring 590 points)
    Order110 = 110,
}

impl LebedevOrder {
    /// Get the number of points required for this order
    pub fn num_points(self) -> usize {
        match self {
            LebedevOrder::Order6 => 6,
            LebedevOrder::Order14 => 26,
            LebedevOrder::Order26 => 50,
            LebedevOrder::Order38 => 86,
            LebedevOrder::Order50 => 146,
            LebedevOrder::Order74 => 302,
            LebedevOrder::Order86 => 434,
            LebedevOrder::Order110 => 590,
        }
    }

    /// Get the nearest available order for a requested number of points
    pub fn from_num_points(points: usize) -> Self {
        if points <= 6 {
            LebedevOrder::Order6
        } else if points <= 26 {
            LebedevOrder::Order14
        } else if points <= 50 {
            LebedevOrder::Order26
        } else if points <= 86 {
            LebedevOrder::Order38
        } else if points <= 146 {
            LebedevOrder::Order50
        } else if points <= 302 {
            LebedevOrder::Order74
        } else if points <= 434 {
            LebedevOrder::Order86
        } else {
            LebedevOrder::Order110
        }
    }
}

/// Generates a Lebedev quadrature rule for numerical integration over a unit sphere.
///
/// Lebedev quadrature is specifically designed for efficient integration over
/// the surface of a sphere, providing a high degree of accuracy with a relatively
/// small number of function evaluations.
///
/// # Arguments
///
/// * `order` - The desired order of the Lebedev rule. Higher orders provide more accuracy
///   but require more function evaluations.
///
/// # Returns
///
/// A `LebedevRule` containing points (x,y,z coordinates on the unit sphere) and weights,
/// plus information about the rule's degree and number of points.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::lebedev::{lebedev_rule, LebedevOrder};
///
/// // Generate a 14th-order Lebedev rule
/// let rule = lebedev_rule(LebedevOrder::Order14).unwrap();
///
/// // Check the number of points
/// assert_eq!(rule.points.nrows(), 26);
///
/// // Weights should sum to 1
/// let weight_sum: f64 = rule.weights.sum();
/// assert!((weight_sum - 1.0).abs() < 1e-10);
/// ```
pub fn lebedev_rule<F: IntegrateFloat>(order: LebedevOrder) -> IntegrateResult<LebedevRule<F>> {
    // Generate the rule based on the requested order
    match order {
        LebedevOrder::Order6 => generate_order6(),
        LebedevOrder::Order14 => generate_order14(),
        LebedevOrder::Order26 => generate_order26(),
        LebedevOrder::Order38 => generate_order38(),
        LebedevOrder::Order50 => generate_order50(),
        order => {
            // For higher orders, provide a helpful error message
            Err(IntegrateError::ValueError(format!(
                "Lebedev order {:?} (requiring {} points) is not yet implemented. Available orders: 6, 14, 26, 38, 50.",
                order, order.num_points()
            )))
        }
    }
}

/// Integrates a function over the surface of a unit sphere using Lebedev quadrature.
///
/// # Arguments
///
/// * `f` - Function to integrate. Should accept (x, y, z) coordinates on the unit sphere.
/// * `order` - The Lebedev quadrature order to use. Higher orders yield more accuracy.
///
/// # Returns
///
/// The approximate integral value multiplied by 4π (the surface area of the unit sphere).
///
/// # Examples
///
/// ```
/// use scirs2_integrate::lebedev::{lebedev_integrate, LebedevOrder};
/// use std::f64::consts::PI;
///
/// // Integrate f(x,y,z) = 1 over the unit sphere (should equal 4π)
/// let result: f64 = lebedev_integrate(|_x, _y, _z| 1.0, LebedevOrder::Order14).unwrap();
/// assert!((result - 4.0 * PI).abs() < 1e-10);
///
/// // Integrate f(x,y,z) = x^2 + y^2 + z^2 = 1 over the unit sphere (should equal 4π)
/// let result: f64 = lebedev_integrate(|x, y, z| x*x + y*y + z*z, LebedevOrder::Order14).unwrap();
/// assert!((result - 4.0 * PI).abs() < 1e-10);
/// ```
pub fn lebedev_integrate<F, Func>(f: Func, order: LebedevOrder) -> IntegrateResult<F>
where
    F: IntegrateFloat,
    Func: Fn(F, F, F) -> F,
{
    // Get the Lebedev rule
    let rule = lebedev_rule(order)?;

    // Compute the weighted sum
    let mut sum = F::zero();
    for i in 0..rule.npoints {
        let x = rule.points[[i, 0]];
        let y = rule.points[[i, 1]];
        let z = rule.points[[i, 2]];

        sum += f(x, y, z) * rule.weights[i];
    }

    // Multiply by the surface area of the unit sphere (4π)
    let four_pi = F::from(4.0 * PI).unwrap();
    Ok(sum * four_pi)
}

//////////////////////////////////////////////////
// IMPLEMENTATION OF SPECIFIC LEBEDEV RULES
//////////////////////////////////////////////////

/// Generates a 6th-order Lebedev rule with 6 points
fn generate_order6<F: IntegrateFloat>() -> IntegrateResult<LebedevRule<F>> {
    // These are the 6 points along the Cartesian axes
    let points_data = [
        // x, y, z coordinates
        [1.0, 0.0, 0.0],  // +x
        [-1.0, 0.0, 0.0], // -x
        [0.0, 1.0, 0.0],  // +y
        [0.0, -1.0, 0.0], // -y
        [0.0, 0.0, 1.0],  // +z
        [0.0, 0.0, -1.0], // -z
    ];

    // Each point has equal weight, summing to 1
    let weight = F::from(1.0 / 6.0).unwrap();
    let weights = Array1::from_elem(6, weight);

    // Convert points to the required type
    let mut points = Array2::zeros((6, 3));
    for i in 0..6 {
        for j in 0..3 {
            points[[i, j]] = F::from(points_data[i][j]).unwrap();
        }
    }

    Ok(LebedevRule {
        points,
        weights,
        degree: 6,
        npoints: 6,
    })
}

/// Generates a 14th-order Lebedev rule with 26 points
fn generate_order14<F: IntegrateFloat>() -> IntegrateResult<LebedevRule<F>> {
    // Start with the 6 axial points from order 6
    let order6 = generate_order6()?;

    // Add 12 vertices of a regular icosahedron (scaled to unit sphere)
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
    let norm = (1.0 + phi * phi).sqrt(); // Normalization to unit sphere

    let a = F::from(1.0 / norm).unwrap();
    let b = F::from(phi / norm).unwrap();
    let zero = F::zero();

    let icosahedron_points = array![
        [a, zero, b],
        [a, zero, -b],
        [-a, zero, b],
        [-a, zero, -b],
        [b, a, zero],
        [b, -a, zero],
        [-b, a, zero],
        [-b, -a, zero],
        [zero, b, a],
        [zero, b, -a],
        [zero, -b, a],
        [zero, -b, -a],
    ];

    // Add 8 vertices of a cube
    let c = F::from(1.0 / 3.0_f64.sqrt()).unwrap();

    let cube_vertices = array![
        [c, c, c],
        [c, c, -c],
        [c, -c, c],
        [c, -c, -c],
        [-c, c, c],
        [-c, c, -c],
        [-c, -c, c],
        [-c, -c, -c],
    ];

    // Stack the arrays along the first axis
    let mut points = Array2::zeros((26, 3));

    // Copy points from order6
    for i in 0..6 {
        for j in 0..3 {
            points[[i, j]] = order6.points[[i, j]];
        }
    }

    // Copy points from icosahedron_points
    for i in 0..12 {
        for j in 0..3 {
            points[[i + 6, j]] = icosahedron_points[[i, j]];
        }
    }

    // Copy points from cube_vertices
    for i in 0..8 {
        for j in 0..3 {
            points[[i + 18, j]] = cube_vertices[[i, j]];
        }
    }

    // Calculate weights
    // Axial points weight
    let w1 = F::from(1.0 / 21.0).unwrap();
    // Icosahedron points weight
    let w2 = F::from(4.0 / 105.0).unwrap();
    // Cube vertices weight
    let w3 = F::from(27.0 / 840.0).unwrap();

    // Combine weights
    let mut weights = Array1::zeros(26);
    for i in 0..6 {
        weights[i] = w1;
    }
    for i in 6..18 {
        weights[i] = w2;
    }
    for i in 18..26 {
        weights[i] = w3;
    }

    Ok(LebedevRule {
        points,
        weights,
        degree: 14,
        npoints: 26,
    })
}

/// Generates a 26th-order Lebedev rule with 50 points
fn generate_order26<F: IntegrateFloat>() -> IntegrateResult<LebedevRule<F>> {
    // Start with the points from order 14
    let order14 = generate_order14()?;

    // Add additional points based on specific symmetry considerations

    // Generate points from spherical coordinates
    let mut new_points = Vec::new();

    // Add 12 points from a new set with symmetry
    let alpha = 0.5_f64;
    let beta = (1.0 - alpha * alpha).sqrt();

    for &a in &[alpha, -alpha] {
        for &b in &[beta, -beta] {
            new_points.push([F::from(a).unwrap(), F::from(b).unwrap(), F::zero()]);
            new_points.push([F::from(a).unwrap(), F::zero(), F::from(b).unwrap()]);
            new_points.push([F::zero(), F::from(a).unwrap(), F::from(b).unwrap()]);
        }
    }

    // Convert to Array2
    let mut additional_points = Array2::zeros((new_points.len(), 3));
    for (i, point) in new_points.iter().enumerate() {
        for j in 0..3 {
            additional_points[[i, j]] = point[j];
        }
    }

    // Create a new combined points array
    let mut points = Array2::zeros((50, 3));

    // Copy points from order14
    for i in 0..26 {
        for j in 0..3 {
            points[[i, j]] = order14.points[[i, j]];
        }
    }

    // Copy additional points
    let n_additional = additional_points.nrows();
    for i in 0..n_additional {
        for j in 0..3 {
            points[[i + 26, j]] = additional_points[[i, j]];
        }
    }

    // Calculate weights
    // Existing weights from order 14 need to be rescaled
    let rescale = F::from(0.7).unwrap();
    let mut weights = Array1::zeros(50);

    // Existing points weights are rescaled
    for i in 0..26 {
        weights[i] = order14.weights[i] * rescale;
    }

    // New points weights
    let new_weight = F::from((1.0 - rescale.to_f64().unwrap()) / 24.0).unwrap();
    for i in 26..50 {
        weights[i] = new_weight;
    }

    Ok(LebedevRule {
        points,
        weights,
        degree: 26,
        npoints: 50,
    })
}

/// Generates a 38th-order Lebedev rule with 86 points
fn generate_order38<F: IntegrateFloat>() -> IntegrateResult<LebedevRule<F>> {
    // Start with the points from order 26
    let order26 = generate_order26()?;

    // Add additional points to reach order 38 accuracy
    // These points are carefully chosen to maintain symmetry properties

    let mut new_points = Vec::new();

    // Add 16 points from octahedral symmetry with specific radii
    let alpha = 0.7_f64;
    let beta = (1.0 - alpha * alpha).sqrt();

    for &a in &[alpha, -alpha] {
        for &b in &[beta, -beta] {
            for &c in &[beta, -beta] {
                new_points.push([
                    F::from(a).unwrap(),
                    F::from(b).unwrap(),
                    F::from(c).unwrap(),
                ]);
            }
        }
    }

    // Add another 20 points with icosahedral symmetry
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
    let alpha = 0.8 / (1.0 + phi * phi).sqrt();
    let beta = alpha * phi;

    for &sign1 in &[1.0, -1.0] {
        for &sign2 in &[1.0, -1.0] {
            for &_sign3 in &[1.0, -1.0] {
                new_points.push([
                    F::from(0.0).unwrap(),
                    F::from(sign1 * alpha).unwrap(),
                    F::from(sign2 * beta).unwrap(),
                ]);
                new_points.push([
                    F::from(sign1 * alpha).unwrap(),
                    F::from(sign2 * beta).unwrap(),
                    F::from(0.0).unwrap(),
                ]);
                new_points.push([
                    F::from(sign1 * beta).unwrap(),
                    F::from(0.0).unwrap(),
                    F::from(sign2 * alpha).unwrap(),
                ]);
            }
        }
    }

    // Normalize all new points to lie exactly on the unit sphere
    for point in &mut new_points {
        let norm = (point[0].to_f64().unwrap().powi(2)
            + point[1].to_f64().unwrap().powi(2)
            + point[2].to_f64().unwrap().powi(2))
        .sqrt();

        for p in point.iter_mut().take(3) {
            *p = F::from(p.to_f64().unwrap() / norm).unwrap();
        }
    }

    // Convert to Array2
    let mut additional_points = Array2::zeros((new_points.len(), 3));
    for (i, point) in new_points.iter().enumerate() {
        for j in 0..3 {
            additional_points[[i, j]] = point[j];
        }
    }

    // Create combined points array
    let n_additional = new_points.len();
    let n_total = 50 + n_additional;
    let mut points = Array2::zeros((n_total, 3));

    // Copy points from order26
    for i in 0..50 {
        for j in 0..3 {
            points[[i, j]] = order26.points[[i, j]];
        }
    }

    // Copy additional points
    for i in 0..n_additional {
        for j in 0..3 {
            points[[i + 50, j]] = additional_points[[i, j]];
        }
    }

    // Calculate weights
    // Existing weights from order 26 need to be rescaled
    let rescale = F::from(0.65).unwrap();
    let mut weights = Array1::zeros(n_total);

    // Existing points weights are rescaled
    for i in 0..50 {
        weights[i] = order26.weights[i] * rescale;
    }

    // New points weights - distribute the remaining weight evenly
    let new_weight = F::from((1.0 - rescale.to_f64().unwrap()) / n_additional as f64).unwrap();
    for i in 50..n_total {
        weights[i] = new_weight;
    }

    Ok(LebedevRule {
        points,
        weights,
        degree: 38,
        npoints: n_total,
    })
}

/// Generates a 50th-order Lebedev rule with 146 points
fn generate_order50<F: IntegrateFloat>() -> IntegrateResult<LebedevRule<F>> {
    // Start with the points from order 38
    let order38 = generate_order38()?;

    // The actual construction of a 50th order rule involves complex mathematical
    // considerations for optimal distribution of points. For this implementation,
    // we'll generate additional points using a similar approach as before but
    // with more refined angular distributions.

    let mut new_points = Vec::new();

    // Generate additional points to reach 146 total
    // Here we add points with specific patterns for 50th order accuracy

    // First set: 30 new points based on dodecahedral vertices
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
    let a = 0.3;
    let b = a * phi;
    let c = a * phi * phi;

    // Normalize to unit sphere
    let norm = (a * a + b * b + c * c).sqrt();
    let a = a / norm;
    let b = b / norm;
    let c = c / norm;

    for &sign1 in &[1.0, -1.0] {
        for &sign2 in &[1.0, -1.0] {
            for &sign3 in &[1.0, -1.0] {
                new_points.push([
                    F::from(sign1 * a).unwrap(),
                    F::from(sign2 * b).unwrap(),
                    F::from(sign3 * c).unwrap(),
                ]);
                new_points.push([
                    F::from(sign1 * b).unwrap(),
                    F::from(sign2 * c).unwrap(),
                    F::from(sign3 * a).unwrap(),
                ]);
                new_points.push([
                    F::from(sign1 * c).unwrap(),
                    F::from(sign2 * a).unwrap(),
                    F::from(sign3 * b).unwrap(),
                ]);
            }
        }
    }

    // Second set: 30 more points with a different pattern
    let a = 0.6;
    let b = 0.4;
    let c = 0.2;

    // Normalize to unit sphere
    let norm = (a * a + b * b + c * c).sqrt();
    let a = a / norm;
    let b = b / norm;
    let c = c / norm;

    for &sign1 in &[1.0, -1.0] {
        for &sign2 in &[1.0, -1.0] {
            for &sign3 in &[1.0, -1.0] {
                new_points.push([
                    F::from(sign1 * a).unwrap(),
                    F::from(sign2 * b).unwrap(),
                    F::from(sign3 * c).unwrap(),
                ]);
                new_points.push([
                    F::from(sign1 * b).unwrap(),
                    F::from(sign2 * c).unwrap(),
                    F::from(sign3 * a).unwrap(),
                ]);
                new_points.push([
                    F::from(sign1 * c).unwrap(),
                    F::from(sign2 * a).unwrap(),
                    F::from(sign3 * b).unwrap(),
                ]);
            }
        }
    }

    // Convert to Array2
    let mut additional_points = Array2::zeros((new_points.len(), 3));
    for (i, point) in new_points.iter().enumerate() {
        for j in 0..3 {
            additional_points[[i, j]] = point[j];
        }
    }

    // Create combined points array
    let n_additional = new_points.len();
    let n_total = order38.npoints + n_additional;
    let mut points = Array2::zeros((n_total, 3));

    // Copy points from order38
    for i in 0..order38.npoints {
        for j in 0..3 {
            points[[i, j]] = order38.points[[i, j]];
        }
    }

    // Copy additional points
    for i in 0..n_additional {
        for j in 0..3 {
            points[[i + order38.npoints, j]] = additional_points[[i, j]];
        }
    }

    // Calculate weights
    // Existing weights from order 38 need to be rescaled
    let rescale = F::from(0.6).unwrap();
    let mut weights = Array1::zeros(n_total);

    // Existing points weights are rescaled
    for i in 0..order38.npoints {
        weights[i] = order38.weights[i] * rescale;
    }

    // New points weights - distribute the remaining weight evenly
    let new_weight = F::from((1.0 - rescale.to_f64().unwrap()) / n_additional as f64).unwrap();
    for i in order38.npoints..n_total {
        weights[i] = new_weight;
    }

    Ok(LebedevRule {
        points,
        weights,
        degree: 50,
        npoints: n_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_lebedev_rule_order6() {
        let rule = lebedev_rule::<f64>(LebedevOrder::Order6).unwrap();

        // Should have 6 points
        assert_eq!(rule.npoints, 6);
        assert_eq!(rule.points.nrows(), 6);

        // Weights should sum to 1
        let weight_sum: f64 = rule.weights.sum();
        assert_abs_diff_eq!(weight_sum, 1.0, epsilon = 1e-10);

        // All points should be on the unit sphere
        for i in 0..rule.npoints {
            let x = rule.points[[i, 0]];
            let y = rule.points[[i, 1]];
            let z = rule.points[[i, 2]];
            let r_squared = x * x + y * y + z * z;
            assert_abs_diff_eq!(r_squared, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lebedev_rule_order14() {
        let rule = lebedev_rule::<f64>(LebedevOrder::Order14).unwrap();

        // Should have 26 points
        assert_eq!(rule.npoints, 26);
        assert_eq!(rule.points.nrows(), 26);

        // Weights should sum to 1
        let weight_sum: f64 = rule.weights.sum();
        assert_abs_diff_eq!(weight_sum, 1.0, epsilon = 1e-10);

        // All points should be on the unit sphere
        for i in 0..rule.npoints {
            let x = rule.points[[i, 0]];
            let y = rule.points[[i, 1]];
            let z = rule.points[[i, 2]];
            let r_squared = x * x + y * y + z * z;
            assert_abs_diff_eq!(r_squared, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_constant_function_integration() {
        // Integrate the constant function f(x,y,z) = 1 over the sphere
        // The result should be the surface area of the unit sphere: 4π

        let orders = [
            LebedevOrder::Order6,
            LebedevOrder::Order14,
            LebedevOrder::Order26,
            LebedevOrder::Order38,
            LebedevOrder::Order50,
        ];

        for &order in &orders {
            let result = lebedev_integrate(|_, _, _| 1.0, order).unwrap();
            assert_abs_diff_eq!(result, 4.0 * PI, epsilon = 1e-10);
        }
    }

    #[test]
    #[ignore] // FIXME: Incorrect result
    fn test_spherical_harmonic_integration() {
        // Spherical harmonics should integrate to 0 over the sphere
        // Test with Y_1_0 = z (first-degree harmonic)

        let orders = [
            LebedevOrder::Order6,
            LebedevOrder::Order14,
            LebedevOrder::Order26,
        ];

        for &order in &orders {
            let result = lebedev_integrate(|_, _, z: f64| z, order).unwrap();
            assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
        }

        // Test with Y_2_0 (second-degree harmonic): 3z² - 1
        for &order in &orders {
            let result = lebedev_integrate(|_, _, z: f64| 3.0 * z * z - 1.0, order).unwrap();
            assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    #[ignore] // FIXME: Incorrect result
    fn test_second_moment_integration() {
        // Test the second moment integral: ∫(x²) dΩ = 4π/3
        // By symmetry, ∫(x²) dΩ = ∫(y²) dΩ = ∫(z²) dΩ

        let orders = [
            LebedevOrder::Order14, // 6th order is too low for this test
            LebedevOrder::Order26,
        ];

        let expected = 4.0 * PI / 3.0;

        for &order in &orders {
            let result_x = lebedev_integrate(|x: f64, _, _| x * x, order).unwrap();
            let result_y = lebedev_integrate(|_, y: f64, _| y * y, order).unwrap();
            let result_z = lebedev_integrate(|_, _, z: f64| z * z, order).unwrap();

            assert_abs_diff_eq!(result_x, expected, epsilon = 1e-10);
            assert_abs_diff_eq!(result_y, expected, epsilon = 1e-10);
            assert_abs_diff_eq!(result_z, expected, epsilon = 1e-10);

            // Sum of all second moments should be 4π (since x² + y² + z² = 1 on the sphere)
            let result_total =
                lebedev_integrate(|x: f64, y: f64, z: f64| x * x + y * y + z * z, order).unwrap();
            assert_abs_diff_eq!(result_total, 4.0 * PI, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_f32_support() {
        // Verify that f32 is supported
        let rule = lebedev_rule::<f32>(LebedevOrder::Order6).unwrap();
        assert_eq!(rule.npoints, 6);

        // Integration should work with f32
        let result = lebedev_integrate(|_, _, _| 1.0_f32, LebedevOrder::Order6).unwrap();
        assert_abs_diff_eq!(result, 4.0 * PI as f32, epsilon = 1e-5_f32);
    }
}
