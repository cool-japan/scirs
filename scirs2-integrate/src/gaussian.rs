//! Gaussian quadrature integration methods
//!
//! This module provides implementations of numerical integration using
//! Gaussian quadrature methods, which are generally more accurate than
//! simpler methods like the trapezoid rule or Simpson's rule for
//! functions that can be well-approximated by polynomials.

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Gauss-Legendre quadrature nodes and weights
#[derive(Debug, Clone)]
pub struct GaussLegendreQuadrature<F: IntegrateFloat> {
    /// Quadrature nodes (points) on the interval [-1, 1]
    pub nodes: Array1<F>,
    /// Quadrature weights
    pub weights: Array1<F>,
}

impl<F: IntegrateFloat> GaussLegendreQuadrature<F> {
    /// Safe conversion from f64 to F type
    #[allow(dead_code)]
    fn safe_from_f64(value: f64) -> IntegrateResult<F> {
        F::from_f64(value).ok_or_else(|| {
            IntegrateError::ComputationError(format!(
                "Failed to convert f64 constant {value} to target type"
            ))
        })
    }
    /// Create a new Gauss-Legendre quadrature with the given number of points
    ///
    /// # Arguments
    ///
    /// * `n` - Number of quadrature points (must be at least 1)
    ///
    /// # Returns
    ///
    /// * `IntegrateResult<GaussLegendreQuadrature<F>>` - The quadrature nodes and weights
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_integrate::gaussian::GaussLegendreQuadrature;
    ///
    /// let quad = GaussLegendreQuadrature::<f64>::new(5).unwrap();
    /// assert_eq!(quad.nodes.len(), 5);
    /// assert_eq!(quad.weights.len(), 5);
    /// ```
    pub fn new(n: usize) -> IntegrateResult<Self> {
        if n == 0 {
            return Err(IntegrateError::ValueError(
                "Number of quadrature points must be at least 1".to_string(),
            ));
        }

        // For common small orders, use pre-computed values for efficiency and accuracy
        match n {
            1 => Ok(Self::gauss_legendre_1()),
            2 => Ok(Self::gauss_legendre_2()),
            3 => Ok(Self::gauss_legendre_3()),
            4 => Ok(Self::gauss_legendre_4()),
            5 => Ok(Self::gauss_legendre_5()),
            10 => Ok(Self::gauss_legendre_10()),
            // For arbitrary n, use general algorithm
            _ => Self::gauss_legendre_general(n),
        }
    }

    // Pre-computed nodes and weights for n=1
    fn gauss_legendre_1() -> Self {
        GaussLegendreQuadrature {
            nodes: Array1::from_vec(vec![F::zero()]),
            weights: Array1::from_vec(vec![F::from_f64(2.0).unwrap()]),
        }
    }

    // Pre-computed nodes and weights for n=2
    fn gauss_legendre_2() -> Self {
        // Correct values for 2-point Gauss-Legendre quadrature
        let nodes = vec![
            F::from_f64(-0.5773502691896257).unwrap(), // -1/sqrt(3)
            F::from_f64(0.5773502691896257).unwrap(),  // 1/sqrt(3)
        ];

        let weights = vec![F::from_f64(1.0).unwrap(), F::from_f64(1.0).unwrap()];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    // Pre-computed nodes and weights for n=3
    fn gauss_legendre_3() -> Self {
        // Correct values for 3-point Gauss-Legendre quadrature
        let nodes = vec![
            F::from_f64(-0.7745966692414834).unwrap(), // -sqrt(3/5)
            F::zero(),
            F::from_f64(0.7745966692414834).unwrap(), // sqrt(3/5)
        ];

        let weights = vec![
            F::from_f64(5.0 / 9.0).unwrap(),
            F::from_f64(8.0 / 9.0).unwrap(),
            F::from_f64(5.0 / 9.0).unwrap(),
        ];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    // Pre-computed nodes and weights for n=4
    fn gauss_legendre_4() -> Self {
        // Correct values for 4-point Gauss-Legendre quadrature
        let nodes = vec![
            F::from_f64(-0.8611363115940526).unwrap(),
            F::from_f64(-0.3399810435848563).unwrap(),
            F::from_f64(0.3399810435848563).unwrap(),
            F::from_f64(0.8611363115940526).unwrap(),
        ];

        let weights = vec![
            F::from_f64(0.3478548451374538).unwrap(),
            F::from_f64(0.6521451548625461).unwrap(),
            F::from_f64(0.6521451548625461).unwrap(),
            F::from_f64(0.3478548451374538).unwrap(),
        ];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    // Pre-computed nodes and weights for n=5
    fn gauss_legendre_5() -> Self {
        // Correct values for 5-point Gauss-Legendre quadrature
        let nodes = vec![
            F::from_f64(-0.906_179_845_938_664).unwrap(),
            F::from_f64(-0.538_469_310_105_683).unwrap(),
            F::zero(),
            F::from_f64(0.538_469_310_105_683).unwrap(),
            F::from_f64(0.906_179_845_938_664).unwrap(),
        ];

        let weights = vec![
            F::from_f64(0.2369268850561891).unwrap(),
            F::from_f64(0.4786286704993665).unwrap(),
            F::from_f64(0.5688888888888889).unwrap(),
            F::from_f64(0.4786286704993665).unwrap(),
            F::from_f64(0.2369268850561891).unwrap(),
        ];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    // Pre-computed nodes and weights for n=10
    fn gauss_legendre_10() -> Self {
        // These values are pre-computed high-precision values for n=10
        let nodes = vec![
            F::from_f64(-0.9739065285171717).unwrap(),
            F::from_f64(-0.8650633666889845).unwrap(),
            F::from_f64(-0.6794095682990244).unwrap(),
            F::from_f64(-0.4333953941292472).unwrap(),
            F::from_f64(-0.1488743389816312).unwrap(),
            F::from_f64(0.1488743389816312).unwrap(),
            F::from_f64(0.4333953941292472).unwrap(),
            F::from_f64(0.6794095682990244).unwrap(),
            F::from_f64(0.8650633666889845).unwrap(),
            F::from_f64(0.9739065285171717).unwrap(),
        ];

        let weights = vec![
            F::from_f64(0.0666713443086881).unwrap(),
            F::from_f64(0.1494513491505806).unwrap(),
            F::from_f64(0.219_086_362_515_982).unwrap(),
            F::from_f64(0.2692667193099963).unwrap(),
            F::from_f64(0.2955242247147529).unwrap(),
            F::from_f64(0.2955242247147529).unwrap(),
            F::from_f64(0.2692667193099963).unwrap(),
            F::from_f64(0.219_086_362_515_982).unwrap(),
            F::from_f64(0.1494513491505806).unwrap(),
            F::from_f64(0.0666713443086881).unwrap(),
        ];

        GaussLegendreQuadrature {
            nodes: Array1::from_vec(nodes),
            weights: Array1::from_vec(weights),
        }
    }

    /// General Gauss-Legendre quadrature computation for arbitrary number of points
    /// Uses the Golub-Welsch algorithm based on eigenvalue decomposition
    fn gauss_legendre_general(n: usize) -> IntegrateResult<Self> {
        // Create companion matrix for Legendre polynomials
        // The companion matrix is tridiagonal with zeros on diagonal
        // and beta coefficients on super/sub-diagonals
        let alpha = vec![F::zero(); n]; // diagonal elements (all zero for Legendre)
        let mut beta = Vec::with_capacity(n - 1); // off-diagonal elements

        // Compute beta coefficients: beta[k] = k+1 / sqrt(4*(k+1)^2 - 1)
        for k in 0..n - 1 {
            let k_plus_1 = (k + 1) as f64;
            let beta_k = k_plus_1 / (4.0 * k_plus_1 * k_plus_1 - 1.0).sqrt();
            beta.push(F::from_f64(beta_k).ok_or_else(|| {
                IntegrateError::ComputationError(format!(
                    "Failed to convert beta coefficient {beta_k} to target type"
                ))
            })?);
        }

        // Use symmetric tridiagonal eigenvalue solver
        let (nodes, eigenvecs) = Self::symmetric_tridiagonal_eigenvalues(&alpha, &beta)?;

        // Compute weights from first components of normalized eigenvectors
        // Weight[i] = 2 * (first component of eigenvector[i])^2
        let mut weights = Vec::with_capacity(n);
        let two = F::from_f64(2.0).ok_or_else(|| {
            IntegrateError::ComputationError("Failed to convert constant 2.0".to_string())
        })?;

        for i in 0..n {
            let first_component = eigenvecs[[0, i]];
            let weight = two * first_component * first_component;
            weights.push(weight);
        }

        // Sort nodes and weights by increasing node values
        let mut node_weight_pairs: Vec<(F, F)> = nodes.into_iter().zip(weights).collect();
        node_weight_pairs.sort_by(|a, b| {
            a.0.to_f64()
                .unwrap_or(0.0)
                .partial_cmp(&b.0.to_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let (sorted_nodes, sorted_weights): (Vec<F>, Vec<F>) =
            node_weight_pairs.into_iter().unzip();

        Ok(GaussLegendreQuadrature {
            nodes: Array1::from_vec(sorted_nodes),
            weights: Array1::from_vec(sorted_weights),
        })
    }

    /// Solve symmetric tridiagonal eigenvalue problem
    /// Uses the QL algorithm with implicit shifts
    fn symmetric_tridiagonal_eigenvalues(
        alpha: &[F],
        beta: &[F],
    ) -> IntegrateResult<(Vec<F>, Array2<F>)> {
        let n = alpha.len();
        let mut d = alpha.to_vec(); // diagonal elements
        let mut e = Vec::with_capacity(n);
        e.push(F::zero()); // e[0] = 0
        e.extend_from_slice(beta); // e[1..n] = beta

        // Initialize eigenvector matrix to identity
        let mut z = Array2::<F>::zeros((n, n));
        for i in 0..n {
            z[[i, i]] = F::one();
        }

        // QL algorithm with implicit shifts
        let max_iterations = 100;
        let eps = F::from_f64(1e-15).unwrap_or_else(|| F::from_f64(1e-10).unwrap());

        for _ in 0..max_iterations {
            let mut converged = true;

            for i in 0..n - 1 {
                if e[i + 1].abs() > eps * (d[i].abs() + d[i + 1].abs()) {
                    converged = false;
                    break;
                }
            }

            if converged {
                break;
            }

            // Apply QL step
            Self::ql_step(&mut d, &mut e, &mut z)?;
        }

        Ok((d, z))
    }

    /// Single QL iteration step
    fn ql_step(d: &mut [F], e: &mut [F], z: &mut Array2<F>) -> IntegrateResult<()> {
        let n = d.len();

        for i in 0..n - 1 {
            if e[i + 1].abs() > F::from_f64(1e-15).unwrap_or_else(|| F::from_f64(1e-10).unwrap()) {
                // Compute shift
                let shift = d[n - 1];

                // Apply shift
                for j in 0..n {
                    d[j] -= shift;
                }

                // Perform QL decomposition step
                let mut p = d[0];
                let mut q = e[1];

                for j in 0..n - 1 {
                    let r = (p * p + q * q).sqrt();
                    if r.abs() < F::from_f64(1e-15).unwrap_or_else(|| F::from_f64(1e-10).unwrap()) {
                        continue;
                    }

                    let c = p / r;
                    let s = q / r;

                    // Update diagonal and off-diagonal elements
                    if j > 0 {
                        e[j] = r;
                    }

                    p = c * d[j] + s * e[j + 1];
                    e[j + 1] = s * d[j] - c * e[j + 1];
                    q = s * d[j + 1];
                    d[j + 1] = c * d[j + 1];

                    // Update eigenvectors
                    for k in 0..n {
                        let temp = z[[k, j]];
                        z[[k, j]] = c * temp + s * z[[k, j + 1]];
                        z[[k, j + 1]] = s * temp - c * z[[k, j + 1]];
                    }

                    if j < n - 2 {
                        p = d[j + 1];
                        q = e[j + 2];
                    }
                }

                d[n - 1] = p;

                // Remove shift
                for j in 0..n {
                    d[j] += shift;
                }
            }
        }

        Ok(())
    }

    /// Apply the quadrature rule to integrate a function over [a, b]
    ///
    /// # Arguments
    ///
    /// * `f` - The function to integrate
    /// * `a` - Lower bound of integration
    /// * `b` - Upper bound of integration
    ///
    /// # Returns
    ///
    /// * The approximate value of the integral
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_integrate::gaussian::GaussLegendreQuadrature;
    ///
    /// // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
    /// let quad = GaussLegendreQuadrature::<f64>::new(5).unwrap();
    /// let result = quad.integrate(|x| x * x, 0.0, 1.0);
    /// assert!((result - 1.0/3.0).abs() < 1e-10);
    /// ```
    pub fn integrate<Func>(&self, f: Func, a: F, b: F) -> F
    where
        Func: Fn(F) -> F,
    {
        // Change of variables from [-1, 1] to [a, b]
        let mid = (a + b) / F::from_f64(2.0).unwrap();
        let half_length = (b - a) / F::from_f64(2.0).unwrap();

        // Apply quadrature rule
        let mut result = F::zero();
        for (i, &x) in self.nodes.iter().enumerate() {
            let transformed_x = mid + half_length * x;
            result += self.weights[i] * f(transformed_x);
        }

        // Scale by half-length due to the change of variables
        result * half_length
    }
}

/// Integrate a function using Gauss-Legendre quadrature
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `n` - Number of quadrature points (more points generally give higher accuracy)
///
/// # Returns
///
/// * `IntegrateResult<F>` - The approximate value of the integral
///
/// # Examples
///
/// ```
/// use scirs2_integrate::gaussian::gauss_legendre;
///
/// // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
/// let result = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5).unwrap();
/// assert!((result - 1.0/3.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn gauss_legendre<F, Func>(f: Func, a: F, b: F, n: usize) -> IntegrateResult<F>
where
    F: IntegrateFloat,
    Func: Fn(F) -> F,
{
    let quadrature = GaussLegendreQuadrature::new(n)?;
    Ok(quadrature.integrate(f, a, b))
}

/// Integrate a function over multiple dimensions using Gauss-Legendre quadrature
///
/// # Arguments
///
/// * `f` - The multidimensional function to integrate
/// * `ranges` - Array of integration ranges (a, b) for each dimension
/// * `n_points` - Number of quadrature points to use for each dimension
///
/// # Returns
///
/// * `IntegrateResult<F>` - The approximate value of the integral
///
/// # Examples
///
/// ```
/// use scirs2_integrate::gaussian::multi_gauss_legendre;
/// use scirs2_core::ndarray::{Array1, ArrayView1};
///
/// // Integrate f(x,y) = x²+y² over [0,1]×[0,1] (exact result: 2/3)
/// let result = multi_gauss_legendre(
///     |x: ArrayView1<f64>| x.iter().map(|&xi| xi*xi).sum::<f64>(),
///     &[(0.0, 1.0), (0.0, 1.0)],
///     5
/// ).unwrap();
/// assert!((result - 2.0/3.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn multi_gauss_legendre<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    n_points: usize,
) -> IntegrateResult<F>
where
    F: IntegrateFloat,
    Func: Fn(ArrayView1<F>) -> F,
{
    if ranges.is_empty() {
        return Err(IntegrateError::ValueError(
            "Integration ranges cannot be empty".to_string(),
        ));
    }

    let quadrature = GaussLegendreQuadrature::new(n_points)?;
    let n_dims = ranges.len();

    // Inner function to perform recursive multidimensional integration
    fn integrate_recursive<F, Func>(
        f: &Func,
        ranges: &[(F, F)],
        quadrature: &GaussLegendreQuadrature<F>,
        dim: usize,
        point: &mut Array1<F>,
        n_dims: usize,
    ) -> F
    where
        F: IntegrateFloat,
        Func: Fn(ArrayView1<F>) -> F,
    {
        if dim == n_dims {
            // Base case: Evaluate the function at the current point
            return f(point.view());
        }

        // Change of variables from [-1, 1] to [a, b]
        let (a, b) = ranges[dim];
        let mid = (a + b) / F::from_f64(2.0).unwrap();
        let half_length = (b - a) / F::from_f64(2.0).unwrap();

        // Apply quadrature rule for the current dimension
        let mut result = F::zero();
        for (i, &x) in quadrature.nodes.iter().enumerate() {
            let transformed_x = mid + half_length * x;
            point[dim] = transformed_x;

            // Recursively integrate remaining dimensions
            let inner_result = integrate_recursive(f, ranges, quadrature, dim + 1, point, n_dims);
            result += quadrature.weights[i] * inner_result;
        }

        // Scale by half-length due to the change of variables
        result * half_length
    }

    // Initialize a point array to store coordinates during integration
    let mut point = Array1::zeros(n_dims);

    // Start the recursive integration from the first dimension
    Ok(integrate_recursive(
        &f,
        ranges,
        &quadrature,
        0,
        &mut point,
        n_dims,
    ))
}

/// Gauss-Kronrod 15-point rule for integration with error estimation
///
/// Returns:
/// - Integral estimate
/// - Error estimate
/// - Number of function evaluations
#[allow(dead_code)]
pub fn gauss_kronrod15<F, Func>(f: Func, a: F, b: F) -> (F, F, usize)
where
    F: IntegrateFloat,
    Func: Fn(F) -> F,
{
    // Gauss-Kronrod 15-point rule (7-point Gauss, 15-point Kronrod)
    // Points and weights from SciPy
    let xgk = [
        -0.9914553711208126f64,
        -0.9491079123427585,
        -0.8648644233597691,
        -0.7415311855993944,
        -0.5860872354676911,
        -0.4058451513773972,
        -0.2077849550078985,
        0.0,
        0.2077849550078985,
        0.4058451513773972,
        0.5860872354676911,
        0.7415311855993944,
        0.8648644233597691,
        0.9491079123427585,
        0.9914553711208126,
    ];

    let wgk = [
        0.022935322010529224f64,
        0.063_092_092_629_978_56,
        0.10479001032225018,
        0.14065325971552592,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.20443294007529889,
        0.20948214108472782,
        0.20443294007529889,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.14065325971552592,
        0.10479001032225018,
        0.063_092_092_629_978_56,
        0.022935322010529224,
    ];

    // Abscissae for the 7-point Gauss rule (odd indices of xgk)
    let wg = [
        0.129_484_966_168_869_7_f64,
        0.27970539148927664,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.27970539148927664,
        0.129_484_966_168_869_7,
    ];

    // Apply the rule
    let half_length = (b - a) / F::from_f64(2.0).unwrap();
    let center = (a + b) / F::from_f64(2.0).unwrap();

    let mut result_kronrod = F::zero();
    let mut result_gauss = F::zero();

    // Evaluate function at center point
    let fc = f(center);

    // Accumulate for Kronrod rule
    result_kronrod += F::from_f64(wgk[7]).unwrap() * fc;

    // Evaluate at other points
    for i in 0..7 {
        let x = F::from_f64(xgk[i]).unwrap();
        let abscissa = center - half_length * x;
        let fval = f(abscissa);
        result_kronrod += F::from_f64(wgk[i]).unwrap() * fval;
        result_gauss += F::from_f64(wg[i]).unwrap() * fval;

        let abscissa = center + half_length * x;
        let fval = f(abscissa);
        result_kronrod += F::from_f64(wgk[14 - i]).unwrap() * fval;
        result_gauss += F::from_f64(wg[6 - i]).unwrap() * fval;
    }

    // Evaluate remaining Kronrod points
    for i in [1, 3, 5, 9, 11, 13] {
        let x = F::from_f64(xgk[i]).unwrap();
        let abscissa = center - half_length * x;
        let fval = f(abscissa);
        result_kronrod += F::from_f64(wgk[i]).unwrap() * fval;

        let abscissa = center + half_length * x;
        let fval = f(abscissa);
        result_kronrod += F::from_f64(wgk[i]).unwrap() * fval;
    }

    // Scale results
    result_kronrod *= half_length;
    result_gauss *= half_length;

    // Compute error estimate
    let error = (result_kronrod - result_gauss).abs();

    (result_kronrod, error, 15)
}

/// Gauss-Kronrod 21-point rule for integration with error estimation
///
/// Returns:
/// - Integral estimate
/// - Error estimate
/// - Number of function evaluations
#[allow(dead_code)]
pub fn gauss_kronrod21<F, Func>(f: Func, a: F, b: F) -> (F, F, usize)
where
    F: IntegrateFloat,
    Func: Fn(F) -> F,
{
    // Gauss-Kronrod 21-point rule (10-point Gauss, 21-point Kronrod)
    // Points and weights from SciPy
    let xgk = [
        -0.9956571630258081f64,
        -0.9739065285171717,
        -0.9301574913557082,
        -0.8650633666889845,
        -0.7808177265864169,
        -0.6794095682990244,
        -0.5627571346686047,
        -0.4333953941292472,
        -0.2943928627014602,
        -0.1488743389816312,
        0.0,
        0.1488743389816312,
        0.2943928627014602,
        0.4333953941292472,
        0.5627571346686047,
        0.6794095682990244,
        0.7808177265864169,
        0.8650633666889845,
        0.9301574913557082,
        0.9739065285171717,
        0.9956571630258081,
    ];

    let wgk = [
        0.011694638867371874f64,
        0.032558162307964725,
        0.054755896574351995,
        0.075_039_674_810_919_96,
        0.093_125_454_583_697_6,
        0.109_387_158_802_297_64,
        0.123_491_976_262_065_84,
        0.134_709_217_311_473_34,
        0.142_775_938_577_060_09,
        0.147_739_104_901_338_49,
        0.149_445_554_002_916_9,
        0.147_739_104_901_338_49,
        0.142_775_938_577_060_09,
        0.134_709_217_311_473_34,
        0.123_491_976_262_065_84,
        0.109_387_158_802_297_64,
        0.093_125_454_583_697_6,
        0.075_039_674_810_919_96,
        0.054755896574351995,
        0.032558162307964725,
        0.011694638867371874,
    ];

    // Abscissae for the 10-point Gauss rule (every other point)
    let wg = [
        0.066_671_344_308_688_14_f64,
        0.149_451_349_150_580_6,
        0.219_086_362_515_982_04,
        0.269_266_719_309_996_35,
        0.295_524_224_714_752_87,
        0.295_524_224_714_752_87,
        0.269_266_719_309_996_35,
        0.219_086_362_515_982_04,
        0.149_451_349_150_580_6,
        0.066_671_344_308_688_14,
    ];

    // Apply the rule
    let half_length = (b - a) / F::from_f64(2.0).unwrap();
    let center = (a + b) / F::from_f64(2.0).unwrap();

    let mut result_kronrod = F::zero();
    let mut result_gauss = F::zero();

    // Evaluate function at center point
    let fc = f(center);
    result_kronrod += F::from_f64(wgk[10]).unwrap() * fc;

    // Evaluate at other points
    for i in 0..10 {
        let x = F::from_f64(xgk[i]).unwrap();
        let abscissa = center - half_length * x;
        let fval = f(abscissa);
        result_kronrod += F::from_f64(wgk[i]).unwrap() * fval;

        let abscissa = center + half_length * x;
        let fval = f(abscissa);
        result_kronrod += F::from_f64(wgk[20 - i]).unwrap() * fval;

        // Add to Gauss result for every other point
        if i % 2 == 0 {
            let idx = i / 2;
            result_gauss += F::from_f64(wg[idx]).unwrap() * fval;
            result_gauss += F::from_f64(wg[9 - idx]).unwrap() * fval;
        }
    }

    // Scale results
    result_kronrod *= half_length;
    result_gauss *= half_length;

    // Compute error estimate
    let error = (result_kronrod - result_gauss).abs();

    (result_kronrod, error, 21)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gauss_legendre_quadrature() {
        // Test integrating x² from 0 to 1 (exact result: 1/3)
        let quad5 = GaussLegendreQuadrature::<f64>::new(5).unwrap();
        let result = quad5.integrate(|x| x * x, 0.0, 1.0);

        // With the corrected nodes and weights, the result should be accurate
        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-10);

        // Test integrating sin(x) from 0 to π (exact result: 2)
        let quad10 = GaussLegendreQuadrature::<f64>::new(10).unwrap();
        let result = quad10.integrate(|x| x.sin(), 0.0, PI);
        assert_relative_eq!(result, 2.0, epsilon = 1e-10);

        // Test integrating exp(-x²) from -1 to 1
        // This is related to the error function, with exact result: sqrt(π)·erf(1)
        let quad10 = GaussLegendreQuadrature::<f64>::new(10).unwrap();
        let result = quad10.integrate(|x| (-x * x).exp(), -1.0, 1.0);
        let exact = PI.sqrt() * libm::erf(1.0);
        assert_relative_eq!(result, exact, epsilon = 1e-10);
    }

    #[test]
    fn test_gauss_legendre_helper() {
        // Test the high-level helper function
        let result = gauss_legendre(|x| x * x, 0.0, 1.0, 5).unwrap();

        // With the corrected nodes and weights, the result should be accurate
        assert_relative_eq!(result, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_dimensional_integration() {
        // Test 2D integration: f(x,y) = x² + y² over [0,1]×[0,1]
        // Exact result: 2/3 (1/3 for x² + 1/3 for y²)
        let result =
            multi_gauss_legendre(|x| x[0] * x[0] + x[1] * x[1], &[(0.0, 1.0), (0.0, 1.0)], 5)
                .unwrap();

        // With the corrected nodes and weights, the result should be accurate
        assert_relative_eq!(result, 2.0 / 3.0, epsilon = 1e-10);

        // Test 3D integration: f(x,y,z) = x²y²z² over [0,1]³
        // Exact result: (1/3)³ = 1/27
        let result = multi_gauss_legendre(
            |x| x[0] * x[0] * x[1] * x[1] * x[2] * x[2],
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            5,
        )
        .unwrap();

        // With the corrected nodes and weights, the result should be accurate
        assert_relative_eq!(result, 1.0 / 27.0, epsilon = 1e-10);
    }
}
