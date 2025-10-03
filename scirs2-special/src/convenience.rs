//! Domain-specific convenience functions
//!
//! This module provides high-level convenience functions that combine
//! basic special functions for common use cases in physics, engineering,
//! and data science.

#![allow(dead_code)]

use crate::{bessel, erf, orthogonal, spherical_harmonics, statistical, SpecialResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Physics-related convenience functions
pub mod physics {
    use super::*;

    /// Calculate the wave function for a particle in a box
    ///
    /// # Arguments
    /// * `n` - Quantum number (n >= 1)
    /// * `x` - Position array (normalized to box length)
    /// * `normalize` - Whether to normalize the wave function
    pub fn particle_in_box_wavefunction(
        n: usize,
        x: &ArrayView1<f64>,
        normalize: bool,
    ) -> SpecialResult<Array1<f64>> {
        if n == 0 {
            return Err(crate::SpecialError::ValueError(
                "Quantum number n must be >= 1".to_string(),
            ));
        }

        let mut psi = Array1::zeros(x.len());
        let normalization = if normalize { (2.0_f64).sqrt() } else { 1.0 };

        for (i, &xi) in x.iter().enumerate() {
            psi[i] = normalization * (n as f64 * PI * xi).sin();
        }

        Ok(psi)
    }

    /// Calculate the radial wave function for hydrogen-like atoms
    ///
    /// # Arguments
    /// * `n` - Principal quantum number
    /// * `l` - Azimuthal quantum number
    /// * `r` - Radial distance array (in units of Bohr radius)
    /// * `z` - Nuclear charge
    pub fn hydrogen_radial_wavefunction(
        n: usize,
        l: usize,
        r: &ArrayView1<f64>,
        z: f64,
    ) -> SpecialResult<Array1<f64>> {
        if n == 0 || l >= n {
            return Err(crate::SpecialError::ValueError(
                "Invalid quantum numbers".to_string(),
            ));
        }

        let mut psi_r = Array1::zeros(r.len());
        let a0 = 1.0; // Bohr radius in atomic units

        // Normalization factor
        let norm_factor = ((2.0 * z / (n as f64 * a0)).powi(3) * factorial(n - l - 1) as f64
            / (2.0 * n as f64 * factorial(n + l) as f64))
            .sqrt();

        for (i, &ri) in r.iter().enumerate() {
            let rho = 2.0 * z * ri / (n as f64 * a0);

            // Generalized Laguerre polynomial L_n^α(x)
            // For hydrogen, we need L_{n-l-1}^{2l+1}(rho)
            // Using regular Laguerre since we don't have generalized version
            let laguerre = orthogonal::laguerre(n - l - 1, rho);

            psi_r[i] = norm_factor * (-rho / 2.0).exp() * rho.powi(l as i32) * laguerre;
        }

        Ok(psi_r)
    }

    /// Calculate the spherical harmonic for given angles
    ///
    /// Convenience wrapper that handles both real and complex cases
    pub fn spherical_harmonic_convenience(
        l: i32,
        m: i32,
        theta: f64,
        phi: f64,
        real: bool,
    ) -> Complex64 {
        if l < 0 {
            return Complex64::new(0.0, 0.0);
        }

        let l_usize = l as usize;

        if real {
            // Real spherical harmonics
            match spherical_harmonics::sph_harm(l_usize, m, theta, phi) {
                Ok(val) => Complex64::new(val, 0.0),
                Err(_) => Complex64::new(0.0, 0.0),
            }
        } else {
            // Complex spherical harmonics
            match spherical_harmonics::sph_harm_complex(l_usize, m, theta, phi) {
                Ok((re, im)) => Complex64::new(re, im),
                Err(_) => Complex64::new(0.0, 0.0),
            }
        }
    }

    /// Calculate the Planck radiation law (spectral radiance)
    ///
    /// # Arguments
    /// * `wavelength` - Wavelength array in meters
    /// * `temperature` - Temperature in Kelvin
    pub fn planck_radiation(
        wavelength: &ArrayView1<f64>,
        temperature: f64,
    ) -> SpecialResult<Array1<f64>> {
        if temperature <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Temperature must be positive".to_string(),
            ));
        }

        let h = 6.62607015e-34; // Planck constant
        let c = 299792458.0; // Speed of light
        let k_b = 1.380649e-23; // Boltzmann constant

        let mut radiance = Array1::zeros(wavelength.len());

        for (i, &lambda) in wavelength.iter().enumerate() {
            if lambda <= 0.0 {
                continue;
            }

            let exp_arg = h * c / (lambda * k_b * temperature);
            radiance[i] = 2.0 * h * c * c / (lambda.powi(5) * (exp_arg.exp() - 1.0));
        }

        Ok(radiance)
    }

    /// Calculate the Fermi-Dirac distribution
    ///
    /// # Arguments
    /// * `energy` - Energy array in eV
    /// * `fermienergy` - Fermi energy in eV
    /// * `temperature` - Temperature in Kelvin
    pub fn fermi_dirac_distribution(
        energy: &ArrayView1<f64>,
        fermienergy: f64,
        temperature: f64,
    ) -> SpecialResult<Array1<f64>> {
        if temperature < 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Temperature cannot be negative".to_string(),
            ));
        }

        let k_b_ev = 8.617333262e-5; // Boltzmann constant in eV/K
        let mut distribution = Array1::zeros(energy.len());

        if temperature == 0.0 {
            // Zero temperature limit
            for (i, &e) in energy.iter().enumerate() {
                distribution[i] = if e < fermienergy { 1.0 } else { 0.0 };
            }
        } else {
            for (i, &e) in energy.iter().enumerate() {
                let x = (e - fermienergy) / (k_b_ev * temperature);
                distribution[i] = statistical::logistic(-x);
            }
        }

        Ok(distribution)
    }

    /// Simple factorial helper (internal use)
    fn factorial(n: usize) -> usize {
        match n {
            0 | 1 => 1,
            _ => n * factorial(n - 1),
        }
    }
}

/// Engineering-related convenience functions
pub mod engineering {
    use super::*;

    /// Calculate the error function for signal-to-noise ratio
    ///
    /// Useful for bit error rate calculations in communications
    pub fn q_function(x: &ArrayView1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.len());
        let sqrt2 = 2.0_f64.sqrt();

        for (i, &xi) in x.iter().enumerate() {
            result[i] = 0.5 * erf::erfc(xi / sqrt2);
        }

        result
    }

    /// Calculate the Rice distribution PDF
    ///
    /// Used in wireless communications for fading channels
    pub fn rice_pdf(x: &ArrayView1<f64>, nu: f64, sigma: f64) -> SpecialResult<Array1<f64>> {
        if sigma <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Sigma must be positive".to_string(),
            ));
        }

        let mut pdf = Array1::zeros(x.len());
        let sigma2 = sigma * sigma;

        for (i, &xi) in x.iter().enumerate() {
            if xi >= 0.0 {
                let i0_arg = xi * nu / sigma2;
                pdf[i] = (xi / sigma2)
                    * (-(xi * xi + nu * nu) / (2.0 * sigma2)).exp()
                    * bessel::i0(i0_arg);
            }
        }

        Ok(pdf)
    }

    /// Calculate antenna array factor
    ///
    /// # Arguments
    /// * `theta` - Angle array in radians
    /// * `n_elements` - Number of antenna elements
    /// * `spacing` - Element spacing in wavelengths
    /// * `phase_shift` - Progressive phase shift between elements
    pub fn antenna_array_factor(
        theta: &ArrayView1<f64>,
        n_elements: usize,
        spacing: f64,
        phase_shift: f64,
    ) -> Array1<Complex64> {
        let mut af = Array1::zeros(theta.len());
        let k = 2.0 * PI; // Wave number (normalized)

        for (i, &th) in theta.iter().enumerate() {
            let psi = k * spacing * th.cos() + phase_shift;

            // Array factor using geometric series formula
            if (psi.sin()).abs() > 1e-10 {
                let numerator = (n_elements as f64 * psi / 2.0).sin();
                let denominator = (psi / 2.0).sin();
                af[i] = Complex64::new(numerator / denominator, 0.0);
            } else {
                af[i] = Complex64::new(n_elements as f64, 0.0);
            }
        }

        af
    }

    /// Calculate the complementary error function for Gaussian noise
    pub fn gaussian_tail_probability(x: &ArrayView1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            result[i] = 0.5 * erf::erfc(xi / 2.0_f64.sqrt());
        }

        result
    }
}

/// Data science and machine learning convenience functions
pub mod data_science {
    use super::*;

    /// Calculate the Gini coefficient from a probability distribution
    ///
    /// Measures inequality in a distribution (0 = perfect equality, 1 = perfect inequality)
    pub fn gini_coefficient(probabilities: &ArrayView1<f64>) -> SpecialResult<f64> {
        if probabilities.iter().any(|&p| p < 0.0) {
            return Err(crate::SpecialError::ValueError(
                "Probabilities must be non-negative".to_string(),
            ));
        }

        let sum: f64 = probabilities.sum();
        if sum == 0.0 {
            return Ok(0.0);
        }

        // Normalize _probabilities
        let n = probabilities.len();
        let mut sorted_probs: Vec<f64> = probabilities.to_vec();
        sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut gini = 0.0;
        for (i, &p) in sorted_probs.iter().enumerate() {
            gini += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * p;
        }

        Ok(gini / (n as f64 * sum))
    }

    /// Calculate Shannon entropy
    ///
    /// # Arguments
    /// * `probabilities` - Probability distribution
    /// * `base` - Logarithm base (2 for bits, e for nats)
    pub fn shannon_entropy(probabilities: &ArrayView1<f64>, base: f64) -> SpecialResult<f64> {
        if base <= 0.0 || base == 1.0 {
            return Err(crate::SpecialError::ValueError(
                "Base must be positive and not equal to 1".to_string(),
            ));
        }

        let mut entropy = 0.0;
        let log_base = base.ln();

        for &p in probabilities.iter() {
            if p > 0.0 {
                entropy -= p * p.ln() / log_base;
            }
        }

        Ok(entropy)
    }

    /// Calculate the softmax temperature-scaled function
    ///
    /// Useful for controlling the "sharpness" of the distribution
    pub fn softmax_temperature(
        logits: &ArrayView1<f64>,
        temperature: f64,
    ) -> SpecialResult<Array1<f64>> {
        if temperature <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Temperature must be positive".to_string(),
            ));
        }

        let scaled_logits = logits.mapv(|x| x / temperature);
        statistical::softmax(scaled_logits.view())
    }

    /// Calculate the Kullback-Leibler divergence
    ///
    /// KL(P||Q) measures how P diverges from reference distribution Q
    pub fn kl_divergence(p: &ArrayView1<f64>, q: &ArrayView1<f64>) -> SpecialResult<f64> {
        if p.len() != q.len() {
            return Err(crate::SpecialError::ValueError(
                "Distributions must have the same length".to_string(),
            ));
        }

        let mut kl = 0.0;

        for (i, (&pi, &qi)) in p.iter().zip(q.iter()).enumerate() {
            if pi < 0.0 || qi < 0.0 {
                return Err(crate::SpecialError::ValueError(format!(
                    "Negative probability at index {i}"
                )));
            }

            if pi > 0.0 {
                if qi == 0.0 {
                    return Ok(f64::INFINITY);
                }
                kl += pi * (pi / qi).ln();
            }
        }

        Ok(kl)
    }

    /// Calculate the Wasserstein distance (1D case)
    ///
    /// Also known as Earth Mover's Distance
    pub fn wasserstein_distance_1d(
        values: &ArrayView1<f64>,
        p: &ArrayView1<f64>,
        q: &ArrayView1<f64>,
    ) -> SpecialResult<f64> {
        if values.len() != p.len() || values.len() != q.len() {
            return Err(crate::SpecialError::ValueError(
                "All arrays must have the same length".to_string(),
            ));
        }

        // Sort by values
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());

        let mut cum_p = 0.0;
        let mut cum_q = 0.0;
        let mut distance = 0.0;

        for i in 0..indices.len() - 1 {
            cum_p += p[indices[i]];
            cum_q += q[indices[i]];

            let delta_x = values[indices[i + 1]] - values[indices[i]];
            distance += (cum_p - cum_q).abs() * delta_x;
        }

        Ok(distance)
    }
}

/// Signal processing convenience functions
pub mod signal_processing {
    use super::*;

    /// Generate a Gaussian window function
    ///
    /// # Arguments
    /// * `n` - Number of points
    /// * `sigma` - Standard deviation (typically 0.1 to 0.5)
    pub fn gaussian_window(n: usize, sigma: f64) -> Array1<f64> {
        let mut window = Array1::zeros(n);
        let center = (n - 1) as f64 / 2.0;

        for i in 0..n {
            let x = (i as f64 - center) / (sigma * center);
            window[i] = (-0.5 * x * x).exp();
        }

        window
    }

    /// Generate a Kaiser window function
    ///
    /// # Arguments
    /// * `n` - Number of points
    /// * `beta` - Shape parameter (0 = rectangular, 5 = Hamming-like, 8.6 = Blackman-like)
    pub fn kaiser_window(n: usize, beta: f64) -> Array1<f64> {
        let mut window = Array1::zeros(n);
        let alpha = (n - 1) as f64 / 2.0;
        let i0_beta = bessel::i0(beta);

        for i in 0..n {
            let x = 2.0 * (i as f64 - alpha) / (n - 1) as f64;
            let arg = beta * (1.0 - x * x).sqrt();
            window[i] = bessel::i0(arg) / i0_beta;
        }

        window
    }

    /// Calculate the spectrogram time-frequency kernel
    ///
    /// Combines window function with complex exponential for STFT
    pub fn stft_kernel(
        n: usize,
        window_type: &str,
        freq_bins: usize,
    ) -> SpecialResult<Array2<Complex64>> {
        let window = match window_type {
            "gaussian" => gaussian_window(n, 0.4),
            "kaiser" => kaiser_window(n, 8.6),
            _ => {
                return Err(crate::SpecialError::ValueError(
                    "Unknown window _type".to_string(),
                ))
            }
        };

        let mut kernel = Array2::zeros((freq_bins, n));

        for k in 0..freq_bins {
            let freq = 2.0 * PI * k as f64 / freq_bins as f64;
            for i in 0..n {
                let phase = -freq * i as f64;
                kernel[[k, i]] = window[i] * Complex64::new(phase.cos(), phase.sin());
            }
        }

        Ok(kernel)
    }
}

/// Financial mathematics convenience functions
pub mod finance {
    use super::*;

    /// Calculate the Black-Scholes call option price
    ///
    /// # Arguments
    /// * `spot` - Current asset price
    /// * `strike` - Strike price
    /// * `rate` - Risk-free rate
    /// * `volatility` - Asset volatility
    /// * `time` - Time to maturity
    pub fn black_scholes_call(
        spot: f64,
        strike: f64,
        rate: f64,
        volatility: f64,
        time: f64,
    ) -> SpecialResult<f64> {
        if spot <= 0.0 || strike <= 0.0 || volatility < 0.0 || time < 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Invalid option parameters".to_string(),
            ));
        }

        if time == 0.0 {
            return Ok((spot - strike).max(0.0));
        }

        let sqrt_t = time.sqrt();
        let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility * volatility) * time)
            / (volatility * sqrt_t);
        let d2 = d1 - volatility * sqrt_t;

        let n_d1 = 0.5 * (1.0 + erf::erf(d1 / 2.0_f64.sqrt()));
        let n_d2 = 0.5 * (1.0 + erf::erf(d2 / 2.0_f64.sqrt()));

        Ok(spot * n_d1 - strike * (-rate * time).exp() * n_d2)
    }

    /// Calculate the implied volatility using Newton-Raphson method
    ///
    /// # Arguments
    /// * `option_price` - Market price of the option
    /// * `spot` - Current asset price
    /// * `strike` - Strike price
    /// * `rate` - Risk-free rate
    /// * `time` - Time to maturity
    /// * `max_iter` - Maximum iterations
    #[allow(clippy::too_many_arguments)]
    pub fn implied_volatility(
        option_price: f64,
        spot: f64,
        strike: f64,
        rate: f64,
        time: f64,
        max_iter: usize,
    ) -> SpecialResult<f64> {
        if option_price <= 0.0 || spot <= 0.0 || strike <= 0.0 || time <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Invalid parameters".to_string(),
            ));
        }

        // Initial guess using Brenner-Subrahmanyam approximation
        let mut vol = (2.0 * PI / time).sqrt() * option_price / spot;

        for _ in 0..max_iter {
            let _price = black_scholes_call(spot, strike, rate, vol, time)?;
            let vega = black_scholes_vega(spot, strike, rate, vol, time)?;

            if vega.abs() < 1e-10 {
                break;
            }

            let diff = option_price - _price;
            if diff.abs() < 1e-8 {
                return Ok(vol);
            }

            vol += diff / vega;
            vol = vol.max(1e-8); // Ensure positive
        }

        Ok(vol)
    }

    /// Calculate Black-Scholes vega (sensitivity to volatility)
    fn black_scholes_vega(
        spot: f64,
        strike: f64,
        rate: f64,
        volatility: f64,
        time: f64,
    ) -> SpecialResult<f64> {
        let sqrt_t = time.sqrt();
        let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility * volatility) * time)
            / (volatility * sqrt_t);

        let phi_d1 = (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt();

        Ok(spot * phi_d1 * sqrt_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_particle_in_box() {
        let x = Array1::linspace(0.0, 1.0, 100);
        let psi = physics::particle_in_box_wavefunction(1, &x.view(), true).unwrap();

        // Check normalization using trapezoidal rule
        let dx = 1.0 / (100.0 - 1.0); // spacing between points
        let norm: f64 = psi
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let weight = if i == 0 || i == psi.len() - 1 {
                    0.5
                } else {
                    1.0
                };
                weight * p * p * dx
            })
            .sum();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_q_function() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let q = engineering::q_function(&x.view());

        // Q(0) should be 0.5
        assert!((q[0] - 0.5).abs() < 1e-10);

        // Q function should decrease
        assert!(q[1] < q[0]);
        assert!(q[2] < q[1]);
        assert!(q[3] < q[2]);
    }

    #[test]
    fn test_shannon_entropy() {
        let uniform = array![0.25, 0.25, 0.25, 0.25];
        let entropy = data_science::shannon_entropy(&uniform.view(), 2.0).unwrap();

        // Uniform distribution over 4 outcomes has entropy of 2 bits
        assert!((entropy - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_black_scholes() {
        // Test case with known result
        let call_price = finance::black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0).unwrap();

        // Price should be reasonable
        assert!(call_price > 0.0);
        assert!(call_price < 100.0);
    }

    #[test]
    fn test_gaussian_window() {
        let window = signal_processing::gaussian_window(21, 0.5);

        // Window should be symmetric
        for i in 0..10 {
            assert!((window[i] - window[20 - i]).abs() < 1e-10);
        }

        // Center should be maximum
        assert_eq!(window[10], 1.0);
    }
}

/// Bioinformatics and computational biology convenience functions
pub mod bioinformatics {
    use super::*;

    /// Calculate the Jukes-Cantor distance for DNA sequences
    ///
    /// # Arguments
    /// * `p` - Proportion of differing sites
    pub fn jukes_cantor_distance(p: f64) -> SpecialResult<f64> {
        if !(0.0..0.75).contains(&p) {
            return Err(crate::SpecialError::ValueError(
                "Proportion must be in [0, 0.75)".to_string(),
            ));
        }

        Ok(-0.75 * (1.0 - 4.0 * p / 3.0).ln())
    }

    /// Calculate the Kimura 2-parameter distance
    ///
    /// # Arguments
    /// * `p` - Proportion of transitions
    /// * `q` - Proportion of transversions
    pub fn kimura_distance(p: f64, q: f64) -> SpecialResult<f64> {
        if p < 0.0 || q < 0.0 || p + q >= 1.0 {
            return Err(crate::SpecialError::ValueError(
                "Invalid proportions".to_string(),
            ));
        }

        let a = 1.0 - 2.0 * p - q;
        let b = 1.0 - 2.0 * q;

        if a <= 0.0 || b <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Distance undefined for given parameters".to_string(),
            ));
        }

        Ok(-0.5 * a.ln() - 0.25 * b.ln())
    }

    /// Calculate the Michaelis-Menten enzyme kinetics
    ///
    /// # Arguments
    /// * `substrate` - Substrate concentration array
    /// * `vmax` - Maximum reaction velocity
    /// * `km` - Michaelis constant
    pub fn michaelis_menten(
        substrate: &ArrayView1<f64>,
        vmax: f64,
        km: f64,
    ) -> SpecialResult<Array1<f64>> {
        if vmax <= 0.0 || km <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Vmax and Km must be positive".to_string(),
            ));
        }

        let mut velocity = Array1::zeros(substrate.len());

        for (i, &s) in substrate.iter().enumerate() {
            if s < 0.0 {
                return Err(crate::SpecialError::ValueError(
                    "Substrate concentration must be non-negative".to_string(),
                ));
            }
            velocity[i] = vmax * s / (km + s);
        }

        Ok(velocity)
    }

    /// Hill equation for cooperative binding
    ///
    /// # Arguments
    /// * `ligand` - Ligand concentration array
    /// * `kd` - Dissociation constant
    /// * `n` - Hill coefficient
    pub fn hill_equation(ligand: &ArrayView1<f64>, kd: f64, n: f64) -> SpecialResult<Array1<f64>> {
        if kd <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Kd must be positive".to_string(),
            ));
        }

        let mut fraction = Array1::zeros(ligand.len());

        for (i, &l) in ligand.iter().enumerate() {
            if l < 0.0 {
                return Err(crate::SpecialError::ValueError(
                    "Ligand concentration must be non-negative".to_string(),
                ));
            }
            let ln_kd = kd.powf(n);
            fraction[i] = l.powf(n) / (ln_kd + l.powf(n));
        }

        Ok(fraction)
    }

    /// Calculate the logistic growth model (Verhulst equation)
    ///
    /// # Arguments
    /// * `t` - Time array
    /// * `k` - Carrying capacity
    /// * `r` - Growth rate
    /// * `p0` - Initial population
    pub fn logistic_growth(
        t: &ArrayView1<f64>,
        k: f64,
        r: f64,
        p0: f64,
    ) -> SpecialResult<Array1<f64>> {
        if k <= 0.0 || p0 < 0.0 || p0 > k {
            return Err(crate::SpecialError::ValueError(
                "Invalid population parameters".to_string(),
            ));
        }

        let mut population = Array1::zeros(t.len());
        let a = (k - p0) / p0;

        for (i, &ti) in t.iter().enumerate() {
            population[i] = k / (1.0 + a * (-r * ti).exp());
        }

        Ok(population)
    }
}

/// Geophysics and earth science convenience functions
pub mod geophysics {
    use super::*;

    /// Calculate seismic wave velocity using the Zoeppritz equation parameters
    ///
    /// # Arguments
    /// * `vp` - P-wave velocity
    /// * `vs` - S-wave velocity
    /// * `density` - Rock density
    pub fn acoustic_impedance(vp: f64, density: f64) -> SpecialResult<f64> {
        if vp <= 0.0 || density <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Velocity and density must be positive".to_string(),
            ));
        }

        Ok(vp * density)
    }

    /// Calculate the Richter magnitude from amplitude
    ///
    /// # Arguments
    /// * `amplitude` - Maximum amplitude in micrometers
    /// * `distance` - Distance from epicenter in km
    pub fn richter_magnitude(amplitude: f64, distance: f64) -> SpecialResult<f64> {
        if amplitude <= 0.0 || distance <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Amplitude and distance must be positive".to_string(),
            ));
        }

        // Empirical attenuation function
        let attenuation = -1.6 * distance.log10() + 3.0;
        Ok(amplitude.log10() + attenuation)
    }

    /// Calculate atmospheric pressure with altitude (barometric formula)
    ///
    /// # Arguments
    /// * `altitude` - Altitude array in meters
    /// * `sea_level_pressure` - Pressure at sea level in Pa
    pub fn barometric_pressure(
        altitude: &ArrayView1<f64>,
        sea_level_pressure: f64,
    ) -> SpecialResult<Array1<f64>> {
        if sea_level_pressure <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Sea level pressure must be positive".to_string(),
            ));
        }

        let g = 9.80665; // m/s²
        let m = 0.0289644; // kg/mol
        let r = 8.31447; // J/(mol·K)
        let t0 = 288.15; // K
        let l = 0.0065; // K/m

        let mut _pressure = Array1::zeros(altitude.len());

        for (i, &h) in altitude.iter().enumerate() {
            if !(-500.0..=11000.0).contains(&h) {
                return Err(crate::SpecialError::ValueError(
                    "Altitude out of troposphere range".to_string(),
                ));
            }

            let temp_ratio = 1.0 - l * h / t0;
            _pressure[i] = sea_level_pressure * temp_ratio.powf(g * m / (r * l));
        }

        Ok(_pressure)
    }

    /// Calculate geostrophic wind speed
    ///
    /// # Arguments
    /// * `pressure_gradient` - Horizontal pressure gradient in Pa/m
    /// * `latitude` - Latitude in degrees
    /// * `density` - Air density in kg/m³
    pub fn geostrophic_wind(
        pressure_gradient: f64,
        latitude: f64,
        density: f64,
    ) -> SpecialResult<f64> {
        if density <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Density must be positive".to_string(),
            ));
        }

        if latitude.abs() >= 90.0 {
            return Err(crate::SpecialError::ValueError(
                "Invalid latitude".to_string(),
            ));
        }

        let omega = 7.2921159e-5; // Earth rotation rate (rad/s)
        let f = 2.0 * omega * (latitude * PI / 180.0).sin(); // Coriolis parameter

        if f.abs() < 1e-10 {
            return Err(crate::SpecialError::ValueError(
                "Geostrophic balance undefined at equator".to_string(),
            ));
        }

        Ok(pressure_gradient.abs() / (density * f.abs()))
    }
}

/// Chemistry and materials science convenience functions
pub mod chemistry {
    use super::*;

    /// Calculate the Arrhenius reaction rate
    ///
    /// # Arguments
    /// * `temperature` - Temperature array in Kelvin
    /// * `ea` - Activation energy in J/mol
    /// * `a` - Pre-exponential factor
    pub fn arrhenius_rate(
        temperature: &ArrayView1<f64>,
        ea: f64,
        a: f64,
    ) -> SpecialResult<Array1<f64>> {
        if ea < 0.0 || a <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Invalid Arrhenius parameters".to_string(),
            ));
        }

        let r = 8.314462618; // J/(mol·K)
        let mut rate = Array1::zeros(temperature.len());

        for (i, &t) in temperature.iter().enumerate() {
            if t <= 0.0 {
                return Err(crate::SpecialError::ValueError(
                    "Temperature must be positive".to_string(),
                ));
            }
            rate[i] = a * (-ea / (r * t)).exp();
        }

        Ok(rate)
    }

    /// Calculate the Debye-Hückel activity coefficient
    ///
    /// # Arguments
    /// * `ionic_strength` - Ionic strength in mol/L
    /// * `charge` - Ion charge
    /// * `temperature` - Temperature in Kelvin
    pub fn debye_huckel_activity(
        ionic_strength: f64,
        charge: f64,
        temperature: f64,
    ) -> SpecialResult<f64> {
        if ionic_strength < 0.0 || temperature <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Invalid parameters".to_string(),
            ));
        }

        // Debye-Hückel limiting law constant
        let a = 0.509 * (temperature / 298.15).powf(-1.5);
        let log_gamma =
            -a * charge * charge * ionic_strength.sqrt() / (1.0 + ionic_strength.sqrt());

        Ok(10.0_f64.powf(log_gamma))
    }

    /// Calculate the Langmuir adsorption isotherm
    ///
    /// # Arguments
    /// * `pressure` - Pressure array in Pa
    /// * `k` - Adsorption constant
    pub fn langmuir_isotherm(pressure: &ArrayView1<f64>, k: f64) -> SpecialResult<Array1<f64>> {
        if k <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Adsorption constant must be positive".to_string(),
            ));
        }

        let mut coverage = Array1::zeros(pressure.len());

        for (i, &p) in pressure.iter().enumerate() {
            if p < 0.0 {
                return Err(crate::SpecialError::ValueError(
                    "Pressure must be non-negative".to_string(),
                ));
            }
            coverage[i] = k * p / (1.0 + k * p);
        }

        Ok(coverage)
    }

    /// Calculate the van der Waals equation of state
    ///
    /// # Arguments
    /// * `pressure` - Pressure in Pa
    /// * `volume` - Molar volume in m³/mol
    /// * `temperature` - Temperature in Kelvin
    /// * `a` - van der Waals constant a
    /// * `b` - van der Waals constant b
    pub fn van_der_waals_pressure(
        volume: f64,
        temperature: f64,
        a: f64,
        b: f64,
    ) -> SpecialResult<f64> {
        if volume <= b || temperature <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Invalid state parameters".to_string(),
            ));
        }

        let r = 8.314462618; // J/(mol·K)
        Ok(r * temperature / (volume - b) - a / (volume * volume))
    }
}

/// Astronomy and astrophysics convenience functions
pub mod astronomy {
    use super::*;

    /// Calculate the Planck function for blackbody radiation
    ///
    /// # Arguments
    /// * `wavelength` - Wavelength array in meters
    /// * `temperature` - Temperature in Kelvin
    pub fn planck_function(
        wavelength: &ArrayView1<f64>,
        temperature: f64,
    ) -> SpecialResult<Array1<f64>> {
        physics::planck_radiation(wavelength, temperature)
    }

    /// Calculate the Saha ionization equation
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    /// * `electron_density` - Electron density in m⁻³
    /// * `ionizationenergy` - Ionization energy in eV
    pub fn saha_equation(
        temperature: f64,
        electron_density: f64,
        ionizationenergy: f64,
    ) -> SpecialResult<f64> {
        if temperature <= 0.0 || electron_density <= 0.0 || ionizationenergy <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "All parameters must be positive".to_string(),
            ));
        }

        let k_ev = 8.617333262e-5; // Boltzmann constant in eV/K
        let h = 6.62607015e-34; // Planck constant
        let m_e = 9.1093837015e-31; // Electron mass

        // Thermal de Broglie wavelength
        let lambda_th = h / (2.0 * PI * m_e * k_ev * temperature * 1.602e-19).sqrt();

        // Saha equation
        let g_ratio = 2.0; // Statistical weight ratio (simplified)
        let exponent = -ionizationenergy / (k_ev * temperature);

        Ok(g_ratio * electron_density * lambda_th.powi(3) * exponent.exp())
    }

    /// Calculate stellar luminosity from radius and temperature
    ///
    /// # Arguments
    /// * `radius` - Stellar radius in solar radii
    /// * `temperature` - Effective temperature in Kelvin
    pub fn stellar_luminosity(radius: f64, temperature: f64) -> SpecialResult<f64> {
        if radius <= 0.0 || temperature <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "Radius and temperature must be positive".to_string(),
            ));
        }

        let sigma = 5.670374419e-8; // Stefan-Boltzmann constant
        let r_sun = 6.96e8; // Solar radius in meters
        let l_sun = 3.828e26; // Solar luminosity in watts

        let area = 4.0 * PI * (radius * r_sun).powi(2);
        let luminosity = area * sigma * temperature.powi(4);

        Ok(luminosity / l_sun) // Return in solar luminosities
    }

    /// Calculate the Jeans mass for gravitational collapse
    ///
    /// # Arguments
    /// * `temperature` - Temperature in Kelvin
    /// * `density` - Mass density in kg/m³
    /// * `mean_molecular_weight` - Mean molecular weight
    pub fn jeans_mass(
        temperature: f64,
        density: f64,
        mean_molecular_weight: f64,
    ) -> SpecialResult<f64> {
        if temperature <= 0.0 || density <= 0.0 || mean_molecular_weight <= 0.0 {
            return Err(crate::SpecialError::ValueError(
                "All parameters must be positive".to_string(),
            ));
        }

        let k_b = 1.380649e-23; // Boltzmann constant
        let g = 6.67430e-11; // Gravitational constant
        let m_p = 1.67262192369e-27; // Proton mass

        let cs = (k_b * temperature / (mean_molecular_weight * m_p)).sqrt();
        let jeans_length = cs * (PI / (g * density)).sqrt();
        let mass = (4.0 / 3.0) * PI * (jeans_length / 2.0).powi(3) * density;

        Ok(mass / 1.9885e30) // Return in solar masses
    }

    /// Calculate redshift from recession velocity
    ///
    /// # Arguments
    /// * `velocity` - Recession velocity in m/s
    /// * `relativistic` - Use relativistic formula if true
    pub fn velocity_to_redshift(velocity: f64, relativistic: bool) -> SpecialResult<f64> {
        let c = 299792458.0; // Speed of light

        if velocity >= c {
            return Err(crate::SpecialError::ValueError(
                "Velocity cannot exceed speed of light".to_string(),
            ));
        }

        if relativistic {
            let beta = velocity / c;
            Ok(((1.0 + beta) / (1.0 - beta)).sqrt() - 1.0)
        } else {
            Ok(velocity / c)
        }
    }
}
