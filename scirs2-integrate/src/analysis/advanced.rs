//! Advanced analysis tools for complex dynamical systems
//!
//! This module contains sophisticated analysis methods for dynamical systems including:
//! - Poincaré section analysis for periodic orbit detection
//! - Lyapunov exponent calculation for chaos detection
//! - Fractal dimension analysis for strange attractors
//! - Recurrence analysis for pattern detection
//! - Continuation analysis for bifurcation detection
//! - Monodromy analysis for periodic orbit stability

use crate::analysis::types::*;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::random::Rng;
use std::collections::HashSet;

/// Poincaré section analysis for periodic orbit detection
pub struct PoincareAnalyzer {
    /// Section definition (hyperplane normal)
    pub section_normal: Array1<f64>,
    /// Point on the section
    pub section_point: Array1<f64>,
    /// Crossing direction (1: positive, -1: negative, 0: both)
    pub crossing_direction: i8,
    /// Tolerance for section crossing detection
    pub crossing_tolerance: f64,
}

impl PoincareAnalyzer {
    /// Create a new Poincaré analyzer
    pub fn new(
        section_normal: Array1<f64>,
        section_point: Array1<f64>,
        crossing_direction: i8,
    ) -> Self {
        Self {
            section_normal,
            section_point,
            crossing_direction,
            crossing_tolerance: 1e-8,
        }
    }

    /// Analyze trajectory to find Poincaré map
    pub fn analyze_trajectory(
        &self,
        trajectory: &[Array1<f64>],
        times: &[f64],
    ) -> IntegrateResult<PoincareMap> {
        let mut crossings = Vec::new();
        let mut crossing_times = Vec::new();

        for i in 1..trajectory.len() {
            if let Some((crossing_point, crossing_time)) =
                self.detect_crossing(&trajectory[i - 1], &trajectory[i], times[i - 1], times[i])?
            {
                crossings.push(crossing_point);
                crossing_times.push(crossing_time);
            }
        }

        // Compute return map if sufficient crossings
        let return_map = if crossings.len() > 1 {
            Some(self.compute_return_map(&crossings)?)
        } else {
            None
        };

        // Detect periodic orbits
        let periodic_orbits = self.detect_periodic_orbits(&crossings)?;

        Ok(PoincareMap {
            crossings,
            crossing_times,
            return_map,
            periodic_orbits,
            section_normal: self.section_normal.clone(),
            section_point: self.section_point.clone(),
        })
    }

    /// Detect crossing of Poincaré section
    fn detect_crossing(
        &self,
        point1: &Array1<f64>,
        point2: &Array1<f64>,
        t1: f64,
        t2: f64,
    ) -> IntegrateResult<Option<(Array1<f64>, f64)>> {
        // Calculate distances from section
        let d1 = self.distance_from_section(point1);
        let d2 = self.distance_from_section(point2);

        // Check for crossing
        let crossed = match self.crossing_direction {
            1 => d1 < 0.0 && d2 > 0.0,  // Positive crossing
            -1 => d1 > 0.0 && d2 < 0.0, // Negative crossing
            0 => d1 * d2 < 0.0,         // Any crossing
            _ => false,
        };

        if !crossed {
            return Ok(None);
        }

        // Interpolate crossing point
        let alpha = d1.abs() / (d1.abs() + d2.abs());
        let crossing_point = (1.0 - alpha) * point1 + alpha * point2;
        let crossing_time = (1.0 - alpha) * t1 + alpha * t2;

        Ok(Some((crossing_point, crossing_time)))
    }

    /// Calculate distance from point to section
    fn distance_from_section(&self, point: &Array1<f64>) -> f64 {
        let relative_pos = point - &self.section_point;
        relative_pos.dot(&self.section_normal)
    }

    /// Compute return map from crossings
    fn compute_return_map(&self, crossings: &[Array1<f64>]) -> IntegrateResult<ReturnMap> {
        let mut current_points = Vec::new();
        let mut next_points = Vec::new();

        for i in 0..crossings.len() - 1 {
            // Project points onto section (remove normal component)
            let current_projected = self.project_to_section(&crossings[i]);
            let next_projected = self.project_to_section(&crossings[i + 1]);

            current_points.push(current_projected);
            next_points.push(next_projected);
        }

        Ok(ReturnMap {
            current_points,
            next_points,
        })
    }

    /// Project point onto Poincaré section
    fn project_to_section(&self, point: &Array1<f64>) -> Array1<f64> {
        let distance = self.distance_from_section(point);
        point - distance * &self.section_normal
    }

    /// Detect periodic orbits from crossings
    fn detect_periodic_orbits(
        &self,
        crossings: &[Array1<f64>],
    ) -> IntegrateResult<Vec<PeriodicOrbit>> {
        let mut periodic_orbits = Vec::new();
        let tolerance = 1e-6;

        // Look for approximate returns to previous crossing points
        for i in 0..crossings.len() {
            for j in (i + 2)..crossings.len() {
                let distance = self.euclidean_distance(&crossings[i], &crossings[j]);
                if distance < tolerance {
                    // Found potential periodic orbit
                    let period_length = j - i;
                    let representative_point = crossings[i].clone();

                    // Verify periodicity by checking intermediate points
                    let mut is_periodic = true;
                    for k in 1..period_length {
                        if i + k < crossings.len() && j + k < crossings.len() {
                            let dist =
                                self.euclidean_distance(&crossings[i + k], &crossings[j + k]);
                            if dist > tolerance {
                                is_periodic = false;
                                break;
                            }
                        }
                    }

                    if is_periodic {
                        // Calculate approximate period in time
                        let period = (j - i) as f64; // Simplified period estimate

                        periodic_orbits.push(PeriodicOrbit {
                            representative_point,
                            period,
                            stability: StabilityType::Stable, // Would need proper analysis
                            floquet_multipliers: vec![],      // Would need computation
                        });
                    }
                }
            }
        }

        Ok(periodic_orbits)
    }

    fn euclidean_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        (a - b).iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

/// Poincaré map data structure
#[derive(Debug, Clone)]
pub struct PoincareMap {
    /// Section crossing points
    pub crossings: Vec<Array1<f64>>,
    /// Times of crossings
    pub crossing_times: Vec<f64>,
    /// Return map data
    pub return_map: Option<ReturnMap>,
    /// Detected periodic orbits
    pub periodic_orbits: Vec<PeriodicOrbit>,
    /// Section normal vector
    pub section_normal: Array1<f64>,
    /// Point on section
    pub section_point: Array1<f64>,
}

/// Return map data
#[derive(Debug, Clone)]
pub struct ReturnMap {
    /// Current points
    pub current_points: Vec<Array1<f64>>,
    /// Next points in return map
    pub next_points: Vec<Array1<f64>>,
}

/// Lyapunov exponent calculator for chaos detection
pub struct LyapunovCalculator {
    /// Number of exponents to calculate
    pub n_exponents: usize,
    /// Perturbation magnitude for tangent vectors
    pub perturbation_magnitude: f64,
    /// Renormalization interval
    pub renormalization_interval: usize,
    /// Integration time step
    pub dt: f64,
}

impl LyapunovCalculator {
    /// Create new Lyapunov calculator
    pub fn new(nexponents: usize, dt: f64) -> Self {
        Self {
            n_exponents: nexponents,
            perturbation_magnitude: 1e-8,
            renormalization_interval: 100,
            dt,
        }
    }

    /// Calculate Lyapunov exponents using tangent space evolution
    pub fn calculate_lyapunov_exponents<F>(
        &self,
        system: F,
        initial_state: &Array1<f64>,
        total_time: f64,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
    {
        let n_steps = (total_time / self.dt) as usize;
        let dim = initial_state.len();

        if self.n_exponents > dim {
            return Err(IntegrateError::ValueError(
                "Number of exponents cannot exceed system dimension".to_string(),
            ));
        }

        // Initialize state and tangent vectors
        let mut state = initial_state.clone();
        let mut tangent_vectors = self.initialize_tangent_vectors(dim)?;
        let mut lyapunov_sums = Array1::zeros(self.n_exponents);

        // Main integration loop
        for step in 0..n_steps {
            // Evolve main trajectory
            let derivative = system(&state);
            state += &(derivative * self.dt);

            // Evolve tangent vectors
            for i in 0..self.n_exponents {
                let jacobian = self.compute_jacobian(&system, &state)?;
                let tangent_derivative = jacobian.dot(&tangent_vectors.column(i));
                let old_tangent = tangent_vectors.column(i).to_owned();
                tangent_vectors
                    .column_mut(i)
                    .assign(&(&old_tangent + &(tangent_derivative * self.dt)));
            }

            // Renormalization to prevent overflow
            if step % self.renormalization_interval == 0 && step > 0 {
                let (q, r) = self.qr_decomposition(&tangent_vectors)?;
                tangent_vectors = q;

                // Add to Lyapunov sum
                for i in 0..self.n_exponents {
                    lyapunov_sums[i] += r[[i, i]].abs().ln();
                }
            }
        }

        // Final normalization
        let lyapunov_exponents = lyapunov_sums / total_time;

        Ok(lyapunov_exponents)
    }

    /// Initialize orthonormal tangent vectors
    fn initialize_tangent_vectors(&self, dim: usize) -> IntegrateResult<Array2<f64>> {
        let mut vectors = Array2::zeros((dim, self.n_exponents));

        // Initialize with random vectors
        let mut rng = scirs2_core::random::rng();
        for i in 0..self.n_exponents {
            for j in 0..dim {
                vectors[[j, i]] = rng.random::<f64>() - 0.5;
            }
        }

        // Gram-Schmidt orthogonalization
        for i in 0..self.n_exponents {
            // Orthogonalize against previous vectors
            for j in 0..i {
                let projection = vectors.column(i).dot(&vectors.column(j));
                let col_j = vectors.column(j).to_owned();
                let mut col_i = vectors.column_mut(i);
                col_i -= &(projection * &col_j);
            }

            // Normalize
            let norm = vectors.column(i).iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                vectors.column_mut(i).mapv_inplace(|x| x / norm);
            }
        }

        Ok(vectors)
    }

    /// Compute Jacobian matrix using finite differences
    fn compute_jacobian<F>(&self, system: &F, state: &Array1<f64>) -> IntegrateResult<Array2<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let dim = state.len();
        let mut jacobian = Array2::zeros((dim, dim));
        let h = self.perturbation_magnitude;

        for j in 0..dim {
            let mut state_plus = state.clone();
            let mut state_minus = state.clone();

            state_plus[j] += h;
            state_minus[j] -= h;

            let f_plus = system(&state_plus);
            let f_minus = system(&state_minus);

            for i in 0..dim {
                jacobian[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * h);
            }
        }

        Ok(jacobian)
    }

    /// QR decomposition using Gram-Schmidt
    fn qr_decomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> IntegrateResult<(Array2<f64>, Array2<f64>)> {
        let (m, n) = matrix.dim();
        let mut q = matrix.clone();
        let mut r = Array2::zeros((n, n));

        for j in 0..n {
            // Orthogonalize against previous columns
            for i in 0..j {
                r[[i, j]] = q.column(j).dot(&q.column(i));
                let col_i = q.column(i).to_owned();
                let mut col_j = q.column_mut(j);
                col_j -= &(r[[i, j]] * &col_i);
            }

            // Normalize
            r[[j, j]] = q.column(j).iter().map(|&x| x * x).sum::<f64>().sqrt();
            if r[[j, j]] > 1e-12 {
                q.column_mut(j).mapv_inplace(|x| x / r[[j, j]]);
            }
        }

        Ok((q, r))
    }

    /// Calculate largest Lyapunov exponent using Wolf's algorithm
    pub fn calculate_largest_lyapunov_exponent<F>(
        &self,
        system: F,
        initial_state: &Array1<f64>,
        total_time: f64,
        min_separation: f64,
        max_separation: f64,
    ) -> IntegrateResult<f64>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync,
    {
        let n_steps = (total_time / self.dt) as usize;
        let dim = initial_state.len();

        // Initialize reference trajectory
        let mut reference_state = initial_state.clone();

        // Initialize nearby trajectory with small perturbation
        let mut nearby_state = initial_state.clone();
        nearby_state[0] += self.perturbation_magnitude;

        let mut lyapunov_sum = 0.0;
        let mut n_rescales = 0;

        for _step in 0..n_steps {
            // Evolve both trajectories
            let ref_derivative = system(&reference_state);
            let nearby_derivative = system(&nearby_state);

            reference_state += &(ref_derivative * self.dt);
            nearby_state += &(nearby_derivative * self.dt);

            // Calculate separation
            let separation_vector = &nearby_state - &reference_state;
            let separation = separation_vector.iter().map(|&x| x * x).sum::<f64>().sqrt();

            // Check if rescaling is needed
            if (separation > max_separation || separation < min_separation) && separation > 1e-15 {
                // Add to Lyapunov sum
                lyapunov_sum += separation.ln();
                n_rescales += 1;

                // Rescale the separation vector
                let scale_factor = self.perturbation_magnitude / separation;
                nearby_state = &reference_state + &(separation_vector * scale_factor);
            }
        }

        if n_rescales > 0 {
            Ok(lyapunov_sum / total_time)
        } else {
            Ok(0.0) // No chaos detected
        }
    }

    /// Estimate Lyapunov exponent from time series using delay embedding
    pub fn estimate_lyapunov_from_timeseries(
        &self,
        timeseries: &Array1<f64>,
        embedding_dimension: usize,
        delay: usize,
    ) -> IntegrateResult<f64> {
        let n = timeseries.len();
        if n < embedding_dimension * delay + 1 {
            return Err(IntegrateError::ValueError(
                "Time series too short for embedding".to_string(),
            ));
        }

        // Create delay embedding
        let n_vectors = n - embedding_dimension * delay;
        let mut embedded_vectors = Vec::new();

        for i in 0..n_vectors {
            let mut vector = Array1::zeros(embedding_dimension);
            for j in 0..embedding_dimension {
                vector[j] = timeseries[i + j * delay];
            }
            embedded_vectors.push(vector);
        }

        // Calculate nearest neighbor distances and their evolution
        let mut lyapunov_sum = 0.0;
        let mut count = 0;
        let min_time_separation = 2 * delay; // Avoid temporal correlations

        for i in 0..embedded_vectors.len() - 1 {
            // Find nearest neighbor with sufficient time separation
            let mut min_distance = f64::INFINITY;
            let mut nearest_index = None;

            for j in 0..embedded_vectors.len() - 1 {
                if (j as i32 - i as i32).abs() >= min_time_separation as i32 {
                    let distance =
                        self.euclidean_distance_arrays(&embedded_vectors[i], &embedded_vectors[j]);
                    if distance < min_distance && distance > 1e-12 {
                        min_distance = distance;
                        nearest_index = Some(j);
                    }
                }
            }

            if let Some(j) = nearest_index {
                // Calculate distance after one time step
                if i + 1 < embedded_vectors.len() && j + 1 < embedded_vectors.len() {
                    let initial_distance = min_distance;
                    let final_distance = self.euclidean_distance_arrays(
                        &embedded_vectors[i + 1],
                        &embedded_vectors[j + 1],
                    );

                    if final_distance > 1e-12 && initial_distance > 1e-12 {
                        lyapunov_sum += (final_distance / initial_distance).ln();
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            Ok(lyapunov_sum / (count as f64))
        } else {
            Ok(0.0)
        }
    }

    /// Helper function for distance calculation between arrays
    fn euclidean_distance_arrays(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        (a - b).iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// Calculate Lyapunov spectrum with error estimates
    pub fn calculate_lyapunov_spectrum_with_errors<F>(
        &self,
        system: F,
        initial_state: &Array1<f64>,
        total_time: f64,
        n_trials: usize,
    ) -> IntegrateResult<(Array1<f64>, Array1<f64>)>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + Clone,
    {
        let dim = initial_state.len();
        let n_exponents = self.n_exponents.min(dim);

        let mut all_exponents = Array2::zeros((n_trials, n_exponents));

        // Calculate Lyapunov exponents multiple times with slightly different initial conditions
        let mut rng = scirs2_core::random::rng();

        for trial in 0..n_trials {
            // Add small random perturbation to initial state
            let mut perturbed_initial = initial_state.clone();
            for i in 0..dim {
                perturbed_initial[i] += (rng.random::<f64>() - 0.5) * 1e-6;
            }

            let exponents =
                self.calculate_lyapunov_exponents(system.clone(), &perturbed_initial, total_time)?;

            for i in 0..n_exponents {
                all_exponents[[trial, i]] = exponents[i];
            }
        }

        // Calculate mean and standard deviation
        let mut means = Array1::zeros(n_exponents);
        let mut std_devs = Array1::zeros(n_exponents);

        for i in 0..n_exponents {
            let column = all_exponents.column(i);
            let mean = column.sum() / n_trials as f64;
            let variance =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_trials as f64;

            means[i] = mean;
            std_devs[i] = variance.sqrt();
        }

        Ok((means, std_devs))
    }
}

/// Fractal dimension analyzer for strange attractors
pub struct FractalAnalyzer {
    /// Range of scales to analyze
    pub scale_range: (f64, f64),
    /// Number of scale points
    pub n_scales: usize,
    /// Box-counting parameters
    pub box_counting_method: BoxCountingMethod,
}

/// Box counting methods
#[derive(Debug, Clone, Copy)]
pub enum BoxCountingMethod {
    /// Standard box counting
    Standard,
    /// Differential box counting
    Differential,
    /// Correlation dimension
    Correlation,
}

impl Default for FractalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl FractalAnalyzer {
    /// Create new fractal analyzer
    pub fn new() -> Self {
        Self {
            scale_range: (1e-4, 1e-1),
            n_scales: 20,
            box_counting_method: BoxCountingMethod::Standard,
        }
    }

    /// Calculate fractal dimension of attractor
    pub fn calculate_fractal_dimension(
        &self,
        attractor_points: &[Array1<f64>],
    ) -> IntegrateResult<FractalDimension> {
        match self.box_counting_method {
            BoxCountingMethod::Standard => self.box_counting_dimension(attractor_points),
            BoxCountingMethod::Differential => self.differential_box_counting(attractor_points),
            BoxCountingMethod::Correlation => self.correlation_dimension(attractor_points),
        }
    }

    /// Standard box counting dimension
    fn box_counting_dimension(&self, points: &[Array1<f64>]) -> IntegrateResult<FractalDimension> {
        if points.is_empty() {
            return Err(IntegrateError::ValueError(
                "Cannot analyze empty point set".to_string(),
            ));
        }

        let dim = points[0].len();

        // Find bounding box
        let (min_bounds, max_bounds) = self.find_bounding_box(points);
        let domain_size = max_bounds
            .iter()
            .zip(min_bounds.iter())
            .map(|(&max, &min)| max - min)
            .fold(0.0f64, |acc, x| acc.max(x));

        let mut scales = Vec::new();
        let mut counts = Vec::new();

        // Analyze different scales
        for i in 0..self.n_scales {
            let t = i as f64 / (self.n_scales - 1) as f64;
            let scale = self.scale_range.0 * (self.scale_range.1 / self.scale_range.0).powf(t);

            let box_size = scale * domain_size;
            let count = self.count_occupied_boxes(points, &min_bounds, box_size, dim)?;

            scales.push(scale);
            counts.push(count as f64);
        }

        // Linear regression on log-log plot
        // For box counting: dimension = -slope of log(count) vs log(scale)
        let slope = self.calculate_slope_from_log_data(&scales, &counts)?;
        let dimension = -slope;

        let r_squared = self.calculate_r_squared(&scales, &counts, slope)?;

        Ok(FractalDimension {
            dimension,
            method: self.box_counting_method,
            scales,
            counts,
            r_squared,
        })
    }

    /// Differential box counting for higher accuracy
    fn differential_box_counting(
        &self,
        points: &[Array1<f64>],
    ) -> IntegrateResult<FractalDimension> {
        // Simplified implementation - would need full differential box counting
        self.box_counting_dimension(points)
    }

    /// Correlation dimension using Grassberger-Procaccia algorithm
    fn correlation_dimension(&self, points: &[Array1<f64>]) -> IntegrateResult<FractalDimension> {
        let n_points = points.len();
        let mut scales = Vec::new();
        let mut correlations = Vec::new();

        for i in 0..self.n_scales {
            let t = i as f64 / (self.n_scales - 1) as f64;
            let r = self.scale_range.0 * (self.scale_range.1 / self.scale_range.0).powf(t);

            let mut count = 0;
            for i in 0..n_points {
                for j in i + 1..n_points {
                    let distance = self.euclidean_distance(&points[i], &points[j]);
                    if distance < r {
                        count += 1;
                    }
                }
            }

            let correlation = 2.0 * count as f64 / (n_points * (n_points - 1)) as f64;

            scales.push(r);
            correlations.push(correlation);
        }

        // Filter out zero correlations for log calculation
        let filtered_data: Vec<(f64, f64)> = scales
            .iter()
            .zip(correlations.iter())
            .filter(|(_, &c)| c > 0.0)
            .map(|(&s, &c)| (s, c))
            .collect();

        if filtered_data.len() < 2 {
            return Err(IntegrateError::ComputationError(
                "Insufficient data for correlation dimension calculation".to_string(),
            ));
        }

        let filtered_scales: Vec<f64> = filtered_data.iter().map(|(s, _)| *s).collect();
        let filtered_correlations: Vec<f64> = filtered_data.iter().map(|(_, c)| *c).collect();

        let dimension =
            self.calculate_slope_from_log_data(&filtered_scales, &filtered_correlations)?;

        Ok(FractalDimension {
            dimension,
            method: BoxCountingMethod::Correlation,
            scales,
            counts: correlations,
            r_squared: self.calculate_r_squared(
                &filtered_scales,
                &filtered_correlations,
                dimension,
            )?,
        })
    }

    /// Helper functions
    fn find_bounding_box(&self, points: &[Array1<f64>]) -> (Array1<f64>, Array1<f64>) {
        let dim = points[0].len();
        let mut min_bounds = Array1::from_elem(dim, f64::INFINITY);
        let mut max_bounds = Array1::from_elem(dim, f64::NEG_INFINITY);

        for point in points {
            for i in 0..dim {
                min_bounds[i] = min_bounds[i].min(point[i]);
                max_bounds[i] = max_bounds[i].max(point[i]);
            }
        }

        (min_bounds, max_bounds)
    }

    fn count_occupied_boxes(
        &self,
        points: &[Array1<f64>],
        min_bounds: &Array1<f64>,
        box_size: f64,
        dim: usize,
    ) -> IntegrateResult<usize> {
        let mut occupied_boxes = HashSet::new();

        for point in points {
            let mut box_index = Vec::with_capacity(dim);
            for i in 0..dim {
                let index = ((point[i] - min_bounds[i]) / box_size).floor() as i64;
                box_index.push(index);
            }
            occupied_boxes.insert(box_index);
        }

        Ok(occupied_boxes.len())
    }

    fn calculate_slope_from_log_data(
        &self,
        x_data: &[f64],
        y_data: &[f64],
    ) -> IntegrateResult<f64> {
        if x_data.len() != y_data.len() || x_data.len() < 2 {
            return Err(IntegrateError::ValueError(
                "Insufficient data for slope calculation".to_string(),
            ));
        }

        let n = x_data.len() as f64;
        let log_x: Vec<f64> = x_data.iter().map(|&x| x.ln()).collect();
        let log_y: Vec<f64> = y_data.iter().map(|&y| y.ln()).collect();

        let sum_x: f64 = log_x.iter().sum();
        let sum_y: f64 = log_y.iter().sum();
        let sum_xy: f64 = log_x.iter().zip(log_y.iter()).map(|(&x, &y)| x * y).sum();
        let sum_xx: f64 = log_x.iter().map(|&x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        Ok(slope)
    }

    fn calculate_r_squared(
        &self,
        x_data: &[f64],
        y_data: &[f64],
        slope: f64,
    ) -> IntegrateResult<f64> {
        let log_x: Vec<f64> = x_data.iter().map(|&x| x.ln()).collect();
        let log_y: Vec<f64> = y_data.iter().map(|&y| y.ln()).collect();

        let mean_y = log_y.iter().sum::<f64>() / log_y.len() as f64;
        let mean_x = log_x.iter().sum::<f64>() / log_x.len() as f64;
        let intercept = mean_y - slope * mean_x;

        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for i in 0..log_y.len() {
            let y_pred = slope * log_x[i] + intercept;
            ss_res += (log_y[i] - y_pred).powi(2);
            ss_tot += (log_y[i] - mean_y).powi(2);
        }

        let r_squared = 1.0 - (ss_res / ss_tot);
        Ok(r_squared)
    }

    fn euclidean_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Fractal dimension result
#[derive(Debug, Clone)]
pub struct FractalDimension {
    /// Calculated dimension
    pub dimension: f64,
    /// Method used
    pub method: BoxCountingMethod,
    /// Scale values used
    pub scales: Vec<f64>,
    /// Count/correlation values
    pub counts: Vec<f64>,
    /// Quality of fit (R²)
    pub r_squared: f64,
}

/// Recurrence analysis for detecting patterns and periodicities
pub struct RecurrenceAnalyzer {
    /// Recurrence threshold
    pub threshold: f64,
    /// Embedding dimension for delay coordinate embedding
    pub embedding_dimension: usize,
    /// Time delay for embedding
    pub time_delay: usize,
    /// Distance metric
    pub distance_metric: DistanceMetric,
}

/// Distance metrics for recurrence analysis
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Maximum (Chebyshev) distance
    Maximum,
    /// Manhattan distance
    Manhattan,
}

impl RecurrenceAnalyzer {
    /// Create new recurrence analyzer
    pub fn new(threshold: f64, embedding_dimension: usize, timedelay: usize) -> Self {
        Self {
            threshold,
            embedding_dimension,
            time_delay: timedelay,
            distance_metric: DistanceMetric::Euclidean,
        }
    }

    /// Perform recurrence analysis
    pub fn analyze_recurrence(&self, timeseries: &[f64]) -> IntegrateResult<RecurrenceAnalysis> {
        // Create delay coordinate embedding
        let embedded_vectors = self.create_embedding(timeseries)?;

        // Compute recurrence matrix
        let recurrence_matrix = self.compute_recurrence_matrix(&embedded_vectors)?;

        // Calculate recurrence quantification measures
        let rqa_measures = self.calculate_rqa_measures(&recurrence_matrix)?;

        Ok(RecurrenceAnalysis {
            recurrence_matrix,
            embedded_vectors,
            rqa_measures,
            threshold: self.threshold,
            embedding_dimension: self.embedding_dimension,
            time_delay: self.time_delay,
        })
    }

    /// Create delay coordinate embedding
    fn create_embedding(&self, timeseries: &[f64]) -> IntegrateResult<Vec<Array1<f64>>> {
        let n = timeseries.len();
        let embedded_length = n - (self.embedding_dimension - 1) * self.time_delay;

        if embedded_length <= 0 {
            return Err(IntegrateError::ValueError(
                "Time series too short for given embedding parameters".to_string(),
            ));
        }

        let mut embedded_vectors = Vec::with_capacity(embedded_length);

        for i in 0..embedded_length {
            let mut vector = Array1::zeros(self.embedding_dimension);
            for j in 0..self.embedding_dimension {
                vector[j] = timeseries[i + j * self.time_delay];
            }
            embedded_vectors.push(vector);
        }

        Ok(embedded_vectors)
    }

    /// Compute recurrence matrix
    fn compute_recurrence_matrix(&self, vectors: &[Array1<f64>]) -> IntegrateResult<Array2<bool>> {
        let n = vectors.len();
        let mut recurrence_matrix = Array2::from_elem((n, n), false);

        for i in 0..n {
            for j in 0..n {
                let distance = self.calculate_distance(&vectors[i], &vectors[j]);
                recurrence_matrix[[i, j]] = distance <= self.threshold;
            }
        }

        Ok(recurrence_matrix)
    }

    /// Calculate distance between vectors
    fn calculate_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        match self.distance_metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt(),
            DistanceMetric::Maximum => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .fold(0.0, f64::max),
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum(),
        }
    }

    /// Calculate Recurrence Quantification Analysis measures
    fn calculate_rqa_measures(
        &self,
        recurrence_matrix: &Array2<bool>,
    ) -> IntegrateResult<RQAMeasures> {
        let n = recurrence_matrix.nrows();
        let total_points = (n * n) as f64;

        // Recurrence rate
        let recurrent_points = recurrence_matrix.iter().filter(|&&x| x).count() as f64;
        let recurrence_rate = recurrent_points / total_points;

        // Determinism (percentage of recurrent points forming diagonal lines)
        let diagonal_lines = self.find_diagonal_lines(recurrence_matrix, 2)?;
        let diagonal_points: usize = diagonal_lines.iter().map(|line| line.length).sum();
        let determinism = diagonal_points as f64 / recurrent_points;

        // Average diagonal line length
        let avg_diagonal_length = if !diagonal_lines.is_empty() {
            diagonal_points as f64 / diagonal_lines.len() as f64
        } else {
            0.0
        };

        // Maximum diagonal line length
        let max_diagonal_length = diagonal_lines
            .iter()
            .map(|line| line.length)
            .max()
            .unwrap_or(0) as f64;

        // Laminarity (percentage of recurrent points forming vertical lines)
        let vertical_lines = self.find_vertical_lines(recurrence_matrix, 2)?;
        let vertical_points: usize = vertical_lines.iter().map(|line| line.length).sum();
        let laminarity = vertical_points as f64 / recurrent_points;

        // Trapping time (average vertical line length)
        let trapping_time = if !vertical_lines.is_empty() {
            vertical_points as f64 / vertical_lines.len() as f64
        } else {
            0.0
        };

        Ok(RQAMeasures {
            recurrence_rate,
            determinism,
            avg_diagonal_length,
            max_diagonal_length,
            laminarity,
            trapping_time,
        })
    }

    /// Find diagonal lines in recurrence matrix
    fn find_diagonal_lines(
        &self,
        matrix: &Array2<bool>,
        min_length: usize,
    ) -> IntegrateResult<Vec<RecurrentLine>> {
        let n = matrix.nrows();
        let mut lines = Vec::new();

        // Check all diagonals
        for k in -(n as i32 - 1)..(n as i32) {
            let mut current_length = 0;
            let mut start_i = 0;
            let mut start_j = 0;

            let (start_row, start_col) = if k >= 0 {
                (0, k as usize)
            } else {
                ((-k) as usize, 0)
            };

            let max_steps = n - start_row.max(start_col);

            for step in 0..max_steps {
                let i = start_row + step;
                let j = start_col + step;

                if i < n && j < n && matrix[[i, j]] {
                    if current_length == 0 {
                        start_i = i;
                        start_j = j;
                    }
                    current_length += 1;
                } else {
                    if current_length >= min_length {
                        lines.push(RecurrentLine {
                            start_i,
                            start_j,
                            length: current_length,
                            line_type: LineType::Diagonal,
                        });
                    }
                    current_length = 0;
                }
            }

            // Check end of diagonal
            if current_length >= min_length {
                lines.push(RecurrentLine {
                    start_i,
                    start_j,
                    length: current_length,
                    line_type: LineType::Diagonal,
                });
            }
        }

        Ok(lines)
    }

    /// Find vertical lines in recurrence matrix
    fn find_vertical_lines(
        &self,
        matrix: &Array2<bool>,
        min_length: usize,
    ) -> IntegrateResult<Vec<RecurrentLine>> {
        let n = matrix.nrows();
        let mut lines = Vec::new();

        for j in 0..n {
            let mut current_length = 0;
            let mut start_i = 0;

            for i in 0..n {
                if matrix[[i, j]] {
                    if current_length == 0 {
                        start_i = i;
                    }
                    current_length += 1;
                } else {
                    if current_length >= min_length {
                        lines.push(RecurrentLine {
                            start_i,
                            start_j: j,
                            length: current_length,
                            line_type: LineType::Vertical,
                        });
                    }
                    current_length = 0;
                }
            }

            // Check end of column
            if current_length >= min_length {
                lines.push(RecurrentLine {
                    start_i,
                    start_j: j,
                    length: current_length,
                    line_type: LineType::Vertical,
                });
            }
        }

        Ok(lines)
    }
}

/// Recurrence analysis result
#[derive(Debug, Clone)]
pub struct RecurrenceAnalysis {
    /// Recurrence matrix
    pub recurrence_matrix: Array2<bool>,
    /// Embedded vectors
    pub embedded_vectors: Vec<Array1<f64>>,
    /// RQA measures
    pub rqa_measures: RQAMeasures,
    /// Analysis parameters
    pub threshold: f64,
    pub embedding_dimension: usize,
    pub time_delay: usize,
}

/// Recurrence Quantification Analysis measures
#[derive(Debug, Clone)]
pub struct RQAMeasures {
    /// Recurrence rate
    pub recurrence_rate: f64,
    /// Determinism
    pub determinism: f64,
    /// Average diagonal line length
    pub avg_diagonal_length: f64,
    /// Maximum diagonal line length
    pub max_diagonal_length: f64,
    /// Laminarity
    pub laminarity: f64,
    /// Trapping time
    pub trapping_time: f64,
}

/// Recurrent line structure
#[derive(Debug, Clone)]
pub struct RecurrentLine {
    pub start_i: usize,
    pub start_j: usize,
    pub length: usize,
    pub line_type: LineType,
}

/// Line types in recurrence plot
#[derive(Debug, Clone, Copy)]
pub enum LineType {
    Diagonal,
    Vertical,
    Horizontal,
}

/// Advanced continuation and monodromy analysis for bifurcation detection
pub struct ContinuationAnalyzer {
    /// Parameter range for continuation
    pub param_range: (f64, f64),
    /// Number of continuation steps
    pub n_steps: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum Newton iterations
    pub max_newton_iter: usize,
}

impl ContinuationAnalyzer {
    /// Create new continuation analyzer
    pub fn new(paramrange: (f64, f64), n_steps: usize) -> Self {
        Self {
            param_range: paramrange,
            n_steps,
            tol: 1e-8,
            max_newton_iter: 50,
        }
    }

    /// Perform parameter continuation to trace bifurcation curves
    pub fn trace_bifurcation_curve<F>(
        &self,
        system: F,
        initial_state: &Array1<f64>,
    ) -> IntegrateResult<ContinuationResult>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let mut bifurcation_points = Vec::new();
        let mut fixed_points = Vec::new();

        let (param_start, param_end) = self.param_range;
        let step = (param_end - param_start) / self.n_steps as f64;

        let mut current_state = initial_state.clone();

        for i in 0..=self.n_steps {
            let param = param_start + i as f64 * step;

            // Find fixed point at current parameter
            let fixed_point = self.find_fixed_point(&system, &current_state, param)?;

            // Compute stability via numerical Jacobian
            let jac = self.numerical_jacobian(&system, &fixed_point, param)?;
            let eigenvalues = self.compute_eigenvalues(&jac)?;

            // Check for bifurcations
            if let Some(bif_type) = self.detect_bifurcation(&eigenvalues) {
                bifurcation_points.push(BifurcationPointData {
                    parameter: param,
                    state: fixed_point.clone(),
                    bifurcation_type: bif_type,
                });
            }

            fixed_points.push(FixedPointData {
                parameter: param,
                state: fixed_point.clone(),
                eigenvalues: eigenvalues.clone(),
                stability: self.classify_stability(&eigenvalues),
            });

            current_state = fixed_point;
        }

        Ok(ContinuationResult {
            bifurcation_points,
            fixed_points,
            parameter_range: self.param_range,
        })
    }

    /// Find fixed point using Newton's method
    fn find_fixed_point<F>(
        &self,
        system: &F,
        initial_guess: &Array1<f64>,
        parameter: f64,
    ) -> IntegrateResult<Array1<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let mut x = initial_guess.clone();

        for _ in 0..self.max_newton_iter {
            let f = system(&x, parameter);
            let norm_f = f.iter().map(|&v| v * v).sum::<f64>().sqrt();

            if norm_f < self.tol {
                return Ok(x);
            }

            let jac = self.numerical_jacobian(system, &x, parameter)?;
            let delta_x = self.solve_linear_system(&jac, &f)?;

            x = &x - &delta_x;
        }

        Err(IntegrateError::ConvergenceError(
            "Fixed point not found".to_string(),
        ))
    }

    /// Compute numerical Jacobian
    fn numerical_jacobian<F>(
        &self,
        system: &F,
        x: &Array1<f64>,
        parameter: f64,
    ) -> IntegrateResult<Array2<f64>>
    where
        F: Fn(&Array1<f64>, f64) -> Array1<f64>,
    {
        let n = x.len();
        let mut jac = Array2::zeros((n, n));
        let h = 1e-8;

        for j in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[j] += h;
            x_minus[j] -= h;

            let f_plus = system(&x_plus, parameter);
            let f_minus = system(&x_minus, parameter);

            for i in 0..n {
                jac[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * h);
            }
        }

        Ok(jac)
    }

    /// Solve linear system using Gaussian elimination
    fn solve_linear_system(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> IntegrateResult<Array1<f64>> {
        let n = a.nrows();
        let mut aug = Array2::zeros((n, n + 1));

        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for k in 0..n {
            let mut max_row = k;
            for i in k + 1..n {
                if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                    max_row = i;
                }
            }

            for j in 0..n + 1 {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }

            for i in k + 1..n {
                let factor = aug[[i, k]] / aug[[k, k]];
                for j in k..n + 1 {
                    aug[[i, j]] -= factor * aug[[k, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in i + 1..n {
                x[i] -= aug[[i, j]] * x[j];
            }
            x[i] /= aug[[i, i]];
        }

        Ok(x)
    }

    /// Compute eigenvalues for 2x2 matrices
    fn compute_eigenvalues(&self, matrix: &Array2<f64>) -> IntegrateResult<Vec<Complex64>> {
        let n = matrix.nrows();

        if n == 2 {
            let a = matrix[[0, 0]];
            let b = matrix[[0, 1]];
            let c = matrix[[1, 0]];
            let d = matrix[[1, 1]];

            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                Ok(vec![
                    Complex64::new((trace + sqrt_disc) / 2.0, 0.0),
                    Complex64::new((trace - sqrt_disc) / 2.0, 0.0),
                ])
            } else {
                let sqrt_disc = (-discriminant).sqrt();
                Ok(vec![
                    Complex64::new(trace / 2.0, sqrt_disc / 2.0),
                    Complex64::new(trace / 2.0, -sqrt_disc / 2.0),
                ])
            }
        } else {
            Err(IntegrateError::InvalidInput(
                "Only 2x2 matrices supported".to_string(),
            ))
        }
    }

    /// Detect bifurcation types
    fn detect_bifurcation(&self, eigenvalues: &[Complex64]) -> Option<BifurcationType> {
        for eigenval in eigenvalues {
            if eigenval.im == 0.0 && eigenval.re.abs() < 1e-6 {
                return Some(BifurcationType::SaddleNode);
            }

            if eigenval.im != 0.0 && eigenval.re.abs() < 1e-6 {
                return Some(BifurcationType::Hopf);
            }
        }
        None
    }

    /// Classify stability
    fn classify_stability(&self, eigenvalues: &[Complex64]) -> StabilityType {
        for eigenval in eigenvalues {
            if eigenval.re > 1e-12 {
                return StabilityType::Unstable;
            }
            if eigenval.re.abs() < 1e-12 {
                return StabilityType::Marginally;
            }
        }
        StabilityType::Stable
    }
}

/// Monodromy matrix analyzer for periodic orbits
pub struct MonodromyAnalyzer {
    pub period: f64,
    pub tol: f64,
    pub n_steps: usize,
}

impl MonodromyAnalyzer {
    /// Create new monodromy analyzer
    pub fn new(period: f64, nsteps: usize) -> Self {
        Self {
            period,
            tol: 1e-8,
            n_steps: nsteps,
        }
    }

    /// Compute monodromy matrix
    pub fn compute_monodromy_matrix<F>(
        &self,
        system: F,
        initial_state: &Array1<f64>,
    ) -> IntegrateResult<MonodromyResult>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let n = initial_state.len();
        let dt = self.period / self.n_steps as f64;

        // Integrate fundamental matrix
        let mut fundamental_matrix = Array2::eye(n);
        let mut current_state = initial_state.clone();

        for _ in 0..self.n_steps {
            let jac = self.numerical_jacobian(&system, &current_state)?;

            // Euler step for fundamental matrix: dΦ/dt = J(t)Φ
            fundamental_matrix = &fundamental_matrix + &(jac.dot(&fundamental_matrix) * dt);

            // RK4 for state evolution
            let k1 = system(&current_state);
            let k2 = system(&(&current_state + &(&k1 * (dt / 2.0))));
            let k3 = system(&(&current_state + &(&k2 * (dt / 2.0))));
            let k4 = system(&(&current_state + &(&k3 * dt)));

            current_state = &current_state + &((&k1 + &k2 * 2.0 + &k3 * 2.0 + &k4) * (dt / 6.0));
        }

        let eigenvalues = self.compute_eigenvalues(&fundamental_matrix)?;
        let stability = self.classify_periodic_stability(&eigenvalues);

        Ok(MonodromyResult {
            monodromy_matrix: fundamental_matrix,
            eigenvalues,
            stability,
            period: self.period,
            final_state: current_state,
        })
    }

    /// Compute numerical Jacobian
    fn numerical_jacobian<F>(&self, system: &F, x: &Array1<f64>) -> IntegrateResult<Array2<f64>>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let n = x.len();
        let mut jac = Array2::zeros((n, n));
        let h = 1e-8;

        for j in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[j] += h;
            x_minus[j] -= h;

            let f_plus = system(&x_plus);
            let f_minus = system(&x_minus);

            for i in 0..n {
                jac[[i, j]] = (f_plus[i] - f_minus[i]) / (2.0 * h);
            }
        }

        Ok(jac)
    }

    /// Compute eigenvalues
    fn compute_eigenvalues(&self, matrix: &Array2<f64>) -> IntegrateResult<Vec<Complex64>> {
        let n = matrix.nrows();

        if n == 2 {
            let a = matrix[[0, 0]];
            let b = matrix[[0, 1]];
            let c = matrix[[1, 0]];
            let d = matrix[[1, 1]];

            let trace = a + d;
            let det = a * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                Ok(vec![
                    Complex64::new((trace + sqrt_disc) / 2.0, 0.0),
                    Complex64::new((trace - sqrt_disc) / 2.0, 0.0),
                ])
            } else {
                let sqrt_disc = (-discriminant).sqrt();
                Ok(vec![
                    Complex64::new(trace / 2.0, sqrt_disc / 2.0),
                    Complex64::new(trace / 2.0, -sqrt_disc / 2.0),
                ])
            }
        } else {
            Err(IntegrateError::InvalidInput(
                "Only 2x2 matrices supported".to_string(),
            ))
        }
    }

    /// Classify periodic stability
    fn classify_periodic_stability(&self, multipliers: &[Complex64]) -> PeriodicStabilityType {
        let max_magnitude = multipliers.iter().map(|m| m.norm()).fold(0.0, f64::max);

        if max_magnitude > 1.0 + 1e-6 {
            PeriodicStabilityType::Unstable
        } else if (max_magnitude - 1.0).abs() < 1e-6 {
            PeriodicStabilityType::Marginally
        } else {
            PeriodicStabilityType::Stable
        }
    }
}

/// Continuation analysis result
#[derive(Debug, Clone)]
pub struct ContinuationResult {
    pub bifurcation_points: Vec<BifurcationPointData>,
    pub fixed_points: Vec<FixedPointData>,
    pub parameter_range: (f64, f64),
}

/// Fixed point with stability data
#[derive(Debug, Clone)]
pub struct FixedPointData {
    pub parameter: f64,
    pub state: Array1<f64>,
    pub eigenvalues: Vec<Complex64>,
    pub stability: StabilityType,
}

/// Bifurcation point data
#[derive(Debug, Clone)]
pub struct BifurcationPointData {
    pub parameter: f64,
    pub state: Array1<f64>,
    pub bifurcation_type: BifurcationType,
}

/// Monodromy analysis result
#[derive(Debug, Clone)]
pub struct MonodromyResult {
    pub monodromy_matrix: Array2<f64>,
    pub eigenvalues: Vec<Complex64>,
    pub stability: PeriodicStabilityType,
    pub period: f64,
    pub final_state: Array1<f64>,
}

/// Extended bifurcation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BifurcationType {
    SaddleNode,
    Hopf,
    PeriodDoubling,
    Transcritical,
    Pitchfork,
}

/// Periodic orbit stability
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PeriodicStabilityType {
    Stable,
    Unstable,
    Marginally,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_analyzer() {
        // Test with helical trajectory that crosses z = 0 plane
        let mut trajectory = Vec::new();
        let mut times = Vec::new();

        for i in 0..100 {
            let t = i as f64 * 0.1;
            let x = t.cos();
            let y = t.sin();
            let z = 0.1 * t.sin(); // Oscillates around z = 0, creating crossings

            trajectory.push(Array1::from_vec(vec![x, y, z]));
            times.push(t);
        }

        // Define Poincaré section as z = 0 plane
        let section_normal = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        let section_point = Array1::from_vec(vec![0.0, 0.0, 0.0]);

        let analyzer = PoincareAnalyzer::new(section_normal, section_point, 1);
        let result = analyzer.analyze_trajectory(&trajectory, &times).unwrap();

        // Should find crossings for this trajectory
        assert!(!result.crossings.is_empty());
    }

    #[test]
    fn test_lyapunov_calculator() {
        // Test with simple linear system (should have negative Lyapunov exponent)
        let system =
            |state: &Array1<f64>| -> Array1<f64> { Array1::from_vec(vec![-state[0], -state[1]]) };

        let calculator = LyapunovCalculator::new(2, 0.01);
        let initial_state = Array1::from_vec(vec![1.0, 1.0]);

        let exponents = calculator
            .calculate_lyapunov_exponents(system, &initial_state, 10.0)
            .unwrap();

        // Both exponents should be negative for stable linear system
        assert!(exponents[0] < 0.0);
        assert!(exponents[1] < 0.0);
    }

    #[test]
    fn test_fractal_analyzer() {
        // Create a simple 2D filled area for testing - should have dimension close to 2
        let mut points = Vec::new();
        let mut rng = scirs2_core::random::rng();

        // Generate points uniformly distributed in a square with some noise
        for _i in 0..500 {
            let x = rng.random::<f64>() * 2.0 - 1.0; // range [-1, 1]
            let y = rng.random::<f64>() * 2.0 - 1.0; // range [-1, 1]
            let point = Array1::from_vec(vec![x, y]);
            points.push(point);
        }

        // Optimized analyzer configuration for better performance
        let mut analyzer = FractalAnalyzer::new();
        analyzer.scale_range = (0.1, 0.5); // Adjusted range for better scale coverage
        analyzer.n_scales = 5; // Increased scales for more stable slope calculation
        analyzer.box_counting_method = BoxCountingMethod::Standard; // Use standard method

        let result = analyzer.calculate_fractal_dimension(&points).unwrap();

        // Verify the results are mathematically valid
        assert!(result.dimension.is_finite(), "Dimension should be finite");
        assert!(
            result.dimension > 0.0,
            "Dimension should be positive, got: {}",
            result.dimension
        );
        assert!(
            result.dimension <= 3.0,
            "Dimension should not exceed embedding dimension, got: {}",
            result.dimension
        );
        assert!(
            result.dimension >= 1.5 && result.dimension <= 2.5,
            "2D filled area should have dimension between 1.5 and 2.5, got: {}",
            result.dimension
        );
        assert!(
            result.r_squared >= 0.0 && result.r_squared <= 1.0,
            "R-squared should be in [0,1], got: {}",
            result.r_squared
        );

        // Verify that the fractal dimension makes sense for a spiral pattern
        println!(
            "Fractal dimension: {}, R²: {}",
            result.dimension, result.r_squared
        );
    }

    #[test]
    fn test_recurrence_analyzer() {
        // Test with sinusoidal time series
        let timeseries: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        let analyzer = RecurrenceAnalyzer::new(0.1, 3, 1);
        let result = analyzer.analyze_recurrence(&timeseries).unwrap();

        // Should have reasonable recurrence measures
        assert!(result.rqa_measures.recurrence_rate > 0.0);
        assert!(result.rqa_measures.recurrence_rate <= 1.0);
        assert!(result.rqa_measures.determinism >= 0.0);
        assert!(result.rqa_measures.determinism <= 1.0);
    }
}
