//! Specialized visualization tools for different scientific domains
//!
//! This module provides visualization tools for specialized scientific domains
//! including quantum mechanics, fluid dynamics, and financial analysis.

use super::types::*;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};

/// Quantum state visualization tools
pub struct QuantumVisualizer;

impl QuantumVisualizer {
    /// Create wave function visualization
    pub fn visualize_wavefunction(
        x: &Array1<f64>,
        probability_density: &Array1<f64>,
        time: f64,
    ) -> IntegrateResult<HeatMapPlot> {
        let mut metadata = PlotMetadata::default();
        metadata.title = format!("Quantum State at t = {:.3}", time);
        metadata.xlabel = "Position".to_string();
        metadata.ylabel = "Probability Density".to_string();

        Ok(HeatMapPlot {
            x: x.clone(),
            y: Array1::from_elem(1, 0.0), // 1D visualization
            z: Array2::from_shape_vec((1, probability_density.len()), probability_density.to_vec())
                .map_err(|e| IntegrateError::ComputationError(format!("Shape error: {e}")))?,
            metadata,
        })
    }

    /// Create complex phase visualization
    pub fn visualize_complex_phase(
        real_parts: &[f64],
        imag_parts: &[f64],
        phases: &[f64],
    ) -> IntegrateResult<PhaseSpacePlot> {
        let mut metadata = PlotMetadata::default();
        metadata.title = "Complex Wave Function Phase".to_string();
        metadata.xlabel = "Real Part".to_string();
        metadata.ylabel = "Imaginary Part".to_string();

        Ok(PhaseSpacePlot {
            x: real_parts.to_vec(),
            y: imag_parts.to_vec(),
            colors: Some(phases.to_vec()),
            metadata,
        })
    }

    /// Create expectation value evolution plot
    pub fn visualize_expectation_evolution(
        times: &[f64],
        positions: &[f64],
        momenta: &[f64],
    ) -> IntegrateResult<PhaseSpacePlot> {
        let mut metadata = PlotMetadata::default();
        metadata.title = "Quantum Expectation Values Evolution".to_string();
        metadata.xlabel = "Position Expectation".to_string();
        metadata.ylabel = "Momentum Expectation".to_string();

        Ok(PhaseSpacePlot {
            x: positions.to_vec(),
            y: momenta.to_vec(),
            colors: Some(times.to_vec()),
            metadata,
        })
    }

    /// Create energy level diagram
    pub fn visualize_energy_levels(
        energies: &Array1<f64>,
        wavefunctions: &Array2<f64>,
    ) -> IntegrateResult<VectorFieldPlot> {
        let n_levels = energies.len().min(5); // Show up to 5 levels
        let n_points = wavefunctions.nrows();

        let x_coords = Array1::linspace(-1.0, 1.0, n_points);
        let mut x_grid = Array2::zeros((n_levels, n_points));
        let mut y_grid = Array2::zeros((n_levels, n_points));
        let mut u = Array2::zeros((n_levels, n_points));
        let mut v = Array2::zeros((n_levels, n_points));
        let mut magnitude = Array2::zeros((n_levels, n_points));

        for level in 0..n_levels {
            for i in 0..n_points {
                x_grid[[level, i]] = x_coords[i];
                y_grid[[level, i]] = energies[level];
                u[[level, i]] = wavefunctions[[i, level]];
                v[[level, i]] = 0.0; // No y-component for energy levels
                magnitude[[level, i]] = wavefunctions[[i, level]].abs();
            }
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Energy Level Diagram".to_string();
        metadata.xlabel = "Position".to_string();
        metadata.ylabel = "Energy".to_string();

        Ok(VectorFieldPlot {
            x_grid,
            y_grid,
            u,
            v,
            magnitude,
            metadata,
        })
    }
}

/// Fluid dynamics visualization tools
pub struct FluidVisualizer;

impl FluidVisualizer {
    /// Create velocity field visualization
    pub fn visualize_velocity_field(state: &FluidState) -> IntegrateResult<VectorFieldPlot> {
        if state.velocity.len() < 2 {
            return Err(IntegrateError::ValueError(
                "Need at least 2 velocity components".to_string(),
            ));
        }

        let u = &state.velocity[0];
        let v = &state.velocity[1];
        let (ny, nx) = u.dim();

        let mut x_grid = Array2::zeros((ny, nx));
        let mut y_grid = Array2::zeros((ny, nx));
        let mut magnitude = Array2::zeros((ny, nx));

        for i in 0..ny {
            for j in 0..nx {
                x_grid[[i, j]] = j as f64 * state.dx;
                y_grid[[i, j]] = i as f64 * state.dy;
                magnitude[[i, j]] = (u[[i, j]].powi(2) + v[[i, j]].powi(2)).sqrt();
            }
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = format!("Velocity Field at t = {:.3}", state.time);
        metadata.xlabel = "X Position".to_string();
        metadata.ylabel = "Y Position".to_string();

        Ok(VectorFieldPlot {
            x_grid,
            y_grid,
            u: u.clone(),
            v: v.clone(),
            magnitude,
            metadata,
        })
    }

    /// Create pressure field heatmap
    pub fn visualize_pressure_field(state: &FluidState) -> IntegrateResult<HeatMapPlot> {
        let (ny, nx) = state.pressure.dim();
        let x = Array1::from_iter((0..nx).map(|i| i as f64 * state.dx));
        let y = Array1::from_iter((0..ny).map(|i| i as f64 * state.dy));

        let mut metadata = PlotMetadata::default();
        metadata.title = format!("Pressure Field at t = {:.3}", state.time);
        metadata.xlabel = "X Position".to_string();
        metadata.ylabel = "Y Position".to_string();

        Ok(HeatMapPlot {
            x,
            y,
            z: state.pressure.clone(),
            metadata,
        })
    }

    /// Create vorticity visualization
    pub fn visualize_vorticity(state: &FluidState) -> IntegrateResult<HeatMapPlot> {
        if state.velocity.len() < 2 {
            return Err(IntegrateError::ValueError(
                "Need at least 2 velocity components".to_string(),
            ));
        }

        let u = &state.velocity[0];
        let v = &state.velocity[1];
        let (ny, nx) = u.dim();

        let mut vorticity = Array2::zeros((ny, nx));

        // Compute vorticity using finite differences
        for i in 1..ny - 1 {
            for j in 1..nx - 1 {
                let dvdx = (v[[i, j + 1]] - v[[i, j - 1]]) / (2.0 * state.dx);
                let dudy = (u[[i + 1, j]] - u[[i - 1, j]]) / (2.0 * state.dy);
                vorticity[[i, j]] = dvdx - dudy;
            }
        }

        let x = Array1::from_iter((0..nx).map(|i| i as f64 * state.dx));
        let y = Array1::from_iter((0..ny).map(|i| i as f64 * state.dy));

        let mut metadata = PlotMetadata::default();
        metadata.title = format!("Vorticity Field at t = {:.3}", state.time);
        metadata.xlabel = "X Position".to_string();
        metadata.ylabel = "Y Position".to_string();

        Ok(HeatMapPlot {
            x,
            y,
            z: vorticity,
            metadata,
        })
    }

    /// Create streamline visualization  
    pub fn visualize_streamlines(
        state: &FluidState,
        n_streamlines: usize,
    ) -> IntegrateResult<Vec<PhaseSpacePlot>> {
        if state.velocity.len() < 2 {
            return Err(IntegrateError::ValueError(
                "Need at least 2 velocity components".to_string(),
            ));
        }

        let u = &state.velocity[0];
        let v = &state.velocity[1];
        let (ny, nx) = u.dim();

        let mut streamlines = Vec::new();

        // Create evenly spaced starting points
        for i in 0..n_streamlines {
            let start_x = (i as f64 / (n_streamlines - 1) as f64) * (nx - 1) as f64 * state.dx;
            let start_y = 0.5 * (ny - 1) as f64 * state.dy; // Start at middle height

            let mut x_line = vec![start_x];
            let mut y_line = vec![start_y];

            let mut current_x = start_x;
            let mut current_y = start_y;

            // Integrate streamline using simple Euler method
            let dt = 0.01 * state.dx.min(state.dy);
            for _ in 0..1000 {
                // Maximum steps
                let i_idx = (current_y / state.dy) as usize;
                let j_idx = (current_x / state.dx) as usize;

                if i_idx >= ny - 1 || j_idx >= nx - 1 || i_idx == 0 || j_idx == 0 {
                    break;
                }

                let vel_x = u[[i_idx, j_idx]];
                let vel_y = v[[i_idx, j_idx]];

                current_x += vel_x * dt;
                current_y += vel_y * dt;

                x_line.push(current_x);
                y_line.push(current_y);

                // Stop if velocity is too small
                if vel_x.abs() + vel_y.abs() < 1e-6 {
                    break;
                }
            }

            let mut metadata = PlotMetadata::default();
            metadata.title = format!("Streamline {} at t = {:.3}", i, state.time);
            metadata.xlabel = "X Position".to_string();
            metadata.ylabel = "Y Position".to_string();

            streamlines.push(PhaseSpacePlot {
                x: x_line,
                y: y_line,
                colors: None,
                metadata,
            });
        }

        Ok(streamlines)
    }

    /// Create 3D fluid visualization
    pub fn visualize_3d_velocity_magnitude(state: &FluidState3D) -> IntegrateResult<SurfacePlot> {
        if state.velocity.len() < 3 {
            return Err(IntegrateError::ValueError(
                "Need 3 velocity components for 3D".to_string(),
            ));
        }

        let u = &state.velocity[0];
        let v = &state.velocity[1];
        let w = &state.velocity[2];
        let (nz, ny, nx) = u.dim();

        // Take a slice at z = nz/2
        let z_slice = nz / 2;
        let mut x_grid = Array2::zeros((ny, nx));
        let mut y_grid = Array2::zeros((ny, nx));
        let mut magnitude = Array2::zeros((ny, nx));

        for i in 0..ny {
            for j in 0..nx {
                x_grid[[i, j]] = j as f64 * state.dx;
                y_grid[[i, j]] = i as f64 * state.dy;
                let vel_mag = (u[[z_slice, i, j]].powi(2)
                    + v[[z_slice, i, j]].powi(2)
                    + w[[z_slice, i, j]].powi(2))
                .sqrt();
                magnitude[[i, j]] = vel_mag;
            }
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = format!("3D Velocity Magnitude at t = {:.3}", state.time);
        metadata.xlabel = "X Position".to_string();
        metadata.ylabel = "Y Position".to_string();

        Ok(SurfacePlot {
            x: x_grid,
            y: y_grid,
            z: magnitude,
            metadata,
        })
    }
}

/// Financial analysis visualization tools
pub struct FinanceVisualizer;

impl FinanceVisualizer {
    /// Create option price surface
    pub fn visualize_option_surface(
        strikes: &Array1<f64>,
        maturities: &Array1<f64>,
        prices: &Array2<f64>,
    ) -> IntegrateResult<SurfacePlot> {
        let (n_maturities, n_strikes) = prices.dim();
        let mut x_grid = Array2::zeros((n_maturities, n_strikes));
        let mut y_grid = Array2::zeros((n_maturities, n_strikes));

        for i in 0..n_maturities {
            for j in 0..n_strikes {
                x_grid[[i, j]] = strikes[j];
                y_grid[[i, j]] = maturities[i];
            }
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Option Price Surface".to_string();
        metadata.xlabel = "Strike Price".to_string();
        metadata.ylabel = "Time to Maturity".to_string();

        Ok(SurfacePlot {
            x: x_grid,
            y: y_grid,
            z: prices.clone(),
            metadata,
        })
    }

    /// Create Greeks surface visualization
    pub fn visualize_greeks_surface(
        strikes: &Array1<f64>,
        spot_prices: &Array1<f64>,
        greek_values: &Array2<f64>,
        greek_name: &str,
    ) -> IntegrateResult<HeatMapPlot> {
        let mut metadata = PlotMetadata::default();
        metadata.title = format!("{} Surface", greek_name);
        metadata.xlabel = "Strike Price".to_string();
        metadata.ylabel = "Spot Price".to_string();

        Ok(HeatMapPlot {
            x: strikes.clone(),
            y: spot_prices.clone(),
            z: greek_values.clone(),
            metadata,
        })
    }

    /// Create volatility smile visualization
    pub fn visualize_volatility_smile(
        strikes: &Array1<f64>,
        implied_volatilities: &Array1<f64>,
        maturity: f64,
    ) -> IntegrateResult<PhaseSpacePlot> {
        let mut metadata = PlotMetadata::default();
        metadata.title = format!("Volatility Smile (T = {:.3})", maturity);
        metadata.xlabel = "Strike Price".to_string();
        metadata.ylabel = "Implied Volatility".to_string();

        Ok(PhaseSpacePlot {
            x: strikes.to_vec(),
            y: implied_volatilities.to_vec(),
            colors: None,
            metadata,
        })
    }

    /// Create risk metrics visualization
    pub fn visualize_risk_metrics(
        time_points: &Array1<f64>,
        var_values: &Array1<f64>,
        cvar_values: &Array1<f64>,
    ) -> IntegrateResult<PhaseSpacePlot> {
        let mut metadata = PlotMetadata::default();
        metadata.title = "Risk Metrics Evolution".to_string();
        metadata.xlabel = "Value at Risk".to_string();
        metadata.ylabel = "Conditional Value at Risk".to_string();

        Ok(PhaseSpacePlot {
            x: var_values.to_vec(),
            y: cvar_values.to_vec(),
            colors: Some(time_points.to_vec()),
            metadata,
        })
    }

    /// Create portfolio performance visualization
    pub fn visualize_portfolio_performance(
        dates: &[String],
        returns: &Array1<f64>,
        benchmark_returns: &Array1<f64>,
    ) -> IntegrateResult<PhaseSpacePlot> {
        // Calculate cumulative returns
        let mut cum_returns = vec![1.0]; // Start with 1.0
        let mut cum_benchmark = vec![1.0];

        for i in 0..returns.len() {
            cum_returns.push(cum_returns[i] * (1.0 + returns[i]));
            cum_benchmark.push(cum_benchmark[i] * (1.0 + benchmark_returns[i]));
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Portfolio vs Benchmark Performance".to_string();
        metadata.xlabel = "Portfolio Cumulative Return".to_string();
        metadata.ylabel = "Benchmark Cumulative Return".to_string();

        Ok(PhaseSpacePlot {
            x: cum_returns,
            y: cum_benchmark,
            colors: Some((0..dates.len() + 1).map(|i| i as f64).collect()),
            metadata,
        })
    }
}

/// Create specialized quantum visualization
pub fn specialized_visualizations(
    visualization_type: &str,
    data: &Array2<f64>,
) -> IntegrateResult<HeatMapPlot> {
    match visualization_type {
        "quantum_probability" => {
            let x = Array1::linspace(-5.0, 5.0, data.ncols());
            let probability_density = data.row(0).to_owned();
            QuantumVisualizer::visualize_wavefunction(&x, &probability_density, 0.0)
        }
        _ => Err(IntegrateError::ValueError(format!(
            "Unknown visualization type: {}",
            visualization_type
        ))),
    }
}

/// Create bifurcation diagram generator for specialized systems
pub struct BifurcationDiagramGenerator {
    /// Parameter range for bifurcation analysis
    pub parameter_range: (f64, f64),
    /// Number of parameter samples
    pub n_parameter_samples: usize,
    /// Number of initial transient steps to skip
    pub transient_steps: usize,
    /// Number of sampling steps after transients
    pub sampling_steps: usize,
    /// Tolerance for detecting fixed points
    pub fixed_point_tolerance: f64,
    /// Tolerance for detecting periodic orbits
    pub period_tolerance: f64,
}

impl BifurcationDiagramGenerator {
    /// Create new bifurcation diagram generator
    pub fn new(parameterrange: (f64, f64), n_parameter_samples: usize) -> Self {
        Self {
            parameter_range: parameterrange,
            n_parameter_samples,
            transient_steps: 1000,
            sampling_steps: 500,
            fixed_point_tolerance: 1e-8,
            period_tolerance: 1e-6,
        }
    }

    /// Generate enhanced bifurcation diagram
    pub fn generate_enhanced_diagram<F>(
        &self,
        map_function: F,
        initial_condition: f64,
    ) -> IntegrateResult<BifurcationDiagram>
    where
        F: Fn(f64, f64) -> f64, // (x, parameter) -> x_next
    {
        let mut parameter_values = Vec::new();
        let mut state_values = Vec::new();
        let mut stability_flags = Vec::new();

        let param_step = (self.parameter_range.1 - self.parameter_range.0)
            / (self.n_parameter_samples - 1) as f64;

        for i in 0..self.n_parameter_samples {
            let param = self.parameter_range.0 + i as f64 * param_step;

            // Run transients
            let mut x = initial_condition;
            for _ in 0..self.transient_steps {
                x = map_function(x, param);
            }

            // Sample attractor
            let mut attractor_states = Vec::new();
            for _ in 0..self.sampling_steps {
                x = map_function(x, param);
                attractor_states.push(x);
            }

            // Simple attractor analysis
            let unique_count = self.count_unique_states(&attractor_states);
            let is_stable = unique_count <= 2;

            // Store representative states
            if unique_count == 1 {
                parameter_values.push(param);
                state_values.push(attractor_states[attractor_states.len() - 1]);
                stability_flags.push(is_stable);
            } else {
                // Store multiple points for periodic/chaotic attractors
                let sample_rate = (attractor_states.len() / 10).max(1);
                for (idx, &state) in attractor_states.iter().step_by(sample_rate).enumerate() {
                    if idx < 10 {
                        // Limit number of points per parameter
                        parameter_values.push(param);
                        state_values.push(state);
                        stability_flags.push(is_stable);
                    }
                }
            }
        }

        Ok(BifurcationDiagram {
            parameters: parameter_values,
            states: vec![state_values],
            stability: stability_flags,
            bifurcation_points: vec![], // Simplified - not computing bifurcation points
        })
    }

    fn count_unique_states(&self, states: &[f64]) -> usize {
        let mut unique_states = states.to_vec();
        unique_states.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_states.dedup_by(|a, b| (*a - *b).abs() < self.fixed_point_tolerance);
        unique_states.len()
    }
}
