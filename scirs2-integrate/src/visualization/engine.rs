//! Core visualization engine for numerical integration
//!
//! This module provides the main VisualizationEngine struct and its implementation
//! for creating various types of plots from numerical integration results.

use super::types::*;
use crate::analysis::{BasinAnalysis, BifurcationPoint};
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::ODEResult;
use scirs2_core::ndarray::{Array1, Array2};

/// Main visualization engine for creating plots from numerical data
#[derive(Debug, Clone)]
pub struct VisualizationEngine {
    /// Output format preference
    pub output_format: OutputFormat,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Figure size
    pub figure_size: (f64, f64),
}

impl Default for VisualizationEngine {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::ASCII,
            color_scheme: ColorScheme::Viridis,
            figure_size: (800.0, 600.0),
        }
    }
}

impl VisualizationEngine {
    /// Create a new visualization engine
    pub fn new() -> Self {
        Default::default()
    }

    /// Create phase space plot from ODE result
    pub fn create_phase_spaceplot<F: crate::common::IntegrateFloat>(
        &self,
        ode_result: &ODEResult<F>,
        x_index: usize,
        y_index: usize,
    ) -> IntegrateResult<PhaseSpacePlot> {
        let n_points = ode_result.t.len();
        let n_vars = if !ode_result.y.is_empty() {
            ode_result.y[0].len()
        } else {
            0
        };

        if x_index >= n_vars || y_index >= n_vars {
            return Err(IntegrateError::ValueError(
                "Variable _index out of bounds".to_string(),
            ));
        }

        let x: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][x_index].to_f64().unwrap_or(0.0))
            .collect();

        let y: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][y_index].to_f64().unwrap_or(0.0))
            .collect();

        // Color by time for trajectory visualization
        let colors: Vec<f64> = ode_result
            .t
            .iter()
            .map(|t| t.to_f64().unwrap_or(0.0))
            .collect();

        let mut metadata = PlotMetadata::default();
        metadata.title = "Phase Space Plot".to_string();
        metadata.xlabel = format!("Variable {x_index}");
        metadata.ylabel = format!("Variable {y_index}");

        Ok(PhaseSpacePlot {
            x,
            y,
            colors: Some(colors),
            metadata,
        })
    }

    /// Create bifurcation diagram from analysis results
    pub fn create_bifurcation_diagram(
        &self,
        bifurcation_points: &[BifurcationPoint],
        parameter_range: (f64, f64),
        n_points: usize,
    ) -> IntegrateResult<BifurcationDiagram> {
        let mut parameters = Vec::new();
        let mut states = Vec::new();
        let mut stability = Vec::new();

        // Create parameter grid
        let param_step = (parameter_range.1 - parameter_range.0) / (n_points - 1) as f64;
        for i in 0..n_points {
            let param = parameter_range.0 + i as f64 * param_step;
            parameters.push(param);

            // Find corresponding state (simplified)
            let mut found = false;
            for bif_point in bifurcation_points {
                if (bif_point.parameter_value - param).abs() < param_step {
                    states.push(bif_point.state.to_vec());
                    // Simplified stability check based on eigenvalues
                    let is_stable = bif_point.eigenvalues.iter().all(|eig| eig.re < 0.0);
                    stability.push(is_stable);
                    found = true;
                    break;
                }
            }

            if !found {
                states.push(vec![0.0]); // Default value
                stability.push(true);
            }
        }

        Ok(BifurcationDiagram {
            parameters,
            states,
            stability,
            bifurcation_points: bifurcation_points.to_vec(),
        })
    }

    /// Create vector field plot for 2D dynamical systems
    pub fn create_vector_fieldplot<F>(
        &self,
        system: F,
        x_range: (f64, f64),
        y_range: (f64, f64),
        grid_size: (usize, usize),
    ) -> IntegrateResult<VectorFieldPlot>
    where
        F: Fn(&Array1<f64>) -> Array1<f64>,
    {
        let (nx, ny) = grid_size;
        let mut x_grid = Array2::zeros((ny, nx));
        let mut y_grid = Array2::zeros((ny, nx));
        let mut u = Array2::zeros((ny, nx));
        let mut v = Array2::zeros((ny, nx));
        let mut magnitude = Array2::zeros((ny, nx));

        let dx = (x_range.1 - x_range.0) / (nx - 1) as f64;
        let dy = (y_range.1 - y_range.0) / (ny - 1) as f64;

        for i in 0..ny {
            for j in 0..nx {
                let x = x_range.0 + j as f64 * dx;
                let y = y_range.0 + i as f64 * dy;

                x_grid[[i, j]] = x;
                y_grid[[i, j]] = y;

                let state = Array1::from_vec(vec![x, y]);
                let derivative = system(&state);

                if derivative.len() >= 2 {
                    u[[i, j]] = derivative[0];
                    v[[i, j]] = derivative[1];
                    magnitude[[i, j]] = (derivative[0].powi(2) + derivative[1].powi(2)).sqrt();
                }
            }
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Vector Field Plot".to_string();
        metadata.xlabel = "X".to_string();
        metadata.ylabel = "Y".to_string();

        Ok(VectorFieldPlot {
            x_grid,
            y_grid,
            u,
            v,
            magnitude,
            metadata,
        })
    }

    /// Create basin of attraction visualization
    pub fn create_basinplot(_basinanalysis: &BasinAnalysis) -> IntegrateResult<HeatMapPlot> {
        let grid_size = _basinanalysis.attractor_indices.nrows();
        let x = Array1::linspace(0.0, 1.0, grid_size);
        let y = Array1::linspace(0.0, 1.0, grid_size);

        // Convert attractor indices to f64 for plotting
        let z = _basinanalysis.attractor_indices.mapv(|x| x as f64);

        let mut metadata = PlotMetadata::default();
        metadata.title = "Basin of Attraction".to_string();
        metadata.xlabel = "X".to_string();
        metadata.ylabel = "Y".to_string();

        Ok(HeatMapPlot { x, y, z, metadata })
    }

    /// Generate ASCII art representation of a 2D plot
    pub fn render_asciiplot(data: &[(f64, f64)], width: usize, height: usize) -> String {
        if data.is_empty() {
            return "No data to plot".to_string();
        }

        // Find data bounds
        let x_min = data.iter().map(|(x_, _)| *x_).fold(f64::INFINITY, f64::min);
        let x_max = data
            .iter()
            .map(|(x_, _)| *x_)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = data
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max);

        // Create character grid
        let mut grid = vec![vec![' '; width]; height];

        // Map data points to grid
        for (x, y) in data {
            let i = ((y - y_min) / (y_max - y_min) * (height - 1) as f64) as usize;
            let j = ((x - x_min) / (x_max - x_min) * (width - 1) as f64) as usize;

            if i < height && j < width {
                grid[height - 1 - i][j] = '*'; // Flip y-axis for proper orientation
            }
        }

        // Convert grid to string
        let mut result = String::new();
        for row in grid {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }

        // Add axis labels
        result.push_str(&format!("\nX range: [{x_min:.3}, {x_max:.3}]\n"));
        result.push_str(&format!("Y range: [{y_min:.3}, {y_max:.3}]\n"));

        result
    }

    /// Export plot data to CSV format
    pub fn export_csv(plot: &PhaseSpacePlot) -> IntegrateResult<String> {
        let mut csv = String::new();

        // Header
        csv.push_str("x,y");
        if plot.colors.is_some() {
            csv.push_str(",color");
        }
        csv.push('\n');

        // Data
        for i in 0..plot.x.len() {
            csv.push_str(&format!("{},{}", plot.x[i], plot.y[i]));
            if let Some(ref colors) = plot.colors {
                csv.push_str(&format!(",{}", colors[i]));
            }
            csv.push('\n');
        }

        Ok(csv)
    }

    /// Create learning curve plot for optimization algorithms
    pub fn create_learning_curve(
        &self,
        iterations: &[usize],
        values: &[f64],
        title: &str,
    ) -> IntegrateResult<PhaseSpacePlot> {
        let x: Vec<f64> = iterations.iter().map(|&i| i as f64).collect();
        let y = values.to_vec();

        let mut metadata = PlotMetadata::default();
        metadata.title = title.to_string();
        metadata.xlabel = "Iteration".to_string();
        metadata.ylabel = "Value".to_string();

        Ok(PhaseSpacePlot {
            x,
            y,
            colors: None,
            metadata,
        })
    }

    /// Create convergence analysis plot
    pub fn create_convergenceplot(
        &self,
        step_sizes: &[f64],
        errors: &[f64],
        theoretical_order: f64,
    ) -> IntegrateResult<PhaseSpacePlot> {
        let x: Vec<f64> = step_sizes.iter().map(|h| h.log10()).collect();
        let y: Vec<f64> = errors.iter().map(|e| e.log10()).collect();

        let mut metadata = PlotMetadata::default();
        metadata.title = "Convergence Analysis".to_string();
        metadata.xlabel = "log10(step size)".to_string();
        metadata.ylabel = "log10(error)".to_string();
        metadata.annotations.insert(
            "theoretical_slope".to_string(),
            theoretical_order.to_string(),
        );

        Ok(PhaseSpacePlot {
            x,
            y,
            colors: None,
            metadata,
        })
    }

    /// Create interactive parameter space exploration plot
    pub fn create_parameter_exploration(
        &self,
        param_ranges: &[(f64, f64)], // [(min1, max1), (min2, max2), ...]
        param_names: &[String],
        evaluation_function: &dyn Fn(&[f64]) -> f64,
        resolution: usize,
    ) -> IntegrateResult<ParameterExplorationPlot> {
        if param_ranges.len() != 2 {
            return Err(IntegrateError::ValueError(
                "Parameter exploration currently supports only 2D parameter spaces".to_string(),
            ));
        }

        let (x_min, x_max) = param_ranges[0];
        let (y_min, y_max) = param_ranges[1];

        let dx = (x_max - x_min) / (resolution - 1) as f64;
        let dy = (y_max - y_min) / (resolution - 1) as f64;

        let mut x_grid = Array2::zeros((resolution, resolution));
        let mut y_grid = Array2::zeros((resolution, resolution));
        let mut z_values = Array2::zeros((resolution, resolution));

        for i in 0..resolution {
            for j in 0..resolution {
                let x = x_min + i as f64 * dx;
                let y = y_min + j as f64 * dy;

                x_grid[[i, j]] = x;
                y_grid[[i, j]] = y;
                z_values[[i, j]] = evaluation_function(&[x, y]);
            }
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Parameter Space Exploration".to_string();
        metadata.xlabel = param_names
            .get(0)
            .cloned()
            .unwrap_or_else(|| "Parameter 1".to_string());
        metadata.ylabel = param_names
            .get(1)
            .cloned()
            .unwrap_or_else(|| "Parameter 2".to_string());

        Ok(ParameterExplorationPlot {
            x_grid,
            y_grid,
            z_values,
            param_ranges: param_ranges.to_vec(),
            param_names: param_names.to_vec(),
            metadata,
        })
    }

    /// Create real-time bifurcation diagram
    pub fn create_real_time_bifurcation(
        &self,
        system: &dyn Fn(&Array1<f64>, f64) -> Array1<f64>,
        parameter_range: (f64, f64),
        initial_conditions: &[Array1<f64>],
        transient_steps: usize,
        record_steps: usize,
    ) -> IntegrateResult<RealTimeBifurcationPlot> {
        let n_params = 200;
        let param_step = (parameter_range.1 - parameter_range.0) / (n_params - 1) as f64;

        let mut parameter_values = Vec::new();
        let mut attractordata = Vec::new();
        let mut stabilitydata = Vec::new();

        for i in 0..n_params {
            let param = parameter_range.0 + i as f64 * param_step;
            parameter_values.push(param);

            let mut param_attractors = Vec::new();
            let mut param_stability = Vec::new();

            for initial in initial_conditions {
                // Evolve system to let transients die out
                let mut state = initial.clone();
                for _ in 0..transient_steps {
                    let derivative = system(&state, param);
                    state += &(&derivative * 0.01); // Small time step
                }

                // Record attractor points
                let mut attractor_points = Vec::new();
                let mut local_maxima = Vec::new();

                for step in 0..record_steps {
                    let derivative = system(&state, param);
                    let derivative_scaled = &derivative * 0.01;
                    let new_state = &state + &derivative_scaled;

                    // Simple local maxima detection for period identification
                    if step > 2
                        && new_state[0] > state[0]
                        && state[0] > (state.clone() - &derivative_scaled)[0]
                    {
                        local_maxima.push(state[0]);
                    }

                    attractor_points.push(state[0]);
                    state = new_state;
                }

                // Determine stability based on attractor behavior
                let stability = if local_maxima.len() == 1 {
                    AttractorStability::FixedPoint
                } else if local_maxima.len() == 2 {
                    AttractorStability::PeriodTwo
                } else if local_maxima.len() > 2 && local_maxima.len() < 10 {
                    AttractorStability::Periodic(local_maxima.len())
                } else {
                    AttractorStability::Chaotic
                };

                param_attractors.push(attractor_points);
                param_stability.push(stability);
            }

            attractordata.push(param_attractors);
            stabilitydata.push(param_stability);
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Real-time Bifurcation Diagram".to_string();
        metadata.xlabel = "Parameter".to_string();
        metadata.ylabel = "Attractor Values".to_string();

        Ok(RealTimeBifurcationPlot {
            parameter_values,
            attractor_data: attractordata,
            stability_data: stabilitydata,
            parameter_range,
            metadata,
        })
    }

    /// Create 3D phase space trajectory
    pub fn create_3d_phase_space<F: crate::common::IntegrateFloat>(
        &self,
        ode_result: &ODEResult<F>,
        x_index: usize,
        y_index: usize,
        z_index: usize,
    ) -> IntegrateResult<PhaseSpace3D> {
        let n_points = ode_result.t.len();
        let n_vars = if !ode_result.y.is_empty() {
            ode_result.y[0].len()
        } else {
            0
        };

        if x_index >= n_vars || y_index >= n_vars || z_index >= n_vars {
            return Err(IntegrateError::ValueError(
                "Variable _index out of bounds".to_string(),
            ));
        }

        let x: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][x_index].to_f64().unwrap_or(0.0))
            .collect();

        let y: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][y_index].to_f64().unwrap_or(0.0))
            .collect();

        let z: Vec<f64> = (0..n_points)
            .map(|i| ode_result.y[i][z_index].to_f64().unwrap_or(0.0))
            .collect();

        // Color by time or by distance from initial point
        let colors: Vec<f64> = ode_result
            .t
            .iter()
            .map(|t| t.to_f64().unwrap_or(0.0))
            .collect();

        let mut metadata = PlotMetadata::default();
        metadata.title = "3D Phase Space Trajectory".to_string();
        metadata.xlabel = format!("Variable {x_index}");
        metadata.ylabel = format!("Variable {y_index}");
        metadata
            .annotations
            .insert("zlabel".to_string(), format!("Variable {z_index}"));

        Ok(PhaseSpace3D {
            x,
            y,
            z,
            colors: Some(colors),
            metadata,
        })
    }

    /// Create interactive sensitivity analysis plot
    pub fn create_sensitivity_analysis(
        &self,
        base_parameters: &[f64],
        parameter_names: &[String],
        sensitivity_function: &dyn Fn(&[f64]) -> f64,
        perturbation_percent: f64,
    ) -> IntegrateResult<SensitivityPlot> {
        let n_params = base_parameters.len();
        let mut sensitivities = Vec::with_capacity(n_params);
        let base_value = sensitivity_function(base_parameters);

        for i in 0..n_params {
            let mut perturbed_params = base_parameters.to_vec();
            let perturbation = base_parameters[i] * perturbation_percent / 100.0;

            // Forward difference
            perturbed_params[i] += perturbation;
            let perturbed_value = sensitivity_function(&perturbed_params);

            // Calculate normalized sensitivity
            let sensitivity = if perturbation.abs() > 1e-12 {
                (perturbed_value - base_value) / perturbation * base_parameters[i] / base_value
            } else {
                0.0
            };

            sensitivities.push(sensitivity);
        }

        let mut metadata = PlotMetadata::default();
        metadata.title = "Parameter Sensitivity Analysis".to_string();
        metadata.xlabel = "Parameters".to_string();
        metadata.ylabel = "Normalized Sensitivity".to_string();

        Ok(SensitivityPlot {
            parameter_names: parameter_names.to_vec(),
            sensitivities,
            base_parameters: base_parameters.to_vec(),
            base_value,
            metadata,
        })
    }
}
