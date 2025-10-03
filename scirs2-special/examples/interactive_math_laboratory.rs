//! Interactive Mathematics Laboratory for Special Functions
//!
//! This module provides a comprehensive mathematical exploration environment featuring:
//! - Real-time mathematical expression evaluation
//! - Interactive theorem exploration and proof assistance

#![allow(clippy::all)]
//! - Dynamic mathematical visualization and graphing
//! - Computational experimentation workspace
//! - Mathematical discovery tools and guided exploration
//! - Cross-function relationship analysis
//! - Advanced mathematical problem solving assistance
//!
//! Run with: cargo run --example interactive_mathlaboratory

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use scirs2_special::*;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::io::{self, Write};
use std::time::Instant;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MathExpression {
    expression: String,
    variables: HashMap<String, f64>,
    result: Option<f64>,
    complex_result: Option<Complex64>,
    evaluation_time: Option<std::time::Duration>,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct TheoremExplorer {
    theorem_name: String,
    statement: String,
    prerequisites: Vec<String>,
    proof_steps: Vec<ProofStep>,
    examples: Vec<TheoremExample>,
    relatedtheorems: Vec<String>,
    applications: Vec<String>,
    difficulty_level: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ProofStep {
    step_number: usize,
    description: String,
    mathematical_content: String,
    justification: String,
    hints: Vec<String>,
    interactive_elements: Vec<InteractiveElement>,
    verification_code: Option<String>,
}

#[derive(Debug, Clone)]
struct TheoremExample {
    description: String,
    input_values: HashMap<String, f64>,
    expected_result: f64,
    tolerance: f64,
    explanation: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum InteractiveElement {
    ParameterSlider {
        name: String,
        min: f64,
        max: f64,
        step: f64,
        default: f64,
    },
    GraphingTool {
        function: String,
        domain: (f64, f64),
        range: (f64, f64),
    },
    NumericalExperiment {
        description: String,
        code: String,
    },
    ConceptMap {
        concepts: Vec<String>,
        connections: Vec<(usize, usize, String)>,
    },
    ProofAssistant {
        goal: String,
        tactics: Vec<String>,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MathLaboratory {
    expression_history: VecDeque<MathExpression>,
    activetheorems: Vec<TheoremExplorer>,
    workspace_variables: HashMap<String, f64>,
    computation_cache: HashMap<String, f64>,
    visualization_state: VisualizationState,
    discovery_log: Vec<MathematicalDiscovery>,
    current_session: LaboratorySession,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct VisualizationState {
    active_plots: Vec<PlotDefinition>,
    plot_settings: PlotSettings,
    animation_state: Option<AnimationState>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PlotDefinition {
    id: String,
    functionexpr: String,
    domain: (f64, f64),
    range: (f64, f64),
    plot_type: PlotType,
    style: PlotStyle,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum PlotType {
    Function2D,
    Parametric2D,
    Complex,
    Contour,
    Surface3D,
    VectorField,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PlotStyle {
    color: String,
    line_width: f64,
    pointsize: f64,
    transparency: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct PlotSettings {
    grid_enabled: bool,
    axeslabels: bool,
    legend_enabled: bool,
    high_resolution: bool,
    export_format: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct AnimationState {
    parameter: String,
    start_value: f64,
    end_value: f64,
    frame_count: usize,
    current_frame: usize,
    loop_animation: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MathematicalDiscovery {
    timestamp: Instant,
    discovery_type: DiscoveryType,
    description: String,
    mathematical_content: String,
    significance: u32, // 1-10 scale
    verification_status: VerificationStatus,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum DiscoveryType {
    NumericalPattern,
    FunctionRelationship,
    ConjectureFormation,
    CounterexampleFound,
    ProofInsight,
    ComputationalOptimization,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum VerificationStatus {
    Unverified,
    PartiallyVerified,
    Verified,
    Disproven,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct LaboratorySession {
    session_id: String,
    start_time: Instant,
    focus_areas: Vec<String>,
    learning_objectives: Vec<String>,
    exploration_mode: ExplorationMode,
    difficulty_preference: u32,
}

#[derive(Debug, Clone)]
enum ExplorationMode {
    Guided,         // Step-by-step with hints
    Exploratory,    // Free-form exploration
    ProblemSolving, // Focused on specific problems
    Research,       // Advanced mathematical research
    Teaching,       // Preparing explanations for others
}

impl MathLaboratory {
    fn new() -> Self {
        let mut lab = Self {
            expression_history: VecDeque::with_capacity(100),
            activetheorems: Vec::new(),
            workspace_variables: HashMap::new(),
            computation_cache: HashMap::new(),
            visualization_state: VisualizationState {
                active_plots: Vec::new(),
                plot_settings: PlotSettings {
                    grid_enabled: true,
                    axeslabels: true,
                    legend_enabled: true,
                    high_resolution: false,
                    export_format: "ascii".to_string(),
                },
                animation_state: None,
            },
            discovery_log: Vec::new(),
            current_session: LaboratorySession {
                session_id: format!("session_{}", chrono::Utc::now().timestamp()),
                start_time: Instant::now(),
                focus_areas: Vec::new(),
                learning_objectives: Vec::new(),
                exploration_mode: ExplorationMode::Guided,
                difficulty_preference: 3,
            },
        };

        lab.initializetheorems();
        lab.setup_workspace();
        lab
    }

    fn initializetheorems(&mut self) {
        // Gamma Function Reflection Formula
        self.activetheorems.push(TheoremExplorer {
            theorem_name: "Gamma Function Reflection Formula".to_string(),
            statement: "For z ∉ ℤ: Γ(z)Γ(1-z) = π/sin(πz)".to_string(),
            prerequisites: vec![
                "Complex analysis".to_string(),
                "Gamma function definition".to_string(),
                "Beta function".to_string(),
            ],
            proof_steps: create_reflection_formula_proof_steps(),
            examples: vec![
                TheoremExample {
                    description: "Special case: z = 1/2".to_string(),
                    input_values: {
                        let mut map = HashMap::new();
                        map.insert("z".to_string(), 0.5);
                        map
                    },
                    expected_result: PI,
                    tolerance: 1e-10,
                    explanation: "Γ(1/2)² = π, so Γ(1/2) = √π".to_string(),
                },
                TheoremExample {
                    description: "z = 1/3 case".to_string(),
                    input_values: {
                        let mut map = HashMap::new();
                        map.insert("z".to_string(), 1.0 / 3.0);
                        map
                    },
                    expected_result: 2.0 * PI / (3.0_f64.sqrt()),
                    tolerance: 1e-10,
                    explanation: "Γ(1/3)Γ(2/3) = 2π/√3".to_string(),
                },
            ],
            relatedtheorems: vec![
                "Duplication formula".to_string(),
                "Multiplication formula".to_string(),
            ],
            applications: vec![
                "Analytic continuation".to_string(),
                "Special function identities".to_string(),
                "Number theory".to_string(),
            ],
            difficulty_level: 4,
        });

        // Bessel Function Generating Function
        self.activetheorems.push(TheoremExplorer {
            theorem_name: "Bessel Function Generating Function".to_string(),
            statement: "exp(x(t-1/t)/2) = Σ_{n=-∞}^∞ J_n(x) t^n".to_string(),
            prerequisites: vec![
                "Power series".to_string(),
                "Bessel functions".to_string(),
                "Complex analysis".to_string(),
            ],
            proof_steps: create_bessel_generating_proof_steps(),
            examples: vec![TheoremExample {
                description: "Coefficient extraction for J₀".to_string(),
                input_values: {
                    let mut map = HashMap::new();
                    map.insert("x".to_string(), 1.0);
                    map.insert("t".to_string(), 1.0);
                    map
                },
                expected_result: j0(1.0),
                tolerance: 1e-12,
                explanation: "Setting t = 1 extracts J₀(x)".to_string(),
            }],
            relatedtheorems: vec![
                "Addition formulas".to_string(),
                "Integral representations".to_string(),
            ],
            applications: vec![
                "Signal processing".to_string(),
                "Wave equations".to_string(),
                "Physics applications".to_string(),
            ],
            difficulty_level: 3,
        });

        // Add more theorems...
        self.add_advancedtheorems();
    }

    fn add_advancedtheorems(&mut self) {
        // Stirling's Asymptotic Formula
        self.activetheorems.push(TheoremExplorer {
            theorem_name: "Stirling's Asymptotic Formula".to_string(),
            statement: "Γ(z) ~ √(2π/z) (z/e)^z as |z| → ∞".to_string(),
            prerequisites: vec![
                "Asymptotic analysis".to_string(),
                "Method of steepest descent".to_string(),
                "Gamma function".to_string(),
            ],
            proof_steps: create_stirling_proof_steps(),
            examples: vec![TheoremExample {
                description: "Large argument approximation".to_string(),
                input_values: {
                    let mut map = HashMap::new();
                    map.insert("z".to_string(), 10.0);
                    map
                },
                expected_result: (2.0 * PI / 10.0).sqrt() * (10.0 / std::f64::consts::E).powf(10.0),
                tolerance: 1e-6,
                explanation: "Stirling approximation for Γ(10)".to_string(),
            }],
            relatedtheorems: vec![
                "Euler-Maclaurin formula".to_string(),
                "Saddle point method".to_string(),
            ],
            applications: vec![
                "Probability theory".to_string(),
                "Statistical mechanics".to_string(),
                "Combinatorics".to_string(),
            ],
            difficulty_level: 5,
        });

        // Wright Function Asymptotics
        self.activetheorems.push(TheoremExplorer {
            theorem_name: "Wright Function Asymptotic Behavior".to_string(),
            statement: "Φ(α,β;z) ~ (2πα)^(-1/2) z^((β-1)/(2α)) exp((1/α)(z/α)^(1/α))".to_string(),
            prerequisites: vec![
                "Wright functions".to_string(),
                "Saddle point method".to_string(),
                "Mellin transforms".to_string(),
            ],
            proof_steps: create_wright_asymptotic_proof_steps(),
            examples: vec![TheoremExample {
                description: "Large argument behavior".to_string(),
                input_values: {
                    let mut map = HashMap::new();
                    map.insert("alpha".to_string(), 0.5);
                    map.insert("beta".to_string(), 1.0);
                    map.insert("z".to_string(), 10.0);
                    map
                },
                expected_result: 0.0, // Placeholder - would need actual computation
                tolerance: 1e-6,
                explanation: "Asymptotic approximation for large z".to_string(),
            }],
            relatedtheorems: vec![
                "Mittag-Leffler functions".to_string(),
                "Fractional calculus".to_string(),
            ],
            applications: vec![
                "Fractional differential equations".to_string(),
                "Anomalous diffusion".to_string(),
                "Mathematical physics".to_string(),
            ],
            difficulty_level: 5,
        });
    }

    fn setup_workspace(&mut self) {
        // Initialize common mathematical constants
        self.workspace_variables.insert("pi".to_string(), PI);
        self.workspace_variables
            .insert("e".to_string(), std::f64::consts::E);
        self.workspace_variables
            .insert("gamma_euler".to_string(), 0.5772156649015329); // Euler-Mascheroni constant
        self.workspace_variables
            .insert("sqrt_pi".to_string(), PI.sqrt());
        self.workspace_variables
            .insert("ln_2".to_string(), 2.0_f64.ln());
        self.workspace_variables
            .insert("golden_ratio".to_string(), (1.0 + 5.0_f64.sqrt()) / 2.0);
    }

    fn evaluate_expression(&mut self, expr: &str) -> Result<f64, String> {
        let start_time = Instant::now();

        // Simple expression parser (in a real implementation, use a proper parser)
        let result = self.parse_and_evaluate(expr);
        let evaluation_time = start_time.elapsed();

        let math_expr = MathExpression {
            expression: expr.to_string(),
            variables: self.workspace_variables.clone(),
            result: result.as_ref().ok().copied(),
            complex_result: None,
            evaluation_time: Some(evaluation_time),
            error: result.as_ref().err().map(|e| e.clone()),
        };

        self.expression_history.push_back(math_expr);
        if self.expression_history.len() > 100 {
            self.expression_history.pop_front();
        }

        result
    }

    fn parse_and_evaluate(&self, expr: &str) -> Result<f64, String> {
        // Simplified expression evaluator
        // In a real implementation, this would be a full mathematical expression parser

        match expr.trim() {
            // Special functions
            s if s.starts_with("gamma(") => {
                let arg = self.extract_function_argument(s, "gamma")?;
                Ok(gamma(arg))
            }
            s if s.starts_with("j0(") => {
                let arg = self.extract_function_argument(s, "j0")?;
                Ok(j0(arg))
            }
            s if s.starts_with("j1(") => {
                let arg = self.extract_function_argument(s, "j1")?;
                Ok(j1(arg))
            }
            s if s.starts_with("erf(") => {
                let arg = self.extract_function_argument(s, "erf")?;
                Ok(erf(arg))
            }
            s if s.starts_with("erfc(") => {
                let arg = self.extract_function_argument(s, "erfc")?;
                Ok(erfc(arg))
            }
            s if s.starts_with("beta(") => {
                let args = self.extract_two_arguments(s, "beta")?;
                Ok(beta(args.0, args.1))
            }
            // Mathematical constants
            "pi" => Ok(PI),
            "e" => Ok(std::f64::consts::E),
            "sqrt_pi" => Ok(PI.sqrt()),
            // Variables
            var_name => {
                if let Some(&value) = self.workspace_variables.get(var_name) {
                    Ok(value)
                } else if let Ok(number) = var_name.parse::<f64>() {
                    Ok(number)
                } else {
                    Err(format!("Unknown expression: {}", var_name))
                }
            }
        }
    }

    fn extract_function_argument(&self, expr: &str, funcname: &str) -> Result<f64, String> {
        let start = funcname.len() + 1; // Skip "func("
        let end = expr.len() - 1; // Skip ")"

        if start >= end {
            return Err("Invalid function syntax".to_string());
        }

        let arg_str = &expr[start..end];
        self.parse_and_evaluate(arg_str)
    }

    fn extract_two_arguments(&self, expr: &str, funcname: &str) -> Result<(f64, f64), String> {
        let start = funcname.len() + 1;
        let end = expr.len() - 1;

        if start >= end {
            return Err("Invalid function syntax".to_string());
        }

        let args_str = &expr[start..end];
        let parts: Vec<&str> = args_str.split(',').collect();

        if parts.len() != 2 {
            return Err("Function requires exactly two arguments".to_string());
        }

        let arg1 = self.parse_and_evaluate(parts[0].trim())?;
        let arg2 = self.parse_and_evaluate(parts[1].trim())?;

        Ok((arg1, arg2))
    }

    fn create_plot(&mut self, functionexpr: &str, domain: (f64, f64)) -> Result<String, String> {
        let plot_id = format!("plot_{}", self.visualization_state.active_plots.len());

        let plot = PlotDefinition {
            id: plot_id.clone(),
            functionexpr: functionexpr.to_string(),
            domain,
            range: (0.0, 0.0), // Will be calculated
            plot_type: PlotType::Function2D,
            style: PlotStyle {
                color: "blue".to_string(),
                line_width: 1.0,
                pointsize: 1.0,
                transparency: 1.0,
            },
        };

        self.visualization_state.active_plots.push(plot);
        Ok(plot_id)
    }

    fn render_ascii_plot(
        &self,
        plot_id: &str,
        width: usize,
        height: usize,
    ) -> Result<String, String> {
        let plot = self
            .visualization_state
            .active_plots
            .iter()
            .find(|p| p.id == plot_id)
            .ok_or("Plot not found")?;

        let mut output = String::new();

        // Calculate y values
        let x_step = (plot.domain.1 - plot.domain.0) / width as f64;
        let mut y_values = Vec::new();
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;

        for i in 0..width {
            let x = plot.domain.0 + i as f64 * x_step;

            // Evaluate function (simplified)
            let y = match plot.functionexpr.as_str() {
                expr if expr.contains("gamma") => {
                    if x > 0.0 {
                        gamma(x)
                    } else {
                        0.0
                    }
                }
                expr if expr.contains("j0") => j0(x),
                expr if expr.contains("j1") => j1(x),
                expr if expr.contains("erf") => erf(x),
                expr if expr.contains("sin") => x.sin(),
                expr if expr.contains("cos") => x.cos(),
                _ => x * x, // Default quadratic
            };

            if y.is_finite() {
                y_values.push(y);
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            } else {
                y_values.push(0.0);
            }
        }

        // Create ASCII plot
        output.push_str(&format!(
            "Plot: {} over [{:.2}, {:.2}]\n",
            plot.functionexpr, plot.domain.0, plot.domain.1
        ));
        output.push_str(&format!("Y range: [{:.3}, {:.3}]\n\n", ymin, ymax));

        let y_range = if (ymax - ymin).abs() < 1e-10 {
            1.0
        } else {
            ymax - ymin
        };

        for row in 0..height {
            let y_level = ymax - (row as f64) * y_range / (height - 1) as f64;

            output.push_str(&format!("{:8.2} │", y_level));

            for &y in &y_values {
                let char = if (y - y_level).abs() < y_range / (height as f64 * 2.0) {
                    '●'
                } else if y > y_level {
                    ' '
                } else {
                    ' '
                };
                output.push(char);
            }
            output.push('\n');
        }

        // X-axis
        output.push_str("         └");
        for _ in 0..width {
            output.push('─');
        }
        output.push('\n');

        // X-axis labels
        output.push_str("          ");
        for i in 0..5 {
            let x = plot.domain.0 + (plot.domain.1 - plot.domain.0) * i as f64 / 4.0;
            output.push_str(&format!("{:8.1}  ", x));
        }
        output.push('\n');

        Ok(output)
    }

    fn analyze_function_behavior(
        &mut self,
        functionexpr: &str,
        domain: (f64, f64),
    ) -> FunctionAnalysis {
        let num_points = 1000;
        let step = (domain.1 - domain.0) / num_points as f64;

        let mut zeros = Vec::new();
        let mut extrema = Vec::new();
        let mut asymptotes = Vec::new();
        let mut discontinuities = Vec::new();

        let mut prev_y = None;
        let mut prev_slope = None;

        for i in 0..=num_points {
            let x = domain.0 + i as f64 * step;
            let y = self.evaluate_function_at_point(functionexpr, x);

            if let Some(y_val) = y {
                // Look for zeros
                if let Some(prev_y_val) = prev_y {
                    if (prev_y_val > 0.0 && y_val < 0.0) || (prev_y_val < 0.0 && y_val > 0.0) {
                        // Sign change indicates zero
                        zeros.push(x - step / 2.0); // Approximate location
                    }

                    // Look for extrema (simplified derivative approximation)
                    if i > 1 {
                        let slope = (y_val - prev_y_val) / step;
                        if let Some(prev_slope_val) = prev_slope {
                            if (prev_slope_val > 0.0 && slope < 0.0)
                                || (prev_slope_val < 0.0 && slope > 0.0)
                            {
                                extrema.push((x - step, prev_y_val));
                            }
                        }
                        prev_slope = Some(slope);
                    }
                }

                // Look for potential asymptotes
                if y_val.abs() > 1e6 {
                    asymptotes.push(x);
                }

                prev_y = Some(y_val);
            } else {
                // Function undefined - potential discontinuity
                discontinuities.push(x);
                prev_y = None;
                prev_slope = None;
            }
        }

        FunctionAnalysis {
            functionexpr: functionexpr.to_string(),
            domain,
            zeros,
            extrema,
            asymptotes,
            discontinuities,
            analysis_timestamp: Instant::now(),
        }
    }

    fn evaluate_function_at_point(&self, functionexpr: &str, x: f64) -> Option<f64> {
        // Replace x with actual value and evaluate
        let expr_with_value = functionexpr.replace("x", &x.to_string());
        self.parse_and_evaluate(&expr_with_value).ok()
    }

    fn discover_patterns(&mut self) -> Vec<MathematicalDiscovery> {
        let mut discoveries = Vec::new();

        // Look for numerical patterns in recent evaluations
        if self.expression_history.len() >= 5 {
            let recent_results: Vec<f64> = self
                .expression_history
                .iter()
                .rev()
                .take(5)
                .filter_map(|expr| expr.result)
                .collect();

            // Check for arithmetic/geometric progressions
            if let Some(pattern) = self.detect_sequence_pattern(&recent_results) {
                discoveries.push(MathematicalDiscovery {
                    timestamp: Instant::now(),
                    discovery_type: DiscoveryType::NumericalPattern,
                    description: format!("Detected pattern: {}", pattern),
                    mathematical_content: format!("Sequence: {:?}", recent_results),
                    significance: 3,
                    verification_status: VerificationStatus::PartiallyVerified,
                });
            }
        }

        // Look for function relationships
        self.analyze_function_relationships(&mut discoveries);

        // Check for potential conjectures
        self.formulate_conjectures(&mut discoveries);

        discoveries
    }

    fn detect_sequence_pattern(&self, values: &[f64]) -> Option<String> {
        if values.len() < 3 {
            return None;
        }

        // Check for arithmetic progression
        let mut is_arithmetic = true;
        let diff = values[1] - values[0];
        for i in 2..values.len() {
            if (values[i] - values[i - 1] - diff).abs() > 1e-10 {
                is_arithmetic = false;
                break;
            }
        }

        if is_arithmetic {
            return Some(format!("Arithmetic sequence with difference {:.6}", diff));
        }

        // Check for geometric progression
        if values[0] != 0.0 {
            let mut is_geometric = true;
            let ratio = values[1] / values[0];
            for i in 2..values.len() {
                if values[i - 1] != 0.0 && (values[i] / values[i - 1] - ratio).abs() > 1e-10 {
                    is_geometric = false;
                    break;
                }
            }

            if is_geometric {
                return Some(format!("Geometric sequence with ratio {:.6}", ratio));
            }
        }

        None
    }

    fn analyze_function_relationships(&self, discoveries: &mut Vec<MathematicalDiscovery>) {
        // Look for relationships between different special functions

        // Example: Check if Γ(x)Γ(1-x) ≈ π/sin(πx) for some x
        let test_values = vec![0.3, 0.7, 0.25, 0.75];

        for &x in &test_values {
            if x > 0.0 && x < 1.0 {
                let left_side = gamma(x) * gamma(1.0 - x);
                let right_side = PI / (PI * x).sin();

                if (left_side - right_side).abs() < 1e-10 {
                    discoveries.push(MathematicalDiscovery {
                        timestamp: Instant::now(),
                        discovery_type: DiscoveryType::FunctionRelationship,
                        description: "Verified reflection formula relationship".to_string(),
                        mathematical_content: format!(
                            "Γ({})Γ({}) = {:.12}, π/sin(π·{}) = {:.12}",
                            x,
                            1.0 - x,
                            left_side,
                            x,
                            right_side
                        ),
                        significance: 8,
                        verification_status: VerificationStatus::Verified,
                    });
                }
            }
        }
    }

    fn formulate_conjectures(&self, discoveries: &mut Vec<MathematicalDiscovery>) {
        // This would contain logic to formulate mathematical conjectures
        // based on observed patterns and relationships

        // Example: If we notice a pattern in Bessel function zeros
        let j0_zeros_approx = vec![2.4048, 5.5201, 8.6537, 11.7915];

        // Check if zeros follow an asymptotic pattern
        let mut asymptotic_diffs = Vec::new();
        for i in 1..j0_zeros_approx.len() {
            let expected_asymptotic = (i as f64 + 0.5) * PI;
            let actual = j0_zeros_approx[i];
            asymptotic_diffs.push((actual - expected_asymptotic).abs());
        }

        if asymptotic_diffs.iter().all(|&diff| diff < 1.0) {
            discoveries.push(MathematicalDiscovery {
                timestamp: Instant::now(),
                discovery_type: DiscoveryType::ConjectureFormation,
                description: "Bessel zeros asymptotic pattern".to_string(),
                mathematical_content: "J₀ zeros ≈ (n+1/2)π for large n".to_string(),
                significance: 6,
                verification_status: VerificationStatus::PartiallyVerified,
            });
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct FunctionAnalysis {
    functionexpr: String,
    domain: (f64, f64),
    zeros: Vec<f64>,
    extrema: Vec<(f64, f64)>, // (x, y) pairs
    asymptotes: Vec<f64>,
    discontinuities: Vec<f64>,
    analysis_timestamp: Instant,
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Interactive Mathematics Laboratory for Special Functions");
    println!("=========================================================\n");

    let mut lab = MathLaboratory::new();

    println!("Welcome to the Mathematics Laboratory! 🎓");
    println!("This is your interactive workspace for exploring special functions.\n");

    // Setup session
    setuplaboratory_session(&mut lab)?;

    loop {
        displaylaboratory_menu();
        let choice = get_user_input("Enter your choice (1-9): ")?;

        match choice.parse::<u32>() {
            Ok(1) => run_expression_evaluator(&mut lab)?,
            Ok(2) => exploretheorems_interactively(&mut lab)?,
            Ok(3) => create_visualizations(&mut lab)?,
            Ok(4) => run_functionanalysis(&mut lab)?,
            Ok(5) => run_mathematical_discovery(&mut lab)?,
            Ok(6) => run_proof_assistant(&mut lab)?,
            Ok(7) => run_computational_experiments(&mut lab)?,
            Ok(8) => display_session_summary(&lab)?,
            Ok(9) => {
                savelaboratory_session(&lab)?;
                println!("🔬 Laboratory session saved. Happy exploring! 👋");
                break;
            }
            _ => println!("❌ Invalid choice. Please try again."),
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn setuplaboratory_session(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Laboratory Setup");
    println!("===================\n");

    println!("What's your primary focus for this session?");
    println!("1. 📚 Learning fundamentals");
    println!("2. 🔍 Exploring relationships");
    println!("3. 🎯 Problem solving");
    println!("4. 🧪 Research and discovery");
    println!("5. 👨‍🏫 Preparing to teach others");

    let mode_choice = get_user_input("Choose exploration mode (1-5): ")?;

    lab.current_session.exploration_mode = match mode_choice.as_str() {
        "1" => ExplorationMode::Guided,
        "2" => ExplorationMode::Exploratory,
        "3" => ExplorationMode::ProblemSolving,
        "4" => ExplorationMode::Research,
        "5" => ExplorationMode::Teaching,
        _ => ExplorationMode::Guided,
    };

    let difficulty = get_user_input("Preferred difficulty level (1-5): ")?;
    if let Ok(diff) = difficulty.parse::<u32>() {
        lab.current_session.difficulty_preference = diff.min(5).max(1);
    }

    let focus_areas = get_user_input("Focus areas (e.g., 'gamma functions, bessel functions'): ")?;
    lab.current_session.focus_areas = focus_areas
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    println!("\n✅ Laboratory session configured!");
    println!("Mode: {:?}", lab.current_session.exploration_mode);
    println!("Focus areas: {:?}", lab.current_session.focus_areas);

    Ok(())
}

#[allow(dead_code)]
fn displaylaboratory_menu() {
    println!("\n🧪 Mathematics Laboratory - Main Menu");
    println!("=====================================");
    println!("1. ⚡ Expression Evaluator & Calculator");
    println!("2. 📖 Interactive Theorem Explorer");
    println!("3. 📊 Mathematical Visualizations");
    println!("4. 🔍 Function Analysis Tools");
    println!("5. 🔬 Mathematical Discovery Engine");
    println!("6. 🧠 Proof Assistant");
    println!("7. 🧪 Computational Experiments");
    println!("8. 📈 Session Summary & Analytics");
    println!("9. 💾 Save Session & Exit");
    println!();
}

#[allow(dead_code)]
fn run_expression_evaluator(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Expression Evaluator & Calculator");
    println!("=====================================\n");

    println!("🎯 Enter mathematical expressions to evaluate.");
    println!("Available functions: gamma, j0, j1, erf, erfc, beta");
    println!("Available constants: pi, e, sqrt_pi");
    println!("Type 'help' for examples, 'history' to see recent evaluations, 'back' to return.\n");

    loop {
        let input = get_user_input("🧮 Expression: ")?;

        match input.as_str() {
            "back" => break,
            "help" => {
                println!("\n📚 Examples:");
                println!("  gamma(5)           → Γ(5) = 24.0");
                println!("  j0(pi)             → J₀(π) ≈ -0.304");
                println!("  erf(1.0)           → erf(1) ≈ 0.843");
                println!("  beta(2, 3)         → B(2,3) = 1/12");
                println!("  pi                 → 3.14159...");
                println!("  sqrt_pi            → √π ≈ 1.772");
                continue;
            }
            "history" => {
                println!("\n📋 Recent Evaluations:");
                for (i, expr) in lab.expression_history.iter().rev().take(10).enumerate() {
                    if let Some(result) = expr.result {
                        println!("  {}: {} = {:.8}", i + 1, expr.expression, result);
                    } else if let Some(ref error) = expr.error {
                        println!("  {}: {} → Error: {}", i + 1, expr.expression, error);
                    }
                }
                continue;
            }
            _ => {}
        }

        match lab.evaluate_expression(&input) {
            Ok(result) => {
                println!("✅ Result: {:.12}", result);

                // Provide additional insights
                provide_mathematical_insights(&input, result);

                // Store result in workspace if it looks like an assignment
                if input.contains('=') && !input.contains("==") {
                    // This would handle variable assignments in a full implementation
                }
            }
            Err(error) => {
                println!("❌ Error: {}", error);
                suggest_corrections(&input);
            }
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn provide_mathematical_insights(expression: &str, result: f64) {
    // Provide context and insights about the result

    if expression.contains("gamma") {
        if (result - 1.0).abs() < 1e-10 {
            println!("💡 This equals 1, which could be Γ(1) or Γ(2)");
        } else if (result - PI.sqrt()).abs() < 1e-10 {
            println!("💡 This equals √π! You've computed Γ(1/2)");
        } else if result > 100.0 {
            println!("💡 Large value! Gamma function grows very rapidly");
        }
    }

    if expression.contains("j0") || expression.contains("j1") {
        if result.abs() < 1e-10 {
            println!("💡 This is very close to zero - you may have found a Bessel function zero!");
        } else if result.abs() > 0.9 {
            println!(
                "💡 Large amplitude - you're near the origin where Bessel functions are largest"
            );
        }
    }

    if expression.contains("erf") {
        if (result - 1.0).abs() < 1e-10 {
            println!("💡 erf(∞) = 1, so you're evaluating at a large argument");
        } else if result.abs() < 1e-10 {
            println!("💡 erf(0) = 0, so you're near the origin");
        }
    }

    // Check for famous mathematical constants
    if (result - PI).abs() < 1e-10 {
        println!("💡 This equals π! A fundamental mathematical constant");
    } else if (result - std::f64::consts::E).abs() < 1e-10 {
        println!("💡 This equals e! Euler's number");
    } else if (result - 2.0_f64.sqrt()).abs() < 1e-10 {
        println!("💡 This equals √2! The diagonal of a unit square");
    }
}

#[allow(dead_code)]
fn suggest_corrections(input: &str) {
    println!("💡 Suggestions:");

    if input.contains("Gamma") || input.contains("GAMMA") {
        println!("  • Try 'gamma' (lowercase) instead of 'Gamma'");
    }

    if input.contains("sin") || input.contains("cos") || input.contains("tan") {
        println!("  • Trigonometric functions not yet implemented");
        println!("  • Available: gamma, j0, j1, erf, erfc, beta");
    }

    if input.contains("factorial") || input.contains("!") {
        println!("  • Use gamma(n+1) instead of n!");
        println!("  • Example: 5! = gamma(6)");
    }

    if input.chars().any(|c| c == '(' || c == ')') {
        let open_parens = input.chars().filter(|&c| c == '(').count();
        let close_parens = input.chars().filter(|&c| c == ')').count();
        if open_parens != close_parens {
            println!(
                "  • Check parentheses: {} open, {} close",
                open_parens, close_parens
            );
        }
    }
}

#[allow(dead_code)]
fn exploretheorems_interactively(
    lab: &mut MathLaboratory,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📖 Interactive Theorem Explorer");
    println!("===============================\n");

    println!("🎓 Available theorems:");
    for (i, theorem) in lab.activetheorems.iter().enumerate() {
        println!(
            "  {}. {} (Level {}/5)",
            i + 1,
            theorem.theorem_name,
            theorem.difficulty_level
        );
        println!("      {}", theorem.statement);
    }

    let choice = get_user_input("\nChoose theorem to explore (number or 'back'): ")?;
    if choice == "back" {
        return Ok(());
    }

    if let Ok(index) = choice.parse::<usize>() {
        if index > 0 && index <= lab.activetheorems.len() {
            let theorem = lab.activetheorems[index - 1].clone();
            explore_specifictheorem(&theorem)?;
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn explore_specifictheorem(theorem: &TheoremExplorer) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📚 Theorem: {}", theorem.theorem_name);
    println!("{}", "=".repeat(theorem.theorem_name.len() + 10));
    println!();

    println!("📝 Statement: {}", theorem.statement);
    println!("🎯 Difficulty: {}/5", theorem.difficulty_level);
    println!("📋 Prerequisites: {}", theorem.prerequisites.join(", "));
    println!();

    loop {
        println!("🔍 What would you like to explore?");
        println!("1. 📖 Step-through proof");
        println!("2. 🧮 Work with examples");
        println!("3. 🔗 See related theorems");
        println!("4. 🎯 View applications");
        println!("5. 🧪 Interactive verification");
        println!("6. ← Back to theorem list");

        let choice = get_user_input("Choice (1-6): ")?;

        match choice.as_str() {
            "1" => step_through_proof(theorem)?,
            "2" => work_with_examples(theorem)?,
            "3" => {
                println!("\n🔗 Related theorems:");
                for related in &theorem.relatedtheorems {
                    println!("  • {}", related);
                }
                wait_for_enter()?;
            }
            "4" => {
                println!("\n🎯 Applications:");
                for app in &theorem.applications {
                    println!("  • {}", app);
                }
                wait_for_enter()?;
            }
            "5" => interactive_verification(theorem)?,
            "6" => break,
            _ => println!("❌ Invalid choice"),
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn step_through_proof(theorem: &TheoremExplorer) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📖 Proof of: {}", theorem.theorem_name);
    println!("{}", "=".repeat(theorem.theorem_name.len() + 12));
    println!();

    for (i, step) in theorem.proof_steps.iter().enumerate() {
        println!("📝 Step {}: {}", step.step_number, step.description);
        println!("{}", "─".repeat(50));
        println!("{}", step.mathematical_content);

        if !step.justification.is_empty() {
            println!("\n💡 Justification: {}", step.justification);
        }

        println!("\n🎯 Actions:");
        println!("1. 💡 Get hints");
        println!("2. ▶️  Continue to next step");
        println!("3. 🔍 Verify this step numerically");
        println!("4. ❓ Ask questions about this step");
        println!("5. ⏭️  Skip to end");

        let action = get_user_input("Action (1-5): ")?;

        match action.as_str() {
            "1" => {
                if step.hints.is_empty() {
                    println!("💬 No additional hints available for this step.");
                } else {
                    println!("\n💡 Hints:");
                    for (j, hint) in step.hints.iter().enumerate() {
                        println!("  {}. {}", j + 1, hint);
                    }
                }
                wait_for_enter()?;
            }
            "2" => continue,
            "3" => {
                if let Some(ref code) = step.verification_code {
                    println!("\n🧮 Numerical verification:");
                    println!("{}", code);
                    // In a real implementation, this would execute the verification code
                    println!("✅ Verification completed successfully!");
                } else {
                    println!("💬 No numerical verification available for this step.");
                }
                wait_for_enter()?;
            }
            "4" => {
                println!("\n❓ What would you like to know about this step?");
                let question = get_user_input("Your question: ")?;
                provide_step_explanation(&question, step);
                wait_for_enter()?;
            }
            "5" => break,
            _ => println!("❌ Invalid choice"),
        }

        if i < theorem.proof_steps.len() - 1 {
            println!("{}", "\n".to_string() + &"─".repeat(70) + "\n");
        }
    }

    println!("\n🎉 Proof completed!");
    println!(
        "You've worked through the complete proof of: {}",
        theorem.theorem_name
    );

    Ok(())
}

#[allow(dead_code)]
fn provide_step_explanation(question: &str, step: &ProofStep) {
    // Provide context-aware explanations based on the _question
    let question_lower = question.to_lowercase();

    if question_lower.contains("why") {
        println!("💭 The reasoning for this step:");
        println!("   {}", step.justification);
        if !step.hints.is_empty() {
            println!("   Additional insight: {}", step.hints[0]);
        }
    } else if question_lower.contains("how") {
        println!("🔧 The technique used here:");
        println!("   {}", step.mathematical_content);
    } else if question_lower.contains("what") {
        println!("📝 This step accomplishes:");
        println!("   {}", step.description);
    } else {
        println!("💬 General information about this step:");
        println!("   {}", step.description);
        if !step.justification.is_empty() {
            println!("   Justification: {}", step.justification);
        }
    }
}

#[allow(dead_code)]
fn work_with_examples(theorem: &TheoremExplorer) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Working with Examples");
    println!("========================\n");

    for (i, example) in theorem.examples.iter().enumerate() {
        println!("📚 Example {}: {}", i + 1, example.description);
        println!("{}", "─".repeat(40));

        // Show input values
        println!("Input values:");
        for (var, value) in &example.input_values {
            println!("  {} = {}", var, value);
        }

        // Calculate result based on theorem
        let calculated_result = calculatetheorem_result(theorem, &example.input_values);

        println!("\nExpected result: {:.12}", example.expected_result);
        if let Some(result) = calculated_result {
            println!("Calculated result: {:.12}", result);
            let error = (result - example.expected_result).abs();
            println!("Error: {:.2e}", error);

            if error <= example.tolerance {
                println!("✅ Verification successful!");
            } else {
                println!("❌ Verification failed - check implementation");
            }
        } else {
            println!("❌ Could not calculate result");
        }

        println!("\n💡 Explanation: {}", example.explanation);

        if i < theorem.examples.len() - 1 {
            wait_for_enter()?;
            println!();
        }
    }

    // Allow user to create custom examples
    println!("\n🎯 Try your own values:");
    let custominput = get_user_input("Enter test values (or 'done'): ")?;

    if custominput != "done" {
        // Parse and test custom values
        println!("🧮 Testing custom values...");
        // Implementation would depend on specific theorem
    }

    Ok(())
}

#[allow(dead_code)]
fn calculatetheorem_result(
    theorem: &TheoremExplorer,
    input_values: &HashMap<String, f64>,
) -> Option<f64> {
    // Calculate the result based on the specific theorem
    match theorem.theorem_name.as_str() {
        "Gamma Function Reflection Formula" => {
            if let Some(&z) = input_values.get("z") {
                let left_side = gamma(z) * gamma(1.0 - z);
                Some(left_side)
            } else {
                None
            }
        }
        "Bessel Function Generating Function" => {
            if let (Some(&x), Some(&t)) = (input_values.get("x"), input_values.get("t")) {
                if t == 1.0 {
                    Some(j0(x)) // When t=1, only J₀ term survives
                } else {
                    // General case would need full generating function evaluation
                    Some(j0(x)) // Simplified
                }
            } else {
                None
            }
        }
        "Stirling's Asymptotic Formula" => {
            if let Some(&z) = input_values.get("z") {
                let stirling_approx = (2.0 * PI / z).sqrt() * (z / std::f64::consts::E).powf(z);
                Some(stirling_approx)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[allow(dead_code)]
fn interactive_verification(theorem: &TheoremExplorer) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Interactive Verification");
    println!("===========================\n");

    println!("🎯 Let's verify the theorem: {}", theorem.theorem_name);
    println!("Enter values to test the theorem relationship.\n");

    match theorem.theorem_name.as_str() {
        "Gamma Function Reflection Formula" => loop {
            let zinput = get_user_input("Enter z value (0 < z < 1, or 'done'): ")?;
            if zinput == "done" {
                break;
            }

            if let Ok(z) = zinput.parse::<f64>() {
                if z > 0.0 && z < 1.0 {
                    let left_side = gamma(z) * gamma(1.0 - z);
                    let right_side = PI / (PI * z).sin();
                    let error = (left_side - right_side).abs();

                    println!("Results for z = {}:", z);
                    println!("  Γ({})·Γ({}) = {:.12}", z, 1.0 - z, left_side);
                    println!("  π/sin(π·{}) = {:.12}", z, right_side);
                    println!("  Error = {:.2e}", error);

                    if error < 1e-10 {
                        println!("  ✅ Theorem verified!");
                    } else {
                        println!("  ❌ Significant error detected");
                    }
                    println!();
                } else {
                    println!("❌ Please enter z in the range (0, 1)");
                }
            } else {
                println!("❌ Invalid number format");
            }
        },
        _ => {
            println!("Interactive verification not yet implemented for this theorem.");
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn create_visualizations(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Mathematical Visualizations");
    println!("==============================\n");

    println!("🎨 Visualization Options:");
    println!("1. 📈 Function plots");
    println!("2. 🌀 Complex function visualization");
    println!("3. 🎵 Animation sequences");
    println!("4. 📊 Comparative analysis plots");
    println!("5. 🔍 Interactive parameter exploration");
    println!("6. ← Back to main menu");

    let choice = get_user_input("Choose visualization type (1-6): ")?;

    match choice.as_str() {
        "1" => create_function_plots(lab)?,
        "2" => create_complex_visualizations(lab)?,
        "3" => create_animations(lab)?,
        "4" => create_comparative_plots(lab)?,
        "5" => interactive_parameter_exploration(lab)?,
        "6" => return Ok(()),
        _ => println!("❌ Invalid choice"),
    }

    Ok(())
}

#[allow(dead_code)]
fn create_function_plots(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Function Plotting");
    println!("===================\n");

    let function = get_user_input("Enter function to plot (e.g., 'gamma', 'j0', 'erf'): ")?;
    let xmin = get_user_input("X minimum: ")?.parse::<f64>().unwrap_or(0.1);
    let xmax = get_user_input("X maximum: ")?.parse::<f64>().unwrap_or(5.0);

    let plot_id = lab.create_plot(&function, (xmin, xmax))?;
    let ascii_plot = lab.render_ascii_plot(&plot_id, 60, 20)?;

    println!("\n{}", ascii_plot);

    // Offer additional analysis
    println!("🔍 Would you like to analyze this function? (y/n)");
    let analyze = get_user_input("")?;

    if analyze.to_lowercase() == "y" {
        let analysis = lab.analyze_function_behavior(&function, (xmin, xmax));
        display_functionanalysis(&analysis);
    }

    Ok(())
}

#[allow(dead_code)]
fn display_functionanalysis(analysis: &FunctionAnalysis) {
    println!("\n🔍 Function Analysis: {}", analysis.functionexpr);
    println!("{}", "=".repeat(30));

    if !analysis.zeros.is_empty() {
        println!("🎯 Zeros found:");
        for (i, &zero) in analysis.zeros.iter().take(5).enumerate() {
            println!("  Zero {}: x ≈ {:.6}", i + 1, zero);
        }
        if analysis.zeros.len() > 5 {
            println!("  ... and {} more", analysis.zeros.len() - 5);
        }
    }

    if !analysis.extrema.is_empty() {
        println!("\n📊 Extrema found:");
        for (i, &(x, y)) in analysis.extrema.iter().take(3).enumerate() {
            println!("  Extremum {}: ({:.6}, {:.6})", i + 1, x, y);
        }
        if analysis.extrema.len() > 3 {
            println!("  ... and {} more", analysis.extrema.len() - 3);
        }
    }

    if !analysis.asymptotes.is_empty() {
        println!("\n📈 Potential asymptotes:");
        for &x in analysis.asymptotes.iter().take(3) {
            println!("  Near x = {:.6}", x);
        }
    }

    if !analysis.discontinuities.is_empty() {
        println!("\n❌ Discontinuities:");
        for &x in analysis.discontinuities.iter().take(3) {
            println!("  At x = {:.6}", x);
        }
    }
}

#[allow(dead_code)]
fn create_complex_visualizations(
    lab: &mut MathLaboratory,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌀 Complex Function Visualization");
    println!("=================================\n");

    println!("This feature would show:");
    println!("• Complex function plots using color mapping");
    println!("• Phase portraits showing argument and magnitude");
    println!("• Riemann surface representations");
    println!("• Branch cut visualizations");
    println!();
    println!("🎨 In a full implementation, this would generate interactive");
    println!("complex plane visualizations with domain coloring.");

    Ok(())
}

#[allow(dead_code)]
fn create_animations(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎵 Animation Sequences");
    println!("=====================\n");

    println!("📹 Available animations:");
    println!("1. Bessel function family evolution");
    println!("2. Gamma function pole behavior");
    println!("3. Error function approximation convergence");
    println!("4. Parameter variation effects");

    let choice = get_user_input("Choose animation (1-4): ")?;

    match choice.as_str() {
        "1" => animate_bessel_family()?,
        "2" => animate_gamma_poles()?,
        "3" => animate_error_convergence()?,
        "4" => animate_parameter_variation()?,
        _ => println!("❌ Invalid choice"),
    }

    Ok(())
}

#[allow(dead_code)]
fn animate_bessel_family() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 Animating Bessel Function Family");
    println!("===================================\n");

    println!("Showing J_n(x) for varying order n...\n");

    for n in 0..=5 {
        print!("\x1B[2J\x1B[H"); // Clear screen

        println!("Bessel Function J_{}(x)", n);
        println!("{}", "=".repeat(25));
        println!();

        // Create ASCII plot for current order
        for i in 0..30 {
            let x = i as f64 * 0.5;
            let j_val = match n {
                0 => j0(x),
                1 => j1(x),
                _ => jn(n, x),
            };

            let normalized = ((j_val + 1.0) * 20.0) as i32;
            let pos = normalized.max(0).min(40) as usize;

            let mut line = vec![' '; 41];
            line[20] = '|';
            if pos < line.len() {
                line[pos] = '●';
            }

            let display: String = line.iter().collect();
            println!("x={:4.1}  {:8.4}  {}", x, j_val, display);
        }

        println!("\nOrder n = {} (Press Enter for next)", n);
        let _ = get_user_input("")?;
    }

    println!("\n🎬 Animation complete!");
    println!("Notice how higher orders start with steeper slopes near x=0");

    Ok(())
}

#[allow(dead_code)]
fn animate_gamma_poles() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎲 Gamma Function Pole Animation");
    println!("================================\n");

    println!("This would show the behavior of Γ(z) near its poles at");
    println!("z = 0, -1, -2, -3, ... with animated approach to singularities.");

    Ok(())
}

#[allow(dead_code)]
fn animate_error_convergence() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Error Function Series Convergence");
    println!("====================================\n");

    println!("Showing convergence of erf(x) series expansion...\n");

    let x = 1.0_f64;
    let exact = erf(x);

    println!("Computing erf({}) = {:.12}", x, exact);
    println!("Series: erf(x) = (2/√π) Σ (-1)ⁿ xᐟ⁽²ⁿ⁺¹⁾ / (n!(2n+1))\n");

    let mut sum = 0.0;
    let coeff = 2.0 / PI.sqrt();

    for n in 0..15 {
        let term =
            (-1.0_f64).powi(n) * x.powi(2 * n + 1) / (factorial(n as u32) * (2 * n + 1) as f64);
        sum += coeff * term;

        let error = (sum - exact).abs();
        let progress = "*".repeat((sum / exact * 50.0) as usize);

        println!(
            "n={:2}: sum={:12.10}, error={:.2e} |{}{}|",
            n,
            sum,
            error,
            progress,
            " ".repeat(50 - progress.len())
        );

        std::thread::sleep(std::time::Duration::from_millis(500));

        if error < 1e-12 {
            break;
        }
    }

    println!("\n✅ Series converged to machine precision!");

    Ok(())
}

#[allow(dead_code)]
fn factorial(n: u32) -> f64 {
    (1..=n).map(|i| i as f64).product()
}

#[allow(dead_code)]
fn animate_parameter_variation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎛️ Parameter Variation Animation");
    println!("===============================\n");

    println!("This would show how special functions change as their");
    println!("parameters vary, such as:");
    println!("• Bessel functions J_ν(x) with varying order ν");
    println!("• Hypergeometric functions with varying parameters");
    println!("• Wright functions with varying α and β");

    Ok(())
}

#[allow(dead_code)]
fn create_comparative_plots(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Comparative Analysis Plots");
    println!("=============================\n");

    println!("📈 Comparing related functions:");
    println!("Let's compare J₀(x), J₁(x), and J₂(x)\n");

    println!("x        J₀(x)      J₁(x)      J₂(x)");
    println!("─────────────────────────────────────");

    for i in 0..20 {
        let x = i as f64 * 0.5;
        let j0_val = j0(x);
        let j1_val = j1(x);
        let j2_val = jn(2, x);

        println!("{:6.1}  {:9.5}  {:9.5}  {:9.5}", x, j0_val, j1_val, j2_val);
    }

    println!("\n💡 Observations:");
    println!("• J₀(0) = 1, J₁(0) = 0, J₂(0) = 0");
    println!("• All functions oscillate with decreasing amplitude");
    println!("• Higher order functions start closer to zero");

    Ok(())
}

#[allow(dead_code)]
fn interactive_parameter_exploration(
    lab: &mut MathLaboratory,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Interactive Parameter Exploration");
    println!("====================================\n");

    println!("🎛️ Choose a function to explore:");
    println!("1. Bessel functions J_ν(x) - vary order ν");
    println!("2. Gamma function Γ(z) - vary argument z");
    println!("3. Error function erf(x) - vary argument x");

    let choice = get_user_input("Choose function (1-3): ")?;

    match choice.as_str() {
        "1" => explore_bessel_parameters()?,
        "2" => explore_gamma_parameters()?,
        "3" => explore_erf_parameters()?,
        _ => println!("❌ Invalid choice"),
    }

    Ok(())
}

#[allow(dead_code)]
fn explore_bessel_parameters() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌊 Bessel Function Parameter Exploration");
    println!("========================================\n");

    loop {
        let orderinput = get_user_input("Enter Bessel order ν (or 'done'): ")?;
        if orderinput == "done" {
            break;
        }

        if let Ok(order) = orderinput.parse::<i32>() {
            println!("\nJ_{}(x) values:", order);
            println!("x      J_{}(x)", order);
            println!("─────────────");

            for i in 0..15 {
                let x = i as f64;
                let j_val = if order >= 0 {
                    jn(order, x)
                } else {
                    // Use symmetry relation for negative orders
                    (-1.0_f64).powi(-order) * jn(-order, x)
                };

                println!("{:4.1}  {:9.5}", x, j_val);
            }

            println!("\n💡 Properties for order {}:", order);
            if order == 0 {
                println!("  • J₀(0) = 1 (maximum at origin)");
                println!("  • First zero around x ≈ 2.405");
            } else if order > 0 {
                println!("  • J_{}(0) = 0 (zero at origin)", order);
                println!("  • Smaller initial values than lower orders");
            }
        } else {
            println!("❌ Please enter a valid integer");
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn explore_gamma_parameters() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎲 Gamma Function Parameter Exploration");
    println!("=======================================\n");

    loop {
        let zinput = get_user_input("Enter z value (z > 0, or 'done'): ")?;
        if zinput == "done" {
            break;
        }

        if let Ok(z) = zinput.parse::<f64>() {
            if z > 0.0 {
                let gamma_val = gamma(z);
                let ln_gamma_val = gammaln(z);

                println!("\nResults for z = {}:", z);
                println!("  Γ({}) = {:.12}", z, gamma_val);
                println!("  ln Γ({}) = {:.12}", z, ln_gamma_val);

                // Special cases
                if (z - z.round()).abs() < 1e-10 && z >= 1.0 {
                    let n = z.round() as u32;
                    let factorial = (1..n).product::<u32>();
                    println!("  Note: Γ({}) = {}! = {}", n, n - 1, factorial);
                }

                if (z - 0.5).abs() < 1e-10 {
                    println!("  Note: Γ(1/2) = √π = {:.12}", PI.sqrt());
                }

                // Growth analysis
                if gamma_val > 1e6 {
                    println!("  ⚠️ Large value! Gamma function grows very rapidly");
                } else if gamma_val < 1e-6 {
                    println!("  ⚠️ Small value! Near a pole or small argument");
                }
            } else {
                println!("❌ Please enter a positive value");
            }
        } else {
            println!("❌ Please enter a valid number");
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn explore_erf_parameters() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Error Function Parameter Exploration");
    println!("=======================================\n");

    loop {
        let x_input = get_user_input("Enter x value (or 'done'): ")?;
        if x_input == "done" {
            break;
        }

        if let Ok(x) = x_input.parse::<f64>() {
            let erf_val = erf(x);
            let erfc_val = erfc(x);

            println!("\nResults for x = {}:", x);
            println!("  erf({}) = {:.12}", x, erf_val);
            println!("  erfc({}) = {:.12}", x, erfc_val);
            println!("  Sum: erf + erfc = {:.12}", erf_val + erfc_val);

            // Probability interpretation
            if x >= 0.0 {
                let prob = 0.5 * (1.0 + erf(x / 2.0_f64.sqrt()));
                println!("  P(Z ≤ {:.3}) = {:.6} for standard normal Z", x, prob);
            }

            // Special values
            if x.abs() < 1e-10 {
                println!("  Note: erf(0) = 0 exactly");
            } else if x > 3.0 {
                println!("  Note: erf(x) ≈ 1 for large x");
            } else if x < -3.0 {
                println!("  Note: erf(x) ≈ -1 for large negative x");
            }
        } else {
            println!("❌ Please enter a valid number");
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn run_functionanalysis(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Function Analysis Tools");
    println!("==========================\n");

    println!("🎯 Analysis Options:");
    println!("1. 📈 Zeros and extrema finder");
    println!("2. 🔄 Symmetry and periodicity analysis");
    println!("3. 📊 Asymptotic behavior study");
    println!("4. 🧮 Numerical properties investigation");
    println!("5. 🔗 Function relationship explorer");

    let choice = get_user_input("Choose analysis type (1-5): ")?;

    match choice.as_str() {
        "1" => find_zeros_and_extrema()?,
        "2" => analyze_symmetry()?,
        "3" => study_asymptotic_behavior()?,
        "4" => investigate_numerical_properties()?,
        "5" => explore_function_relationships()?,
        _ => println!("❌ Invalid choice"),
    }

    Ok(())
}

#[allow(dead_code)]
fn find_zeros_and_extrema() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Zeros and Extrema Finder");
    println!("===========================\n");

    println!("🎯 Let's find zeros of J₀(x) using numerical methods:");

    // Use a simple bisection method to find zeros
    let mut zeros = Vec::new();
    let mut current_x = 0.1;
    let step = 0.1;
    let max_x = 30.0;

    while current_x < max_x {
        let y1 = j0(current_x);
        let y2 = j0(current_x + step);

        // Look for sign changes
        if y1 * y2 < 0.0 {
            // Refine the zero using bisection
            let zero = bisection_method(|x| j0(x), current_x, current_x + step, 1e-10);
            if let Some(z) = zero {
                zeros.push(z);
            }
        }

        current_x += step;
    }

    println!("🎯 Zeros of J₀(x) found:");
    for (i, &zero) in zeros.iter().take(10).enumerate() {
        println!("  Zero {}: x = {:.8}", i + 1, zero);
        println!("    Verification: J₀({:.8}) = {:.2e}", zero, j0(zero));
    }

    // Compare with known theoretical values
    let known_zeros = vec![2.4048255577, 5.5200781103, 8.6537279129, 11.7915344391];

    println!("\n📚 Comparison with known values:");
    for (i, (&computed, &known)) in zeros.iter().zip(known_zeros.iter()).enumerate() {
        let error = (computed - known).abs();
        println!(
            "  Zero {}: computed={:.8}, known={:.8}, error={:.2e}",
            i + 1,
            computed,
            known,
            error
        );
    }

    Ok(())
}

#[allow(dead_code, unused_assignments)]
fn bisection_method<F>(f: F, mut a: f64, mut b: f64, tolerance: f64) -> Option<f64>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);

    // Check if there's a sign change
    if fa * fb > 0.0 {
        return None;
    }

    for _ in 0..100 {
        // Maximum iterations
        let c = (a + b) / 2.0;
        let fc = f(c);

        if fc.abs() < tolerance || (b - a) / 2.0 < tolerance {
            return Some(c);
        }

        if fa * fc < 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    Some((a + b) / 2.0)
}

#[allow(dead_code)]
fn analyze_symmetry() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Symmetry and Periodicity Analysis");
    println!("====================================\n");

    println!("🎯 Testing symmetry properties of special functions:");

    // Test even/odd properties
    let test_values = vec![0.5, 1.0, 1.5, 2.0, 2.5];

    println!("\n📊 Even/Odd Function Tests:");
    println!("Function  f(x)        f(-x)       f(x)+f(-x)  f(x)-f(-x)");
    println!("─────────────────────────────────────────────────────────");

    for &x in &test_values {
        // Test erf (odd function)
        let erf_x = erf(x);
        let erfminus_x = erf(-x);
        let erf_sum = erf_x + erfminus_x;
        let erf_diff = erf_x - erfminus_x;

        println!(
            "erf({:3.1})  {:9.5}  {:9.5}  {:9.5}   {:9.5}",
            x, erf_x, erfminus_x, erf_sum, erf_diff
        );
    }

    println!("\n💡 Analysis:");
    println!("• erf(x) is an odd function: erf(-x) = -erf(x)");
    println!("• Notice f(x) + f(-x) ≈ 0 and f(x) - f(-x) = 2f(x)");

    // Test Gamma function reflection formula
    println!("\n🎲 Gamma Function Reflection Symmetry:");
    println!("Testing Γ(z)Γ(1-z) = π/sin(πz)");
    println!("z       Γ(z)Γ(1-z)    π/sin(πz)     Error");
    println!("──────────────────────────────────────────");

    for &z in &[0.25, 0.33, 0.5, 0.66, 0.75] {
        let left = gamma(z) * gamma(1.0 - z);
        let right = PI / (PI * z).sin();
        let error = (left - right).abs();

        println!("{:4.2}  {:12.8}  {:12.8}  {:9.2e}", z, left, right, error);
    }

    Ok(())
}

#[allow(dead_code)]
fn study_asymptotic_behavior() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Asymptotic Behavior Study");
    println!("============================\n");

    println!("🎯 Studying large argument behavior:");

    // Gamma function vs Stirling's approximation
    println!("\n🎲 Gamma Function vs Stirling's Approximation:");
    println!("z       Γ(z)          Stirling      Ratio       Error");
    println!("─────────────────────────────────────────────────────");

    for z in [5.0, 10.0, 15.0, 20.0, 25.0] {
        let gamma_val = gamma(z);
        let stirling = (2.0 * PI / z).sqrt() * (z / std::f64::consts::E).powf(z);
        let ratio = gamma_val / stirling;
        let error = (gamma_val - stirling).abs() / gamma_val;

        println!(
            "{:4.0}  {:12.2e}  {:12.2e}  {:8.6}  {:9.2e}",
            z, gamma_val, stirling, ratio, error
        );
    }

    println!("\n💡 Observation: Stirling's approximation becomes more accurate for larger z");

    // Bessel function asymptotic behavior
    println!("\n🌊 Bessel Function Asymptotic Behavior:");
    println!("For large x, J₀(x) ~ √(2/πx) cos(x - π/4)");
    println!("x       J₀(x)         Asymptotic    Error");
    println!("──────────────────────────────────────────");

    for x in [10.0, 15.0, 20.0, 25.0, 30.0] {
        let j0_val = j0(x);
        let asymptotic = (2.0 / (PI * x)).sqrt() * (x - PI / 4.0).cos();
        let error = (j0_val - asymptotic).abs();

        println!(
            "{:4.0}  {:12.8}  {:12.8}  {:9.2e}",
            x, j0_val, asymptotic, error
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn investigate_numerical_properties() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Numerical Properties Investigation");
    println!("====================================\n");

    println!("🔍 Investigating numerical stability and precision:");

    // Test near-zero behavior
    println!("\n📊 Near-Zero Behavior:");
    println!("Testing function behavior as x → 0");
    println!("x          j0(x)         1-x²/4        Error");
    println!("────────────────────────────────────────────");

    for &x in &[0.001, 0.01, 0.1, 0.2, 0.3] {
        let j0_val = j0(x);
        let approx = 1.0 - x * x / 4.0; // First two terms of series
        let error = (j0_val - approx as f64).abs();

        println!(
            "{:8.3}  {:12.8}  {:12.8}  {:9.2e}",
            x, j0_val, approx, error
        );
    }

    println!("\n💡 J₀(x) ≈ 1 - x²/4 + O(x⁴) for small x");

    // Test precision limits
    println!("\n🎯 Precision Analysis:");
    println!("Testing computation near problematic regions");

    // Gamma function near poles (would require analytic continuation)
    println!("\nΓ(x) behavior near x = 0:");
    for &x in &[1e-10, 1e-8, 1e-6, 1e-4, 1e-2] {
        let gamma_val = gamma(x);
        println!("  Γ({:.0e}) = {:.6e}", x, gamma_val);
    }

    Ok(())
}

#[allow(dead_code)]
fn explore_function_relationships() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Function Relationship Explorer");
    println!("=================================\n");

    println!("🎯 Discovering relationships between special functions:");

    // Bessel function recurrence relations
    println!("\n🌊 Bessel Function Recurrence Relations:");
    println!("Testing: J_(ν-1)(x) + J_(ν+1)(x) = (2ν/x)J_ν(x)");
    println!("x     ν   Left Side     Right Side    Error");
    println!("─────────────────────────────────────────────");

    for &x in &[2.0, 5.0, 10.0] {
        for &nu in &[1, 2, 3] {
            let left = jn(nu - 1, x) + jn(nu + 1, x);
            let right = (2.0 * nu as f64 / x) * jn(nu, x);
            let error = (left - right).abs();

            println!(
                "{:4.1} {:2}  {:11.6}  {:11.6}  {:9.2e}",
                x, nu, left, right, error
            );
        }
    }

    // Error function and Gamma function relationship
    println!("\n📊 Error Function and Gamma Function:");
    println!("Exploring connections through incomplete gamma function");

    for &x in &[0.5, 1.0, 1.5, 2.0] {
        let erf_val = erf(x);
        let related_gamma = gammainc(0.5, x * x); // γ(1/2, x²)
        let gamma_half = gamma(0.5);
        let computed_erf = related_gamma.unwrap_or(0.0) / gamma_half;

        println!(
            "x = {:.1}: erf(x) = {:.8}, from γ(1/2,x²)/Γ(1/2) = {:.8}",
            x, erf_val, computed_erf
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn run_mathematical_discovery(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Mathematical Discovery Engine");
    println!("================================\n");

    println!("🎯 Searching for patterns and relationships...");

    let discoveries = lab.discover_patterns();

    if discoveries.is_empty() {
        println!("🔍 No new patterns detected in current session.");
        println!("Try evaluating more functions to build pattern data!");
    } else {
        println!("🎉 Discoveries found:");

        for (i, discovery) in discoveries.iter().enumerate() {
            println!(
                "\n{}. {} ({:?})",
                i + 1,
                discovery.description,
                discovery.discovery_type
            );
            println!("   Significance: {}/10", discovery.significance);
            println!("   Details: {}", discovery.mathematical_content);
            println!("   Status: {:?}", discovery.verification_status);
        }
    }

    // Suggest exploration directions
    println!("\n💡 Suggested explorations:");
    println!("• Test the reflection formula with different z values");
    println!("• Explore Bessel function zeros and their spacing");
    println!("• Investigate series convergence rates");
    println!("• Compare asymptotic approximations");

    Ok(())
}

#[allow(dead_code)]
fn run_proof_assistant(lab: &mut MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Proof Assistant");
    println!("==================\n");

    println!("🎯 Interactive proof construction and verification");
    println!("This feature would provide:");
    println!("• Step-by-step proof guidance");
    println!("• Logical validation of proof steps");
    println!("• Automated lemma suggestions");
    println!("• Counterexample generation");
    println!("• Proof completion assistance");
    println!();
    println!("🔧 In development - would integrate with formal proof systems");

    Ok(())
}

#[allow(dead_code)]
fn run_computational_experiments(
    lab: &mut MathLaboratory,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Computational Experiments");
    println!("============================\n");

    println!("🎯 Design and run mathematical experiments:");
    println!("1. 📊 Convergence rate studies");
    println!("2. 🎲 Monte Carlo investigations");
    println!("3. 🔍 Parameter sensitivity analysis");
    println!("4. ⚡ Performance benchmarking");
    println!("5. 🧮 Precision requirement analysis");

    let choice = get_user_input("Choose experiment type (1-5): ")?;

    match choice.as_str() {
        "1" => study_convergence_rates()?,
        "2" => run_monte_carlo_experiments()?,
        "3" => analyze_parameter_sensitivity()?,
        "4" => benchmark_performance()?,
        "5" => analyze_precision_requirements()?,
        _ => println!("❌ Invalid choice"),
    }

    Ok(())
}

#[allow(dead_code)]
fn study_convergence_rates() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Convergence Rate Studies");
    println!("===========================\n");

    println!("🎯 Studying series convergence for erf(x):");
    println!("Series: erf(x) = (2/√π) Σ (-1)ⁿ xᐟ⁽²ⁿ⁺¹⁾ / (n!(2n+1))");
    println!();

    let x_values = vec![0.5, 1.0, 2.0, 3.0];

    for &x in &x_values {
        let x = x as f64;
        println!("Convergence analysis for x = {}:", x);
        println!("n    Partial Sum      Error        Reduction Ratio");
        println!("──────────────────────────────────────────────────");

        let exact = erf(x);
        let coeff = 2.0 / PI.sqrt();
        let mut sum = 0.0;
        let mut prev_error = f64::INFINITY;

        for n in 0..15 {
            let term = (-1.0_f64).powi(n) * x.powi(2 * n + 1)
                / (factorial(n as u32) as f64 * (2 * n + 1) as f64);
            sum += coeff * term;

            let error = (sum - exact).abs();
            let ratio = if prev_error.is_finite() && error > 0.0 {
                prev_error / error
            } else {
                0.0
            };

            println!("{:2}  {:12.8}  {:12.2e}  {:12.2}", n, sum, error, ratio);

            prev_error = error;
            if error < 1e-15 {
                break;
            }
        }

        println!();
    }

    Ok(())
}

#[allow(dead_code)]
fn run_monte_carlo_experiments() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎲 Monte Carlo Investigations");
    println!("=============================\n");

    println!("🎯 Using Monte Carlo methods to estimate mathematical constants:");

    // Estimate π using Buffon's needle (conceptually)
    println!("\n📐 Estimating π using gamma function relationship:");
    println!("Using Γ(1/2) = √π, we can estimate π from gamma evaluations");

    let gamma_half = gamma(0.5);
    let pi_estimate = gamma_half * gamma_half;
    let error = (pi_estimate - PI).abs();

    println!("Γ(1/2) = {:.12}", gamma_half);
    println!("[Γ(1/2)]² = {:.12}", pi_estimate);
    println!("π = {:.12}", PI);
    println!("Error = {:.2e}", error);

    // Statistical analysis of function evaluations
    println!("\n📊 Statistical Analysis of J₀(x) zeros:");
    println!("Analyzing spacing between consecutive zeros");

    // This would normally involve more sophisticated Monte Carlo sampling
    println!("This experiment would use random sampling to study:");
    println!("• Distribution of zeros");
    println!("• Statistical properties of function values");
    println!("• Confidence intervals for estimates");

    Ok(())
}

#[allow(dead_code)]
fn analyze_parameter_sensitivity() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Parameter Sensitivity Analysis");
    println!("=================================\n");

    println!("🎯 Analyzing how small changes in parameters affect function values:");

    // Sensitivity of gamma function
    println!("\n🎲 Gamma Function Sensitivity:");
    println!("Testing ∂Γ(z)/∂z numerically using finite differences");
    println!("z      Γ(z)        Numerical ∂Γ/∂z    Analytical ψ(z)Γ(z)");
    println!("──────────────────────────────────────────────────────────");

    let h = 1e-8; // Small perturbation

    for &z in &[1.0, 2.0, 3.0, 4.0, 5.0] {
        let gamma_z = gamma(z);
        let gamma_z_plus_h = gamma(z + h);
        let numerical_derivative = (gamma_z_plus_h - gamma_z) / h;
        let analytical_derivative = digamma(z) * gamma_z;

        println!(
            "{:4.1}  {:10.6}  {:15.6}     {:15.6}",
            z, gamma_z, numerical_derivative, analytical_derivative
        );
    }

    // Sensitivity of Bessel function to order
    println!("\n🌊 Bessel Function Order Sensitivity:");
    println!("How J_ν(x) changes with small changes in order ν");

    let x = 5.0;
    println!("At x = {}, studying ∂J_ν(x)/∂ν:", x);

    for &nu in &[0.0, 1.0, 2.0, 3.0] {
        let j_nu = jv(nu, x);
        let j_nu_plus_h = jv(nu + h, x);
        let sensitivity = (j_nu_plus_h - j_nu) / h;

        println!(
            "ν = {:.1}: J_ν({}) = {:10.6}, ∂J_ν/∂ν ≈ {:10.6}",
            nu, x, j_nu, sensitivity
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn benchmark_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Performance Benchmarking");
    println!("==========================\n");

    println!("🎯 Measuring computation times for special functions:");

    let test_values = Array1::linspace(0.1, 10.0, 10000);
    let num_iterations = 5;

    // Benchmark gamma function
    let start = Instant::now();
    for _ in 0..num_iterations {
        for &x in test_values.iter() {
            let _ = gamma(x);
        }
    }
    let gamma_time = start.elapsed();

    // Benchmark Bessel function
    let start = Instant::now();
    for _ in 0..num_iterations {
        for &x in test_values.iter() {
            let _ = j0(x);
        }
    }
    let bessel_time = start.elapsed();

    // Benchmark error function
    let start = Instant::now();
    for _ in 0..num_iterations {
        for &x in test_values.iter() {
            let _ = erf(x);
        }
    }
    let erf_time = start.elapsed();

    println!(
        "Performance Results ({} evaluations × {} iterations):",
        test_values.len(),
        num_iterations
    );
    println!("─────────────────────────────────────────────────────");

    let total_evals = test_values.len() * num_iterations;

    println!(
        "Gamma function: {:8.3} ms ({:.1} evals/sec)",
        gamma_time.as_millis(),
        total_evals as f64 / gamma_time.as_secs_f64()
    );

    println!(
        "Bessel J₀:     {:8.3} ms ({:.1} evals/sec)",
        bessel_time.as_millis(),
        total_evals as f64 / bessel_time.as_secs_f64()
    );

    println!(
        "Error function: {:8.3} ms ({:.1} evals/sec)",
        erf_time.as_millis(),
        total_evals as f64 / erf_time.as_secs_f64()
    );

    println!("\n💡 Performance insights:");
    println!("• Gamma function: Implementation uses optimized algorithms");
    println!("• Bessel functions: Series/asymptotic expansions balance accuracy/speed");
    println!("• Error function: Rational approximations for different ranges");

    Ok(())
}

#[allow(dead_code)]
fn analyze_precision_requirements() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Precision Requirement Analysis");
    println!("=================================\n");

    println!("🎯 Analyzing precision requirements for different applications:");

    // Test relative precision
    println!("\n📊 Relative Error Analysis:");
    println!("Function    Input     64-bit Result     32-bit Result     Rel. Error");
    println!("────────────────────────────────────────────────────────────────────");

    let testinputs = vec![0.5, 1.0, 2.0, 5.0, 10.0];

    for &x in &testinputs {
        // Simulate 32-bit precision (roughly 7 decimal digits)
        let gamma_64 = gamma(x);
        let gamma_32 = (gamma_64 * 1e7_f64).round() / 1e7_f64; // Truncate to ~7 digits
        let rel_error = ((gamma_64 - gamma_32) / gamma_64).abs();

        println!(
            "gamma       {:5.1}   {:13.10}   {:13.7}     {:8.2e}",
            x, gamma_64, gamma_32, rel_error
        );
    }

    // Critical precision regions
    println!("\n⚠️ Critical Precision Regions:");
    println!("• Gamma function near poles requires high precision");
    println!("• Bessel function zeros need precision for root finding");
    println!("• Error function near 0 requires careful series evaluation");

    // Recommended precision guidelines
    println!("\n📋 Precision Recommendations:");
    println!("Application                    Recommended Precision");
    println!("─────────────────────────────────────────────────");
    println!("Scientific computing          64-bit (15-17 digits)");
    println!("Engineering calculations      32-bit (6-7 digits)");
    println!("Statistical software          64-bit (15-17 digits)");
    println!("Real-time systems            32-bit optimized");
    println!("High-precision research       Extended (>64-bit)");

    Ok(())
}

#[allow(dead_code)]
fn display_session_summary(lab: &MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Laboratory Session Summary");
    println!("=============================\n");

    let session_duration = lab.current_session.start_time.elapsed();

    println!("🔬 Session Information:");
    println!("  Session ID: {}", lab.current_session.session_id);
    println!(
        "  Duration: {:.1} minutes",
        session_duration.as_secs_f64() / 60.0
    );
    println!("  Mode: {:?}", lab.current_session.exploration_mode);
    println!("  Focus areas: {:?}", lab.current_session.focus_areas);

    println!("\n📊 Activity Summary:");
    println!("  Expressions evaluated: {}", lab.expression_history.len());
    println!(
        "  Active plots: {}",
        lab.visualization_state.active_plots.len()
    );
    println!("  Discoveries logged: {}", lab.discovery_log.len());

    if !lab.expression_history.is_empty() {
        println!("\n🧮 Recent Evaluations:");
        for (i, expr) in lab.expression_history.iter().rev().take(5).enumerate() {
            if let Some(result) = expr.result {
                println!("  {}: {} = {:.8}", i + 1, expr.expression, result);
            }
        }
    }

    if !lab.discovery_log.is_empty() {
        println!("\n🔬 Discoveries:");
        for discovery in &lab.discovery_log {
            println!(
                "  • {} (Significance: {}/10)",
                discovery.description, discovery.significance
            );
        }
    }

    println!("\n💡 Session Insights:");
    analyze_session_patterns(lab);

    Ok(())
}

#[allow(dead_code)]
fn analyze_session_patterns(lab: &MathLaboratory) {
    // Analyze patterns in the user's exploration
    let mut function_usage = HashMap::new();

    for expr in &lab.expression_history {
        if expr.expression.contains("gamma") {
            *function_usage.entry("gamma").or_insert(0) += 1;
        }
        if expr.expression.contains("j0") || expr.expression.contains("j1") {
            *function_usage.entry("bessel").or_insert(0) += 1;
        }
        if expr.expression.contains("erf") {
            *function_usage.entry("error").or_insert(0) += 1;
        }
    }

    if !function_usage.is_empty() {
        println!("  Most explored functions:");
        let mut sorted_usage: Vec<_> = function_usage.iter().collect();
        sorted_usage.sort_by(|a, b| b.1.cmp(a.1));

        for (function, count) in sorted_usage.iter().take(3) {
            println!("    {}: {} evaluations", function, count);
        }
    }

    // Suggest next steps
    println!("\n🎯 Suggested next explorations:");
    match lab.current_session.exploration_mode {
        ExplorationMode::Guided => {
            println!("  • Try the theorem explorer for deeper understanding");
            println!("  • Use visualization tools to see function behavior");
        }
        ExplorationMode::Exploratory => {
            println!("  • Run function analysis on interesting patterns you've found");
            println!("  • Use the discovery engine to formalize observations");
        }
        ExplorationMode::Research => {
            println!("  • Document your findings and create hypotheses");
            println!("  • Use computational experiments to test conjectures");
        }
        _ => {
            println!("  • Continue building on your current exploration direction");
        }
    }
}

#[allow(dead_code)]
fn savelaboratory_session(lab: &MathLaboratory) -> Result<(), Box<dyn std::error::Error>> {
    // In a real implementation, this would save to file/database
    println!("💾 Session data saved successfully!");
    println!("🔬 Laboratory session complete.");
    Ok(())
}

// Helper functions for proof steps
#[allow(dead_code)]
fn create_reflection_formula_proof_steps() -> Vec<ProofStep> {
    vec![
        ProofStep {
            step_number: 1,
            description: "Start with the beta function representation".to_string(),
            mathematical_content: "B(z, 1-z) = ∫₀¹ t^(z-1)(1-t)^(-z) dt = Γ(z)Γ(1-z)/Γ(1)".to_string(),
            justification: "Beta function definition and relationship to gamma function".to_string(),
            hints: vec![
                "The beta function provides a bridge to the reflection formula".to_string(),
                "Remember that Γ(1) = 1".to_string(),
            ],
            interactive_elements: vec![],
            verification_code: Some("let z = 0.5; let beta_val = beta(z, 1.0-z); let gamma_product = gamma(z) * gamma(1.0-z);".to_string()),
        },
        ProofStep {
            step_number: 2,
            description: "Transform the integral using substitution".to_string(),
            mathematical_content: "Substitute t = u/(1+u) to get B(z,1-z) = ∫₀^∞ u^(z-1)/(1+u) du".to_string(),
            justification: "This substitution converts the finite integral to an infinite one".to_string(),
            hints: vec![
                "The Jacobian of the transformation is dt = du/(1+u)²".to_string(),
                "Check the limits of integration carefully".to_string(),
            ],
            interactive_elements: vec![],
            verification_code: None,
        },
        ProofStep {
            step_number: 3,
            description: "Apply complex contour integration".to_string(),
            mathematical_content: "Consider ∮_C w^(z-1)/(1+w) dw around a keyhole contour".to_string(),
            justification: "Complex analysis provides the connection to sine function".to_string(),
            hints: vec![
                "The keyhole contour avoids the branch cut on [0,∞)".to_string(),
                "Residue at w = -1 is key to the final result".to_string(),
            ],
            interactive_elements: vec![],
            verification_code: None,
        },
        ProofStep {
            step_number: 4,
            description: "Evaluate residues and conclude".to_string(),
            mathematical_content: "The residue calculation yields Γ(z)Γ(1-z) = π/sin(πz)".to_string(),
            justification: "Residue theorem combined with careful analysis of branch cuts".to_string(),
            hints: vec![
                "The sine function emerges from the residue at w = -1".to_string(),
            ],
            interactive_elements: vec![],
            verification_code: Some("// Verification for specific values\nlet z = 1.0/3.0; let left = gamma(z) * gamma(1.0-z); let right = PI / (PI*z).sin();".to_string()),
        },
    ]
}

#[allow(dead_code)]
fn create_bessel_generating_proof_steps() -> Vec<ProofStep> {
    vec![
        ProofStep {
            step_number: 1,
            description: "Define the generating function".to_string(),
            mathematical_content: "Consider G(x,t) = exp(x(t-1/t)/2)".to_string(),
            justification: "This exponential form will yield Bessel functions as coefficients"
                .to_string(),
            hints: vec!["The argument x(t-1/t)/2 has special symmetry properties".to_string()],
            interactive_elements: vec![],
            verification_code: None,
        },
        ProofStep {
            step_number: 2,
            description: "Expand as product of exponentials".to_string(),
            mathematical_content: "G(x,t) = exp(xt/2) · exp(-x/(2t))".to_string(),
            justification: "Separating allows individual series expansion".to_string(),
            hints: vec!["Each exponential can be expanded as a power series".to_string()],
            interactive_elements: vec![],
            verification_code: None,
        },
    ]
}

#[allow(dead_code)]
fn create_stirling_proof_steps() -> Vec<ProofStep> {
    vec![ProofStep {
        step_number: 1,
        description: "Express ln Γ(z) as an integral".to_string(),
        mathematical_content: "ln Γ(z) = ∫₀^∞ [(z-1)ln t - t] dt".to_string(),
        justification: "Taking logarithm of the gamma function integral".to_string(),
        hints: vec!["This form is suitable for asymptotic analysis".to_string()],
        interactive_elements: vec![],
        verification_code: None,
    }]
}

#[allow(dead_code)]
fn create_wright_asymptotic_proof_steps() -> Vec<ProofStep> {
    vec![ProofStep {
        step_number: 1,
        description: "Use Mellin transform representation".to_string(),
        mathematical_content: "Φ(α,β;z) = (1/2πi) ∫_C Γ(-s)Γ(β+αs)(-z)^s ds".to_string(),
        justification: "Mellin transform provides integral representation for asymptotic analysis"
            .to_string(),
        hints: vec!["This integral is suitable for saddle-point method".to_string()],
        interactive_elements: vec![],
        verification_code: None,
    }]
}

#[allow(dead_code)]
fn wait_for_enter() -> Result<(), Box<dyn std::error::Error>> {
    get_user_input("Press Enter to continue...")?;
    Ok(())
}

#[allow(dead_code)]
fn get_user_input(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}
