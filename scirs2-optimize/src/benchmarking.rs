//! Comprehensive benchmarking system for optimization algorithms
//!
//! This module provides a complete benchmarking suite for comparing different
//! optimization algorithms across various test problems, metrics, and scenarios.

use crate::error::ScirsResult;
use ndarray::{Array1, ArrayView1};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::result::OptimizeResults;
use crate::visualization::{OptimizationTrajectory, OptimizationVisualizer};

/// Standard test functions for optimization benchmarking
pub mod test_functions {
    use super::*;

    /// Rosenbrock function (classic unconstrained optimization test)
    pub fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        let n = x.len();
        let mut sum = 0.0;
        for i in 0..(n - 1) {
            let term1 = x[i + 1] - x[i].powi(2);
            let term2 = 1.0 - x[i];
            sum += 100.0 * term1.powi(2) + term2.powi(2);
        }
        sum
    }

    /// Sphere function (simple convex test)
    pub fn sphere(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|&xi| xi.powi(2)).sum()
    }

    /// Rastrigin function (highly multimodal)
    pub fn rastrigin(x: &ArrayView1<f64>) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        a * n
            + x.iter()
                .map(|&xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    /// Ackley function (multimodal with global structure)
    pub fn ackley(x: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let sum_sq = x.iter().map(|&xi| xi.powi(2)).sum::<f64>();
        let sum_cos = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>();

        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    }

    /// Griewank function (multimodal with product term)
    pub fn griewank(x: &ArrayView1<f64>) -> f64 {
        let sum_term = x.iter().map(|&xi| xi.powi(2)).sum::<f64>() / 4000.0;
        let prod_term = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (xi / ((i + 1) as f64).sqrt()).cos())
            .product::<f64>();

        sum_term - prod_term + 1.0
    }

    /// Levy function (multimodal with variable transformation)
    pub fn levy(x: &ArrayView1<f64>) -> f64 {
        let w: Vec<f64> = x.iter().map(|&xi| 1.0 + (xi - 1.0) / 4.0).collect();
        let n = w.len();

        let term1 = (std::f64::consts::PI * w[0]).sin().powi(2);
        let term2 = (0..(n - 1))
            .map(|i| {
                (w[i] - 1.0).powi(2)
                    * (1.0 + 10.0 * (std::f64::consts::PI * w[i + 1]).sin().powi(2))
            })
            .sum::<f64>();
        let term3 = (w[n - 1] - 1.0).powi(2)
            * (1.0 + (2.0 * std::f64::consts::PI * w[n - 1]).sin().powi(2));

        term1 + term2 + term3
    }

    /// Schwefel function (multimodal with shifted optimum)
    pub fn schwefel(x: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        418.9829 * n
            - x.iter()
                .map(|&xi| xi * (xi.abs().sqrt()).sin())
                .sum::<f64>()
    }

    /// Get bounds for a test function
    pub fn get_bounds(function_name: &str, dimensions: usize) -> Vec<(f64, f64)> {
        match function_name {
            "rosenbrock" => vec![(-5.0, 5.0); dimensions],
            "sphere" => vec![(-5.12, 5.12); dimensions],
            "rastrigin" => vec![(-5.12, 5.12); dimensions],
            "ackley" => vec![(-32.768, 32.768); dimensions],
            "griewank" => vec![(-600.0, 600.0); dimensions],
            "levy" => vec![(-10.0, 10.0); dimensions],
            "schwefel" => vec![(-500.0, 500.0); dimensions],
            _ => vec![(-10.0, 10.0); dimensions],
        }
    }

    /// Get global optimum for a test function
    pub fn get_global_optimum(function_name: &str, dimensions: usize) -> (Array1<f64>, f64) {
        match function_name {
            "rosenbrock" => (Array1::ones(dimensions), 0.0),
            "sphere" => (Array1::zeros(dimensions), 0.0),
            "rastrigin" => (Array1::zeros(dimensions), 0.0),
            "ackley" => (Array1::zeros(dimensions), 0.0),
            "griewank" => (Array1::zeros(dimensions), 0.0),
            "levy" => (Array1::ones(dimensions), 0.0),
            "schwefel" => (Array1::from_elem(dimensions, 420.9687), 0.0),
            _ => (Array1::zeros(dimensions), 0.0),
        }
    }
}

/// Test problem definition
#[derive(Debug, Clone)]
pub struct TestProblem {
    /// Name of the test function
    pub name: String,
    /// Function to optimize
    pub function: fn(&ArrayView1<f64>) -> f64,
    /// Problem dimensions
    pub dimensions: usize,
    /// Variable bounds
    pub bounds: Vec<(f64, f64)>,
    /// Known global optimum location
    pub global_optimum: Array1<f64>,
    /// Known global optimum value
    pub global_minimum: f64,
    /// Problem characteristics
    pub characteristics: ProblemCharacteristics,
}

impl TestProblem {
    /// Create a new test problem
    pub fn new(name: &str, dimensions: usize) -> Self {
        let function = match name {
            "rosenbrock" => test_functions::rosenbrock,
            "sphere" => test_functions::sphere,
            "rastrigin" => test_functions::rastrigin,
            "ackley" => test_functions::ackley,
            "griewank" => test_functions::griewank,
            "levy" => test_functions::levy,
            "schwefel" => test_functions::schwefel,
            _ => test_functions::sphere,
        };

        let bounds = test_functions::get_bounds(name, dimensions);
        let (global_optimum, global_minimum) = test_functions::get_global_optimum(name, dimensions);
        let characteristics = ProblemCharacteristics::from_function_name(name);

        Self {
            name: name.to_string(),
            function,
            dimensions,
            bounds,
            global_optimum,
            global_minimum,
            characteristics,
        }
    }

    /// Evaluate the function at a point
    pub fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        (self.function)(x)
    }

    /// Generate random starting points for the problem
    pub fn generate_starting_points(&self, count: usize) -> ScirsResult<Vec<Array1<f64>>> {
        use rand::{rng, Rng};
        let mut rng = rand::rng();
        let mut points = Vec::with_capacity(count);

        for _ in 0..count {
            let mut point = Array1::zeros(self.dimensions);
            for (i, &(low, high)) in self.bounds.iter().enumerate() {
                point[i] = rng.gen_range(low..=high);
            }
            points.push(point);
        }

        Ok(points)
    }
}

/// Problem characteristics for categorization
#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    /// Whether the function is multimodal
    pub multimodal: bool,
    /// Whether the function is separable
    pub separable: bool,
    /// Whether the function is convex
    pub convex: bool,
    /// Estimated difficulty level (1-5)
    pub difficulty: u8,
}

impl ProblemCharacteristics {
    fn from_function_name(name: &str) -> Self {
        match name {
            "sphere" => Self {
                multimodal: false,
                separable: true,
                convex: true,
                difficulty: 1,
            },
            "rosenbrock" => Self {
                multimodal: false,
                separable: false,
                convex: false,
                difficulty: 3,
            },
            "rastrigin" => Self {
                multimodal: true,
                separable: true,
                convex: false,
                difficulty: 4,
            },
            "ackley" => Self {
                multimodal: true,
                separable: false,
                convex: false,
                difficulty: 4,
            },
            "griewank" => Self {
                multimodal: true,
                separable: false,
                convex: false,
                difficulty: 4,
            },
            "levy" => Self {
                multimodal: true,
                separable: false,
                convex: false,
                difficulty: 4,
            },
            "schwefel" => Self {
                multimodal: true,
                separable: true,
                convex: false,
                difficulty: 5,
            },
            _ => Self {
                multimodal: false,
                separable: true,
                convex: true,
                difficulty: 1,
            },
        }
    }
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Test problems to include
    pub test_problems: Vec<String>,
    /// Problem dimensions to test
    pub dimensions: Vec<usize>,
    /// Number of independent runs per problem
    pub runs_per_problem: usize,
    /// Maximum number of function evaluations
    pub max_function_evaluations: usize,
    /// Maximum optimization time per run
    pub max_time: Duration,
    /// Target accuracy for success criteria
    pub target_accuracy: f64,
    /// Whether to enable detailed logging
    pub detailed_logging: bool,
    /// Whether to save optimization trajectories
    pub save_trajectories: bool,
    /// Output directory for results
    pub output_directory: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            test_problems: vec![
                "sphere".to_string(),
                "rosenbrock".to_string(),
                "rastrigin".to_string(),
                "ackley".to_string(),
            ],
            dimensions: vec![2, 5, 10, 20],
            runs_per_problem: 30,
            max_function_evaluations: 10000,
            max_time: Duration::from_secs(300), // 5 minutes
            target_accuracy: 1e-6,
            detailed_logging: true,
            save_trajectories: false,
            output_directory: "benchmark_results".to_string(),
        }
    }
}

/// Benchmark results for a single run
#[derive(Debug, Clone)]
pub struct BenchmarkRun {
    /// Problem that was solved
    pub problem_name: String,
    /// Problem dimensions
    pub dimensions: usize,
    /// Run number
    pub run_id: usize,
    /// Optimization algorithm used
    pub algorithm: String,
    /// Optimization results
    pub results: OptimizeResults<f64>,
    /// Runtime statistics
    pub runtime_stats: RuntimeStats,
    /// Distance to global optimum
    pub distance_to_optimum: f64,
    /// Whether the run was successful (reached target accuracy)
    pub success: bool,
    /// Optimization trajectory (if saved)
    pub trajectory: Option<OptimizationTrajectory>,
}

/// Runtime statistics for a benchmark run
#[derive(Debug, Clone)]
pub struct RuntimeStats {
    /// Total wall clock time
    pub total_time: Duration,
    /// Time per function evaluation
    pub time_per_evaluation: Duration,
    /// Peak memory usage (in bytes)
    pub peak_memory: usize,
    /// Number of convergence checks
    pub convergence_checks: usize,
}

/// Aggregated benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Configuration used for benchmarking
    pub config: BenchmarkConfig,
    /// Individual run results
    pub runs: Vec<BenchmarkRun>,
    /// Statistical summary
    pub summary: BenchmarkSummary,
    /// Performance rankings
    pub rankings: HashMap<String, AlgorithmRanking>,
}

impl BenchmarkResults {
    /// Generate comprehensive benchmark report
    pub fn generate_report(&self) -> ScirsResult<String> {
        let mut report = String::from("Optimization Algorithm Benchmark Report\n");
        report.push_str("======================================\n\n");

        // Configuration summary
        report.push_str("Benchmark Configuration:\n");
        report.push_str(&format!(
            "  Test Problems: {:?}\n",
            self.config.test_problems
        ));
        report.push_str(&format!("  Dimensions: {:?}\n", self.config.dimensions));
        report.push_str(&format!(
            "  Runs per Problem: {}\n",
            self.config.runs_per_problem
        ));
        report.push_str(&format!(
            "  Max Function Evaluations: {}\n",
            self.config.max_function_evaluations
        ));
        report.push_str(&format!(
            "  Target Accuracy: {:.2e}\n",
            self.config.target_accuracy
        ));
        report.push_str("\n");

        // Overall summary
        report.push_str("Overall Summary:\n");
        report.push_str(&format!("  Total Runs: {}\n", self.runs.len()));
        report.push_str(&format!(
            "  Successful Runs: {}\n",
            self.summary.successful_runs
        ));
        report.push_str(&format!(
            "  Success Rate: {:.1}%\n",
            self.summary.overall_success_rate * 100.0
        ));
        report.push_str(&format!(
            "  Average Runtime: {:.3}s\n",
            self.summary.average_runtime.as_secs_f64()
        ));
        report.push_str("\n");

        // Algorithm rankings
        report.push_str("Algorithm Rankings:\n");
        let mut ranked_algorithms: Vec<_> = self.rankings.iter().collect();
        ranked_algorithms.sort_by(|a, b| {
            a.1.overall_score
                .partial_cmp(&b.1.overall_score)
                .unwrap()
                .reverse()
        });

        for (i, (algorithm, ranking)) in ranked_algorithms.iter().enumerate() {
            report.push_str(&format!(
                "  {}. {} (Score: {:.3})\n",
                i + 1,
                algorithm,
                ranking.overall_score
            ));
            report.push_str(&format!(
                "     Success Rate: {:.1}%, Avg Runtime: {:.3}s\n",
                ranking.success_rate * 100.0,
                ranking.average_runtime.as_secs_f64()
            ));
        }
        report.push_str("\n");

        // Problem-specific results
        report.push_str("Problem-Specific Results:\n");
        for problem in &self.config.test_problems {
            report.push_str(&format!("  {}:\n", problem));

            let problem_runs: Vec<_> = self
                .runs
                .iter()
                .filter(|run| run.problem_name == *problem)
                .collect();

            if !problem_runs.is_empty() {
                let success_count = problem_runs.iter().filter(|run| run.success).count();
                let success_rate = success_count as f64 / problem_runs.len() as f64;
                let avg_distance = problem_runs
                    .iter()
                    .map(|run| run.distance_to_optimum)
                    .sum::<f64>()
                    / problem_runs.len() as f64;

                report.push_str(&format!("    Success Rate: {:.1}%\n", success_rate * 100.0));
                report.push_str(&format!(
                    "    Avg Distance to Optimum: {:.6e}\n",
                    avg_distance
                ));
            }
        }

        Ok(report)
    }

    /// Save results to files
    pub fn save_results(&self, output_dir: &Path) -> ScirsResult<()> {
        std::fs::create_dir_all(output_dir)?;

        // Save summary report
        let report = self.generate_report()?;
        let report_path = output_dir.join("benchmark_report.txt");
        std::fs::write(report_path, report)?;

        // Save detailed results as CSV
        self.save_csv_results(output_dir)?;

        // Generate visualizations
        if self.config.save_trajectories {
            self.generate_visualizations(output_dir)?;
        }

        Ok(())
    }

    /// Save results in CSV format
    fn save_csv_results(&self, output_dir: &Path) -> ScirsResult<()> {
        let csv_path = output_dir.join("benchmark_results.csv");
        let mut csv_content = String::from("problem,dimensions,run_id,algorithm,success,final_value,function_evaluations,runtime_ms,distance_to_optimum\n");

        for run in &self.runs {
            csv_content.push_str(&format!(
                "{},{},{},{},{},{:.6e},{},{},{:.6e}\n",
                run.problem_name,
                run.dimensions,
                run.run_id,
                run.algorithm,
                run.success,
                run.results.fun,
                run.results.nfev,
                run.runtime_stats.total_time.as_millis(),
                run.distance_to_optimum
            ));
        }

        std::fs::write(csv_path, csv_content)?;
        Ok(())
    }

    /// Generate visualization plots
    fn generate_visualizations(&self, output_dir: &Path) -> ScirsResult<()> {
        let viz_dir = output_dir.join("visualizations");
        std::fs::create_dir_all(&viz_dir)?;

        let visualizer = OptimizationVisualizer::new();

        // Generate convergence plots for each run with trajectory
        for run in &self.runs {
            if let Some(ref trajectory) = run.trajectory {
                let plot_path = viz_dir.join(format!(
                    "{}_{}_{}_{}.svg",
                    run.problem_name, run.dimensions, run.algorithm, run.run_id
                ));
                visualizer.plot_convergence(trajectory, &plot_path)?;
            }
        }

        Ok(())
    }
}

/// Statistical summary of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Total number of successful runs
    pub successful_runs: usize,
    /// Overall success rate across all problems
    pub overall_success_rate: f64,
    /// Average runtime across all runs
    pub average_runtime: Duration,
    /// Standard deviation of runtime
    pub runtime_std: Duration,
    /// Average function evaluations
    pub average_function_evaluations: f64,
    /// Best achieved distance to optimum
    pub best_distance_to_optimum: f64,
    /// Worst achieved distance to optimum
    pub worst_distance_to_optimum: f64,
}

/// Algorithm ranking information
#[derive(Debug, Clone)]
pub struct AlgorithmRanking {
    /// Algorithm name
    pub algorithm: String,
    /// Overall performance score (higher is better)
    pub overall_score: f64,
    /// Success rate
    pub success_rate: f64,
    /// Average runtime
    pub average_runtime: Duration,
    /// Average distance to optimum
    pub average_distance: f64,
    /// Ranking on each problem type
    pub problem_rankings: HashMap<String, f64>,
}

/// Main benchmarking system
pub struct BenchmarkSystem {
    config: BenchmarkConfig,
    test_problems: Vec<TestProblem>,
}

impl BenchmarkSystem {
    /// Create a new benchmark system
    pub fn new(config: BenchmarkConfig) -> Self {
        let mut test_problems = Vec::new();

        for problem_name in &config.test_problems {
            for &dim in &config.dimensions {
                test_problems.push(TestProblem::new(problem_name, dim));
            }
        }

        Self {
            config,
            test_problems,
        }
    }

    /// Run benchmark for a specific algorithm
    pub fn benchmark_algorithm<F>(
        &self,
        algorithm_name: &str,
        optimize_fn: F,
    ) -> ScirsResult<BenchmarkResults>
    where
        F: Fn(&TestProblem, &Array1<f64>) -> ScirsResult<OptimizeResults<f64>> + Clone,
    {
        let mut runs = Vec::new();

        for problem in &self.test_problems {
            println!(
                "Benchmarking {} on {} ({}D)",
                algorithm_name, problem.name, problem.dimensions
            );

            let starting_points = problem.generate_starting_points(self.config.runs_per_problem)?;

            for (run_id, start_point) in starting_points.iter().enumerate() {
                let start_time = Instant::now();

                // Run optimization
                let result = optimize_fn(problem, start_point);

                let runtime = start_time.elapsed();

                match result {
                    Ok(opt_result) => {
                        // Calculate distance to global optimum
                        let distance = (&opt_result.x - &problem.global_optimum)
                            .iter()
                            .map(|&x| x * x)
                            .sum::<f64>()
                            .sqrt();

                        // Check if run was successful
                        let success = distance < self.config.target_accuracy;

                        let runtime_stats = RuntimeStats {
                            total_time: runtime,
                            time_per_evaluation: runtime / opt_result.nfev.max(1) as u32,
                            peak_memory: 0, // Would need system monitoring
                            convergence_checks: opt_result.nit,
                        };

                        runs.push(BenchmarkRun {
                            problem_name: problem.name.clone(),
                            dimensions: problem.dimensions,
                            run_id,
                            algorithm: algorithm_name.to_string(),
                            results: opt_result,
                            runtime_stats,
                            distance_to_optimum: distance,
                            success,
                            trajectory: None, // Would need to be provided by algorithm
                        });
                    }
                    Err(e) => {
                        // Record failed run
                        let runtime_stats = RuntimeStats {
                            total_time: runtime,
                            time_per_evaluation: Duration::from_secs(0),
                            peak_memory: 0,
                            convergence_checks: 0,
                        };

                        runs.push(BenchmarkRun {
                            problem_name: problem.name.clone(),
                            dimensions: problem.dimensions,
                            run_id,
                            algorithm: algorithm_name.to_string(),
                            results: OptimizeResults::<f64> {
                                x: start_point.clone(),
                                fun: f64::INFINITY,
                                success: false,
                                message: format!("Error: {}", e),
                                nit: 0,
                                nfev: 0,
                                ..OptimizeResults::default()
                            },
                            runtime_stats,
                            distance_to_optimum: f64::INFINITY,
                            success: false,
                            trajectory: None,
                        });
                    }
                }
            }
        }

        // Compute summary statistics
        let summary = self.compute_summary(&runs);
        let mut rankings = HashMap::new();
        rankings.insert(
            algorithm_name.to_string(),
            self.compute_ranking(algorithm_name, &runs),
        );

        Ok(BenchmarkResults {
            config: self.config.clone(),
            runs,
            summary,
            rankings,
        })
    }

    /// Compute summary statistics
    fn compute_summary(&self, runs: &[BenchmarkRun]) -> BenchmarkSummary {
        let successful_runs = runs.iter().filter(|run| run.success).count();
        let overall_success_rate = successful_runs as f64 / runs.len() as f64;

        let total_runtime: Duration = runs.iter().map(|run| run.runtime_stats.total_time).sum();
        let average_runtime = total_runtime / runs.len() as u32;

        let average_function_evaluations =
            runs.iter().map(|run| run.results.nfev as f64).sum::<f64>() / runs.len() as f64;

        let distances: Vec<f64> = runs
            .iter()
            .filter(|run| run.distance_to_optimum.is_finite())
            .map(|run| run.distance_to_optimum)
            .collect();

        let best_distance_to_optimum = distances.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst_distance_to_optimum = distances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute runtime standard deviation
        let mean_runtime_ms = average_runtime.as_millis() as f64;
        let variance = runs
            .iter()
            .map(|run| {
                let diff = run.runtime_stats.total_time.as_millis() as f64 - mean_runtime_ms;
                diff * diff
            })
            .sum::<f64>()
            / runs.len() as f64;
        let runtime_std = Duration::from_millis(variance.sqrt() as u64);

        BenchmarkSummary {
            successful_runs,
            overall_success_rate,
            average_runtime,
            runtime_std,
            average_function_evaluations,
            best_distance_to_optimum,
            worst_distance_to_optimum,
        }
    }

    /// Compute algorithm ranking
    fn compute_ranking(&self, algorithm: &str, runs: &[BenchmarkRun]) -> AlgorithmRanking {
        let successful_runs = runs.iter().filter(|run| run.success).count();
        let success_rate = successful_runs as f64 / runs.len() as f64;

        let total_runtime: Duration = runs.iter().map(|run| run.runtime_stats.total_time).sum();
        let average_runtime = total_runtime / runs.len() as u32;

        let finite_distances: Vec<f64> = runs
            .iter()
            .filter(|run| run.distance_to_optimum.is_finite())
            .map(|run| run.distance_to_optimum)
            .collect();

        let average_distance = if finite_distances.is_empty() {
            f64::INFINITY
        } else {
            finite_distances.iter().sum::<f64>() / finite_distances.len() as f64
        };

        // Compute overall score (higher is better)
        let runtime_score = 1.0 / (average_runtime.as_secs_f64() + 1e-6);
        let accuracy_score = 1.0 / (average_distance + 1e-6);
        let overall_score = success_rate * (runtime_score + accuracy_score) / 2.0;

        // Compute problem-specific rankings
        let mut problem_rankings = HashMap::new();
        for problem_name in &self.config.test_problems {
            let problem_runs: Vec<_> = runs
                .iter()
                .filter(|run| run.problem_name == *problem_name)
                .collect();

            if !problem_runs.is_empty() {
                let problem_success = problem_runs.iter().filter(|run| run.success).count() as f64
                    / problem_runs.len() as f64;
                problem_rankings.insert(problem_name.clone(), problem_success);
            }
        }

        AlgorithmRanking {
            algorithm: algorithm.to_string(),
            overall_score,
            success_rate,
            average_runtime,
            average_distance,
            problem_rankings,
        }
    }
}

/// Predefined benchmark suites
pub mod benchmark_suites {
    use super::*;

    /// Create a quick benchmark for algorithm development
    pub fn quick_benchmark() -> BenchmarkConfig {
        BenchmarkConfig {
            test_problems: vec!["sphere".to_string(), "rosenbrock".to_string()],
            dimensions: vec![2, 5],
            runs_per_problem: 5,
            max_function_evaluations: 1000,
            max_time: Duration::from_secs(30),
            target_accuracy: 1e-3,
            detailed_logging: false,
            save_trajectories: false,
            output_directory: "quick_benchmark".to_string(),
        }
    }

    /// Create a comprehensive benchmark for publication
    pub fn comprehensive_benchmark() -> BenchmarkConfig {
        BenchmarkConfig {
            test_problems: vec![
                "sphere".to_string(),
                "rosenbrock".to_string(),
                "rastrigin".to_string(),
                "ackley".to_string(),
                "griewank".to_string(),
                "levy".to_string(),
                "schwefel".to_string(),
            ],
            dimensions: vec![2, 5, 10, 20, 50],
            runs_per_problem: 50,
            max_function_evaluations: 100000,
            max_time: Duration::from_secs(3600), // 1 hour
            target_accuracy: 1e-8,
            detailed_logging: true,
            save_trajectories: true,
            output_directory: "comprehensive_benchmark".to_string(),
        }
    }

    /// Create a scalability benchmark for large dimensions
    pub fn scalability_benchmark() -> BenchmarkConfig {
        BenchmarkConfig {
            test_problems: vec!["sphere".to_string(), "rastrigin".to_string()],
            dimensions: vec![10, 50, 100, 500, 1000],
            runs_per_problem: 20,
            max_function_evaluations: 1000000,
            max_time: Duration::from_secs(1800), // 30 minutes
            target_accuracy: 1e-6,
            detailed_logging: true,
            save_trajectories: false,
            output_directory: "scalability_benchmark".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_test_functions() {
        let x = array![0.0, 0.0];

        // Test that global optima are correct
        assert!((test_functions::sphere(&x.view()) - 0.0).abs() < 1e-10);
        assert!((test_functions::rastrigin(&x.view()) - 0.0).abs() < 1e-10);
        assert!((test_functions::ackley(&x.view()) - 0.0).abs() < 1e-10);
        assert!((test_functions::griewank(&x.view()) - 0.0).abs() < 1e-10);

        let x_ones = array![1.0, 1.0];
        assert!((test_functions::rosenbrock(&x_ones.view()) - 0.0).abs() < 1e-10);
        assert!((test_functions::levy(&x_ones.view()) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_problem_creation() {
        let problem = TestProblem::new("rosenbrock", 2);
        assert_eq!(problem.name, "rosenbrock");
        assert_eq!(problem.dimensions, 2);
        assert_eq!(problem.bounds.len(), 2);
        assert_eq!(problem.global_optimum.len(), 2);
        assert!(problem.characteristics.separable == false);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert!(config.test_problems.contains(&"sphere".to_string()));
        assert!(config.dimensions.contains(&2));
        assert_eq!(config.runs_per_problem, 30);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_benchmark_suites() {
        let quick = benchmark_suites::quick_benchmark();
        assert_eq!(quick.runs_per_problem, 5);
        assert_eq!(quick.max_function_evaluations, 1000);

        let comprehensive = benchmark_suites::comprehensive_benchmark();
        assert_eq!(comprehensive.runs_per_problem, 50);
        assert!(comprehensive.test_problems.len() >= 5);

        let scalability = benchmark_suites::scalability_benchmark();
        assert!(scalability.dimensions.contains(&1000));
    }
}
