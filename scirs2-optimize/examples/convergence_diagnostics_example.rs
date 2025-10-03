//! Example of using convergence diagnostics with optimization

use scirs2_core::ndarray::{array, ArrayView1};
use scirs2_optimize::unconstrained::{
    minimize_bfgs, DiagnosticCollector, DiagnosticOptions, ExportFormat, LineSearchDiagnostic,
    Options,
};

#[allow(dead_code)]
fn main() {
    // Rosenbrock function
    let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };

    // Initial point
    let x0 = array![0.0, 0.0];

    // Create diagnostic collector
    let diagnostic_options = DiagnosticOptions {
        track_nit: true,
        estimate_conditioning: true,
        detect_noise: true,
        analyze_convergence_rate: true,
        history_window: 50,
        track_memory: false,
        export_format: ExportFormat::Csv,
    };

    let mut diagnostic_collector = DiagnosticCollector::new(diagnostic_options);

    // Run optimization with manual diagnostic collection
    // (In practice, this would be integrated into the optimizer)
    println!("Optimizing Rosenbrock function with convergence diagnostics...\n");

    // Simulate some iterations for demonstration
    let grad = array![2.0, 1.0];
    let step = array![0.1, 0.05];
    let direction = array![-2.0, -1.0];

    // Record an iteration
    let ls_diagnostic = LineSearchDiagnostic {
        n_fev: 3,
        n_gev: 1,
        alpha: 0.5,
        alpha_init: 1.0,
        success: true,
        wolfe_satisfied: (true, true),
    };

    diagnostic_collector.record_iteration(
        10.0,
        &grad.view(),
        &step.view(),
        &direction.view(),
        ls_diagnostic,
    );

    // Finalize diagnostics
    let diagnostics = diagnostic_collector.finalize();

    // Print summary report
    println!("{}", diagnostics.summary_report());

    // Export to CSV
    match diagnostics.to_csv() {
        Ok(csv) => {
            println!("\nCSV Export:");
            println!("{}", csv);
        }
        Err(e) => println!("Error exporting to CSV: {:?}", e),
    }

    // Show warnings
    if !diagnostics.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &diagnostics.warnings {
            println!("  [{:?}] {}", warning.severity, warning.message);
            for rec in &warning.recommendations {
                println!("    - {}", rec);
            }
        }
    }

    // Show problem analysis
    println!("\nProblem Analysis:");
    println!(
        "  Difficulty: {:?}",
        diagnostics.problem_analysis.difficulty
    );
    println!("  Features: {:?}", diagnostics.problem_analysis.features);

    // Show performance metrics
    println!("\nPerformance Metrics:");
    println!(
        "  Total iterations: {}",
        diagnostics.performance_metrics.total_nit
    );
    println!(
        "  Total function evaluations: {}",
        diagnostics.performance_metrics.total_fev
    );
    println!(
        "  Average iteration time: {:?}",
        diagnostics.performance_metrics.avg_iteration_time
    );
    println!(
        "  Function evaluation rate: {:.2} per second",
        diagnostics.performance_metrics.fev_rate
    );

    // Show convergence analysis
    println!("\nConvergence Analysis:");
    println!(
        "  Convergence rate: {:?}",
        diagnostics.convergence_analysis.convergence_rate
    );
    println!(
        "  Convergence phase: {:?}",
        diagnostics.convergence_analysis.convergence_phase
    );
    println!(
        "  Confidence score: {:.2}",
        diagnostics.convergence_analysis.confidence_score
    );

    // Now run actual optimization
    println!("\n\nRunning actual optimization...");
    let options = Options::default();

    match minimize_bfgs(rosenbrock, x0, &options) {
        Ok(result) => {
            println!("\nOptimization Result:");
            println!("  Solution: {:?}", result.x);
            println!("  Function value: {}", result.fun);
            println!("  Iterations: {}", result.nit);
            println!("  Success: {}", result.success);
            println!("  Message: {}", result.message);
        }
        Err(e) => println!("Optimization failed: {:?}", e),
    }
}
