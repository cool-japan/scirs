//! Comprehensive example demonstrating the new advanced modules in scirs2-integrate
//!
//! This example showcases:
//! - Advanced AMR (Adaptive Mesh Refinement) for PDE solving
//! - Advanced error estimation with Richardson extrapolation and spectral analysis
//! - Parallel optimization with vectorized operations
//! - Performance monitoring and profiling
//!
//! The example solves a 2D heat equation with adaptive mesh refinement
//! while monitoring performance and using advanced error estimation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_integrate::{
    // Advanced modules
    amr_advanced::{
        AdaptiveCell, AdaptiveMeshLevel, AdvancedAMRManager, CellId, GradientRefinementCriterion,
        RefinementFlag,
    },
    error_estimation::AdvancedErrorEstimator,
    parallel_optimization::{
        ArithmeticOp, ParallelOptimizer, VectorOperation, VectorizedComputeTask,
    },
    performance_monitor::{PerformanceAnalyzer, PerformanceProfiler},

    IntegrateResult,
};
use std::collections::HashMap;

#[allow(dead_code)]
fn main() -> IntegrateResult<()> {
    println!("=== Advanced Features Integration Example ===\n");

    // Initialize performance profiler
    let mut profiler = PerformanceProfiler::new();
    profiler.start_phase("initialization");

    // Example 1: Advanced AMR for 2D heat equation
    demonstrate_advanced_amr(&mut profiler)?;

    // Example 2: Advanced error estimation
    demonstrate_error_estimation(&mut profiler)?;

    // Example 3: Parallel optimization with vectorized operations
    demonstrate_parallel_optimization(&mut profiler)?;

    profiler.end_phase("initialization");

    // Generate performance report
    let metrics = profiler.finalize();
    let report = PerformanceAnalyzer::generate_report(&metrics);

    println!("\n=== Performance Report ===");
    report.print_summary();

    Ok(())
}

/// Demonstrate Advanced Adaptive Mesh Refinement
#[allow(dead_code)]
fn demonstrate_advanced_amr(profiler: &mut PerformanceProfiler) -> IntegrateResult<()> {
    profiler.start_phase("amr_demonstration");
    println!("1. Advanced AMR Demonstration");
    println!("   Setting up adaptive mesh refinement for 2D heat equation...");

    // Create initial mesh level
    let mut cells = HashMap::new();
    let n_cells = 4; // 2x2 grid
    for i in 0..n_cells {
        let cell_id = CellId { level: 0, index: i };
        let x = (i % 2) as f64 * 0.5 + 0.25;
        let y = (i / 2) as f64 * 0.5 + 0.25;

        let cell = AdaptiveCell {
            id: cell_id,
            center: Array1::from_vec(vec![x, y]),
            size: 0.5,
            solution: Array1::from_vec(vec![initial_temperature(x, y)]),
            error_estimate: 0.0,
            refinement_flag: RefinementFlag::None,
            is_active: true,
            neighbors: vec![],
            parent: None,
            children: vec![],
        };
        cells.insert(cell_id, cell);
    }

    let initial_level = AdaptiveMeshLevel {
        level: 0,
        cells,
        grid_spacing: 0.5,
        boundary_cells: std::collections::HashSet::new(),
    };

    // Create AMR manager
    let mut amr_manager = AdvancedAMRManager::new(initial_level, 3, 0.01);

    // Add gradient-based refinement criterion
    let gradient_criterion = GradientRefinementCriterion {
        component: Some(0),
        threshold: 1.0,
        relative_tolerance: 0.1,
    };
    amr_manager.add_criterion(Box::new(gradient_criterion));

    // Simulate solution update
    let solution = Array2::from_shape_fn((4, 1), |(i, _j)| {
        let x = (i % 2) as f64 * 0.5 + 0.25;
        let y = (i / 2) as f64 * 0.5 + 0.25;
        heat_equation_solution(x, y, 0.1)
    });

    // Perform mesh adaptation
    let adaptation_result = amr_manager.adapt_mesh(&solution)?;

    println!("   - Cells refined: {}", adaptation_result.cells_refined);
    println!(
        "   - Cells coarsened: {}",
        adaptation_result.cells_coarsened
    );
    println!(
        "   - Total active cells: {}",
        adaptation_result.total_active_cells
    );
    println!(
        "   - Load balance quality: {:.3}",
        adaptation_result.load_balance_quality
    );

    profiler.end_phase("amr_demonstration");
    Ok(())
}

/// Demonstrate advanced error estimation techniques
#[allow(dead_code)]
fn demonstrate_error_estimation(profiler: &mut PerformanceProfiler) -> IntegrateResult<()> {
    profiler.start_phase("error_estimation_demo");
    println!("\n2. Advanced Error Estimation Demonstration");
    println!("   Using Richardson extrapolation and spectral analysis...");

    // Create advanced error estimator
    let mut error_estimator = AdvancedErrorEstimator::<f64>::new(1e-6, 3);

    // Simulate ODE solving with error analysis
    let ode_fn = |_t: f64, y: &ArrayView1<f64>| Array1::from_vec(vec![-y[0], -2.0 * y[1]]);

    // Simulate multiple solution steps with different step sizes
    let solutions = [
        Array1::from_vec(vec![1.0, 0.5]),   // t=0.0
        Array1::from_vec(vec![0.95, 0.45]), // t=0.1
        Array1::from_vec(vec![0.90, 0.40]), // t=0.2
        Array1::from_vec(vec![0.86, 0.36]), // t=0.3
    ];

    let step_sizes = [0.1, 0.1, 0.1];

    for (i, solution) in solutions.iter().enumerate().skip(1) {
        let step_size = step_sizes[i - 1];
        let embedded_error = Some(1e-4 * (i as f64));

        let error_analysis =
            error_estimator.analyze_error(solution, step_size, ode_fn, embedded_error)?;

        println!(
            "   Step {}: Primary error = {:.2e}, Confidence = {:.3}",
            i, error_analysis.primary_estimate, error_analysis.confidence
        );

        if let Some(richardson_error) = error_analysis.richardson_error {
            println!("            Richardson error = {richardson_error:.2e}");
        }
        if let Some(spectral_error) = error_analysis.spectral_error {
            println!("            Spectral error = {spectral_error:.2e}");
        }
    }

    profiler.end_phase("error_estimation_demo");
    Ok(())
}

/// Demonstrate parallel optimization with vectorized operations
#[allow(dead_code)]
fn demonstrate_parallel_optimization(profiler: &mut PerformanceProfiler) -> IntegrateResult<()> {
    profiler.start_phase("parallel_optimization_demo");
    println!("\n3. Parallel Optimization Demonstration");
    println!("   Performing vectorized matrix operations...");

    // Create parallel optimizer
    let mut optimizer = ParallelOptimizer::new(4);
    optimizer.initialize()?;

    // Create test matrix
    let input_matrix =
        Array2::from_shape_fn((100, 50), |(i, j)| (i as f64 * 0.1 + j as f64 * 0.05).sin());

    // Test different vectorized operations
    let operations = vec![
        (
            "Element-wise exp",
            VectorOperation::ElementWise(ArithmeticOp::Exp),
        ),
        (
            "Element-wise sin",
            VectorOperation::ElementWise(ArithmeticOp::Sin),
        ),
        (
            "Add constant",
            VectorOperation::ElementWise(ArithmeticOp::Add(1.0)),
        ),
        (
            "Multiply by 2",
            VectorOperation::ElementWise(ArithmeticOp::Multiply(2.0)),
        ),
    ];

    for (name, operation) in operations {
        let task = VectorizedComputeTask {
            input: input_matrix.clone(),
            operation,
            chunk_size: 25,
            prefer_simd: true,
        };

        let start_time = std::time::Instant::now();
        let result = optimizer.execute_vectorized(task)?;
        let duration = start_time.elapsed();

        println!(
            "   - {}: {:.3}ms, output shape: {:?}",
            name,
            duration.as_millis(),
            result.dim()
        );

        profiler.record_metric(&format!("{name}_time_ms"), duration.as_millis() as f64);
    }

    profiler.end_phase("parallel_optimization_demo");
    Ok(())
}

/// Initial temperature distribution for heat equation
#[allow(dead_code)]
fn initial_temperature(x: f64, y: f64) -> f64 {
    // Gaussian heat source
    let center_x = 0.5;
    let center_y = 0.5;
    let sigma = 0.2;

    let dx = x - center_x;
    let dy = y - center_y;
    let r_squared = dx * dx + dy * dy;

    100.0 * (-r_squared / (2.0 * sigma * sigma)).exp()
}

/// Analytical solution for heat equation (simplified)
#[allow(dead_code)]
fn heat_equation_solution(x: f64, y: f64, t: f64) -> f64 {
    // Simplified analytical solution for demonstration
    let center_x = 0.5;
    let center_y = 0.5;
    let alpha = 0.1; // thermal diffusivity
    let sigma = 0.2 + 4.0 * alpha * t; // spreading due to diffusion

    let dx = x - center_x;
    let dy = y - center_y;
    let r_squared = dx * dx + dy * dy;

    100.0 * (-r_squared / (2.0 * sigma * sigma)).exp() * (0.2 / sigma).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_features_integration() {
        // This test ensures the example compiles and basic functionality works
        let result = initial_temperature(0.5, 0.5);
        assert!(result > 0.0);

        let solution = heat_equation_solution(0.5, 0.5, 0.1);
        assert!(solution > 0.0 && solution < result);
    }

    #[test]
    fn test_individual_modules() {
        // Test AMR manager creation
        let initial_level = AdaptiveMeshLevel {
            level: 0,
            cells: HashMap::new(),
            grid_spacing: 1.0,
            boundary_cells: std::collections::HashSet::new(),
        };
        let _amr = AdvancedAMRManager::new(initial_level, 5, 0.01);

        // Test error estimator creation
        let _estimator = AdvancedErrorEstimator::<f64>::new(1e-6, 3);

        // Test parallel optimizer creation
        let _optimizer = ParallelOptimizer::new(2);

        // Test performance profiler creation
        let _profiler = PerformanceProfiler::new();
    }
}
