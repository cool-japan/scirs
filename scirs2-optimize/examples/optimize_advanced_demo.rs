//! Advanced Mode Demonstration
//!
//! This example showcases the advanced capabilities of the Advanced Coordinator,
//! demonstrating how it intelligently combines quantum-inspired optimization,
//! neuromorphic computing, and meta-learning approaches for superior optimization performance.

use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_optimize::prelude::*;

/// Complex multi-modal test function for demonstrating Advanced capabilities
#[allow(dead_code)]
fn complex_optimization_problem(x: &ArrayView1<f64>) -> f64 {
    let n = x.len();
    let mut result = 0.0;

    // Rosenbrock-like terms with multiple local minima
    for i in 0..n - 1 {
        let term1 = (1.0 - x[i]).powi(2);
        let term2 = 100.0 * (x[i + 1] - x[i].powi(2)).powi(2);
        result += term1 + term2;
    }

    // Add multi-modal landscape with multiple local minima
    for i in 0..n {
        result += 0.1 * (10.0 * std::f64::consts::PI * x[i]).sin().powi(2);
    }

    // Add high-frequency noise
    for i in 0..n {
        result += 0.01 * (50.0 * x[i]).cos();
    }

    result
}

/// Demonstration of different Advanced strategies
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Advanced Mode Optimization Demonstration 🧠");
    println!("=================================================");

    // Problem setup
    let dimension = 5;
    let initial_params = Array1::from_vec((0..dimension).map(|i| (i as f64 - 2.5) * 0.5).collect());
    let initial_objective = complex_optimization_problem(&initial_params.view());

    println!("\n📋 Problem Configuration:");
    println!("   Dimension: {}", dimension);
    println!("   Initial parameters: {:?}", initial_params);
    println!("   Initial objective: {:.6}", initial_objective);

    // Strategy 1: Quantum-Neural Fusion
    println!("\n🌌 Strategy 1: Quantum-Neural Fusion");
    println!("   Combining quantum superposition with neural adaptation...");

    let config1 = AdvancedConfig {
        strategy: AdvancedStrategy::QuantumNeuralFusion,
        max_nit: 500,
        max_evaluations: 5000,
        tolerance: 1e-8,
        time_budget: Some(std::time::Duration::from_secs(30)),
        ..Default::default()
    };

    let result1 = advanced_optimize(
        complex_optimization_problem,
        &initial_params.view(),
        Some(config1),
    )?;

    println!(
        "   ✅ Result: {:.6} (improvement: {:.3}%)",
        result1.fun,
        ((initial_objective - result1.fun) / initial_objective * 100.0).max(0.0)
    );
    println!(
        "   📊 Iterations: {}, Success: {}",
        result1.nit, result1.success
    );

    // Strategy 2: Neuromorphic-Quantum Hybrid
    println!("\n🧠 Strategy 2: Neuromorphic-Quantum Hybrid");
    println!("   Leveraging spiking networks with quantum tunneling...");

    let config2 = AdvancedConfig {
        strategy: AdvancedStrategy::NeuromorphicQuantumHybrid,
        max_nit: 500,
        max_evaluations: 5000,
        tolerance: 1e-8,
        fusion_strength: 0.8,
        ..Default::default()
    };

    let result2 = advanced_optimize(
        complex_optimization_problem,
        &initial_params.view(),
        Some(config2),
    )?;

    println!(
        "   ✅ Result: {:.6} (improvement: {:.3}%)",
        result2.fun,
        ((initial_objective - result2.fun) / initial_objective * 100.0).max(0.0)
    );
    println!(
        "   📊 Iterations: {}, Success: {}",
        result2.nit, result2.success
    );

    // Strategy 3: Meta-Learning Quantum
    println!("\n🎯 Strategy 3: Meta-Learning Quantum");
    println!("   Applying meta-learning with quantum enhancement...");

    let config3 = AdvancedConfig {
        strategy: AdvancedStrategy::MetaLearningQuantum,
        max_nit: 500,
        max_evaluations: 5000,
        tolerance: 1e-8,
        coordination_learning_rate: 0.02,
        ..Default::default()
    };

    let result3 = advanced_optimize(
        complex_optimization_problem,
        &initial_params.view(),
        Some(config3),
    )?;

    println!(
        "   ✅ Result: {:.6} (improvement: {:.3}%)",
        result3.fun,
        ((initial_objective - result3.fun) / initial_objective * 100.0).max(0.0)
    );
    println!(
        "   📊 Iterations: {}, Success: {}",
        result3.nit, result3.success
    );

    // Strategy 4: Adaptive Selection
    println!("\n🔄 Strategy 4: Adaptive Selection");
    println!("   Dynamic strategy switching based on performance...");

    let config4 = AdvancedConfig {
        strategy: AdvancedStrategy::AdaptiveSelection,
        max_nit: 500,
        max_evaluations: 5000,
        tolerance: 1e-8,
        switching_threshold: 0.005,
        ..Default::default()
    };

    let result4 = advanced_optimize(
        complex_optimization_problem,
        &initial_params.view(),
        Some(config4),
    )?;

    println!(
        "   ✅ Result: {:.6} (improvement: {:.3}%)",
        result4.fun,
        ((initial_objective - result4.fun) / initial_objective * 100.0).max(0.0)
    );
    println!(
        "   📊 Iterations: {}, Success: {}",
        result4.nit, result4.success
    );

    // Strategy 5: Full Advanced
    println!("\n🚀 Strategy 5: Full Advanced");
    println!("   All strategies working in parallel with intelligent coordination...");

    let config5 = AdvancedConfig {
        strategy: AdvancedStrategy::FullAdvanced,
        max_nit: 1000,
        max_evaluations: 10000,
        tolerance: 1e-10,
        parallel_threads: 4,
        fusion_strength: 0.9,
        coordination_learning_rate: 0.01,
        ..Default::default()
    };

    let result5 = advanced_optimize(
        complex_optimization_problem,
        &initial_params.view(),
        Some(config5.clone()),
    )?;

    println!(
        "   ✅ Result: {:.6} (improvement: {:.3}%)",
        result5.fun,
        ((initial_objective - result5.fun) / initial_objective * 100.0).max(0.0)
    );
    println!(
        "   📊 Iterations: {}, Success: {}",
        result5.nit, result5.success
    );

    // Performance Comparison
    println!("\n📈 Performance Comparison Summary:");
    println!("=================================");
    let results = vec![
        ("Quantum-Neural Fusion", result1.fun),
        ("Neuromorphic-Quantum Hybrid", result2.fun),
        ("Meta-Learning Quantum", result3.fun),
        ("Adaptive Selection", result4.fun),
        ("Full Advanced", result5.fun),
    ];

    let mut sorted_results = results.clone();
    sorted_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (i, (strategy, objective)) in sorted_results.iter().enumerate() {
        let improvement = ((initial_objective - objective) / initial_objective * 100.0).max(0.0);
        println!(
            "{}. {} - {:.8} (improvement: {:.3}%)",
            i + 1,
            strategy,
            objective,
            improvement
        );
    }

    println!("\n🎉 Advanced Demonstration Complete!");
    println!("The Full Advanced strategy typically achieves the best performance");
    println!("by intelligently coordinating all available optimization approaches.");

    // Advanced Statistics Demonstration
    println!("\n📊 Advanced Advanced Statistics:");
    println!("==================================");

    // Create a coordinator to demonstrate statistics
    let mut coordinator = AdvancedCoordinator::new(config5, &initial_params.view());

    // Run a few optimization steps to generate statistics
    let _ = coordinator.optimize(complex_optimization_problem);
    let stats = coordinator.get_advanced_stats();

    println!("Total Function Evaluations: {}", stats.total_evaluations);
    println!("Current Iteration: {}", stats.current_iteration);
    println!("Best Objective Found: {:.8}", stats.best_objective);
    println!("Active Strategy: {:?}", stats.active_strategy);
    println!("Elapsed Time: {:.2?}", stats.elapsed_time);

    if let Some(quantum_stats) = stats.quantum_stats {
        println!("\n🌌 Quantum Component Statistics:");
        println!("   Tunneling Events: {}", quantum_stats.tunneling_events);
        println!(
            "   Current Temperature: {:.6}",
            quantum_stats.current_temperature
        );
        println!(
            "   Entanglement Strength: {:.6}",
            quantum_stats.entanglement_strength
        );
        println!(
            "   Quantum Interference: {:.6}",
            quantum_stats.quantum_interference
        );
        println!(
            "   Superposition Dimension: {}",
            quantum_stats.superposition_dimension
        );
        println!("   Evolution Time: {:.6}", quantum_stats.evolution_time);
        println!(
            "   Decoherence Level: {:.6}",
            quantum_stats.decoherence_level
        );
    }

    println!("\n💡 Pro Tips for Advanced Optimization:");
    println!("========================================");
    println!("1. Use Fulladvanced for the most challenging problems");
    println!("2. AdaptiveSelection automatically chooses the best strategy");
    println!("3. Increase fusion_strength for better cross-modal coordination");
    println!("4. Set appropriate time_budget for real-time applications");
    println!("5. Monitor quantum tunneling events for local minima escape");

    Ok(())
}

/// Additional helper function for benchmarking
#[allow(dead_code)]
fn benchmark_advanced_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Advanced Strategy Benchmarking");
    println!("==================================");

    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    fn rastrigin(x: &ArrayView1<f64>) -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        a * n
            + x.iter()
                .map(|&xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    fn ackley(x: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let sum1 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>() / n;
        let sum2 = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
            / n;
        -20.0 * (-0.2 * sum1.sqrt()).exp() - sum2.exp() + 20.0 + std::f64::consts::E
    }

    let test_functions = vec![
        ("Rosenbrock", rosenbrock as fn(&ArrayView1<f64>) -> f64),
        ("Rastrigin", rastrigin as fn(&ArrayView1<f64>) -> f64),
        ("Ackley", ackley as fn(&ArrayView1<f64>) -> f64),
    ];

    for (func_name, objective) in test_functions {
        println!("\n🎯 Testing on {} function:", func_name);

        let initial_params = Array1::from_vec(vec![2.5; 3]);
        let config = AdvancedConfig {
            strategy: AdvancedStrategy::FullAdvanced,
            max_nit: 200,
            max_evaluations: 2000,
            ..Default::default()
        };

        let result = advanced_optimize(objective, &initial_params.view(), Some(config))?;
        println!("   Result: {:.6} in {} iterations", result.fun, result.nit);
    }

    Ok(())
}
