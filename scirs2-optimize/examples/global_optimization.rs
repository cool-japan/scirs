//! Examples of global optimization algorithms

use ndarray::{array, ArrayView1};
use scirs2_optimize::global::{
    basinhopping, differential_evolution, dual_annealing, particle_swarm, simulated_annealing,
    BasinHoppingOptions, DifferentialEvolutionOptions, DualAnnealingOptions, ParticleSwarmOptions,
    SimulatedAnnealingOptions,
};

fn main() {
    println!("Global Optimization Examples\n");

    // Example 1: Differential Evolution
    println!("1. Differential Evolution - Rosenbrock function");
    let rosenbrock = |x: &ndarray::ArrayView1<f64>| {
        let a = 1.0;
        let b = 100.0;
        let x0 = x[0];
        let x1 = x[1];
        (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
    };

    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    let options = DifferentialEvolutionOptions {
        maxiter: 100,
        popsize: 15,
        seed: Some(42),
        ..Default::default()
    };

    let result = differential_evolution(rosenbrock, bounds, Some(options), None).unwrap();
    println!("Differential Evolution result:");
    println!("  Solution: {:?}", result.x);
    println!("  Function value: {}", result.fun);
    println!("  Success: {}\n", result.success);

    // Example 2: Basin-hopping
    println!("2. Basin-hopping - Sphere function");
    let sphere = |x: &ndarray::ArrayView1<f64>| x.iter().map(|&xi| xi.powi(2)).sum::<f64>();

    let x0 = array![2.0, 3.0];

    let options = BasinHoppingOptions {
        niter: 50,
        seed: Some(42),
        ..Default::default()
    };

    let result = basinhopping(sphere, x0.clone(), Some(options), None, None).unwrap();
    println!("Basin-hopping result:");
    println!("  Solution: {:?}", result.x);
    println!("  Function value: {}", result.fun);
    println!("  Success: {}\n", result.success);

    // Example 3: Dual Annealing
    println!("3. Dual Annealing - Rastrigin function");
    let rastrigin = |x: &ndarray::ArrayView1<f64>| {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|&xi| xi.powi(2) - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    };

    let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];

    let options = DualAnnealingOptions {
        maxiter: 100,
        seed: Some(42),
        ..Default::default()
    };

    let result = dual_annealing(rastrigin, x0, bounds, Some(options)).unwrap();
    println!("Dual Annealing result:");
    println!("  Solution: {:?}", result.x);
    println!("  Function value: {}", result.fun);
    println!("  Success: {}\n", result.success);

    // Example 4: Particle Swarm Optimization
    println!("4. Particle Swarm Optimization - Ackley function");
    let ackley = |x: &ArrayView1<f64>| {
        let n = x.len() as f64;
        let a = 20.0;
        let b = 0.2;
        let c = 2.0 * std::f64::consts::PI;

        let sum1 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>() / n;
        let sum2 = x.iter().map(|&xi| (c * xi).cos()).sum::<f64>() / n;

        -a * (-b * sum1.sqrt()).exp() - sum2.exp() + a + std::f64::consts::E
    };

    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    let options = ParticleSwarmOptions {
        swarm_size: 30,
        maxiter: 100,
        seed: Some(42),
        ..Default::default()
    };

    let result = particle_swarm(ackley, bounds, Some(options)).unwrap();
    println!("Particle Swarm result:");
    println!("  Solution: {:?}", result.x);
    println!("  Function value: {}", result.fun);
    println!("  Success: {}\n", result.success);

    // Example 5: Simulated Annealing
    println!("5. Simulated Annealing - Himmelblau's function");
    let himmelblau = |x: &ArrayView1<f64>| {
        let x0 = x[0];
        let x1 = x[1];
        (x0.powi(2) + x1 - 11.0).powi(2) + (x0 + x1.powi(2) - 7.0).powi(2)
    };

    let x0 = array![0.0, 0.0];
    let bounds = Some(vec![(-5.0, 5.0), (-5.0, 5.0)]);

    let options = SimulatedAnnealingOptions {
        maxiter: 1000,
        initial_temp: 10.0,
        seed: Some(42),
        ..Default::default()
    };

    let result = simulated_annealing(himmelblau, x0, bounds, Some(options)).unwrap();
    println!("Simulated Annealing result:");
    println!("  Solution: {:?}", result.x);
    println!("  Function value: {}", result.fun);
    println!("  Success: {}", result.success);
}
