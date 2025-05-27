use scirs2_integrate::symplectic::potential::HamiltonianSystem;
use scirs2_integrate::symplectic::leapfrog::StormerVerlet;
use scirs2_integrate::symplectic::SymplecticIntegrator;
use ndarray::array;

fn main() {
    // Define a simple harmonic oscillator: H = p²/2 + q²/2
    let system = HamiltonianSystem::new(
        |_t, _q, p| p.clone(),  // dq/dt = p
        |_t, q, _p| -q.clone(), // dp/dt = -q
    );

    // Initial conditions: (q0, p0) = (1.0, 0.0)
    let q0 = array![1.0];
    let p0 = array![0.0];
    let t = 0.0;
    let dt = 0.1;

    // Create integrator
    let integrator = StormerVerlet::<f64>::new();

    // Take one step
    let (q1, p1) = integrator.step(&system, t, &q0, &p0, dt).unwrap();

    // Energy should be conserved (approximately)
    let initial_energy = 0.5_f64 * p0.dot(&p0) + 0.5_f64 * q0.dot(&q0);
    let final_energy = 0.5_f64 * p1.dot(&p1) + 0.5_f64 * q1.dot(&q1);
    
    println!("Initial energy: {}", initial_energy);
    println!("Final energy: {}", final_energy);
    println!("Energy difference: {}", (initial_energy - final_energy).abs());
    println!("Relative error: {}", (initial_energy - final_energy).abs() / initial_energy);
}