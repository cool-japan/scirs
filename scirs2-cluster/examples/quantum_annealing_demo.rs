//! Quantum Annealing Clustering Demonstration
//!
//! This example demonstrates the quantum annealing clustering algorithm,
//! showcasing its ability to find global optima using quantum tunneling effects.

use scirs2_cluster::{
    quantum_annealing_clustering, CoolingSchedule, QuantumAnnealingClustering,
    QuantumAnnealingConfig,
};
use scirs2_core::ndarray::Array2;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Quantum Annealing Clustering Demonstration");
    println!("===========================================");

    // Create sample data with three distinct clusters
    let data = Array2::from_shape_vec(
        (12, 2),
        vec![
            // Cluster 1: Around origin
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, // Cluster 2: Around (5, 5)
            5.0, 5.0, 5.1, 5.1, 4.9, 5.1, 5.1, 4.9, // Cluster 3: Around (-3, 3)
            -3.0, 3.0, -2.9, 3.1, -3.1, 2.9, -2.9, 2.9,
        ],
    )?;

    println!(
        "📊 Data points: {} samples, {} features",
        data.nrows(),
        data.ncols()
    );
    println!("🎯 Target clusters: 3");
    println!();

    // Example 1: Basic quantum annealing clustering
    println!("🚀 Example 1: Basic Quantum Annealing Clustering");
    println!("------------------------------------------------");

    let (assignments, energy) = quantum_annealing_clustering(data.view(), 3)?;

    println!("✅ Clustering completed!");
    println!("   Final energy: {:.6}", energy);
    println!("   Cluster assignments: {:?}", assignments.to_vec());
    println!();

    // Example 2: Custom configuration with different cooling schedules
    println!("🔧 Example 2: Custom Configuration");
    println!("----------------------------------");

    let cooling_schedules = vec![
        ("Linear", CoolingSchedule::Linear),
        ("Exponential", CoolingSchedule::Exponential),
        ("Logarithmic", CoolingSchedule::Logarithmic),
        ("Power Law (α=1.5)", CoolingSchedule::PowerLaw(1.5)),
    ];

    for (name, schedule) in cooling_schedules {
        let config = QuantumAnnealingConfig {
            initial_temperature: 20.0,
            final_temperature: 0.001,
            annealing_steps: 500,
            cooling_schedule: schedule,
            mc_sweeps: 50,
            random_seed: Some(42),
        };

        let mut annealer = QuantumAnnealingClustering::new(3, config);
        annealer.fit(data.view())?;

        let assignments = annealer.predict(data.view())?;
        let energy = annealer.best_energy().unwrap_or(0.0);

        println!("🌡️  {} schedule:", name);
        println!("   Energy: {:.6}", energy);
        println!("   Temperature range: [20.0 → 0.001] over 500 steps");
        println!("   Assignments: {:?}", assignments.to_vec());
        println!();
    }

    // Example 3: Performance comparison
    println!("⚡ Example 3: Performance Analysis");
    println!("---------------------------------");

    let mut best_energy = f64::INFINITY;
    let mut best_schedule = "";

    for (name, schedule) in &[
        ("Exponential", CoolingSchedule::Exponential),
        ("Power Law (α=2.0)", CoolingSchedule::PowerLaw(2.0)),
    ] {
        let config = QuantumAnnealingConfig {
            initial_temperature: 10.0,
            final_temperature: 0.01,
            annealing_steps: 1000,
            cooling_schedule: *schedule,
            mc_sweeps: 100,
            random_seed: Some(123),
        };

        let start_time = std::time::Instant::now();
        let mut annealer = QuantumAnnealingClustering::new(3, config);
        annealer.fit(data.view())?;
        let duration = start_time.elapsed();

        let energy = annealer.best_energy().unwrap_or(0.0);
        let temperature_schedule = annealer.temperature_schedule();

        println!("📈 {} schedule:", name);
        println!("   Final energy: {:.6}", energy);
        println!("   Runtime: {:.2?}", duration);
        println!("   Temperature steps: {}", temperature_schedule.len());
        println!("   Initial temp: {:.3}", temperature_schedule[0]);
        println!(
            "   Final temp: {:.6}",
            temperature_schedule[temperature_schedule.len() - 1]
        );

        if energy < best_energy {
            best_energy = energy;
            best_schedule = name;
        }
        println!();
    }

    println!(
        "🏆 Best performing schedule: {} (energy: {:.6})",
        best_schedule, best_energy
    );
    println!();

    // Example 4: Detailed algorithm analysis
    println!("🔍 Example 4: Algorithm Analysis");
    println!("--------------------------------");

    let config = QuantumAnnealingConfig {
        initial_temperature: 15.0,
        final_temperature: 0.005,
        annealing_steps: 800,
        cooling_schedule: CoolingSchedule::Exponential,
        mc_sweeps: 75,
        random_seed: Some(456),
    };

    let mut annealer = QuantumAnnealingClustering::new(3, config);

    println!("⚙️  Configuration:");
    println!("   Initial temperature: 15.0");
    println!("   Final temperature: 0.005");
    println!("   Annealing steps: 800");
    println!("   MC sweeps per step: 75");
    println!("   Cooling: Exponential");
    println!();

    let start_time = std::time::Instant::now();
    annealer.fit(data.view())?;
    let training_time = start_time.elapsed();

    let assignments = annealer.predict(data.view())?;
    let energy = annealer.best_energy().unwrap();
    let temp_schedule = annealer.temperature_schedule();

    println!("📊 Results:");
    println!("   Training time: {:.2?}", training_time);
    println!("   Final energy: {:.8}", energy);
    println!("   Cluster assignments: {:?}", assignments.to_vec());

    // Analyze cluster quality
    let mut cluster_counts = vec![0; 3];
    for &cluster in assignments.iter() {
        cluster_counts[cluster] += 1;
    }

    println!("   Cluster sizes: {:?}", cluster_counts);
    println!(
        "   Temperature decay rate: {:.6}",
        temp_schedule[temp_schedule.len() - 1] / temp_schedule[0]
    );
    println!();

    println!("✨ Quantum annealing clustering demonstration completed!");
    println!("   The algorithm successfully found cluster assignments using");
    println!("   quantum tunneling effects to escape local minima.");

    Ok(())
}
