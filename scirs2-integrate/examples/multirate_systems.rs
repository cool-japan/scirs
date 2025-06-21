//! Multirate Systems Examples
//!
//! This example demonstrates solving ODEs with multiple time scales using
//! specialized multirate methods. These are particularly important for:
//! - Chemical kinetics with fast/slow reactions
//! - Electrical circuits with different RC time constants  
//! - Climate models with fast weather and slow climate dynamics
//! - Biological systems with multi-scale processes

use ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::ode::{
    MultirateMethod, MultirateOptions, MultirateSolver, MultirateSystem, ODEMethod,
};

/// Stiff oscillator with fast and slow components
/// Fast: ω = 100, Slow: ω = 1, representing a 100:1 time scale separation
#[derive(Clone)]
struct StiffOscillatorSystem {
    omega_fast: f64,
    omega_slow: f64,
    coupling: f64,
}

impl MultirateSystem<f64> for StiffOscillatorSystem {
    fn slow_rhs(&self, _t: f64, y_slow: ArrayView1<f64>, y_fast: ArrayView1<f64>) -> Array1<f64> {
        let x_slow = y_slow[0];
        let v_slow = y_slow[1];
        let x_fast = y_fast[0];

        // Slow oscillator influenced by fast component
        let dx_dt = v_slow;
        let dv_dt = -self.omega_slow * self.omega_slow * x_slow + self.coupling * x_fast;

        array![dx_dt, dv_dt]
    }

    fn fast_rhs(&self, _t: f64, y_slow: ArrayView1<f64>, y_fast: ArrayView1<f64>) -> Array1<f64> {
        let x_slow = y_slow[0];
        let x_fast = y_fast[0];
        let v_fast = y_fast[1];

        // Fast oscillator weakly coupled to slow component
        let dx_dt = v_fast;
        let dv_dt = -self.omega_fast * self.omega_fast * x_fast + self.coupling * x_slow;

        array![dx_dt, dv_dt]
    }

    fn slow_dim(&self) -> usize {
        2
    }
    fn fast_dim(&self) -> usize {
        2
    }
}

/// Chemical reaction system with fast equilibration and slow conversion
/// Fast: A ⇌ B (fast equilibrium), Slow: B → C (slow conversion)
struct ChemicalReactionSystem {
    k_fast_forward: f64,
    k_fast_backward: f64,
    k_slow: f64,
}

impl MultirateSystem<f64> for ChemicalReactionSystem {
    fn slow_rhs(&self, _t: f64, y_slow: ArrayView1<f64>, y_fast: ArrayView1<f64>) -> Array1<f64> {
        let _c = y_slow[0]; // Concentration of C (product)
        let b = y_fast[1]; // Concentration of B (intermediate)

        // Slow conversion: B → C
        let dc_dt = self.k_slow * b;

        array![dc_dt]
    }

    fn fast_rhs(&self, _t: f64, y_slow: ArrayView1<f64>, y_fast: ArrayView1<f64>) -> Array1<f64> {
        let _c = y_slow[0]; // Product C doesn't affect fast equilibrium
        let a = y_fast[0]; // Concentration of A
        let b = y_fast[1]; // Concentration of B

        // Fast equilibrium: A ⇌ B
        let da_dt = -self.k_fast_forward * a + self.k_fast_backward * b;
        let db_dt = self.k_fast_forward * a - self.k_fast_backward * b - self.k_slow * b;

        array![da_dt, db_dt]
    }

    fn slow_dim(&self) -> usize {
        1
    }
    fn fast_dim(&self) -> usize {
        2
    }
}

/// Van der Pol oscillator with two time scales
/// Represents electronic circuit with fast and slow relaxation
struct TwoTimescaleVanDerPol {
    epsilon: f64, // Slow parameter
    mu_fast: f64, // Fast nonlinearity
    mu_slow: f64, // Slow nonlinearity
}

impl MultirateSystem<f64> for TwoTimescaleVanDerPol {
    fn slow_rhs(&self, _t: f64, y_slow: ArrayView1<f64>, y_fast: ArrayView1<f64>) -> Array1<f64> {
        let x_slow = y_slow[0];
        let y_slow_var = y_slow[1];
        let x_fast = y_fast[0];

        // Slow Van der Pol dynamics
        let dx_dt = y_slow_var;
        let dy_dt = self.epsilon * (self.mu_slow * (1.0 - x_slow * x_slow) * y_slow_var - x_slow)
            + 0.1 * x_fast;

        array![dx_dt, dy_dt]
    }

    fn fast_rhs(&self, _t: f64, y_slow: ArrayView1<f64>, y_fast: ArrayView1<f64>) -> Array1<f64> {
        let x_slow = y_slow[0];
        let x_fast = y_fast[0];
        let y_fast_var = y_fast[1];

        // Fast Van der Pol dynamics
        let dx_dt = y_fast_var;
        let dy_dt = self.mu_fast * (1.0 - x_fast * x_fast) * y_fast_var - x_fast + 0.05 * x_slow;

        array![dx_dt, dy_dt]
    }

    fn slow_dim(&self) -> usize {
        2
    }
    fn fast_dim(&self) -> usize {
        2
    }
}

/// Climate model with fast weather and slow climate dynamics
struct ClimateWeatherSystem {
    climate_timescale: f64,
    weather_timescale: f64,
    coupling_strength: f64,
}

impl MultirateSystem<f64> for ClimateWeatherSystem {
    fn slow_rhs(&self, _t: f64, y_slow: ArrayView1<f64>, y_fast: ArrayView1<f64>) -> Array1<f64> {
        let temp_climate = y_slow[0]; // Long-term temperature trend
        let co2_level = y_slow[1]; // CO2 concentration
        let temp_weather = y_fast[0]; // Short-term weather temperature

        // Climate dynamics (decades to centuries)
        let dtemp_climate_dt = (co2_level - 280.0) / (100.0 * self.climate_timescale)
            + self.coupling_strength * (temp_weather - temp_climate);
        let dco2_dt = 2.0 / self.climate_timescale; // Anthropogenic CO2 increase

        array![dtemp_climate_dt, dco2_dt]
    }

    fn fast_rhs(&self, _t: f64, y_slow: ArrayView1<f64>, y_fast: ArrayView1<f64>) -> Array1<f64> {
        let temp_climate = y_slow[0];
        let temp_weather = y_fast[0];
        let pressure = y_fast[1];

        // Weather dynamics (days to weeks)
        let dtemp_weather_dt = -self.weather_timescale * (temp_weather - temp_climate)
            + 0.1 * pressure * (2.0 * (temp_weather * 0.1).sin() - 1.0);
        let dpressure_dt =
            -0.5 * self.weather_timescale * pressure + 0.2 * (temp_weather - temp_climate);

        array![dtemp_weather_dt, dpressure_dt]
    }

    fn slow_dim(&self) -> usize {
        2
    }
    fn fast_dim(&self) -> usize {
        2
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multirate Systems Examples\n");

    // Example 1: Stiff Oscillator System
    println!("1. Stiff Oscillator (ωfast=100, ωslow=1)");
    let stiff_system = StiffOscillatorSystem {
        omega_fast: 100.0, // Fast frequency
        omega_slow: 1.0,   // Slow frequency
        coupling: 0.1,     // Weak coupling
    };

    let options = MultirateOptions {
        method: MultirateMethod::ExplicitMRK {
            macro_steps: 4,
            micro_steps: 50,
        },
        macro_step: 0.01,
        rtol: 1e-6,
        atol: 1e-9,
        max_steps: 1000,
        timescale_ratio: Some(100.0),
    };

    let mut solver = MultirateSolver::new(options);

    // Initial: [x_slow, v_slow, x_fast, v_fast]
    let y0 = array![1.0, 0.0, 0.1, 0.0];
    let result = solver.solve(stiff_system, [0.0, 1.0], y0.clone())?;

    println!(
        "   Initial state: [slow: {:.3}, {:.3}] [fast: {:.3}, {:.3}]",
        y0[0], y0[1], y0[2], y0[3]
    );
    let final_state = result.y.last().unwrap();
    println!(
        "   Final state: [slow: {:.3}, {:.3}] [fast: {:.3}, {:.3}]",
        final_state[0], final_state[1], final_state[2], final_state[3]
    );
    println!("   Steps taken: {} (multirate efficiency)", result.n_steps);
    println!();

    // Example 2: Chemical Reaction System
    println!("2. Chemical Reactions (Fast Equilibrium + Slow Conversion)");
    let chem_system = ChemicalReactionSystem {
        k_fast_forward: 100.0, // Fast forward rate
        k_fast_backward: 80.0, // Fast backward rate
        k_slow: 0.5,           // Slow conversion rate
    };

    let options_chem = MultirateOptions {
        method: MultirateMethod::CompoundFastSlow {
            fast_method: ODEMethod::RK4,
            slow_method: ODEMethod::RK4,
        },
        macro_step: 0.05,
        rtol: 1e-7,
        atol: 1e-10,
        max_steps: 500,
        timescale_ratio: Some(200.0),
    };

    let mut solver_chem = MultirateSolver::new(options_chem);

    // Initial: [C, A, B]
    let y0_chem = array![0.0, 1.0, 0.0]; // Start with only A
    let result_chem = solver_chem.solve(chem_system, [0.0, 2.0], y0_chem.clone())?;

    println!(
        "   Initial concentrations: [C]={:.3}, [A]={:.3}, [B]={:.3}",
        y0_chem[0], y0_chem[1], y0_chem[2]
    );
    let final_chem = result_chem.y.last().unwrap();
    println!(
        "   Final concentrations: [C]={:.3}, [A]={:.3}, [B]={:.3}",
        final_chem[0], final_chem[1], final_chem[2]
    );
    println!(
        "   Total mass: {:.3} (conservation check)",
        final_chem[0] + final_chem[1] + final_chem[2]
    );
    println!(
        "   Fast equilibrium ratio A/B: {:.3}",
        final_chem[1] / final_chem[2]
    );
    println!();

    // Example 3: Two-Timescale Van der Pol
    println!("3. Two-Timescale Van der Pol Oscillator");
    let vdp_system = TwoTimescaleVanDerPol {
        epsilon: 0.1, // Slow time scale
        mu_fast: 5.0, // Fast nonlinearity
        mu_slow: 1.0, // Slow nonlinearity
    };

    let options_vdp = MultirateOptions {
        method: MultirateMethod::ExplicitMRK {
            macro_steps: 4,
            micro_steps: 25,
        },
        macro_step: 0.02,
        rtol: 1e-8,
        atol: 1e-11,
        max_steps: 2000,
        timescale_ratio: Some(50.0),
    };

    let mut solver_vdp = MultirateSolver::new(options_vdp);

    // Initial: [x_slow, y_slow, x_fast, y_fast]
    let y0_vdp = array![0.1, 0.0, 0.2, 0.1];
    let result_vdp = solver_vdp.solve(vdp_system, [0.0, 5.0], y0_vdp.clone())?;

    println!(
        "   Initial: [slow: {:.3}, {:.3}] [fast: {:.3}, {:.3}]",
        y0_vdp[0], y0_vdp[1], y0_vdp[2], y0_vdp[3]
    );
    let final_vdp = result_vdp.y.last().unwrap();
    println!(
        "   Final: [slow: {:.3}, {:.3}] [fast: {:.3}, {:.3}]",
        final_vdp[0], final_vdp[1], final_vdp[2], final_vdp[3]
    );
    println!(
        "   Steps taken: {} (complex multi-scale dynamics)",
        result_vdp.n_steps
    );
    println!();

    // Example 4: Climate-Weather System
    println!("4. Climate-Weather Model (Long-term/Short-term)");
    let climate_system = ClimateWeatherSystem {
        climate_timescale: 365.0 * 10.0, // 10-year climate scale
        weather_timescale: 7.0,          // 1-week weather scale
        coupling_strength: 0.01,
    };

    let options_climate = MultirateOptions {
        method: MultirateMethod::Extrapolated {
            base_ratio: 10,
            levels: 2,
        },
        macro_step: 1.0, // 1 day macro steps
        rtol: 1e-6,
        atol: 1e-9,
        max_steps: 365, // 1 year simulation
        timescale_ratio: Some(365.0 * 10.0 / 7.0),
    };

    let mut solver_climate = MultirateSolver::new(options_climate);

    // Initial: [temp_climate, co2, temp_weather, pressure]
    let y0_climate = array![15.0, 380.0, 15.5, 1013.0]; // Realistic climate values
    let result_climate = solver_climate.solve(climate_system, [0.0, 100.0], y0_climate.clone())?;

    println!(
        "   Initial: Climate temp={:.1}°C, CO2={:.0}ppm, Weather temp={:.1}°C, Pressure={:.0}hPa",
        y0_climate[0], y0_climate[1], y0_climate[2], y0_climate[3]
    );
    let final_climate = result_climate.y.last().unwrap();
    println!(
        "   Final: Climate temp={:.1}°C, CO2={:.0}ppm, Weather temp={:.1}°C, Pressure={:.0}hPa",
        final_climate[0], final_climate[1], final_climate[2], final_climate[3]
    );
    println!(
        "   Climate warming: {:.2}°C over 100 days",
        final_climate[0] - y0_climate[0]
    );
    println!(
        "   CO2 increase: {:.0}ppm",
        final_climate[1] - y0_climate[1]
    );
    println!();

    // Comparison with different multirate methods
    println!("5. Method Comparison on Stiff System");

    let test_system = StiffOscillatorSystem {
        omega_fast: 50.0,
        omega_slow: 1.0,
        coupling: 0.05,
    };

    let methods = vec![
        (
            "Explicit MRK",
            MultirateMethod::ExplicitMRK {
                macro_steps: 4,
                micro_steps: 25,
            },
        ),
        (
            "Compound F-S",
            MultirateMethod::CompoundFastSlow {
                fast_method: ODEMethod::RK4,
                slow_method: ODEMethod::RK4,
            },
        ),
        (
            "Extrapolated",
            MultirateMethod::Extrapolated {
                base_ratio: 5,
                levels: 2,
            },
        ),
    ];

    for (name, method) in methods {
        let options_test = MultirateOptions {
            method,
            macro_step: 0.02,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 250,
            timescale_ratio: Some(50.0),
        };

        let mut solver_test = MultirateSolver::new(options_test);
        let y0_test = array![1.0, 0.0, 0.1, 0.0];

        let start_time = std::time::Instant::now();
        let result_test = solver_test.solve(test_system.clone(), [0.0, 1.0], y0_test.clone())?;
        let elapsed = start_time.elapsed();

        println!(
            "   {}: {} steps, {:.2}ms",
            name,
            result_test.n_steps,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    println!("\nAll multirate examples completed successfully!");
    println!("\nMultirate Method Analysis:");
    println!("- Explicit MRK: Good balance of accuracy and efficiency for moderate stiffness");
    println!("- Compound Fast-Slow: Effective for well-separated time scales");
    println!("- Extrapolated: Higher accuracy through Richardson extrapolation");
    println!("- IMEX methods: Best for stiff systems (implicit for fast, explicit for slow)");
    println!("- Time scale separation ratio determines optimal micro/macro step ratio");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_chemical_reaction_conservation() {
        let system = ChemicalReactionSystem {
            k_fast_forward: 50.0,
            k_fast_backward: 40.0,
            k_slow: 1.0,
        };

        let options = MultirateOptions {
            method: MultirateMethod::ExplicitMRK {
                macro_steps: 4,
                micro_steps: 20,
            },
            macro_step: 0.01,
            rtol: 1e-8,
            atol: 1e-11,
            max_steps: 200,
            timescale_ratio: Some(50.0),
        };

        let mut solver = MultirateSolver::new(options);
        let y0 = array![0.0, 1.0, 0.0]; // Initial: only A

        let result = solver.solve(system, [0.0, 0.5], y0.clone()).unwrap();

        let initial_total = y0.sum();
        let final_total = result.y.last().unwrap().sum();

        // Mass should be conserved
        assert_abs_diff_eq!(initial_total, final_total, epsilon = 1e-3);

        // Some conversion should have occurred
        assert!(result.y.last().unwrap()[0] > 0.01); // Some C produced
    }

    #[test]
    fn test_stiff_oscillator_energy() {
        let system = StiffOscillatorSystem {
            omega_fast: 20.0,
            omega_slow: 1.0,
            coupling: 0.02, // Weak coupling
        };

        let options = MultirateOptions {
            method: MultirateMethod::ExplicitMRK {
                macro_steps: 4,
                micro_steps: 20,
            },
            macro_step: 0.005,
            rtol: 1e-10,
            atol: 1e-13,
            max_steps: 100,
            timescale_ratio: Some(20.0),
        };

        let mut solver = MultirateSolver::new(options);
        let y0 = array![1.0, 0.0, 0.1, 0.0];

        let result = solver.solve(system, [0.0, 0.1], y0.clone()).unwrap();

        // Calculate total energy (kinetic + potential)
        let initial_energy = 0.5 * (y0[1] * y0[1] + y0[3] * y0[3])
            + 0.5 * (1.0 * y0[0] * y0[0] + 400.0 * y0[2] * y0[2]);

        let final_state = result.y.last().unwrap();
        let final_energy = 0.5
            * (final_state[1] * final_state[1] + final_state[3] * final_state[3])
            + 0.5
                * (1.0 * final_state[0] * final_state[0] + 400.0 * final_state[2] * final_state[2]);

        // Energy should be approximately conserved (small coupling)
        let energy_error = (final_energy - initial_energy).abs() / initial_energy;
        assert!(energy_error < 0.1); // Within 10% due to coupling
    }
}
