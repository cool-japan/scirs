//! Example of LTI system interconnection: series, parallel, and feedback connections

use scirs2_signal::lti::system::*;
use scirs2_signal::lti::{bode, LtiSystem};

fn main() {
    println!("LTI System Interconnection Example");
    println!("===================================\n");

    // Example 1: Series Connection
    println!("1. Series Connection");
    println!("-------------------");

    // Create two first-order systems
    // G1(s) = 5 / (s + 2)
    let g1 = tf(vec![5.0], vec![1.0, 2.0], None).unwrap();
    println!("G1(s) = 5 / (s + 2)");

    // G2(s) = 1 / (s + 5)
    let g2 = tf(vec![1.0], vec![1.0, 5.0], None).unwrap();
    println!("G2(s) = 1 / (s + 5)");

    // Series connection: H(s) = G2(s) * G1(s)
    let series_sys = series(&g1, &g2).unwrap();
    println!("Series H(s) = G2(s) * G1(s)");
    println!("Numerator coefficients: {:?}", series_sys.num);
    println!("Denominator coefficients: {:?}", series_sys.den);
    // Result should be: H(s) = 5 / ((s+2)(s+5)) = 5 / (s^2 + 7s + 10)

    // Example 2: Parallel Connection
    println!("\n\n2. Parallel Connection");
    println!("---------------------");

    // Create two systems for parallel connection
    // G1(s) = 3 / (s + 1)
    let g1_par = tf(vec![3.0], vec![1.0, 1.0], None).unwrap();
    println!("G1(s) = 3 / (s + 1)");

    // G2(s) = 2 / (s + 4)
    let g2_par = tf(vec![2.0], vec![1.0, 4.0], None).unwrap();
    println!("G2(s) = 2 / (s + 4)");

    // Parallel connection: H(s) = G1(s) + G2(s)
    let parallel_sys = parallel(&g1_par, &g2_par).unwrap();
    println!("Parallel H(s) = G1(s) + G2(s)");
    println!("Numerator coefficients: {:?}", parallel_sys.num);
    println!("Denominator coefficients: {:?}", parallel_sys.den);
    // Result should be: H(s) = (3(s+4) + 2(s+1)) / ((s+1)(s+4)) = (5s+14) / (s^2+5s+4)

    // Example 3: Unity Feedback Control System
    println!("\n\n3. Unity Feedback Control System");
    println!("--------------------------------");

    // Plant: G(s) = 10 / (s(s + 1))
    let plant = tf(vec![10.0], vec![1.0, 1.0, 0.0], None).unwrap();
    println!("Plant G(s) = 10 / (s(s + 1))");

    // Unity feedback (negative feedback)
    let closed_loop = feedback(&plant, None, 1).unwrap();
    println!("Closed-loop T(s) = G(s) / (1 + G(s))");
    println!("Numerator coefficients: {:?}", closed_loop.num);
    println!("Denominator coefficients: {:?}", closed_loop.den);
    // Result: T(s) = 10 / (s^2 + s + 10)

    // Check stability
    match closed_loop.is_stable() {
        Ok(stable) => println!("Closed-loop system is stable: {}", stable),
        Err(_) => println!("Could not determine stability"),
    }

    // Example 4: Feedback with Controller
    println!("\n\n4. Feedback with PID Controller");
    println!("-------------------------------");

    // Plant: G(s) = 1 / ((s + 1)(s + 2))
    let plant_pid = tf(vec![1.0], vec![1.0, 3.0, 2.0], None).unwrap();
    println!("Plant G(s) = 1 / ((s + 1)(s + 2))");

    // PID Controller: C(s) = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki) / s
    // Let's use Kp = 10, Ki = 5, Kd = 2
    let controller = tf(vec![2.0, 10.0, 5.0], vec![1.0, 0.0], None).unwrap();
    println!("PID Controller C(s) = (2s^2 + 10s + 5) / s");

    // Series connection of controller and plant
    let forward_path = series(&controller, &plant_pid).unwrap();
    println!("Forward path = C(s) * G(s)");

    // Closed-loop with unity feedback
    let pid_closed_loop = feedback(&forward_path, None, 1).unwrap();
    println!("PID Closed-loop system:");
    println!("Numerator coefficients: {:?}", pid_closed_loop.num);
    println!("Denominator coefficients: {:?}", pid_closed_loop.den);

    // Example 5: Sensitivity and Complementary Sensitivity
    println!("\n\n5. Sensitivity Functions");
    println!("------------------------");

    // For the unity feedback system from Example 3
    let sens = sensitivity(&plant, None).unwrap();
    let comp_sens = complementary_sensitivity(&plant, None).unwrap();

    println!("Sensitivity S(s) = 1 / (1 + G(s)):");
    println!("Numerator coefficients: {:?}", sens.num);
    println!("Denominator coefficients: {:?}", sens.den);

    println!("Complementary Sensitivity T(s) = G(s) / (1 + G(s)):");
    println!("Numerator coefficients: {:?}", comp_sens.num);
    println!("Denominator coefficients: {:?}", comp_sens.den);

    // Verify that S(s) + T(s) = 1
    let sum_st = parallel(&sens, &comp_sens).unwrap();
    println!("S(s) + T(s) verification:");
    println!("Numerator coefficients: {:?}", sum_st.num);
    println!("Denominator coefficients: {:?}", sum_st.den);
    // Should result in a transfer function equal to 1

    // Example 6: Frequency Response Comparison
    println!("\n\n6. Frequency Response Analysis");
    println!("------------------------------");

    // Compare open-loop vs closed-loop frequency response
    let freqs = vec![0.1, 1.0, 10.0, 100.0];

    println!("Frequency response comparison (|H(jω)| in dB):");
    println!("Frequency (rad/s) | Open-loop | Closed-loop");
    println!("------------------|-----------|------------");

    for &freq in &freqs {
        let open_loop_resp = plant.frequency_response(&[freq]).unwrap();
        let closed_loop_resp = closed_loop.frequency_response(&[freq]).unwrap();

        let open_loop_mag = 20.0 * open_loop_resp[0].norm().log10();
        let closed_loop_mag = 20.0 * closed_loop_resp[0].norm().log10();

        println!(
            "{:16.1} | {:8.2} | {:10.2}",
            freq, open_loop_mag, closed_loop_mag
        );
    }

    // Example 7: Bode Plot Data
    println!("\n\n7. Bode Plot Data");
    println!("-----------------");

    // Generate Bode plot data for the closed-loop system
    let bode_freqs = vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
    let (w, mag, phase) = bode(&closed_loop, Some(&bode_freqs)).unwrap();

    println!("Closed-loop Bode plot data:");
    println!("Frequency (rad/s) | Magnitude (dB) | Phase (deg)");
    println!("------------------|----------------|------------");

    for ((freq, magnitude), ph) in w.iter().zip(mag.iter()).zip(phase.iter()) {
        println!("{:16.1} | {:13.2} | {:10.2}", freq, magnitude, ph);
    }

    // Example 8: Multi-stage System
    println!("\n\n8. Multi-stage System Design");
    println!("----------------------------");

    // Design a multi-stage system: Prefilter -> Controller -> Plant

    // Prefilter for reference shaping: F(s) = 1 / (0.1s + 1)
    let prefilter = tf(vec![1.0], vec![0.1, 1.0], None).unwrap();
    println!("Prefilter F(s) = 1 / (0.1s + 1)");

    // Simple proportional controller: C(s) = 5
    let prop_controller = tf(vec![5.0], vec![1.0], None).unwrap();
    println!("Controller C(s) = 5");

    // Plant: G(s) = 1 / (s + 1)
    let simple_plant = tf(vec![1.0], vec![1.0, 1.0], None).unwrap();
    println!("Plant G(s) = 1 / (s + 1)");

    // Build the system step by step
    let controller_plant = series(&prop_controller, &simple_plant).unwrap();
    let closed_inner = feedback(&controller_plant, None, 1).unwrap();
    let complete_system = series(&prefilter, &closed_inner).unwrap();

    println!("Complete system T(s) = F(s) * (C(s)*G(s))/(1 + C(s)*G(s)):");
    println!("Numerator coefficients: {:?}", complete_system.num);
    println!("Denominator coefficients: {:?}", complete_system.den);

    println!("\n\nSystem interconnection allows for:");
    println!("- Building complex control systems from simple components");
    println!("- Analyzing system properties (stability, performance)");
    println!("- Designing controllers and compensators");
    println!("- Computing sensitivity functions for robustness analysis");
}
