use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("LSODA Method Switching Visualization Example");
    println!("-------------------------------------------");
    println!("This example focuses on collecting detailed information about when and why");
    println!("LSODA switches between methods, to help understand its behavior");
    println!("The output is formatted to be easily parsed for visualization");

    // First example: A system with regions of varying stiffness
    println!("\nRunning stiffness wave example");
    println!("This system's stiffness oscillates between stiff and non-stiff regions");

    // System with oscillating stiffness
    let stiffness_wave = |t: f64, y: ArrayView1<f64>| {
        // The stiffness parameter oscillates between -1 and 1001
        // This causes the system to alternate between stiff and non-stiff regions
        let stiffness = 500.0 + 500.0 * (t * 0.5).sin();

        // A 2D system where the first component has variable stiffness
        // and the second is constant for reference
        array![-stiffness * y[0], -y[1]]
    };

    let result = solve_ivp(
        stiffness_wave,
        [0.0, 40.0], // Long enough to see multiple stiffness cycles
        array![1.0, 1.0],
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-4,
            atol: 1e-6,
            max_steps: 10000,
            h0: Some(0.1),
            min_step: Some(1e-6),
            // No verbose option available
            ..Default::default()
        }),
    );

    match result {
        Ok(res) => {
            println!("Integration successful!");
            println!(
                "Steps: {}, Function evaluations: {}",
                res.n_steps, res.n_eval
            );
            println!("Accepted: {}, Rejected: {}", res.n_accepted, res.n_rejected);

            if let Some(msg) = &res.message {
                println!("{}", msg);
            }

            // Save detailed results for visualization
            let mut file = File::create("lsoda_visualization_data.csv").unwrap();
            writeln!(&mut file, "t,y1,y2,stiffness,h,method").unwrap();

            // Method string representations for CSV output
            let method_names = ["unknown", "Adams", "BDF"];

            // Reconstruct stiffness at each time point and save with solution
            for i in 0..res.t.len() {
                let t = res.t[i];
                let stiffness = 500.0 + 500.0 * (t * 0.5).sin();

                // We can't directly access which method was used at each step from the result
                // But we can infer it based on the step size pattern or just use "unknown"
                // In a real visualization tool, you would need to capture this during integration
                let method_idx = 0; // Will be filled with real method info in enhanced solver

                writeln!(
                    &mut file,
                    "{:.6},{:.10e},{:.10e},{:.6},{:.10e},{}",
                    t,
                    res.y[i][0],
                    res.y[i][1],
                    stiffness,
                    if i > 0 { res.t[i] - res.t[i - 1] } else { 0.1 },
                    method_names[method_idx]
                )
                .unwrap();
            }

            println!("\nData saved to lsoda_visualization_data.csv");
            println!("This file contains t, y values, stiffness, and step size info");
            println!("You can use this data to visualize the solution and method switching");

            // Generate instructions for plotting
            let mut readme = File::create("lsoda_visualization_plotting_guide.txt").unwrap();
            writeln!(
                &mut readme,
                "LSODA Method Switching Visualization Guide\n\
                =========================================\n\
                \n\
                The data files contain detailed information about the LSODA solution\n\
                including times, function values, stiffness measures, and more.\n\
                \n\
                To visualize the method switching behavior, you can use tools like\n\
                Python with matplotlib, gnuplot, or any other plotting software.\n\
                \n\
                Python Example:\n\
                ```python\n\
                import numpy as np\n\
                import matplotlib.pyplot as plt\n\
                \n\
                # Load the data\n\
                data = np.loadtxt('lsoda_visualization_data.csv', delimiter=',', skiprows=1)\n\
                t = data[:, 0]\n\
                y1 = data[:, 1]\n\
                y2 = data[:, 2]\n\
                stiffness = data[:, 3]\n\
                step_size = data[:, 4]\n\
                \n\
                # Create a figure with multiple subplots\n\
                fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)\n\
                \n\
                # Plot solution\n\
                axes[0].semilogy(t, np.abs(y1), label='y1 (variable stiffness)')\n\
                axes[0].semilogy(t, np.abs(y2), label='y2 (reference)')\n\
                axes[0].set_ylabel('Solution (log scale)')\n\
                axes[0].legend()\n\
                axes[0].grid(True)\n\
                \n\
                # Plot stiffness parameter\n\
                axes[1].plot(t, stiffness)\n\
                axes[1].set_ylabel('Stiffness Parameter')\n\
                axes[1].grid(True)\n\
                \n\
                # Plot step size\n\
                axes[2].semilogy(t, step_size)\n\
                axes[2].set_ylabel('Step Size (log scale)')\n\
                axes[2].set_xlabel('Time')\n\
                axes[2].grid(True)\n\
                \n\
                plt.tight_layout()\n\
                plt.savefig('lsoda_method_switching_visualization.png')\n\
                plt.show()\n\
                ```\n\
                \n\
                What to Look For:\n\
                ----------------\n\
                1. Stiffness parameter: Higher values indicate stiffer regions\n\
                2. Step size changes: LSODA typically takes larger steps in smooth regions\n\
                   and smaller steps in rapidly changing regions\n\
                3. Correlation between stiffness and step size: In stiff regions with BDF method,\n\
                   the solver can sometimes take larger steps than Adams would allow\n\
                4. Method switching: If you enhanced the solver to output method information,\n\
                   you'll see transitions between Adams and BDF methods\n\
                \n\
                For optimal visualization of LSODA's behavior, consider enhancing the solver\n\
                to output additional diagnostic information during integration."
            )
            .unwrap();

            println!("A plotting guide has been saved to lsoda_visualization_plotting_guide.txt");
        }
        Err(e) => {
            println!("Integration failed: {}", e);
        }
    }

    // Second example: Robertson chemical reaction system
    // This is a classic stiff problem that's good for demonstrating method switching
    println!("\nRunning Robertson chemical reaction example");
    println!("This is a standard test case for stiff ODE solvers");

    let robertson = |_t: f64, y: ArrayView1<f64>| {
        let k1 = 0.04;
        let k2 = 3.0e7;
        let k3 = 1.0e4;

        array![
            -k1 * y[0] + k3 * y[1] * y[2],
            k1 * y[0] - k2 * y[1].powi(2) - k3 * y[1] * y[2],
            k2 * y[1].powi(2)
        ]
    };

    let result = solve_ivp(
        robertson,
        [0.0, 1000.0], // Long time interval to see full behavior
        array![1.0, 0.0, 0.0],
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-4,
            atol: 1e-7, // Tighter atol for this problem
            max_steps: 10000,
            // No verbose option available
            ..Default::default()
        }),
    );

    match result {
        Ok(res) => {
            println!("Integration successful!");
            println!(
                "Steps: {}, Function evaluations: {}",
                res.n_steps, res.n_eval
            );
            println!("Accepted: {}, Rejected: {}", res.n_accepted, res.n_rejected);

            if let Some(msg) = &res.message {
                println!("{}", msg);
            }

            // Save solution trajectory
            let mut file = File::create("robertson_solution.csv").unwrap();
            writeln!(&mut file, "t,y1,y2,y3,step_size").unwrap();

            for i in 0..res.t.len() {
                let step_size = if i > 0 { res.t[i] - res.t[i - 1] } else { 0.0 };
                writeln!(
                    &mut file,
                    "{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}",
                    res.t[i], res.y[i][0], res.y[i][1], res.y[i][2], step_size
                )
                .unwrap();
            }

            println!("\nSolution saved to robertson_solution.csv");
            println!("Note the characteristic behavior:");
            println!("- y₁ decreases from 1.0 to ~0.0 (most of it converts to y₃)");
            println!("- y₂ rapidly increases then slowly decreases (intermediate species)");
            println!("- y₃ increases from 0.0 toward 1.0 (end product)");
            println!("- The problem is very stiff in the initial transient phase");
            println!("- LSODA should switch to BDF method during the stiff phase");
        }
        Err(e) => {
            println!("Integration failed: {}", e);
        }
    }

    println!("\nAdvanced Method Switching Analysis");
    println!("---------------------------------");
    println!("To truly visualize LSODA's method switching behavior, the solver would");
    println!("need to be enhanced to record and output the following for each step:");
    println!("1. The active method (Adams or BDF) used at each successful step");
    println!("2. Stiffness detection metrics:");
    println!("   - Relative step size (h / abs(t))");
    println!("   - Acceptance ratio and recent rejection rate");
    println!("   - Step size relative to minimum (h / min_step)");
    println!("3. Method switching events with reason for switching");
    println!("4. Order adaptivity information (changing formula order within a method)");
    println!("");
    println!("This information would enable creation of detailed visualizations showing:");
    println!("- When and why the solver switches methods");
    println!("- How the solution, stiffness, and step size correlate");
    println!("- The efficiency gains of adaptive method switching");
}
