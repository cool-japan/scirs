use scirs2_core::ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::dae::{solve_semi_explicit_dae, DAEOptions};
use scirs2_integrate::ode::ODEMethod;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Pendulum as a DAE System Example");
    println!("================================\n");

    // Physical parameters
    let g = 9.81; // Gravity constant (m/s²)
    let l = 1.0; // Pendulum length (m)
    let m = 1.0; // Mass (kg)

    // Initial conditions
    // x, y: Cartesian coordinates of the pendulum bob
    // vx, vy: Velocities in x and y directions
    // lambda: Lagrange multiplier (represents tension / constraint force)

    // Start at angle theta = π/6 (30 degrees)
    let theta0 = PI / 6.0;

    // Differential variables: [x, y, vx, vy]
    let x0 = array![
        l * theta0.sin(),  // x = l*sin(theta)
        -l * theta0.cos(), // y = -l*cos(theta)
        0.0,               // vx = 0 (starting from rest)
        0.0                // vy = 0 (starting from rest)
    ];

    // Algebraic variable: [lambda] (Lagrange multiplier)
    // For a pendulum at rest, lambda = m*g*cos(theta)
    let y0 = array![m * g * theta0.cos()];

    // Time span: 0 to 10 seconds
    let t_span = [0.0, 10.0];

    // Differential equations for the pendulum
    // x' = vx
    // y' = vy
    // vx' = -2*lambda*x/m
    // vy' = -2*lambda*y/m - g
    let f = |_t: f64, x: ArrayView1<f64>, y: ArrayView1<f64>| -> Array1<f64> {
        let lambda = y[0];

        array![
            x[2],                         // x' = vx
            x[3],                         // y' = vy
            -2.0 * lambda * x[0] / m,     // vx' = -2*lambda*x/m
            -2.0 * lambda * x[1] / m - g  // vy' = -2*lambda*y/m - g
        ]
    };

    // Constraint equation: x² + y² = l²
    // This enforces that the pendulum bob remains at a fixed distance l from the origin
    let g_constraint = |_t: f64, x: ArrayView1<f64>, _y: ArrayView1<f64>| -> Array1<f64> {
        array![x[0] * x[0] + x[1] * x[1] - l * l]
    };

    // DAE options: use Radau method (implicit) with a relatively tight tolerance
    let options = DAEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 1000,
        ..Default::default()
    };

    // Solve the DAE system
    println!("Solving pendulum DAE system...");
    let result = solve_semi_explicit_dae(f, g_constraint, t_span, x0, y0, Some(options))?;

    println!("Solution completed with {} steps.\n", result.n_steps);

    // Print the first few and last few points of the solution
    println!(
        "{:<10} {:<12} {:<12} {:<12} {:<12}",
        "Time", "X", "Y", "Angle", "Energy"
    );
    println!("{:-<60}", "");

    let num_print = 6.min(result.t.len());

    // Function to calculate angle from Cartesian coordinates
    let angle = |x: f64, y: f64| -> f64 {
        let theta = y.atan2(x);
        if theta > 0.0 {
            theta - PI
        } else {
            theta + PI
        }
    };

    // Function to calculate energy
    let energy = |x: ArrayView1<f64>, lambda: f64| -> f64 {
        // Kinetic energy: 0.5 * m * (vx² + vy²)
        let kinetic = 0.5 * m * (x[2] * x[2] + x[3] * x[3]);

        // Potential energy: m * g * (l + y)
        let potential = m * g * (l + x[1]);

        // Constraint energy (should be close to zero for a correct solution)
        let constraint = lambda * (x[0] * x[0] + x[1] * x[1] - l * l);

        kinetic + potential + constraint
    };

    // Print first few points
    for i in 0..num_print {
        let t = result.t[i];
        let x = &result.x[i];
        let lambda = result.y[i][0];
        let theta = angle(x[0], x[1]);
        let e = energy(x.view(), lambda);

        println!(
            "{:<10.3} {:<12.6} {:<12.6} {:<12.6} {:<12.6}",
            t, x[0], x[1], theta, e
        );
    }

    if result.t.len() > 2 * num_print {
        println!("{:^60}", "...");
    }

    // Print last few points
    if result.t.len() > num_print {
        for i in (result.t.len() - num_print)..result.t.len() {
            let t = result.t[i];
            let x = &result.x[i];
            let lambda = result.y[i][0];
            let theta = angle(x[0], x[1]);
            let e = energy(x.view(), lambda);

            println!(
                "{:<10.3} {:<12.6} {:<12.6} {:<12.6} {:<12.6}",
                t, x[0], x[1], theta, e
            );
        }
    }

    println!("\nConstraint Satisfaction Check (should be close to zero):");
    for i in [0, result.t.len() / 2, result.t.len() - 1] {
        let t = result.t[i];
        let x = &result.x[i];
        let constraint_value = x[0] * x[0] + x[1] * x[1] - l * l;
        println!(
            "t = {:<8.3}: |x² + y² - l²| = {:.3e}",
            t,
            constraint_value.abs()
        );
    }

    println!("\nPendulum Period Analysis:");
    analyze_period(&result.t, &result.x);

    Ok(())
}

/// Analyze the period of the pendulum by detecting zero crossings
#[allow(dead_code)]
fn analyze_period(t: &[f64], x: &[Array1<f64>]) {
    let mut crossings = Vec::new();

    // Detect x-axis crossings (when x[0] changes sign)
    for i in 1..t.len() {
        if x[i - 1][0] * x[i][0] <= 0.0 && x[i - 1][1] < 0.0 {
            // Linear interpolation to find the crossing time
            let t1 = t[i - 1];
            let t2 = t[i];
            let x1 = x[i - 1][0];
            let x2 = x[i][0];

            // t_cross = t1 + (t2 - t1) * |x1| / (|x1| + |x2|)
            let t_cross = t1 + (t2 - t1) * x1.abs() / (x1.abs() + x2.abs());
            crossings.push(t_cross);
        }
    }

    if crossings.len() >= 2 {
        println!("Detected {} complete oscillations", crossings.len() - 1);

        // Calculate periods
        let mut periods = Vec::new();
        for i in 1..crossings.len() {
            periods.push(crossings[i] - crossings[i - 1]);
        }

        // Print period statistics
        let avg_period = periods.iter().sum::<f64>() / periods.len() as f64;
        let theoretical_period = 2.0 * std::f64::consts::PI * (1.0 / 9.81_f64).sqrt();

        println!("Average period: {avg_period:.6} seconds");
        println!("Theoretical period for small oscillations: {theoretical_period:.6} seconds");
        println!(
            "Difference: {:.3}%",
            100.0 * (avg_period - theoretical_period).abs() / theoretical_period
        );

        if periods.len() >= 2 {
            let first_period = periods[0];
            let last_period = periods[periods.len() - 1];
            println!("First period: {first_period:.6} seconds");
            println!("Last period: {last_period:.6} seconds");
            println!(
                "Period change: {:.6}%",
                100.0 * (last_period - first_period) / first_period
            );
        }
    } else {
        println!("Not enough oscillations detected to calculate period.");
    }
}
