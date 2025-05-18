//! State-Dependent Mass Matrix with Event Detection Example
//!
//! This example demonstrates the combined use of state-dependent mass matrices
//! and event detection in ODE solvers. We model a nonlinear mechanical system
//! where the mass matrix depends on the state of the system.
//!
//! The physical system modeled is a bead sliding on a rotating wire,
//! where the effective mass matrix depends on the bead's position.

use ndarray::{array, Array2, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{
    solve_ivp_with_events, terminal_event, EventAction, EventDirection, EventSpec, MassMatrix,
    ODEMethod, ODEOptions, ODEOptionsWithEvents,
};

fn main() -> IntegrateResult<()> {
    println!("=== Bead on Rotating Wire with Event Detection ===");

    // Physical parameters
    let g = 9.81; // Acceleration due to gravity, m/s^2
    let omega = 2.0; // Angular velocity of the wire, rad/s
    let m = 1.0; // Mass of the bead, kg

    // The system is described by the position of the bead along the wire, r
    // The wire rotates with angular velocity omega around the z-axis
    // The wire is curved according to the function z = h(r) = alpha*r^2
    let alpha = 0.1; // Coefficient for wire shape: z = alpha*r^2

    // The state vector is y = [r, v], where r is the radial position and v is the radial velocity

    // For a bead on a 3D curve with parametric equation r(s) = (r*cos(omega*t), r*sin(omega*t), h(r))
    // The effective mass matrix is:
    // M(r) = [  m    0  ]   where M_11 = m * (1 + (dh/dr)^2)
    //        [  0    m  ]
    // This accounts for the constraint forces keeping the bead on the wire

    // State-dependent mass matrix function
    let state_dependent_mass = move |_t: f64, y: ArrayView1<f64>| {
        let r = y[0];

        // Derivative of height function with respect to r: dh/dr = 2*alpha*r
        let dhdr = 2.0 * alpha * r;

        // Effective mass includes contribution from the wire constraint
        let effective_mass = m * (1.0 + dhdr * dhdr);

        // Create mass matrix
        let mut mass_matrix = Array2::<f64>::eye(2);
        mass_matrix[[0, 0]] = effective_mass;
        mass_matrix[[1, 1]] = m; // Standard mass for velocity component

        mass_matrix
    };

    // Create the mass matrix specification
    let mass = MassMatrix::state_dependent(state_dependent_mass);

    // ODE function for the forces on the bead
    // Force components:
    // 1. Gravity component along the wire
    // 2. Centrifugal force from rotation
    let f = move |_t: f64, y: ArrayView1<f64>| {
        let r = y[0];

        // Derivative of height function: dh/dr = 2*alpha*r
        let dhdr = 2.0 * alpha * r;

        // Gravity force along the wire: -m*g*sin(theta)
        // where sin(theta) = (dh/dr) / sqrt(1 + (dh/dr)^2)
        let gravity_component = -m * g * dhdr / (1.0 + dhdr * dhdr).sqrt();

        // Centrifugal force: m*omega^2*r
        let centrifugal_force = m * omega * omega * r;

        // Net force
        let net_force = gravity_component + centrifugal_force;

        // Return the right-hand side of the ODE
        array![
            y[1],      // r' = v
            net_force  // v' = F/m (mass matrix will divide by effective mass)
        ]
    };

    // Initial conditions: bead starts at r = 0.5 m with v = 0 m/s
    let r0 = 0.5; // Initial radial position
    let v0 = 0.0; // Initial radial velocity
    let y0 = array![r0, v0];

    // Time span for integration
    let t_span = [0.0, 20.0];

    // Event functions to detect:
    // 1. Velocity changes sign (indicating turning points)
    // 2. Bead passes through origin (r = 0)
    // 3. Bead reaches a maximum radial distance
    // 4. Bead reaches a specific height on the wire
    let max_radius = 2.0; // Maximum radius to detect
    let target_height = 0.4; // Target height for event detection

    // Define event functions as closures
    let max_radius_event = {
        let max_radius_copy = max_radius;
        Box::new(move |_t: f64, y: ArrayView1<f64>| max_radius_copy - y[0])
            as Box<dyn Fn(f64, ArrayView1<f64>) -> f64 + Send + Sync>
    };

    let height_event = {
        let alpha_copy = alpha;
        let target_height_copy = target_height;
        Box::new(move |_t: f64, y: ArrayView1<f64>| {
            let r = y[0];
            let height = alpha_copy * r * r;
            height - target_height_copy
        }) as Box<dyn Fn(f64, ArrayView1<f64>) -> f64 + Send + Sync>
    };

    let event_funcs: Vec<Box<dyn Fn(f64, ArrayView1<f64>) -> f64 + Send + Sync>> = vec![
        // Event 1: Velocity changes sign (turning points)
        Box::new(|_t: f64, y: ArrayView1<f64>| y[1]),
        // Event 2: Bead passes through origin
        Box::new(|_t: f64, y: ArrayView1<f64>| y[0]),
        // Event 3: Bead reaches maximum allowed radius (terminal event)
        max_radius_event,
        // Event 4: Bead reaches specific height
        height_event,
    ];

    // Event specifications
    let event_specs = vec![
        // Turning points
        EventSpec {
            id: "turning_point".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Origin crossing
        EventSpec {
            id: "origin_crossing".to_string(),
            direction: EventDirection::Both,
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: None,
            precise_time: true,
        },
        // Maximum radius (terminal event)
        terminal_event::<f64>("max_radius", EventDirection::Falling),
        // Target height reached
        EventSpec {
            id: "target_height".to_string(),
            direction: EventDirection::Rising,
            action: EventAction::Continue,
            threshold: 1e-8,
            max_count: Some(3), // Only detect first few occurrences
            precise_time: true,
        },
    ];

    // Create options with both mass matrix and event detection
    let options = ODEOptionsWithEvents::new(
        ODEOptions {
            method: ODEMethod::Radau, // Implicit method required for state-dependent mass matrix
            rtol: 1e-6,
            atol: 1e-8,
            dense_output: true, // Required for precise event detection
            mass_matrix: Some(mass),
            ..Default::default()
        },
        event_specs,
    );

    println!("Solving the bead on rotating wire system with state-dependent mass matrix...");

    // Solve the system
    let result = solve_ivp_with_events(f, t_span, y0, event_funcs, options)?;

    // Print basic solution info
    println!("\nIntegration results:");
    println!(
        "  Final time: {:.4} s",
        result.base_result.t.last().unwrap()
    );
    println!(
        "  Final position: r = {:.4} m",
        result.base_result.y.last().unwrap()[0]
    );
    println!(
        "  Final velocity: v = {:.4} m/s",
        result.base_result.y.last().unwrap()[1]
    );
    println!("  Steps taken: {}", result.base_result.n_steps);
    println!("  Function evaluations: {}", result.base_result.n_eval);
    println!("  Jacobian evaluations: {}", result.base_result.n_jac);
    println!("  LU decompositions: {}", result.base_result.n_lu);

    // Print event detection summary
    println!("\nEvent detection summary:");
    println!(
        "  Turning points detected: {}",
        result.events.get_count("turning_point")
    );
    println!(
        "  Origin crossings detected: {}",
        result.events.get_count("origin_crossing")
    );
    println!(
        "  Target height events: {}",
        result.events.get_count("target_height")
    );
    println!("  Terminated by max radius: {}", result.event_termination);

    // Analyze turning points
    let turning_points = result.events.get_events("turning_point");
    if !turning_points.is_empty() {
        println!("\nTurning point analysis:");
        println!("  Time (s)\tPosition (m)\tAcceleration (m/s²)");

        for (i, event) in turning_points.iter().take(5).enumerate() {
            let r = event.state[0];
            let _v = event.state[1]; // Should be close to zero

            // Calculate effective mass at this position
            let dhdr = 2.0 * alpha * r;
            let effective_mass = m * (1.0 + dhdr * dhdr);

            // Calculate forces
            let gravity_component = -m * g * dhdr / (1.0 + dhdr * dhdr).sqrt();
            let centrifugal_force = m * omega * omega * r;
            let net_force = gravity_component + centrifugal_force;

            // Calculate acceleration a = F/m_effective
            let acceleration = net_force / effective_mass;

            println!(
                "  {}: {:.4}\t{:.4}\t\t{:.4}",
                i + 1,
                event.time,
                r,
                acceleration
            );
        }

        if turning_points.len() > 5 {
            println!("  ... and {} more turning points", turning_points.len() - 5);
        }
    }

    // Analyze target height events
    let height_events = result.events.get_events("target_height");
    if !height_events.is_empty() {
        println!("\nTarget height events (z = {:.4} m):", target_height);
        println!("  Time (s)\tRadius (m)\tVelocity (m/s)");

        for (i, event) in height_events.iter().enumerate() {
            println!(
                "  {}: {:.4}\t{:.4}\t\t{:.4}",
                i + 1,
                event.time,
                event.state[0],
                event.state[1]
            );
        }
    }

    // Analyze terminal event
    if result.event_termination {
        if let Some(terminal_event) = result.events.get_events("max_radius").first() {
            let r = terminal_event.state[0];
            let v = terminal_event.state[1];
            let height = alpha * r * r;

            println!("\nTerminal event (maximum radius reached):");
            println!("  Time: {:.4} s", terminal_event.time);
            println!("  Final radius: {:.4} m", r);
            println!("  Final velocity: {:.4} m/s", v);
            println!("  Height at termination: {:.4} m", height);

            // Calculate energy at termination
            let kinetic_energy = 0.5 * m * v * v;
            let potential_energy = m * g * height;
            let rotational_energy = -0.5 * m * omega * omega * r * r; // Negative of centrifugal potential
            let total_energy = kinetic_energy + potential_energy + rotational_energy;

            println!("  Energy components at termination:");
            println!("    Kinetic energy: {:.4} J", kinetic_energy);
            println!(
                "    Gravitational potential energy: {:.4} J",
                potential_energy
            );
            println!("    Rotational energy: {:.4} J", rotational_energy);
            println!("    Total energy: {:.4} J", total_energy);
        }
    }

    println!("\nThis example demonstrates the successful integration of state-dependent");
    println!("mass matrices with event detection. The state-dependent mass matrix");
    println!("accounts for the geometric constraint of the bead moving along the");
    println!("rotating wire, while events detect important physical occurrences.");

    Ok(())
}
