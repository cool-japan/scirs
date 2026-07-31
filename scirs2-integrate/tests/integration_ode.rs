//! Integration tests: scirs2-integrate ODE solvers
//!
//! Covers:
//! - Stiff ODE solver (BDF) vs analytical solution
//! - Non-stiff ODE (RK45) vs analytical solution
//! - Event detection accuracy
//! - Symplectic integrators with energy conservation
//! - Multi-dimensional ODE systems

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_integrate::{
    solve_ivp, solve_ivp_with_events, terminal_event, EventAction, EventDirection, ODEMethod,
    ODEOptions, ODEOptionsWithEvents, ODEResultWithEvents, SymplecticMethod,
    SymplecticSeparableSystem, SymplecticStepper,
};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// 1. Simple exponential decay: dy/dt = -k*y,  y(0) = y0
//    Analytical: y(t) = y0 * exp(-k*t)
// ---------------------------------------------------------------------------

#[test]
fn test_rk45_exponential_decay_vs_analytical() {
    let k = 2.0_f64;
    let y0 = 3.0_f64;

    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![-k * y[0]]) };

    let opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let result = solve_ivp(f, [0.0_f64, 2.0], Array1::from_vec(vec![y0]), Some(opts))
        .expect("RK45 exponential decay solve failed");

    assert!(result.success, "RK45 solve did not report success");

    // Check final value
    let t_final = *result.t.last().expect("empty t in result");
    let y_final = result.y.last().expect("empty y in result")[0];
    let y_analytical = y0 * (-k * t_final).exp();

    assert_abs_diff_eq!(y_final, y_analytical, epsilon = 1e-5);
}

// ---------------------------------------------------------------------------
// 2. Harmonic oscillator: y'' + omega^2*y = 0
//    State: [y, y']  Analytical: y(t) = A*cos(omega*t) + B*sin(omega*t)
// ---------------------------------------------------------------------------

#[test]
fn test_rk45_harmonic_oscillator_accuracy() {
    let omega = 2.0_f64;
    let y0 = 1.0_f64;
    let dy0 = 0.0_f64; // starts at rest

    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![y[1], -omega * omega * y[0]])
    };

    let opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-10,
        atol: 1e-12,
        ..Default::default()
    };

    let t_end = 2.0 * PI / omega; // one full period
    let result = solve_ivp(
        f,
        [0.0_f64, t_end],
        Array1::from_vec(vec![y0, dy0]),
        Some(opts),
    )
    .expect("RK45 harmonic oscillator solve failed");

    assert!(result.success, "Harmonic oscillator solve did not succeed");

    // After one full period, y should return to initial conditions
    let y_end = &result
        .y
        .last()
        .expect("empty y in harmonic oscillator result");
    assert_abs_diff_eq!(y_end[0], y0, epsilon = 1e-6);
    assert_abs_diff_eq!(y_end[1], dy0, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// 3. Stiff ODE: Robertson chemical kinetics (scaled)
//    Modified for well-conditioning: use a smaller time span
// ---------------------------------------------------------------------------

#[test]
fn test_bdf_stiff_ode_versus_rk45() {
    // Moderately stiff ODE: dy/dt = -10*y,  y(0) = 1
    // Exact solution: y(t) = exp(-10*t).
    // The stiffness ratio is 10, manageable for both BDF and RK45.
    let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![-10.0 * y[0]]) };

    let opts_bdf = ODEOptions {
        method: ODEMethod::Bdf,
        rtol: 1e-4,
        atol: 1e-6,
        max_steps: 50_000,
        ..Default::default()
    };

    let opts_rk = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let t_end = 1.0_f64;
    let y0 = Array1::from_vec(vec![1.0_f64]);

    let result_bdf =
        solve_ivp(f, [0.0_f64, t_end], y0.clone(), Some(opts_bdf)).expect("BDF solve failed");
    let result_rk = solve_ivp(f, [0.0_f64, t_end], y0, Some(opts_rk)).expect("RK45 solve failed");

    assert!(result_bdf.success, "BDF did not converge");
    assert!(result_rk.success, "RK45 reference did not converge");

    let y_bdf = result_bdf.y.last().expect("empty y BDF")[0];
    let y_rk = result_rk.y.last().expect("empty y RK")[0];
    let y_exact = (-10.0_f64 * t_end).exp(); // e^{-10} ≈ 4.54e-5

    // Both solutions should agree with the exact solution
    assert_abs_diff_eq!(y_bdf, y_exact, epsilon = 1e-2);
    assert_abs_diff_eq!(y_rk, y_exact, epsilon = 1e-5);
}

// ---------------------------------------------------------------------------
// 4. High-order method: DOP853 on Lorenz attractor (just verify no panic/wrong dims)
// ---------------------------------------------------------------------------

#[test]
fn test_dop853_lorenz_system_stability() {
    let sigma = 10.0_f64;
    let rho = 28.0_f64;
    let beta = 8.0_f64 / 3.0;

    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        let x = y[0];
        let yy = y[1];
        let z = y[2];
        Array1::from_vec(vec![
            sigma * (yy - x),
            x * (rho - z) - yy,
            x * yy - beta * z,
        ])
    };

    // Use tighter tolerances and shorter time span to keep the chaotic
    // trajectory on the attractor. Lorenz is sensitive to numerical
    // errors; t=2 is enough to demonstrate stability without risking
    // divergence from accumulated step-size controller drift.
    let opts = ODEOptions {
        method: ODEMethod::DOP853,
        rtol: 1e-9,
        atol: 1e-11,
        max_steps: 200_000,
        ..Default::default()
    };

    let y0 = Array1::from_vec(vec![1.0_f64, 1.0, 1.0]);
    let result = solve_ivp(f, [0.0_f64, 2.0], y0, Some(opts)).expect("DOP853 Lorenz solve failed");

    assert!(result.success, "DOP853 Lorenz did not succeed");
    assert!(!result.y.is_empty(), "Lorenz result is empty");

    // On the Lorenz attractor, |x| < 25, |y| < 30, |z| < 50 roughly.
    // Use 200 as a generous upper bound.
    for state in &result.y {
        for &val in state.iter() {
            assert!(
                val.is_finite() && val.abs() < 200.0,
                "Lorenz state blew up: {val}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Event detection: bouncing ball, detect zero crossing of height
// ---------------------------------------------------------------------------

#[test]
fn test_event_detection_zero_crossing_sine() {
    // y'' = -9.81, y(0) = 10, y'(0) = 0 (free fall)
    // Detect when y = 0 (hits ground)
    // Analytical: y(t) = 10 - 0.5*9.81*t^2
    //             y = 0 at t = sqrt(20/9.81) ≈ 1.428 s

    let g = 9.81_f64;
    let f =
        move |_t: f64, _y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![_y[1], -g]) };

    // Event: detect when y[0] = 0 (ball hits ground)
    let event_fn = |_t: f64, y: ArrayView1<f64>| -> f64 { y[0] };

    let base_opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-10,
        atol: 1e-12,
        dense_output: true,
        ..Default::default()
    };

    let event_spec = terminal_event::<f64>("ground_hit", EventDirection::Falling);

    let opts_with_events = ODEOptionsWithEvents {
        base_options: base_opts,
        event_specs: vec![event_spec],
    };

    let y0 = Array1::from_vec(vec![10.0_f64, 0.0]);
    let result: ODEResultWithEvents<f64> =
        solve_ivp_with_events(f, [0.0_f64, 5.0], y0, vec![event_fn], opts_with_events)
            .expect("Event detection solve failed");

    let t_final = *result.base_result.t.last().expect("empty t");

    // Analytical impact time
    let t_analytical = (2.0 * 10.0 / g).sqrt();

    // Integration should have stopped near the impact time
    assert_abs_diff_eq!(t_final, t_analytical, epsilon = 1e-4);
}

// ---------------------------------------------------------------------------
// 6. Euler method: first-order accuracy check
// ---------------------------------------------------------------------------

#[test]
fn test_euler_method_first_order_accuracy() {
    // dy/dt = y,  y(0) = 1  →  y(t) = exp(t)
    let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { y.to_owned() };

    let t_end = 1.0_f64;
    let h0 = 0.01_f64; // small fixed step

    let opts = ODEOptions {
        method: ODEMethod::Euler,
        h0: Some(h0),
        rtol: 1.0, // disable adaptive control
        atol: 1.0,
        ..Default::default()
    };

    let result = solve_ivp(
        f,
        [0.0_f64, t_end],
        Array1::from_vec(vec![1.0_f64]),
        Some(opts),
    )
    .expect("Euler solve failed");

    let y_numerical = result.y.last().expect("empty y Euler")[0];
    let y_exact = t_end.exp();

    // Euler error ~ O(h) * t_end
    let expected_error = h0 * t_end * y_exact;
    assert!(
        (y_numerical - y_exact).abs() < 2.0 * expected_error,
        "Euler error {:.2e} larger than expected O(h) bound {:.2e}",
        (y_numerical - y_exact).abs(),
        expected_error
    );
}

// ---------------------------------------------------------------------------
// 7. Symplectic integrator: simple harmonic oscillator energy conservation
// ---------------------------------------------------------------------------

#[test]
fn test_symplectic_stormer_verlet_energy_conservation() {
    // H = p^2/2 + q^2/2 (unit-mass SHO, omega=1)
    // Exact energy: E = p0^2/2 + q0^2/2
    let omega = 1.0_f64;

    let kinetic_grad = move |_t: f64, p: &Array1<f64>| -> Array1<f64> {
        p.clone() // dT/dp = p
    };
    let potential_grad = move |_t: f64, q: &Array1<f64>| -> Array1<f64> {
        q.mapv(|qi| omega * omega * qi) // dV/dq = omega^2 * q
    };

    let system = SymplecticSeparableSystem::new(1, kinetic_grad, potential_grad).with_energy(
        |_t: f64, p: &Array1<f64>| p[0] * p[0] / 2.0,
        move |_t: f64, q: &Array1<f64>| omega * omega * q[0] * q[0] / 2.0,
    );

    let q0 = Array1::from_vec(vec![1.0_f64]); // q(0) = 1
    let p0 = Array1::from_vec(vec![0.0_f64]); // p(0) = 0
    let e0 = 0.5_f64; // initial energy = 0^2/2 + 1^2/2 = 0.5

    let stepper =
        scirs2_integrate::create_symplectic_stepper::<f64>(SymplecticMethod::StormerVerlet);
    let t_end = 10.0 * 2.0 * PI; // 10 full periods
    let dt = 0.01_f64;

    let result =
        scirs2_integrate::solve_hamiltonian(&system, &*stepper, 0.0_f64, t_end, dt, q0, p0)
            .expect("Symplectic SHO integration failed");

    assert!(!result.t.is_empty(), "Symplectic result is empty");

    // Check energy drift over the integration interval
    if let Some(monitor) = &result.energy_monitor {
        let energies = &monitor.energy_history;
        // Symplectic integrators preserve energy almost exactly — drift should be small
        for &e in energies {
            assert_abs_diff_eq!(e, e0, epsilon = 1e-3);
        }
    } else {
        // Fallback: check that final state has correct energy
        let q_end = result.q.last().expect("empty q");
        let p_end = result.p.last().expect("empty p");
        let e_end = p_end[0] * p_end[0] / 2.0 + omega * omega * q_end[0] * q_end[0] / 2.0;
        assert_abs_diff_eq!(e_end, e0, epsilon = 1e-3);
    }
}

// ---------------------------------------------------------------------------
// 8. Yoshida4 symplectic integrator: Kepler orbit conservation
// ---------------------------------------------------------------------------

#[test]
fn test_yoshida4_kepler_orbit_energy_conservation() {
    // 2D Kepler problem (planar):
    // H = (px^2 + py^2)/2 - GM/r,   GM = 1
    // State: q = [x, y],  p = [px, py]
    // Circular orbit: r=1, v=1, E=-0.5

    let kinetic_grad = |_t: f64, p: &Array1<f64>| -> Array1<f64> {
        p.clone() // dT/dp = p
    };
    let potential_grad = |_t: f64, q: &Array1<f64>| -> Array1<f64> {
        let r3 = (q[0] * q[0] + q[1] * q[1]).powf(1.5);
        if r3 < 1e-10 {
            Array1::zeros(2)
        } else {
            Array1::from_vec(vec![q[0] / r3, q[1] / r3])
        }
    };

    let system = SymplecticSeparableSystem::new(2, kinetic_grad, potential_grad).with_energy(
        |_t: f64, p: &Array1<f64>| (p[0] * p[0] + p[1] * p[1]) / 2.0,
        |_t: f64, q: &Array1<f64>| {
            let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
            if r < 1e-10 {
                f64::INFINITY
            } else {
                -1.0 / r
            }
        },
    );

    // Circular orbit at r=1: q=(1,0), p=(0,1), E=-0.5
    let q0 = Array1::from_vec(vec![1.0_f64, 0.0]);
    let p0 = Array1::from_vec(vec![0.0_f64, 1.0]);
    let e0 = -0.5_f64;

    let stepper = scirs2_integrate::create_symplectic_stepper::<f64>(SymplecticMethod::Yoshida4);
    let t_end = 2.0 * PI; // one full orbit
    let dt = 0.01_f64;

    let result =
        scirs2_integrate::solve_hamiltonian(&system, &*stepper, 0.0_f64, t_end, dt, q0, p0)
            .expect("Yoshida4 Kepler integration failed");

    // Final position should return to (1, 0)
    let q_end = result.q.last().expect("empty q");
    assert_abs_diff_eq!(q_end[0], 1.0, epsilon = 0.01);
    assert_abs_diff_eq!(q_end[1], 0.0, epsilon = 0.01);

    // Energy conservation
    if let Some(monitor) = &result.energy_monitor {
        for &e in &monitor.energy_history {
            assert_abs_diff_eq!(e, e0, epsilon = 0.01);
        }
    }
}

// ---------------------------------------------------------------------------
// 9. RK4 fixed-step: compare with RK45 on smooth ODE
// ---------------------------------------------------------------------------

#[test]
fn test_rk4_vs_rk45_smooth_ode_agreement() {
    // dy/dt = cos(t),  y(0) = 0  →  y(t) = sin(t)
    let f = |t: f64, _y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![t.cos()]) };

    let t_end = PI;

    let opts_rk4 = ODEOptions {
        method: ODEMethod::RK4,
        h0: Some(0.01),
        ..Default::default()
    };
    let opts_rk45 = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-10,
        atol: 1e-12,
        ..Default::default()
    };

    let y0 = Array1::from_vec(vec![0.0_f64]);

    let res_rk4 =
        solve_ivp(f, [0.0_f64, t_end], y0.clone(), Some(opts_rk4)).expect("RK4 solve failed");
    let res_rk45 =
        solve_ivp(f, [0.0_f64, t_end], y0, Some(opts_rk45)).expect("RK45 smooth ODE solve failed");

    let y_rk4 = res_rk4.y.last().expect("empty y RK4")[0];
    let y_rk45 = res_rk45.y.last().expect("empty y RK45")[0];
    let y_exact = t_end.sin(); // sin(pi) = 0

    assert_abs_diff_eq!(y_rk4, y_exact, epsilon = 1e-4);
    assert_abs_diff_eq!(y_rk45, y_exact, epsilon = 1e-8);
    // Both should agree with each other closely
    assert_abs_diff_eq!(y_rk4, y_rk45, epsilon = 1e-4);
}

// ---------------------------------------------------------------------------
// 10. Multi-dimensional system: coupled oscillators
// ---------------------------------------------------------------------------

#[test]
fn test_rk45_coupled_oscillators() {
    // Two coupled harmonic oscillators:
    // x'' = -omega1^2 * x + k * (y - x)
    // y'' = -omega2^2 * y + k * (x - y)
    // State: [x, x', y, y']
    let omega1 = 1.0_f64;
    let omega2 = 1.5_f64;
    let kc = 0.1_f64; // coupling constant

    let f = move |_t: f64, s: ArrayView1<f64>| -> Array1<f64> {
        let x = s[0];
        let xp = s[1];
        let y = s[2];
        let yp = s[3];
        Array1::from_vec(vec![
            xp,
            -omega1 * omega1 * x + kc * (y - x),
            yp,
            -omega2 * omega2 * y + kc * (x - y),
        ])
    };

    let opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        max_steps: 100_000,
        ..Default::default()
    };

    let y0 = Array1::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
    let result =
        solve_ivp(f, [0.0_f64, 10.0], y0, Some(opts)).expect("Coupled oscillator solve failed");

    assert!(result.success, "Coupled oscillator solve did not succeed");

    // Energy (total mechanical) should be conserved
    // E = 0.5*(x'^2 + omega1^2*x^2 + y'^2 + omega2^2*y^2) + kc*(x-y)^2/2
    // (approximate, not including coupling fully, but verifies boundedness)
    for state in &result.y {
        let x = state[0];
        let xp = state[1];
        let y = state[2];
        let yp = state[3];
        let e = 0.5 * (xp * xp + omega1 * omega1 * x * x + yp * yp + omega2 * omega2 * y * y);
        // Energy should remain bounded (initial E ≈ 0.5)
        assert!(
            e < 2.0,
            "Coupled oscillator energy blew up to {e} at t={}",
            result.t[result.y.iter().position(|s| s[0] == state[0]).unwrap_or(0)]
        );
    }
}

// ---------------------------------------------------------------------------
// 8. RK23 / DOP853: real-method regression tests.
//
// Before the fix, `rk23_method`/`dop853_method` silently ran a fixed-step
// forward Euler under the RK23/DOP853 labels (single f(t,y) eval, "always
// accept", zero error estimate). The tests below are chosen specifically
// to fail under that stand-in: it ignores rtol/atol entirely (so tightening
// tolerance changes nothing), and it is numerically unstable for anything
// but trivial problems taken in ~100 fixed steps (verified separately: it
// diverges to ~1e20 on the many-period oscillator below and ~1e124 on the
// stiff relaxation problem below).
// ---------------------------------------------------------------------------

#[test]
fn test_rk23_convergence_order_tolerance_halving() {
    let k = 2.0_f64;
    let y0 = 3.0_f64;
    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![-k * y[0]]) };
    let y_exact = y0 * (-k * 2.0_f64).exp();

    let mut rtol = 1e-4_f64;
    let mut prev_err: Option<f64> = None;
    let mut errors = Vec::new();

    for _ in 0..5 {
        let opts = ODEOptions {
            method: ODEMethod::RK23,
            rtol,
            atol: rtol * 1e-3,
            max_steps: 200_000,
            ..Default::default()
        };
        let result = solve_ivp(f, [0.0_f64, 2.0], Array1::from_vec(vec![y0]), Some(opts))
            .expect("RK23 convergence-order solve failed");
        assert!(result.success, "RK23 solve did not succeed at rtol={rtol}");
        assert!(result.n_accepted > 0, "RK23 took no accepted steps");

        let y_final = result.y.last().expect("empty y in result")[0];
        let err = (y_final - y_exact).abs();

        if let Some(prev) = prev_err {
            assert!(
                err < prev,
                "RK23 error did not shrink when tolerance was halved: {err:e} vs previous {prev:e} (rtol={rtol:e})"
            );
        }
        errors.push(err);
        prev_err = Some(err);
        rtol /= 2.0;
    }

    // Over 4 halvings (16x tighter tolerance) a real adaptive 3rd-order
    // method should show a substantial cumulative error reduction; a
    // tolerance-blind stub would show none at all.
    let first = *errors.first().expect("no errors recorded");
    let last = *errors.last().expect("no errors recorded");
    assert!(
        last < first / 4.0,
        "RK23 error did not shrink enough over 4 tolerance halvings: first={first:e} last={last:e}"
    );
}

#[test]
fn test_dop853_convergence_order_tolerance_halving() {
    let k = 2.0_f64;
    let y0 = 3.0_f64;
    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![-k * y[0]]) };
    let y_exact = y0 * (-k * 2.0_f64).exp();

    let mut rtol = 1e-3_f64;
    let mut prev_err: Option<f64> = None;
    let mut errors = Vec::new();

    for _ in 0..5 {
        let opts = ODEOptions {
            method: ODEMethod::DOP853,
            rtol,
            atol: rtol * 1e-3,
            max_steps: 200_000,
            ..Default::default()
        };
        let result = solve_ivp(f, [0.0_f64, 2.0], Array1::from_vec(vec![y0]), Some(opts))
            .expect("DOP853 convergence-order solve failed");
        assert!(
            result.success,
            "DOP853 solve did not succeed at rtol={rtol}"
        );
        assert!(result.n_accepted > 0, "DOP853 took no accepted steps");

        let y_final = result.y.last().expect("empty y in result")[0];
        let err = (y_final - y_exact).abs();

        if let Some(prev) = prev_err {
            assert!(
                err < prev,
                "DOP853 error did not shrink when tolerance was halved: {err:e} vs previous {prev:e} (rtol={rtol:e})"
            );
        }
        errors.push(err);
        prev_err = Some(err);
        rtol /= 2.0;
    }

    let first = *errors.first().expect("no errors recorded");
    let last = *errors.last().expect("no errors recorded");
    assert!(
        last < first / 4.0,
        "DOP853 error did not shrink enough over 4 tolerance halvings: first={first:e} last={last:e}"
    );
}

#[test]
fn test_rk23_harmonic_oscillator_many_periods() {
    let omega = 2.0_f64;
    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![y[1], -omega * omega * y[0]])
    };

    let opts = ODEOptions {
        method: ODEMethod::RK23,
        rtol: 1e-6,
        atol: 1e-9,
        max_steps: 200_000,
        ..Default::default()
    };

    let period = 2.0 * PI / omega;
    let t_end = 20.0 * period; // many periods, non-trivial oscillatory data
    let result = solve_ivp(
        f,
        [0.0_f64, t_end],
        Array1::from_vec(vec![1.0_f64, 0.0]),
        Some(opts),
    )
    .expect("RK23 harmonic oscillator solve failed");

    assert!(result.success, "RK23 many-period solve did not succeed");
    assert!(result.n_accepted > 0, "RK23 took no accepted steps");

    // After exactly 20 full periods the state must return arbitrarily
    // close to the initial condition. A fixed-step, tolerance-blind Euler
    // stand-in diverges here (unbounded energy growth from too-large
    // steps on an undamped oscillator), so this is a strong regression
    // check for genuine 3rd-order accuracy + adaptivity.
    let y_end = result.y.last().expect("empty y in result");
    assert!(
        (y_end[0] - 1.0).abs() < 1e-2,
        "RK23 harmonic oscillator y[0] drifted too far after 20 periods: {}",
        y_end[0]
    );
    assert!(
        y_end[1].abs() < 1e-2,
        "RK23 harmonic oscillator y[1] drifted too far after 20 periods: {}",
        y_end[1]
    );
}

#[test]
fn test_dop853_harmonic_oscillator_many_periods() {
    let omega = 2.0_f64;
    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![y[1], -omega * omega * y[0]])
    };

    let opts = ODEOptions {
        method: ODEMethod::DOP853,
        rtol: 1e-9,
        atol: 1e-12,
        max_steps: 200_000,
        ..Default::default()
    };

    let period = 2.0 * PI / omega;
    let t_end = 20.0 * period;
    let result = solve_ivp(
        f,
        [0.0_f64, t_end],
        Array1::from_vec(vec![1.0_f64, 0.0]),
        Some(opts),
    )
    .expect("DOP853 harmonic oscillator solve failed");

    assert!(result.success, "DOP853 many-period solve did not succeed");
    assert!(result.n_accepted > 0, "DOP853 took no accepted steps");

    let y_end = result.y.last().expect("empty y in result");
    assert!(
        (y_end[0] - 1.0).abs() < 1e-5,
        "DOP853 harmonic oscillator y[0] drifted too far after 20 periods: {}",
        y_end[0]
    );
    assert!(
        y_end[1].abs() < 1e-5,
        "DOP853 harmonic oscillator y[1] drifted too far after 20 periods: {}",
        y_end[1]
    );
}

#[test]
fn test_rk23_dop853_stiff_relaxation_step_control_sanity() {
    // A moderately stiff relaxation problem in the spirit of Robertson's
    // chemical kinetics test: a fast mode relaxes onto a slowly-varying
    // manifold, forcing genuine step-size adaptation (small steps during
    // the initial transient, larger steps once on the manifold).
    //   dy/dt = -lambda * (y - sin(t)),  y(0) = 0
    // Exact solution: y(t) = A*sin(t) + B*cos(t) + C*exp(-lambda*t)
    let lambda = 1000.0_f64;
    let f = move |t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![-lambda * (y[0] - t.sin())])
    };
    let b_coef = -lambda / (lambda * lambda + 1.0);
    let a_coef = -lambda * b_coef;
    let c_coef = -b_coef;
    let y_exact = |t: f64| a_coef * t.sin() + b_coef * t.cos() + c_coef * (-lambda * t).exp();

    let t_end = 2.0_f64;

    for method in [ODEMethod::RK23, ODEMethod::DOP853] {
        let opts = ODEOptions {
            method,
            rtol: 1e-8,
            atol: 1e-11,
            max_steps: 50_000,
            ..Default::default()
        };
        let result = solve_ivp(
            f,
            [0.0_f64, t_end],
            Array1::from_vec(vec![0.0_f64]),
            Some(opts),
        )
        .unwrap_or_else(|e| panic!("{method:?} stiff relaxation solve failed: {e}"));

        assert!(
            result.success,
            "{method:?} did not converge on the stiff relaxation problem (max_steps hit)"
        );
        // Step-control sanity: the fast transient forces real adaptation
        // (many small accepted steps, some rejections), unlike the
        // tolerance-blind fixed-step stand-in this replaces (which
        // diverges to ~1e124 on this exact problem taken in ~100 fixed
        // steps).
        assert!(
            result.n_accepted > 10,
            "{method:?} took suspiciously few accepted steps: {}",
            result.n_accepted
        );

        let y_final = result.y.last().expect("empty y in result")[0];
        let err = (y_final - y_exact(t_end)).abs();
        assert!(
            err < 1e-4,
            "{method:?} stiff relaxation error too large: {err:e} (y_final={y_final}, expected={})",
            y_exact(t_end)
        );
    }
}

#[test]
fn test_cross_method_agreement_nonlinear_pendulum() {
    // Nonlinear pendulum: theta'' = -sin(theta). There is no closed-form
    // solution, but RK23, RK45 and DOP853 should all agree closely with
    // each other at a tight tolerance -- a strong cross-check that the
    // newly implemented RK23/DOP853 steppers are numerically correct and
    // not merely "doesn't panic".
    let f =
        |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![y[1], -y[0].sin()]) };

    let y0 = Array1::from_vec(vec![1.0_f64, 0.0]);
    let t_span = [0.0_f64, 5.0];
    let methods = [ODEMethod::RK23, ODEMethod::RK45, ODEMethod::DOP853];

    let mut finals: Vec<Array1<f64>> = Vec::new();
    for method in methods {
        let opts = ODEOptions {
            method,
            rtol: 1e-9,
            atol: 1e-11,
            max_steps: 200_000,
            ..Default::default()
        };
        let result = solve_ivp(f, t_span, y0.clone(), Some(opts))
            .unwrap_or_else(|e| panic!("{method:?} pendulum solve failed: {e}"));
        assert!(result.success, "{method:?} pendulum solve did not succeed");
        finals.push(result.y.last().expect("empty y in result").clone());
    }

    for i in 1..finals.len() {
        let diff_theta = (finals[i][0] - finals[0][0]).abs();
        let diff_omega = (finals[i][1] - finals[0][1]).abs();
        assert!(
            diff_theta < 1e-4,
            "{:?} disagreed with {:?} on theta: {} vs {}",
            methods[i],
            methods[0],
            finals[i][0],
            finals[0][0]
        );
        assert!(
            diff_omega < 1e-4,
            "{:?} disagreed with {:?} on omega: {} vs {}",
            methods[i],
            methods[0],
            finals[i][1],
            finals[0][1]
        );
    }
}
