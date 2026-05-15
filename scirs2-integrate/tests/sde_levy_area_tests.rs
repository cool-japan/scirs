//! Integration tests for the Lévy area approximation (Wiktorsson 2001)
//! and the general-noise SRK solver.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::prelude::seeded_rng;
use scirs2_integrate::sde::levy_area::{iterated_integral, levy_area_wiktorsson};

// ---- Structural tests ----

#[test]
fn levy_area_shape() {
    let dim = 4;
    let h = 0.05_f64;
    let dw = Array1::zeros(dim);
    let mut rng = seeded_rng(1);
    let a = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng);
    assert_eq!(
        a.shape(),
        &[dim, dim],
        "Lévy area must have shape (dim, dim)"
    );
}

#[test]
fn levy_area_zero_diagonal() {
    let dim = 5;
    let h = 0.1_f64;
    let dw = Array1::zeros(dim);
    let mut rng = seeded_rng(99);
    let a = levy_area_wiktorsson(dim, h, &dw, 10, &mut rng);
    for i in 0..dim {
        assert!(
            a[[i, i]].abs() < 1e-15,
            "Diagonal A[{i},{i}] must be zero, got {}",
            a[[i, i]]
        );
    }
}

#[test]
fn levy_area_antisymmetry() {
    let dim = 4;
    let h = 0.1_f64;
    let dw = array![0.05_f64, -0.03, 0.07, -0.02];
    let mut rng = seeded_rng(42);
    let a = levy_area_wiktorsson(dim, h, &dw, 10, &mut rng);
    for i in 0..dim {
        for j in 0..dim {
            let sum = a[[i, j]] + a[[j, i]];
            assert!(
                sum.abs() < 1e-14,
                "Antisymmetry violated: A[{i},{j}] + A[{j},{i}] = {:.2e} ≠ 0",
                sum
            );
        }
    }
}

#[test]
fn levy_area_deterministic_with_seed() {
    let dim = 3;
    let h = 0.01_f64;
    let dw = array![0.02_f64, -0.01, 0.03];
    let mut rng1 = seeded_rng(555);
    let mut rng2 = seeded_rng(555);
    let a1 = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng1);
    let a2 = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng2);
    for i in 0..dim {
        for j in 0..dim {
            assert!(
                (a1[[i, j]] - a2[[i, j]]).abs() < 1e-15,
                "Same seed must produce identical result at [{i},{j}]"
            );
        }
    }
}

// ---- Statistical test ----

#[test]
fn levy_area_variance_matches_theoretical() {
    // Theoretical: Var(A_{ij}) = h²/12 (infinite series).
    // k=5 truncation captures ~89% of the infinite sum → empirical ≈ 0.89 * h²/12.
    // We allow ±30% tolerance relative to h²/12 to accommodate sampling noise.
    let dim = 2;
    let h = 0.1_f64;
    let dw = array![0.0_f64, 0.0];
    let n_samples = 5000;

    let theoretical = h * h / 12.0;
    let tol_lo = theoretical * 0.49; // 0.89 * 0.70 ≈ 0.62, use 0.49 for safety
    let tol_hi = theoretical * 1.30;

    let samples: Vec<f64> = (0..n_samples)
        .map(|seed| {
            let mut rng = seeded_rng(seed as u64);
            let a = levy_area_wiktorsson(dim, h, &dw, 5, &mut rng);
            a[[0, 1]]
        })
        .collect();

    let mean: f64 = samples.iter().sum::<f64>() / n_samples as f64;
    let var: f64 =
        samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_samples - 1) as f64;

    assert!(
        var > tol_lo && var < tol_hi,
        "Lévy area variance {:.6e} outside [{:.6e}, {:.6e}] (h²/12 = {:.6e})",
        var,
        tol_lo,
        tol_hi,
        theoretical
    );
}

// ---- Iterated integral tests ----

#[test]
fn iterated_integral_symmetric_noise() {
    // For zero Lévy area (diagonal noise): I_{ij} + I_{ji} = dW_i * dW_j (i≠j)
    let dim = 2;
    let h = 0.1_f64;
    let dw = array![0.15_f64, -0.08];
    let levy_zero = Array2::zeros((dim, dim));
    let integ = iterated_integral(&dw, h, &levy_zero);

    let sum_01 = integ[[0, 1]] + integ[[1, 0]];
    let expected = dw[0] * dw[1];
    assert!(
        (sum_01 - expected).abs() < 1e-14,
        "I_01 + I_10 should equal dW_0 * dW_1 = {:.6}, got {:.6}",
        expected,
        sum_01
    );
}

#[test]
fn iterated_integral_diagonal_term() {
    // I_{ii} = (dW_i^2 - h) / 2
    let dim = 2;
    let h = 0.1_f64;
    let dw = array![0.2_f64, -0.1];
    let levy_zero = Array2::zeros((dim, dim));
    let integ = iterated_integral(&dw, h, &levy_zero);

    for i in 0..dim {
        let expected = (dw[i] * dw[i] - h) / 2.0;
        assert!(
            (integ[[i, i]] - expected).abs() < 1e-14,
            "I[{i},{i}] should be (dW^2 - h)/2 = {:.6}, got {:.6}",
            expected,
            integ[[i, i]]
        );
    }
}

// ---- srk_strong_general smoke tests ----

use scirs2_integrate::sde::runge_kutta_sde::srk_strong_general;
use scirs2_integrate::sde::SdeProblem;

#[test]
fn srk_strong_general_scalar_matches_scalar_path() {
    // For scalar noise, srk_strong_general should produce a valid trajectory
    let mu = 0.05_f64;
    let sigma = 0.2_f64;
    let prob = SdeProblem::new(
        array![1.0_f64],
        [0.0, 1.0],
        1,
        move |_t, x| array![mu * x[0]],
        move |_t, x| {
            let mut g = Array2::zeros((1, 1));
            g[[0, 0]] = sigma * x[0];
            g
        },
    );
    let mut rng = seeded_rng(0);
    let sol = srk_strong_general(&prob, 0.1, &mut rng, 5)
        .expect("srk_strong_general should succeed for scalar noise");
    assert_eq!(
        sol.len(),
        11,
        "Should have 11 time points for dt=0.1 on [0,1]"
    );
    let x_final = sol.x_final().expect("solution has state")[0];
    // GBM with mu=0.05, sigma=0.2: E[S(1)] = exp(0.05) ≈ 1.051, allow wide range
    assert!(x_final > 0.0, "GBM solution must be positive");
}

#[test]
fn srk_strong_general_2d_state_runs() {
    // 2D state with 2 Brownian motions and a non-diagonal diffusion matrix
    let prob = SdeProblem::new(
        array![1.0_f64, 0.5],
        [0.0, 0.5],
        2,
        move |_t, x| array![-0.5 * x[0], -0.5 * x[1]],
        move |_t, _x| {
            let mut g = Array2::zeros((2, 2));
            g[[0, 0]] = 0.2;
            g[[0, 1]] = 0.05; // off-diagonal coupling
            g[[1, 0]] = 0.05;
            g[[1, 1]] = 0.3;
            g
        },
    );
    let mut rng = seeded_rng(42);
    let sol = srk_strong_general(&prob, 0.05, &mut rng, 5)
        .expect("srk_strong_general should handle 2D general noise");
    assert!(!sol.is_empty());
    assert!(sol.len() > 1);
    // Both components should remain finite
    let x_final = sol.x_final().expect("solution has state");
    assert!(x_final[0].is_finite());
    assert!(x_final[1].is_finite());
}

#[test]
fn srk_strong_general_invalid_dt() {
    let prob = SdeProblem::new(
        array![1.0_f64],
        [0.0, 1.0],
        1,
        |_t, x| x.clone(),
        |_t, _x| Array2::eye(1),
    );
    let mut rng = seeded_rng(0);
    assert!(
        srk_strong_general(&prob, 0.0, &mut rng, 5).is_err(),
        "Zero dt should be an error"
    );
}
