//! Integration tests for the streaming SIR particle filter.
//!
//! Tests cover:
//! - Linear-Gaussian tracking accuracy vs. Kalman filter ground truth
//! - ESS-triggered resampling under degenerate likelihoods
//! - Bounded memory / step-count correctness over 1000 steps
//! - Log marginal likelihood sign and finiteness
//! - Non-linear (bearings-only) tracking

use scirs2_core::ndarray::{arr1, Array1};
use scirs2_integrate::sde::streaming_particle_filter::{SimpleRng, StreamingParticleFilterBuilder};

// ---------------------------------------------------------------------------
// 1. Linear-Gaussian tracking — compare with Kalman filter mean
// ---------------------------------------------------------------------------

/// Kalman filter for the 1D linear-Gaussian model:
///   x_{t+1} = x_t + w_t,   w_t ~ N(0, q)
///   y_t     = x_t + v_t,   v_t ~ N(0, r)
fn kalman_1d_update(x_pred: f64, p_pred: f64, obs: f64, q: f64, r: f64) -> (f64, f64) {
    // Predict
    let x_p = x_pred;
    let p_p = p_pred + q;
    // Update
    let k = p_p / (p_p + r);
    let x_upd = x_p + k * (obs - x_p);
    let p_upd = (1.0 - k) * p_p;
    (x_upd, p_upd)
}

#[test]
fn streaming_filter_tracks_linear_gaussian_matches_kalman() {
    let n = 1000;
    let process_noise = 0.1_f64;
    let obs_noise_std = 0.5_f64;
    let obs_noise_var = obs_noise_std * obs_noise_std;

    let mut filter = StreamingParticleFilterBuilder::new(n)
        .transition(move |x, seed| {
            let mut rng = SimpleRng::new(seed);
            arr1(&[x[0] + rng.normal() * process_noise])
        })
        .log_likelihood(move |x, obs: &Array1<f64>| {
            let diff = x[0] - obs[0];
            -0.5 * diff * diff / obs_noise_var
        })
        .ess_threshold(0.5)
        .seed(42)
        .initial_state(arr1(&[0.0]))
        .initial_spread(1.0)
        .build()
        .expect("build should succeed");

    // Ground-truth 1D random walk (deterministic trajectory for reproducibility)
    let true_states: Vec<f64> = (0..20).map(|t| (t as f64 * 0.05_f64).sin() * 3.0).collect();

    // Kalman filter ground truth
    let mut kf_x = 0.0_f64;
    let mut kf_p = 1.0_f64;
    let q = process_noise * process_noise;
    let r = obs_noise_var;

    let mut rmse_sum = 0.0_f64;
    for &true_x in &true_states {
        let obs_val = true_x; // noiseless observation for simplicity
        let obs = arr1(&[obs_val]);

        let est = filter.step(&obs);
        let (kf_mean, p_next) = kalman_1d_update(kf_x, kf_p, obs_val, q, r);
        kf_x = kf_mean;
        kf_p = p_next;

        let err = est.mean[0] - kf_x;
        rmse_sum += err * err;
    }
    let rmse = (rmse_sum / true_states.len() as f64).sqrt();
    assert!(
        rmse < 0.5,
        "PF-vs-Kalman RMSE {} should be < 0.5 for linear-Gaussian model",
        rmse
    );
}

// ---------------------------------------------------------------------------
// 2. Degenerate likelihood forces resampling
// ---------------------------------------------------------------------------

#[test]
fn streaming_filter_adaptive_resample_triggers_on_ess() {
    // Very peaked likelihood far from initial particles → immediate weight collapse
    let mut filter = StreamingParticleFilterBuilder::new(200)
        .transition(|x, _seed| x.to_owned())
        .log_likelihood(|x, obs: &Array1<f64>| {
            let diff = x[0] - obs[0];
            -500.0 * diff * diff // extremely peaked
        })
        .ess_threshold(0.5)
        .seed(123)
        .initial_state(arr1(&[0.0]))
        .initial_spread(0.1)
        .build()
        .expect("build should succeed");

    // Observe far from the initial cloud
    let obs = arr1(&[100.0_f64]);
    let est = filter.step(&obs);

    // ESS should be very low OR resampling should have triggered
    let resampled = filter.resample_count() > 0;
    let ess_low = est.effective_sample_size < 200.0 * 0.9;
    assert!(
        resampled || ess_low,
        "Expected resampling or low ESS; resample_count={}, ESS={}",
        filter.resample_count(),
        est.effective_sample_size
    );
}

// ---------------------------------------------------------------------------
// 3. Bounded memory — 1000 steps without OOM; step_count is correct
// ---------------------------------------------------------------------------

#[test]
fn streaming_filter_bounded_memory_1000_steps() {
    let mut filter = StreamingParticleFilterBuilder::new(100)
        .transition(|x, _seed| x.to_owned())
        .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
        .ess_threshold(0.5)
        .seed(0)
        .build()
        .expect("build should succeed");

    let obs = arr1(&[0.0_f64]);
    for _ in 0..1000 {
        filter.step(&obs);
    }
    assert_eq!(
        filter.step_count(),
        1000,
        "step_count should equal number of step() calls"
    );
}

// ---------------------------------------------------------------------------
// 4. Log marginal likelihood is finite and accumulates correctly
// ---------------------------------------------------------------------------

#[test]
fn streaming_filter_log_evidence_is_finite() {
    let mut filter = StreamingParticleFilterBuilder::new(200)
        .transition(|x, _| x.to_owned())
        .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
        .seed(7)
        .build()
        .expect("build");

    let obs = arr1(&[0.0_f64]);
    for _ in 0..10 {
        let est = filter.step(&obs);
        assert!(
            est.log_marginal_likelihood.is_finite(),
            "log_marginal_likelihood must be finite"
        );
    }
    // With flat ll=0, increments are 0 (logsumexp(w+0) - logsumexp(w) = 0),
    // so evidence stays near 0.
    assert!(
        filter.log_marginal_likelihood().abs() < 1e-6,
        "evidence with flat ll should be near 0, got {}",
        filter.log_marginal_likelihood()
    );
}

// ---------------------------------------------------------------------------
// 5. Non-linear bearings-only tracking (sanity check on non-Gaussian model)
// ---------------------------------------------------------------------------

#[test]
fn streaming_filter_bearings_only_nonlinear_tracking() {
    // Target moves horizontally at y=10; radar at origin.
    // State = [x_pos], observation = atan2(10, x_pos) (angle in radians).
    // Process: x_{t+1} = x_t + 1 + noise
    // Observation: y_t = atan2(10.0, x_t) + noise

    let obs_noise_std = 0.05_f64;
    let n_particles = 500;

    let mut filter = StreamingParticleFilterBuilder::new(n_particles)
        .transition(|x, seed| {
            let mut rng = SimpleRng::new(seed);
            arr1(&[x[0] + 1.0 + rng.normal() * 0.2])
        })
        .log_likelihood(move |x, obs: &Array1<f64>| {
            let predicted_angle = (10.0_f64).atan2(x[0]);
            let diff = predicted_angle - obs[0];
            -0.5 * diff * diff / (obs_noise_std * obs_noise_std)
        })
        .ess_threshold(0.5)
        .seed(99)
        .initial_state(arr1(&[-5.0]))
        .initial_spread(2.0)
        .build()
        .expect("build");

    // Simulate target at x=-5,-4,...,9 (15 steps)
    let mut rmse = 0.0_f64;
    let steps = 15_usize;
    for t in 0..steps {
        let true_x = -5.0 + t as f64;
        let obs_angle = (10.0_f64).atan2(true_x);
        let obs = arr1(&[obs_angle]);
        let est = filter.step(&obs);
        let err = est.mean[0] - true_x;
        rmse += err * err;
    }
    rmse = (rmse / steps as f64).sqrt();
    assert!(
        rmse < 3.0,
        "Bearings-only RMSE {} should be < 3.0 (non-linear model, generous threshold)",
        rmse
    );
}

// ---------------------------------------------------------------------------
// 6. ESS after uniform-weight step equals N
// ---------------------------------------------------------------------------

#[test]
fn streaming_filter_ess_matches_n_particles_with_flat_likelihood() {
    let n = 300;
    let mut filter = StreamingParticleFilterBuilder::new(n)
        .transition(|x, _| x.to_owned())
        .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
        .seed(55)
        .build()
        .expect("build");

    let obs = arr1(&[0.0_f64]);
    let est = filter.step(&obs);
    // With flat likelihood (all equal), ESS should be exactly N
    let ess = est.effective_sample_size;
    assert!(
        (ess - n as f64).abs() < 1.0,
        "ESS should be ~N with flat likelihood, got {ess}"
    );
}

// ---------------------------------------------------------------------------
// 7. Resample count accessor matches expectations
// ---------------------------------------------------------------------------

#[test]
fn streaming_filter_resample_count_accessor() {
    let mut filter = StreamingParticleFilterBuilder::new(50)
        .transition(|x, _| x.to_owned())
        .log_likelihood(|_, _: &Array1<f64>| 0.0_f64)
        .seed(11)
        .build()
        .expect("build");

    assert_eq!(filter.resample_count(), 0);
    let obs = arr1(&[0.0_f64]);
    filter.step(&obs);
    // With flat likelihood, no resampling should occur
    assert_eq!(filter.resample_count(), 0);
}
