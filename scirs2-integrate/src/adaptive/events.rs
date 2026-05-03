//! Event detection for ODE integration.
//!
//! This module provides infrastructure for detecting and locating **zero
//! crossings** (events) of user-defined scalar functions of the ODE state
//! during integration.  Events are used to stop integration early, record
//! exact crossing times, or switch between different ODE systems.
//!
//! # Event detection algorithm
//!
//! 1. After each accepted ODE step the event function `g(t, y)` is evaluated
//!    at the new time point.
//! 2. If the sign of `g` changes compared with the previous step the solver
//!    knows a zero crossing occurred somewhere in `(t_prev, t_curr)`.
//! 3. The **Illinois algorithm** (a bracket-based secant method with
//!    superlinear convergence) is used to find the exact crossing time to
//!    within a small tolerance.
//! 4. If the event is marked `terminal = true` integration stops at that
//!    point; otherwise the crossing is recorded and integration continues.
//!
//! # Usage
//!
//! Combine `EventSpec` and `EventSet` with `dopri5_with_events` to obtain
//! both the solution trajectory and a list of detected crossings.

use super::embedded_rk::{dopri5, OdeResult};
use crate::error::{IntegrateError, IntegrateResult};

// ─── Public types ────────────────────────────────────────────────────────────

/// Specifies the direction of a zero crossing to be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventDirection {
    /// Detect only rising crossings (g goes from negative to positive).
    Rising,
    /// Detect only falling crossings (g goes from positive to negative).
    Falling,
    /// Detect crossings in either direction.
    Both,
}

/// A single event specification.
///
/// An event is triggered when the scalar function `func(t, y)` passes
/// through zero in the given `direction`.
pub struct EventSpec {
    /// The event function.  An event triggers when this crosses zero.
    pub func: Box<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>,
    /// Which sign change directions to detect.
    pub direction: EventDirection,
    /// Whether to halt integration when this event fires.
    pub terminal: bool,
}

impl std::fmt::Debug for EventSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventSpec")
            .field("direction", &self.direction)
            .field("terminal", &self.terminal)
            .finish()
    }
}

/// Result of a detected event.
#[derive(Debug, Clone)]
pub struct EventResult {
    /// The time at which the event function crossed zero.
    pub t_event: f64,
    /// The interpolated state at `t_event`.
    pub y_event: Vec<f64>,
    /// Index into the `EventSpec` slice that fired.
    pub event_idx: usize,
}

/// A collection of events to be monitored during integration.
pub struct EventSet {
    /// The list of events, indexed starting from zero.
    pub specs: Vec<EventSpec>,
}

impl EventSet {
    /// Create a new `EventSet` from a vector of `EventSpec`.
    pub fn new(specs: Vec<EventSpec>) -> Self {
        Self { specs }
    }
}

// ─── Illinois bracket root finder ────────────────────────────────────────────

/// Maximum number of Illinois iterations for root polishing.
const MAX_ILLINOIS: usize = 50;
/// Tolerance for the Illinois root-finding iteration (in time).
const ILLINOIS_TOL: f64 = 1e-12;

/// Check whether the sign change between `g_prev` and `g_curr` matches the
/// requested `direction`.
fn direction_matches(g_prev: f64, g_curr: f64, direction: EventDirection) -> bool {
    match direction {
        EventDirection::Both => g_prev * g_curr < 0.0,
        EventDirection::Rising => g_prev < 0.0 && g_curr > 0.0,
        EventDirection::Falling => g_prev > 0.0 && g_curr < 0.0,
    }
}

/// Internal Illinois iteration.
///
/// Performs the Illinois secant method on a user-supplied evaluation
/// function `eval(t) -> g`.  The bracket `[ta, tb]` must satisfy
/// `ga * gb < 0`.  Returns the located crossing time and state.
fn illinois_bracket<E>(mut ta: f64, mut tb: f64, mut ga: f64, mut gb: f64, eval: E) -> f64
where
    E: Fn(f64) -> f64,
{
    // Illinois state: which side was most recently *not* updated
    // (we halve that side's function value to improve convergence).
    let mut side = 0i32; // 0 = neutral, +1 = tb stale, -1 = ta stale

    for _ in 0..MAX_ILLINOIS {
        // Secant step
        let dg = gb - ga;
        let t_new = if dg.abs() < 1e-300 {
            (ta + tb) / 2.0
        } else {
            ta - ga * (tb - ta) / dg
        };
        let t_new = t_new.clamp(ta.min(tb), ta.max(tb));

        if (tb - ta).abs() < ILLINOIS_TOL {
            return t_new;
        }

        let g_new = eval(t_new);

        if g_new.abs() < ILLINOIS_TOL {
            return t_new;
        }

        if ga * g_new < 0.0 {
            // Root in [ta, t_new]; tb moves to t_new
            if side == 1 {
                // tb was already stale; halve ga (Illinois modification)
                ga /= 2.0;
            }
            tb = t_new;
            gb = g_new;
            side = 1; // tb just moved → ta is now the stale side
        } else {
            // Root in [t_new, tb]; ta moves to t_new
            if side == -1 {
                // ta was already stale; halve gb
                gb /= 2.0;
            }
            ta = t_new;
            ga = g_new;
            side = -1; // ta just moved → tb is now the stale side
        }
    }

    (ta + tb) / 2.0
}

/// Locate the zero crossing of `event.func` in the interval `[t_prev, t_curr]`
/// using the **Illinois algorithm**.
///
/// The ODE solution at intermediate times is approximated by linearly
/// interpolating the state vectors `y_prev` and `y_curr`.  For higher
/// accuracy use `find_event_root_dense` with a dense-output interpolant.
///
/// Returns `Some(EventResult)` if a crossing is found, `None` if there is no
/// bracketed zero (e.g. the direction filter rejects the crossing).
///
/// # Parameters
///
/// * `g_prev`    – Event function value at `t_prev`.
/// * `g_curr`    – Event function value at `t_curr`.
/// * `t_prev`    – Left bracket time.
/// * `t_curr`    – Right bracket time.
/// * `y_prev`    – State vector at `t_prev`.
/// * `y_curr`    – State vector at `t_curr`.
/// * `event_idx` – Index of the triggering event in the surrounding slice.
/// * `event`     – The `EventSpec` whose zero we are locating.
pub fn find_event_root(
    g_prev: f64,
    g_curr: f64,
    t_prev: f64,
    t_curr: f64,
    y_prev: &[f64],
    y_curr: &[f64],
    event_idx: usize,
    event: &EventSpec,
) -> Option<EventResult> {
    if !direction_matches(g_prev, g_curr, event.direction) {
        return None;
    }

    let n = y_prev.len();
    let dt = t_curr - t_prev;

    // Linear interpolation helper
    let interp = |t: f64| -> Vec<f64> {
        let alpha = if dt.abs() < 1e-300 {
            0.5
        } else {
            (t - t_prev) / dt
        };
        (0..n)
            .map(|i| y_prev[i] + alpha * (y_curr[i] - y_prev[i]))
            .collect()
    };

    let eval = |t: f64| -> f64 {
        let y = interp(t);
        (event.func)(t, &y)
    };

    let t_event = illinois_bracket(t_prev, t_curr, g_prev, g_curr, eval);
    let y_event = interp(t_event);

    Some(EventResult {
        t_event,
        y_event,
        event_idx,
    })
}

/// Locate a zero crossing using a callable ODE solution interpolant instead
/// of linear interpolation between steps.
///
/// `interp(t)` must return the (approximate) ODE state at any time in
/// `[t_prev, t_curr]`.  This is typically the dense-output polynomial from
/// the underlying solver step.
///
/// Returns `Some(EventResult)` if a crossing is found, `None` otherwise.
pub fn find_event_root_dense<I>(
    g_prev: f64,
    g_curr: f64,
    t_prev: f64,
    t_curr: f64,
    interp: I,
    event_idx: usize,
    event: &EventSpec,
) -> Option<EventResult>
where
    I: Fn(f64) -> Vec<f64>,
{
    if !direction_matches(g_prev, g_curr, event.direction) {
        return None;
    }

    let eval = |t: f64| -> f64 {
        let y = interp(t);
        (event.func)(t, &y)
    };

    let t_event = illinois_bracket(t_prev, t_curr, g_prev, g_curr, eval);
    let y_event = interp(t_event);

    Some(EventResult {
        t_event,
        y_event,
        event_idx,
    })
}

// ─── Complete result type ────────────────────────────────────────────────────

/// Combined result from ODE integration with event detection.
#[derive(Debug)]
pub struct OdeEventResult {
    /// The standard ODE trajectory.
    pub ode: OdeResult,
    /// All detected events, in chronological order.
    pub events: Vec<EventResult>,
    /// Whether integration terminated due to a terminal event.
    pub terminated: bool,
}

// ─── High-level solver with events ──────────────────────────────────────────

/// Solve an ODE with DOPRI5 while monitoring a set of events.
///
/// Integration proceeds step by step.  After each accepted step the event
/// functions are evaluated and any zero crossings located with
/// [`find_event_root`].  If a terminal event fires integration stops at the
/// event time; otherwise it continues to `t_end`.
///
/// # Arguments
///
/// * `f`       – Right-hand side `dy/dt = f(t, y)`.
/// * `t0`      – Initial time.
/// * `y0`      – Initial state vector.
/// * `t_end`   – Final time (may not be reached if a terminal event fires).
/// * `rtol`    – Relative tolerance for DOPRI5.
/// * `atol`    – Absolute tolerance for DOPRI5.
/// * `events`  – The set of events to monitor.
///
/// # Errors
///
/// Propagates any errors from the underlying DOPRI5 integrator.
pub fn dopri5_with_events<F>(
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    rtol: f64,
    atol: f64,
    events: EventSet,
) -> IntegrateResult<OdeEventResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64> + Clone,
{
    if y0.is_empty() {
        return Err(IntegrateError::ValueError(
            "y0 must be non-empty".to_string(),
        ));
    }
    if t_end <= t0 {
        return Err(IntegrateError::ValueError("t_end must be > t0".to_string()));
    }

    let mut all_t: Vec<f64> = vec![t0];
    let mut all_y: Vec<Vec<f64>> = vec![y0.to_vec()];
    let mut all_events: Vec<EventResult> = Vec::new();
    let mut n_steps_total: usize = 0;
    let mut n_rejected_total: usize = 0;
    let mut n_evals_total: usize = 0;
    let mut terminated = false;

    // Evaluate all event functions at t0
    let mut g_prev: Vec<f64> = events.specs.iter().map(|s| (s.func)(t0, y0)).collect();

    // Step through using DOPRI5 in segments.  We run one "short" integration
    // at a time to keep the segment granularity coarse; then we scan for
    // events within each returned step.
    //
    // For simplicity we drive DOPRI5 with a per-segment call and inspect the
    // resulting trajectory pairwise.
    let n_seg_max = 10_000_usize;
    let seg_hint = ((t_end - t0) / 0.1).ceil() as usize; // ~100 points per segment
    let n_seg = seg_hint.min(n_seg_max).max(1);

    let dt_seg = (t_end - t0) / n_seg as f64;
    let mut t_start = t0;
    let mut y_start = y0.to_vec();

    for _seg in 0..n_seg {
        if terminated || t_start >= t_end - 1e-14 * (t_end - t0) {
            break;
        }

        let t_seg_end = (t_start + dt_seg).min(t_end);

        let seg_result = dopri5(f.clone(), t_start, &y_start, t_seg_end, rtol, atol)?;

        n_steps_total += seg_result.n_steps;
        n_rejected_total += seg_result.n_rejected;
        n_evals_total += seg_result.n_evals;

        // Scan each consecutive pair in the segment for events
        let seg_len = seg_result.t.len();
        let mut early_stop_idx: Option<usize> = None;

        'step_scan: for step_i in 1..seg_len {
            let t_p = seg_result.t[step_i - 1];
            let t_c = seg_result.t[step_i];
            let y_p = &seg_result.y[step_i - 1];
            let y_c = &seg_result.y[step_i];

            for (ev_idx, spec) in events.specs.iter().enumerate() {
                let g_c = (spec.func)(t_c, y_c);
                let g_p = g_prev[ev_idx];

                if direction_matches(g_p, g_c, spec.direction) {
                    if let Some(ev) = find_event_root(g_p, g_c, t_p, t_c, y_p, y_c, ev_idx, spec) {
                        all_events.push(ev);
                        if spec.terminal {
                            early_stop_idx = Some(step_i);
                            terminated = true;
                            break 'step_scan;
                        }
                    }
                }

                g_prev[ev_idx] = g_c;
            }
        }

        // Append trajectory points
        let append_up_to = early_stop_idx.unwrap_or(seg_len);
        for step_i in 1..append_up_to {
            all_t.push(seg_result.t[step_i]);
            all_y.push(seg_result.y[step_i].clone());
        }

        // If a terminal event fired add the event location as the final point
        if terminated {
            if let Some(last_ev) = all_events.last() {
                all_t.push(last_ev.t_event);
                all_y.push(last_ev.y_event.clone());
            }
            break;
        }

        // Advance to next segment
        if let (Some(t_last), Some(y_last)) = (seg_result.t.last(), seg_result.y.last()) {
            t_start = *t_last;
            y_start = y_last.clone();
        } else {
            break;
        }
    }

    let n_out = all_t.len();
    Ok(OdeEventResult {
        ode: OdeResult {
            t: all_t,
            y: all_y,
            n_steps: n_steps_total,
            n_rejected: n_rejected_total,
            n_evals: n_evals_total + n_out, // approximate
        },
        events: all_events,
        terminated,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Illinois root finder ─────────────────────────────────────────────────

    #[test]
    fn illinois_finds_exact_midpoint() {
        // g(t) = t - 0.5, crosses zero at t = 0.5
        let spec = EventSpec {
            func: Box::new(|t: f64, _y: &[f64]| t - 0.5),
            direction: EventDirection::Rising,
            terminal: false,
        };
        let y_prev = vec![1.0_f64];
        let y_curr = vec![1.0_f64];
        let result = find_event_root(-0.5, 0.5, 0.0, 1.0, &y_prev, &y_curr, 0, &spec)
            .expect("should detect rising crossing");
        assert!(
            (result.t_event - 0.5).abs() < 1e-10,
            "t_event={} expected 0.5",
            result.t_event
        );
        assert_eq!(result.event_idx, 0);
    }

    #[test]
    fn illinois_direction_filter_falling() {
        // g goes from +1 to -1 → falling crossing
        let spec_rising = EventSpec {
            func: Box::new(|t: f64, _y: &[f64]| 1.0 - 2.0 * t), // crosses 0 at 0.5
            direction: EventDirection::Rising,                  // should NOT match
            terminal: false,
        };
        let y = vec![0.0_f64];
        let res = find_event_root(1.0, -1.0, 0.0, 1.0, &y, &y, 0, &spec_rising);
        assert!(
            res.is_none(),
            "Rising filter should reject falling crossing"
        );

        let spec_falling = EventSpec {
            func: Box::new(|t: f64, _y: &[f64]| 1.0 - 2.0 * t),
            direction: EventDirection::Falling,
            terminal: false,
        };
        let res2 = find_event_root(1.0, -1.0, 0.0, 1.0, &y, &y, 0, &spec_falling)
            .expect("Falling filter should accept falling crossing");
        assert!((res2.t_event - 0.5).abs() < 1e-8);
    }

    #[test]
    fn illinois_both_directions() {
        let spec = EventSpec {
            func: Box::new(|t: f64, _y: &[f64]| (t - 0.3).sin()),
            direction: EventDirection::Both,
            terminal: false,
        };
        let y = vec![0.0_f64];
        // Any sign change should be caught
        let res = find_event_root(-0.5, 0.5, 0.0, 0.6, &y, &y, 2, &spec);
        let ev = res.expect("should find crossing");
        assert_eq!(ev.event_idx, 2);
    }

    // ── dopri5_with_events ───────────────────────────────────────────────────

    #[test]
    fn events_detect_zero_crossing_sin() {
        // dy/dt = cos(t), y(0) = 0 → y(t) = sin(t)
        // Event: y crosses zero again at t = π
        let f = |t: f64, _y: &[f64]| vec![t.cos()];
        let event_spec = EventSpec {
            func: Box::new(|_t: f64, y: &[f64]| y[0]),
            direction: EventDirection::Falling, // sin goes positive → negative at π
            terminal: false,
        };
        let events = EventSet::new(vec![event_spec]);
        let result = dopri5_with_events(f, 0.0, &[0.0], 4.0, 1e-8, 1e-10, events)
            .expect("integration failed");

        // Should detect a crossing near t = π ≈ 3.14159
        let pi = std::f64::consts::PI;
        let found = result.events.iter().any(|e| (e.t_event - pi).abs() < 0.05);
        assert!(
            found,
            "Expected crossing near t=π, got events: {:?}",
            result.events.iter().map(|e| e.t_event).collect::<Vec<_>>()
        );
        assert!(!result.terminated);
    }

    #[test]
    fn events_terminal_stops_integration() {
        // dy/dt = -y, y(0) = 1  →  y(t) = exp(-t)
        // Terminal event: y < 0.5 (triggers when exp(-t) = 0.5, i.e. t = ln 2 ≈ 0.693)
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let threshold = EventSpec {
            func: Box::new(|_t: f64, y: &[f64]| y[0] - 0.5), // crosses 0 from above
            direction: EventDirection::Falling,
            terminal: true,
        };
        let events = EventSet::new(vec![threshold]);
        let result = dopri5_with_events(f, 0.0, &[1.0], 5.0, 1e-8, 1e-10, events)
            .expect("integration failed");

        assert!(result.terminated, "Expected terminal stop");
        // Integration should stop well before t = 5
        let t_final = result.ode.t.last().copied().unwrap_or(0.0);
        let ln2 = 2.0_f64.ln();
        assert!(
            (t_final - ln2).abs() < 0.1,
            "Expected termination near t=ln2≈{ln2:.4}, got t={t_final:.4}"
        );
        assert!(!result.events.is_empty());
    }

    #[test]
    fn events_multiple_crossings() {
        // dy/dt = 1, y(0) = 0  →  y(t) = t
        // Detect crossings of thresholds at t = 1, 2, 3
        let f = |_t: f64, _y: &[f64]| vec![1.0];
        let mut specs = Vec::new();
        for thresh in [1.0_f64, 2.0, 3.0] {
            specs.push(EventSpec {
                func: Box::new(move |_t: f64, y: &[f64]| y[0] - thresh),
                direction: EventDirection::Rising,
                terminal: false,
            });
        }
        let events = EventSet::new(specs);
        let result = dopri5_with_events(f, 0.0, &[0.0], 4.0, 1e-8, 1e-10, events)
            .expect("integration failed");

        // Should detect 3 crossings
        assert!(
            result.events.len() >= 3,
            "expected ≥3 events, got {}",
            result.events.len()
        );
    }

    #[test]
    fn events_validates_empty_y0() {
        let f = |_t: f64, _y: &[f64]| vec![];
        let events = EventSet::new(vec![]);
        assert!(dopri5_with_events(f, 0.0, &[], 1.0, 1e-6, 1e-8, events).is_err());
    }

    #[test]
    fn events_validates_t_end_leq_t0() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let events = EventSet::new(vec![]);
        assert!(dopri5_with_events(f, 1.0, &[1.0], 0.5, 1e-6, 1e-8, events).is_err());
    }
}
