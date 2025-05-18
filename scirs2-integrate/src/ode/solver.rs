//! ODE solver interface
//!
//! This module provides the main interface for solving ODEs.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::methods::{
    bdf_method, dop853_method, enhanced_bdf_method, enhanced_lsoda_method, euler_method,
    lsoda_method, radau_method, radau_method_with_mass, rk23_method, rk45_method, rk4_method,
};
use crate::ode::types::{MassMatrix, MassMatrixType, ODEMethod, ODEOptions, ODEResult};
use crate::ode::utils::dense_output::DenseSolution;
use crate::ode::utils::events::{
    EventAction, EventHandler, ODEOptionsWithEvents, ODEResultWithEvents,
};
use crate::ode::utils::interpolation::ContinuousOutputMethod;
use crate::ode::utils::mass_matrix;
use ndarray::{Array1, ArrayView1};

/// Solve an initial value problem (IVP) for a system of ODEs.
///
/// This function solves a system of first-order ODEs of the form dy/dt = f(t, y),
/// given initial conditions y(t0) = y0.
///
/// # Arguments
///
/// * `f` - Function that computes the derivative dy/dt = f(t, y)
/// * `t_span` - The interval of integration [t0, tf]
/// * `y0` - Initial state
/// * `options` - Solver options (optional)
///
/// # Returns
///
/// Result containing the solution at different time points or an error
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::{array, ArrayView1};
/// use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
///
/// // Define ODE system: dy/dt = -y
/// let f = |_t: f64, y: ArrayView1<f64>| array![-y[0]];
///
/// // Solve the ODE
/// let result = solve_ivp(
///     f,
///     [0.0, 2.0],    // time span [t_start, t_end]
///     array![1.0],   // initial condition
///     Some(ODEOptions {
///         method: ODEMethod::RK45,
///         rtol: 1e-6,
///         atol: 1e-8,
///         ..Default::default()
///     }),
/// ).unwrap();
///
/// // Access the solution
/// let final_time = result.t.last().unwrap();
/// let final_value = result.y.last().unwrap()[0];
/// println!("y({}) = {}", final_time, final_value);
/// ```
///
/// # Mass Matrices
///
/// This function can also handle ODEs with mass matrices by specifying a mass matrix
/// in the options. For example:
///
/// ```rust,ignore
/// use ndarray::{array, Array2, ArrayView1};
/// use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions, MassMatrix};
///
/// // Define a constant mass matrix for the ODE system
/// let mut mass_matrix = Array2::<f64>::eye(2);
/// mass_matrix[[0, 1]] = 1.0; // Non-identity mass matrix
///
/// // Create the mass matrix specification
/// let mass = MassMatrix::constant(mass_matrix);
///
/// // Solve the ODE with mass matrix
/// let result = solve_ivp(
///     |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]],
///     [0.0, 10.0],    // time span [t_start, t_end]
///     array![1.0, 0.0],   // initial condition
///     Some(ODEOptions {
///         method: ODEMethod::Radau, // Use an implicit method for mass matrices
///         rtol: 1e-6,
///         atol: 1e-8,
///         mass_matrix: Some(mass),
///         ..Default::default()
///     }),
/// ).unwrap();
/// ```
pub fn solve_ivp<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    options: Option<ODEOptions<F>>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + std::iter::Sum,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    // Use default options if none provided
    let opts = options.unwrap_or_default();

    // Handle mass matrix if provided
    if let Some(mass) = &opts.mass_matrix {
        return solve_ivp_with_mass_internal(f, t_span, y0, mass.clone(), opts);
    }

    // Calculate default initial step size if not provided
    let [t_start, t_end] = t_span;
    let h0 = opts.h0.unwrap_or_else(|| {
        let span = t_end - t_start;
        span * F::from_f64(0.01).unwrap() // 1% of interval
    });

    // Dispatch to the appropriate solver based on the method
    match opts.method {
        ODEMethod::Euler => euler_method(f, t_span, y0, h0, opts),
        ODEMethod::RK4 => rk4_method(f, t_span, y0, h0, opts),
        ODEMethod::RK45 => rk45_method(f, t_span, y0, opts),
        ODEMethod::RK23 => rk23_method(f, t_span, y0, opts),
        ODEMethod::Bdf => bdf_method(f, t_span, y0, opts),
        ODEMethod::DOP853 => dop853_method(f, t_span, y0, opts),
        ODEMethod::Radau => radau_method(f, t_span, y0, opts),
        ODEMethod::LSODA => {
            // For LSODA, use a slightly modified options struct with better defaults
            // to help users get more stable results
            let lsoda_opts = ODEOptions {
                // If h0 not set, use a slightly larger default than other methods
                h0: opts.h0.or_else(|| {
                    let span = t_span[1] - t_span[0];
                    Some(span * F::from_f64(0.05).unwrap()) // 5% of interval instead of default 1%
                }),
                // If min_step not set, use reasonable minimum (keeping consistency)
                min_step: opts.min_step.or_else(|| {
                    let span = t_span[1] - t_span[0];
                    // 0.01% of span as default - match the implementation in lsoda_method
                    Some(span * F::from_f64(0.0001).unwrap())
                }),
                // Otherwise use the original options
                ..opts
            };

            lsoda_method(f, t_span, y0, lsoda_opts)
        }
        ODEMethod::EnhancedLSODA => {
            // For enhanced LSODA, use similar defaults to standard LSODA
            // but with even better initial parameters
            let enhanced_opts = ODEOptions {
                // If h0 not set, use a slightly larger default than other methods
                h0: opts.h0.or_else(|| {
                    let span = t_span[1] - t_span[0];
                    Some(span * F::from_f64(0.05).unwrap()) // 5% of interval
                }),
                // If min_step not set, use reasonable minimum
                min_step: opts.min_step.or_else(|| {
                    let span = t_span[1] - t_span[0];
                    Some(span * F::from_f64(0.0001).unwrap()) // 0.01% of span
                }),
                // Set more generous max_steps by default for the enhanced version
                max_steps: if opts.max_steps == 500 {
                    1000
                } else {
                    opts.max_steps
                },
                // Otherwise use the original options
                ..opts
            };

            enhanced_lsoda_method(f, t_span, y0, enhanced_opts)
        }
        ODEMethod::EnhancedBDF => {
            // For enhanced BDF, use optimized defaults to ensure better convergence
            let enhanced_bdf_opts = ODEOptions {
                // If h0 not set, use an adaptive initial step size (more conservative than default)
                h0: opts.h0.or_else(|| {
                    let span = t_span[1] - t_span[0];
                    // Start with 1% of the interval which is more conservative for stiff problems
                    Some(span * F::from_f64(0.01).unwrap())
                }),
                // More generous max_steps allowance for better convergence
                max_steps: if opts.max_steps == 500 {
                    1000
                } else {
                    opts.max_steps
                },
                // If no max_order specified, start with order 3 as a good balance
                max_order: opts.max_order.or(Some(3)),
                // The rest of the options are kept as provided
                ..opts
            };

            enhanced_bdf_method(f, t_span, y0, enhanced_bdf_opts)
        }
    }
}

/// Internal implementation of solve_ivp with mass matrix
///
/// This function handles the case where a mass matrix is provided.
/// It transforms the ODE system if possible, or selects an appropriate
/// solver that can handle the mass matrix.
fn solve_ivp_with_mass_internal<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    mass_matrix: MassMatrix<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + std::iter::Sum,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    // Check if mass matrix is compatible with the initial state
    mass_matrix::check_mass_compatibility(&mass_matrix, t_span[0], y0.view())?;

    // Handle different types of mass matrices
    match mass_matrix.matrix_type {
        // Identity mass matrix is just the standard case
        MassMatrixType::Identity => {
            // Recursive call without the mass matrix
            let mut new_opts = opts.clone();
            new_opts.mass_matrix = None;
            solve_ivp(f, t_span, y0, Some(new_opts))
        }

        // For constant or time-dependent mass matrices, we can transform to standard form
        // for non-stiff solvers
        MassMatrixType::Constant | MassMatrixType::TimeDependent => {
            // Check if we're using an implicit method that directly supports mass matrices
            match opts.method {
                // Use specialized Radau solver with mass matrix support
                ODEMethod::Radau => {
                    crate::ode::methods::radau_method_with_mass(f, t_span, y0, mass_matrix, opts)
                },

                // For other implicit methods, we don't have direct support yet
                ODEMethod::Bdf | ODEMethod::EnhancedBDF => {
                    Err(IntegrateError::NotImplementedError(
                        "Direct mass matrix support for BDF methods is not yet implemented. \
                        Please use Radau method or an explicit method with constant or time-dependent mass matrices.".to_string()
                    ))
                },

                // For explicit methods, transform to standard form: y' = M⁻¹·f(t,y)
                _ => {
                    // Clone f before moving into the transformation
                    let f_clone = f.clone();
                    let transformed_f = mass_matrix::transform_to_standard_form(f_clone, &mass_matrix);

                    // Create a wrapper function that handles the IntegrateResult return type
                    let wrapper_f = move |t: F, y: ArrayView1<F>| -> Array1<F> {
                        transformed_f(t, y).unwrap_or_else(|_| {
                            // If the transformation fails, return zeros as a fallback
                            // This isn't ideal, but allows the solver to continue
                            Array1::zeros(y.len())
                        })
                    };

                    // Create new options without the mass matrix
                    let mut new_opts = opts.clone();
                    new_opts.mass_matrix = None;

                    // Solve the transformed system by dispatching to the appropriate method
                    let [t_start, t_end] = t_span;
                    let h0 = new_opts.h0.unwrap_or_else(|| {
                        let span = t_end - t_start;
                        span * F::from_f64(0.01).unwrap() // 1% of interval
                    });

                    match new_opts.method {
                        ODEMethod::Euler => euler_method(wrapper_f, t_span, y0, h0, new_opts),
                        ODEMethod::RK4 => rk4_method(wrapper_f, t_span, y0, h0, new_opts),
                        ODEMethod::RK45 => rk45_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::RK23 => rk23_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::DOP853 => dop853_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::Radau => radau_method_with_mass(wrapper_f, t_span, y0, mass_matrix.clone(), new_opts),
                        ODEMethod::Bdf => bdf_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::EnhancedBDF => enhanced_bdf_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::LSODA => lsoda_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::EnhancedLSODA => enhanced_lsoda_method(wrapper_f, t_span, y0, new_opts),
                    }
                }
            }
        }

        // State-dependent mass matrices require special handling
        MassMatrixType::StateDependent => {
            // For state-dependent mass matrices, we need implicit methods
            // that can directly handle them
            match opts.method {
                // Radau method can handle state-dependent mass matrices
                ODEMethod::Radau => {
                    crate::ode::methods::radau_method_with_mass(f, t_span, y0, mass_matrix, opts)
                },

                // For other implicit methods, we don't have direct support yet
                ODEMethod::Bdf | ODEMethod::EnhancedBDF => {
                    Err(IntegrateError::NotImplementedError(
                        "Direct state-dependent mass matrix support for BDF methods is not yet implemented. \
                        Please use Radau method for state-dependent mass matrices.".to_string()
                    ))
                },

                // For explicit methods, we need to transform at each step
                // This is similar to the time-dependent case but requires evaluating
                // the mass matrix at each state point during integration
                _ => {
                    // Create a wrapped function that solves M(t,y)·y' = f(t,y) at each step
                    let wrapper_f = move |t: F, y: ArrayView1<F>| -> Array1<F> {
                        // Compute original RHS: f(t,y)
                        let rhs = f(t, y);

                        // Solve M(t,y)·y' = f(t,y) for y' on the fly
                        match mass_matrix::solve_mass_system(&mass_matrix, t, y, rhs.view()) {
                            Ok(result) => result,
                            Err(_) => {
                                // If solving fails, return zeros as fallback
                                // Not ideal, but allows the solver to continue
                                Array1::zeros(y.len())
                            }
                        }
                    };

                    // Create new options without the mass matrix
                    let mut new_opts = opts.clone();
                    new_opts.mass_matrix = None;

                    // Calculate default initial step size if not provided
                    let h0 = new_opts.h0.unwrap_or_else(|| {
                        let span = t_span[1] - t_span[0];
                        span * F::from_f64(0.01).unwrap() // 1% of interval
                    });

                    // Dispatch directly to the appropriate method
                    match new_opts.method {
                        ODEMethod::Euler => euler_method(wrapper_f, t_span, y0, h0, new_opts),
                        ODEMethod::RK4 => rk4_method(wrapper_f, t_span, y0, h0, new_opts),
                        ODEMethod::RK45 => rk45_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::RK23 => rk23_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::DOP853 => dop853_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::Radau => radau_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::LSODA => lsoda_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::EnhancedLSODA => enhanced_lsoda_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::EnhancedBDF => enhanced_bdf_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::Bdf => {
                            // BDF shouldn't reach here because we already handled it above
                            Err(IntegrateError::NotImplementedError(
                                "BDF method should not reach this case".to_string()
                            ))
                        }
                    }
                }
            }
        }
    }
}

/// Solve an initial value problem (IVP) for a system of ODEs with event detection.
///
/// This function solves a system of first-order ODEs of the form dy/dt = f(t, y)
/// with event detection capabilities. Events allow detecting when specific conditions
/// are met during integration (e.g., when a function crosses zero) and taking appropriate
/// actions (such as stopping the integration or recording the event).
///
/// # Arguments
///
/// * `f` - Function that computes the derivative dy/dt = f(t, y)
/// * `t_span` - The interval of integration [t0, tf]
/// * `y0` - Initial state
/// * `event_funcs` - Vector of functions that compute event conditions g(t, y)
///   An event occurs when g(t, y) crosses zero.
/// * `options` - Solver options with event specifications
///
/// # Returns
///
/// Result containing the solution at different time points, event information, and diagnostics
///
/// # Examples
///
/// ```rust,ignore
/// use ndarray::{array, ArrayView1};
/// use scirs2_integrate::ode::{
///     solve_ivp_with_events, ODEMethod, ODEOptions, EventSpec,
///     EventDirection, ODEOptionsWithEvents
/// };
///
/// // Define ODE system: simple harmonic oscillator
/// // dy1/dt = y2
/// // dy2/dt = -y1
/// let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];
///
/// // Define event function: detect when y1 = 0 (crossing point)
/// let event_funcs = vec![
///     |_t: f64, y: ArrayView1<f64>| y[0]  // Event when y1 crosses zero
/// ];
///
/// // Define event specification
/// let event_specs = vec![
///     EventSpec {
///         id: "zero_crossing".to_string(),
///         direction: EventDirection::Both,
///         action: EventAction::Continue,
///         threshold: 1e-6,
///         max_count: None,
///         precise_time: true,
///     }
/// ];
///
/// // Create ODE options with events
/// let options = ODEOptionsWithEvents::new(
///     ODEOptions {
///         method: ODEMethod::RK45,
///         rtol: 1e-6,
///         atol: 1e-8,
///         dense_output: true,  // Enable dense output for precise event detection
///         ..Default::default()
///     },
///     event_specs,
/// );
///
/// // Solve the ODE with event detection
/// let result = solve_ivp_with_events(
///     f,
///     [0.0, 10.0],    // time span [t_start, t_end]
///     array![1.0, 0.0],   // initial condition
///     event_funcs,
///     options,
/// ).unwrap();
///
/// // Access the detected events
/// for event in result.events.events {
///     println!("Event at t = {}: y = {:?}", event.time, event.state);
/// }
/// ```
pub fn solve_ivp_with_events<F, Func, EventFunc>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    event_funcs: Vec<EventFunc>,
    options: ODEOptionsWithEvents<F>,
) -> IntegrateResult<ODEResultWithEvents<F>>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone + 'static,
    EventFunc: Fn(F, ArrayView1<F>) -> F,
{
    // Extract base options and ensure dense output is enabled for event detection
    let mut base_options = options.base_options.clone();
    base_options.dense_output = true;

    // First, solve the IVP using the standard solver
    let base_result = solve_ivp(f.clone(), t_span, y0.clone(), Some(base_options))?;

    // Create the dense solution for event detection
    let dense_output = if !base_result.t.is_empty() {
        let dense = DenseSolution::new(
            base_result.t.clone(),
            base_result.y.clone(),
            None, // No derivatives provided
            Some(ContinuousOutputMethod::CubicHermite),
            Some(Box::new(f.clone())), // Provide function for derivative calculation
        );
        Some(dense)
    } else {
        None
    };

    // Create event handler
    let mut event_handler = EventHandler::new(options.event_specs.clone());

    // Initialize event handler
    event_handler.initialize(base_result.t[0], &base_result.y[0], &event_funcs)?;

    // Check for events at each time step
    let mut event_termination = false;

    for i in 1..base_result.t.len() {
        let t = base_result.t[i];
        let y = &base_result.y[i];

        // Check for events between the previous and current step
        let action = event_handler.check_events(t, y, dense_output.as_ref(), &event_funcs)?;

        if action == EventAction::Stop {
            event_termination = true;
            break;
        }
    }

    // If an event caused termination, truncate the results
    let final_result = if event_termination {
        // Get the event that caused termination
        let last_event = event_handler.record.events.last().ok_or_else(|| {
            IntegrateError::ValueError("No event found for termination".to_string())
        })?;

        // Find the insertion point for the event time
        let mut event_index = base_result.t.len();
        let mut exact_match = false;

        for (i, &t) in base_result.t.iter().enumerate() {
            if (t - last_event.time).abs() < F::from_f64(1e-10).unwrap() {
                event_index = i + 1;
                exact_match = true;
                break;
            } else if t > last_event.time {
                event_index = i;
                break;
            }
        }

        // Create truncated results
        let mut truncated_t = base_result.t[..event_index].to_vec();
        let mut truncated_y = base_result.y[..event_index].to_vec();

        // If we don't have an exact match, add the event time and state
        if !exact_match && event_index > 0 {
            // Only add if the event time is different from the last time point
            let last_t = truncated_t.last().copied().unwrap_or(F::zero());
            if (last_t - last_event.time).abs() > F::from_f64(1e-10).unwrap() {
                truncated_t.push(last_event.time);
                truncated_y.push(last_event.state.clone());
            }
        }

        // Create the truncated result
        ODEResult {
            t: truncated_t,
            y: truncated_y,
            message: base_result.message,
            success: base_result.success,
            n_steps: base_result.n_steps,
            n_eval: base_result.n_eval,
            n_accepted: base_result.n_accepted,
            n_rejected: base_result.n_rejected,
            n_lu: base_result.n_lu,
            n_jac: base_result.n_jac,
            method: base_result.method,
        }
    } else {
        base_result
    };

    // Create the result with events
    let result_with_events = ODEResultWithEvents::new(
        final_result,
        event_handler.record,
        dense_output,
        event_termination,
    );

    Ok(result_with_events)
}
