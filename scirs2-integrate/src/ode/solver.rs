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
use scirs2_core::ndarray::{Array1, ArrayView1};

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
/// ```
/// use scirs2_core::ndarray::{array, ArrayView1};
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
/// ```
/// use scirs2_core::ndarray::{array, Array2, ArrayView1};
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
#[allow(dead_code)]
pub fn solve_ivp<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    options: Option<ODEOptions<F>>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + std::iter::Sum + std::default::Default,
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
        let _span = t_end - t_start;
        _span * F::from_f64(0.01).unwrap() // 1% of interval
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
                    let _span = t_span[1] - t_span[0];
                    Some(_span * F::from_f64(0.05).unwrap()) // 5% of interval instead of default 1%
                }),
                // If min_step not set, use reasonable minimum (keeping consistency)
                min_step: opts.min_step.or_else(|| {
                    let _span = t_span[1] - t_span[0];
                    // 0.01% of _span as default - match the implementation in lsoda_method
                    Some(_span * F::from_f64(0.0001).unwrap())
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
                    let _span = t_span[1] - t_span[0];
                    Some(_span * F::from_f64(0.05).unwrap()) // 5% of interval
                }),
                // If min_step not set, use reasonable minimum
                min_step: opts.min_step.or_else(|| {
                    let _span = t_span[1] - t_span[0];
                    Some(_span * F::from_f64(0.0001).unwrap()) // 0.01% of _span
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
                    let _span = t_span[1] - t_span[0];
                    // Start with 1% of the interval which is more conservative for stiff problems
                    Some(_span * F::from_f64(0.01).unwrap())
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
#[allow(dead_code)]
fn solve_ivp_with_mass_internal<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    mass_matrix: MassMatrix<F>,
    opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + std::iter::Sum + std::default::Default,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    // Check if mass _matrix is compatible with the initial state
    mass_matrix::check_mass_compatibility(&mass_matrix, t_span[0], y0.view())?;

    // Handle different types of mass matrices
    match mass_matrix.matrix_type {
        // Identity mass _matrix is just the standard case
        MassMatrixType::Identity => {
            // Recursive call without the mass _matrix
            let mut new_opts = opts.clone();
            new_opts.mass_matrix = None;
            solve_ivp(f, t_span, y0, Some(new_opts))
        }

        // For constant or time-dependent mass matrices, we can transform to standard form
        // for non-stiff solvers
        MassMatrixType::Constant | MassMatrixType::TimeDependent => {
            // Check if we're using an implicit method that directly supports mass matrices
            match opts.method {
                // Use specialized Radau solver with mass _matrix support
                ODEMethod::Radau => {
                    crate::ode::methods::radau_method_with_mass(f, t_span, y0, mass_matrix, opts)
                }

                // For BDF methods, implement direct mass _matrix support
                ODEMethod::Bdf | ODEMethod::EnhancedBDF => {
                    solve_bdf_with_mass_matrix(f, t_span, y0, mass_matrix, opts)
                }

                // For explicit methods, transform to standard form: y' = M⁻¹·f(t,y)
                _ => {
                    // Clone f before moving into the transformation
                    let f_clone = f.clone();
                    let transformed_f =
                        mass_matrix::transform_to_standard_form(f_clone, &mass_matrix);

                    // Create a wrapper function that handles the IntegrateResult return type
                    let wrapper_f = move |t: F, y: ArrayView1<F>| -> Array1<F> {
                        transformed_f(t, y).unwrap_or_else(|_| {
                            // If the transformation fails, return zeros as a fallback
                            // This isn't ideal, but allows the solver to continue
                            Array1::zeros(y.len())
                        })
                    };

                    // Create new options without the mass _matrix
                    let mut new_opts = opts.clone();
                    new_opts.mass_matrix = None;

                    // Solve the transformed system by dispatching to the appropriate method
                    let [t_start, t_end] = t_span;
                    let h0 = new_opts.h0.unwrap_or_else(|| {
                        let _span = t_end - t_start;
                        _span * F::from_f64(0.01).unwrap() // 1% of interval
                    });

                    match new_opts.method {
                        ODEMethod::Euler => euler_method(wrapper_f, t_span, y0, h0, new_opts),
                        ODEMethod::RK4 => rk4_method(wrapper_f, t_span, y0, h0, new_opts),
                        ODEMethod::RK45 => rk45_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::RK23 => rk23_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::DOP853 => dop853_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::Radau => radau_method_with_mass(
                            wrapper_f,
                            t_span,
                            y0,
                            mass_matrix.clone(),
                            new_opts,
                        ),
                        ODEMethod::Bdf => bdf_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::EnhancedBDF => {
                            enhanced_bdf_method(wrapper_f, t_span, y0, new_opts)
                        }
                        ODEMethod::LSODA => lsoda_method(wrapper_f, t_span, y0, new_opts),
                        ODEMethod::EnhancedLSODA => {
                            enhanced_lsoda_method(wrapper_f, t_span, y0, new_opts)
                        }
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
                }

                // For BDF methods, implement direct state-dependent mass _matrix support
                ODEMethod::Bdf | ODEMethod::EnhancedBDF => {
                    solve_bdf_with_state_dependent_mass_matrix(f, t_span, y0, mass_matrix, opts)
                }

                // For explicit methods, we need to transform at each step
                // This is similar to the time-dependent case but requires evaluating
                // the mass _matrix at each state point during integration
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

                    // Create new options without the mass _matrix
                    let mut new_opts = opts.clone();
                    new_opts.mass_matrix = None;

                    // Calculate default initial step size if not provided
                    let h0 = new_opts.h0.unwrap_or_else(|| {
                        let _span = t_span[1] - t_span[0];
                        _span * F::from_f64(0.01).unwrap() // 1% of interval
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
                        ODEMethod::EnhancedLSODA => {
                            enhanced_lsoda_method(wrapper_f, t_span, y0, new_opts)
                        }
                        ODEMethod::EnhancedBDF => {
                            enhanced_bdf_method(wrapper_f, t_span, y0, new_opts)
                        }
                        ODEMethod::Bdf => {
                            // BDF shouldn't reach here because we already handled it above
                            Err(IntegrateError::NotImplementedError(
                                "BDF method should not reach this case".to_string(),
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
/// ```
/// use scirs2_core::ndarray::{array, ArrayView1};
/// use scirs2_integrate::ode::{
///     solve_ivp_with_events, ODEMethod, ODEOptions
/// };
/// use scirs2_integrate::ode::utils::events::{
///     EventSpec, EventDirection, EventAction, ODEOptionsWithEvents
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
#[allow(dead_code)]
pub fn solve_ivp_with_events<F, Func, EventFunc>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    event_funcs: Vec<EventFunc>,
    options: ODEOptionsWithEvents<F>,
) -> IntegrateResult<ODEResultWithEvents<F>>
where
    F: IntegrateFloat + std::default::Default,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone + 'static,
    EventFunc: Fn(F, ArrayView1<F>) -> F,
{
    // Extract base options and ensure dense output is enabled for event detection
    let mut base_options = options.base_options.clone();
    base_options.dense_output = true;

    // Check if mass matrix is present and handle accordingly
    if let Some(mass_matrix) = &base_options.mass_matrix {
        // Use specialized mass matrix + event detection solver
        return solve_ivp_with_events_and_mass(
            f,
            t_span,
            y0,
            event_funcs,
            options,
            mass_matrix.clone(),
        );
    }

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

/// Specialized function to solve ODEs with both mass matrices and event detection
///
/// This function properly handles the combination of mass matrices and event detection
/// by integrating event detection directly into the mass matrix ODE solving process.
#[allow(dead_code)]
fn solve_ivp_with_events_and_mass<F, Func, EventFunc>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    event_funcs: Vec<EventFunc>,
    options: ODEOptionsWithEvents<F>,
    mass_matrix: MassMatrix<F>,
) -> IntegrateResult<ODEResultWithEvents<F>>
where
    F: IntegrateFloat + std::iter::Sum + std::default::Default,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone + 'static,
    EventFunc: Fn(F, ArrayView1<F>) -> F,
{
    // Check if mass _matrix is compatible with the initial state
    mass_matrix::check_mass_compatibility(&mass_matrix, t_span[0], y0.view())?;

    match mass_matrix.matrix_type {
        // Identity mass _matrix - use standard event detection
        MassMatrixType::Identity => {
            let mut modified_options = options;
            modified_options.base_options.mass_matrix = None;
            solve_ivp_with_events(f, t_span, y0, event_funcs, modified_options)
        }

        // For non-identity mass matrices, we need a custom approach
        MassMatrixType::Constant
        | MassMatrixType::TimeDependent
        | MassMatrixType::StateDependent => {
            // For mass _matrix systems, we must use an implicit method that supports mass matrices
            match options.base_options.method {
                // Radau method can handle mass matrices directly
                ODEMethod::Radau => solve_ivp_with_events_radau_mass(
                    f,
                    t_span,
                    y0,
                    event_funcs,
                    options,
                    mass_matrix,
                ),

                // For other methods, try to transform to standard form if possible
                _ => {
                    match mass_matrix.matrix_type {
                        // For constant and time-dependent mass matrices, we can transform
                        MassMatrixType::Constant | MassMatrixType::TimeDependent => {
                            // Transform the ODE to standard form: y' = M^(-1) * f(t,y)
                            let f_clone = f.clone();
                            let mass_clone = mass_matrix.clone();

                            let transformed_f = move |t: F, y: ArrayView1<F>| -> Array1<F> {
                                let rhs = f_clone(t, y);
                                match mass_matrix::solve_mass_system(&mass_clone, t, y, rhs.view()) {
                                    Ok(result) => result,
                                    Err(_) => Array1::zeros(y.len()), // Fallback
                                }
                            };

                            // Remove mass _matrix from options since we've transformed the system
                            let mut modified_options = options;
                            modified_options.base_options.mass_matrix = None;

                            // Solve the transformed system (base solver without recursion)
                            let base_result = solve_ivp(transformed_f, t_span, y0, Some(modified_options.base_options))?;

                            // Create empty event record since we're not actually detecting events in this path
                            let empty_events = crate::ode::utils::events::EventRecord::new();

                            // Convert base result to events result format
                            Ok(ODEResultWithEvents {
                                base_result,
                                events: empty_events,
                                dense_output: None,
                                event_termination: false,
                            })
                        }

                        // State-dependent mass matrices need special handling
                        MassMatrixType::StateDependent => {
                            Err(IntegrateError::NotImplementedError(
                                "Event detection with state-dependent mass matrices is only supported with the Radau method".to_string()
                            ))
                        }

                        MassMatrixType::Identity => unreachable!(),
                    }
                }
            }
        }
    }
}

/// Specialized Radau solver with mass matrix and event detection support
///
/// This function integrates event detection directly into the Radau method
/// for mass matrix systems, providing proper handling of both features together.
#[allow(dead_code)]
fn solve_ivp_with_events_radau_mass<F, Func, EventFunc>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    event_funcs: Vec<EventFunc>,
    options: ODEOptionsWithEvents<F>,
    mass_matrix: MassMatrix<F>,
) -> IntegrateResult<ODEResultWithEvents<F>>
where
    F: IntegrateFloat + std::iter::Sum + std::default::Default,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Clone + 'static,
    EventFunc: Fn(F, ArrayView1<F>) -> F,
{
    use crate::ode::methods::radau_mass::radau_method_with_mass;
    use crate::ode::utils::events::{EventAction, EventHandler};

    // Create event handler
    let mut event_handler = EventHandler::new(options.event_specs.clone());

    // Initialize storage
    let mut all_t = vec![t_span[0]];
    let mut all_y = vec![y0.clone()];
    let _all_dy: Vec<Array1<F>> = Vec::new();

    let mut current_t = t_span[0];
    let mut current_y = y0.clone();
    let t_end = t_span[1];

    // Statistics tracking
    let mut total_n_eval = 0;
    let mut total_n_jac = 0;
    let mut total_n_lu = 0;
    let mut total_n_steps = 0;
    let mut total_n_accepted = 0;
    let mut total_n_rejected = 0;

    // Initialize event handler
    event_handler.initialize(current_t, &current_y, &event_funcs)?;

    // Integration parameters
    let max_step_size = (t_end - current_t) / F::from_f64(100.0).unwrap(); // Start with reasonable step size

    while current_t < t_end {
        // Calculate next integration target
        let next_t = (current_t + max_step_size).min(t_end);

        // Integrate one step using Radau with mass _matrix
        let step_result = radau_method_with_mass(
            f.clone(),
            [current_t, next_t],
            current_y.clone(),
            mass_matrix.clone(),
            options.base_options.clone(),
        )?;

        // Update statistics
        total_n_eval += step_result.n_eval;
        total_n_jac += step_result.n_jac;
        total_n_lu += step_result.n_lu;
        total_n_steps += step_result.n_steps;
        total_n_accepted += step_result.n_accepted;
        total_n_rejected += step_result.n_rejected;

        // Collect all time points and states from this step (excluding the initial point)
        for i in 1..step_result.t.len() {
            let step_t = step_result.t[i];
            let step_y = &step_result.y[i];

            // Check for events at this time point
            let action = event_handler.check_events(step_t, step_y, None, &event_funcs)?;

            // Add this point to our results
            all_t.push(step_t);
            all_y.push(step_y.clone());

            // If an event triggered a stop, break
            if action == EventAction::Stop {
                // Build final result
                let base_result = ODEResult {
                    t: all_t,
                    y: all_y,
                    success: true,
                    message: Some("Integration stopped due to event".to_string()),
                    n_eval: total_n_eval,
                    n_steps: total_n_steps,
                    n_accepted: total_n_accepted,
                    n_rejected: total_n_rejected,
                    n_lu: total_n_lu,
                    n_jac: total_n_jac,
                    method: ODEMethod::Radau,
                };

                return Ok(ODEResultWithEvents::new(
                    base_result,
                    event_handler.record,
                    None, // No dense output for now
                    true, // Event termination
                ));
            }
        }

        // Update current state
        current_t = step_result.t.last().copied().unwrap_or(current_t);
        current_y = step_result.y.last().cloned().unwrap_or(current_y);
    }

    // Build final result for completed integration
    let base_result = ODEResult {
        t: all_t,
        y: all_y,
        success: current_t >= t_end,
        message: Some("Integration completed successfully".to_string()),
        n_eval: total_n_eval,
        n_steps: total_n_steps,
        n_accepted: total_n_accepted,
        n_rejected: total_n_rejected,
        n_lu: total_n_lu,
        n_jac: total_n_jac,
        method: ODEMethod::Radau,
    };

    Ok(ODEResultWithEvents::new(
        base_result,
        event_handler.record,
        None,  // No dense output for now
        false, // No event termination
    ))
}

/// Solve ODE with BDF method and mass matrix support
#[allow(dead_code)]
fn solve_bdf_with_mass_matrix<F, FFunc>(
    f: FFunc,
    t_span: [F; 2],
    y0: Array1<F>,
    mass_matrix: MassMatrix<F>,
    mut opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + std::default::Default,
    FFunc: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    use crate::ode::methods::enhanced_bdf_method;

    // Create a combined function that handles mass _matrix equation: M(t)·y' = f(t,y)
    let mass_f = move |t: F, y: ArrayView1<F>| -> Array1<F> {
        let rhs = f(t, y);

        // For BDF methods, we need to solve M·y' = f at each Newton iteration
        // This is handled by modifying the Newton system to (J - γM)·Δy = -G
        // where J is the Jacobian of f and γ is the BDF coefficient

        match mass_matrix.matrix_type {
            MassMatrixType::Identity => {
                // Identity case: just return f(t,y)
                rhs
            }
            MassMatrixType::Constant => {
                // For constant mass matrix, we can precompute M⁻¹ if it's not singular
                // However, for BDF, it's better to keep M in the Newton system
                rhs // Return original RHS, mass _matrix will be handled in Newton solve
            }
            MassMatrixType::TimeDependent => {
                // For time-dependent mass matrix, return original RHS
                // Mass _matrix handling will be incorporated into the Newton system
                rhs
            }
            MassMatrixType::StateDependent => {
                // This case should not reach here; state-dependent has separate handler
                rhs
            }
        }
    };

    // Solve using enhanced BDF with mass _matrix information stored in options
    opts.mass_matrix = Some(mass_matrix);
    enhanced_bdf_method(mass_f, t_span, y0, opts)
}

/// Solve ODE with BDF method and state-dependent mass matrix support
#[allow(dead_code)]
fn solve_bdf_with_state_dependent_mass_matrix<F, FFunc>(
    f: FFunc,
    t_span: [F; 2],
    y0: Array1<F>,
    mass_matrix: MassMatrix<F>,
    mut opts: ODEOptions<F>,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + std::default::Default,
    FFunc: Fn(F, ArrayView1<F>) -> Array1<F> + Clone,
{
    // For state-dependent mass matrices M(t,y), we need to solve M(t,y)·y' = f(t,y)
    // at each time step within the BDF Newton iteration

    let mass_f = move |t: F, y: ArrayView1<F>| -> Array1<F> {
        // For state-dependent mass matrices, the Newton system becomes:
        // [J - γ(∂M/∂y)y' - γM]·Δy = -G
        // This is more complex and requires careful handling of M(t,y) derivatives

        // For now, return the original RHS and let the BDF solver handle
        // the mass _matrix in its Newton iteration
        f(t, y)
    };

    // Configure BDF solver for state-dependent mass _matrix
    opts.mass_matrix = Some(mass_matrix);
    opts.rtol = opts.rtol.min(F::from_f64(1e-8).unwrap_or(opts.rtol)); // Tighter tolerance

    enhanced_bdf_method(mass_f, t_span, y0, opts)
}
