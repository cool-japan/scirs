//! Event detection and handling for ODE solvers
//!
//! This module provides functionality for detecting events during ODE integration
//! and handling them appropriately. Events are defined as conditions where a given
//! function crosses zero. The event can trigger various actions, such as stopping
//! the integration, modifying the state, or recording the event time.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::utils::dense_output::DenseSolution;
use ndarray::{Array1, ArrayView1};

/// Direction of zero-crossing for event detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EventDirection {
    /// Detect crossings from negative to positive
    Rising,
    /// Detect crossings from positive to negative
    Falling,
    /// Detect both crossing directions
    #[default]
    Both,
}

/// Action to take when an event is detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EventAction {
    /// Continue integration without stopping
    #[default]
    Continue,
    /// Stop the integration and return
    Stop,
}

/// Definition of an event to detect during integration
#[derive(Debug, Clone)]
pub struct EventSpec<F: IntegrateFloat> {
    /// Unique identifier for this event type
    pub id: String,
    /// Direction of zero-crossing to detect
    pub direction: EventDirection,
    /// Action to take when event is detected
    pub action: EventAction,
    /// Tolerance for considering event to be triggered (prevents chattering)
    pub threshold: F,
    /// Maximum number of detections (None for unlimited)
    pub max_count: Option<usize>,
    /// Whether to refine the event time with high precision
    pub precise_time: bool,
}

impl<F: IntegrateFloat> EventSpec<F> {
    /// Check if the maximum count has been reached for this event
    pub fn max_count_reached(&self, _id: &str, current_count: Option<usize>) -> bool {
        if let Some(max) = self.max_count {
            if let Some(count) = current_count {
                return count >= max;
            }
        }
        false
    }
}

impl<F: IntegrateFloat> Default for EventSpec<F> {
    fn default() -> Self {
        EventSpec {
            id: "default".to_string(),
            direction: EventDirection::default(),
            action: EventAction::default(),
            threshold: F::from_f64(1e-6).unwrap(),
            max_count: None,
            precise_time: true,
        }
    }
}

/// Represents a detected event during integration
#[derive(Debug, Clone)]
pub struct Event<F: IntegrateFloat> {
    /// ID of the event that was triggered
    pub id: String,
    /// Time at which the event occurred
    pub time: F,
    /// Values of the state at the event time
    pub state: Array1<F>,
    /// Value of the event function at the event time
    pub value: F,
    /// Direction of zero-crossing (1 for rising, -1 for falling)
    pub direction: i8,
}

/// Record of all events detected during integration
#[derive(Debug, Clone)]
pub struct EventRecord<F: IntegrateFloat> {
    /// List of all detected events in chronological order
    pub events: Vec<Event<F>>,
    /// Count of events by ID
    pub counts: std::collections::HashMap<String, usize>,
}

impl<F: IntegrateFloat> Default for EventRecord<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> EventRecord<F> {
    /// Create a new empty event record
    pub fn new() -> Self {
        EventRecord {
            events: Vec::new(),
            counts: std::collections::HashMap::new(),
        }
    }

    /// Add a detected event to the record
    pub fn add_event(&mut self, event: Event<F>) {
        // Update count for this event type
        *self.counts.entry(event.id.clone()).or_insert(0) += 1;

        // Add to the list of events
        self.events.push(event);
    }

    /// Get the count of a specific event type
    pub fn get_count(&self, id: &str) -> usize {
        *self.counts.get(id).unwrap_or(&0)
    }

    /// Get all events of a specific type
    pub fn get_events(&self, id: &str) -> Vec<&Event<F>> {
        self.events.iter().filter(|e| e.id == id).collect()
    }

    /// Check if the maximum event count has been reached for a specific event type
    pub fn max_count_reached(&self, id: &str, max_count: Option<usize>) -> bool {
        if let Some(max) = max_count {
            self.get_count(id) >= max
        } else {
            false
        }
    }
}

/// Event detection and handling during ODE integration
#[derive(Debug)]
pub struct EventHandler<F: IntegrateFloat> {
    /// List of event specifications to detect
    pub specs: Vec<EventSpec<F>>,
    /// Record of detected events
    pub record: EventRecord<F>,
    /// Last values of event functions for each spec
    last_values: Vec<Option<F>>,
    /// States at last step
    last_state: Option<(F, Array1<F>)>,
}

impl<F: IntegrateFloat> EventHandler<F> {
    /// Create a new event handler with the given event specifications
    pub fn new(specs: Vec<EventSpec<F>>) -> Self {
        let last_values = vec![None; specs.len()];

        EventHandler {
            specs,
            record: EventRecord::new(),
            last_values,
            last_state: None,
        }
    }

    /// Initialize the event handler with the initial state
    pub fn initialize<Func>(
        &mut self,
        t: F,
        y: &Array1<F>,
        event_funcs: &[Func],
    ) -> IntegrateResult<()>
    where
        Func: Fn(F, ArrayView1<F>) -> F,
    {
        // Store initial state
        self.last_state = Some((t, y.clone()));

        // Initialize last event function values
        for (i, func) in event_funcs.iter().enumerate() {
            let value = func(t, y.view());
            self.last_values[i] = Some(value);
        }

        Ok(())
    }

    /// Check for events between the last state and the current state
    pub fn check_events<Func>(
        &mut self,
        t: F,
        y: &Array1<F>,
        dense_output: Option<&DenseSolution<F>>,
        event_funcs: &[Func],
    ) -> IntegrateResult<EventAction>
    where
        Func: Fn(F, ArrayView1<F>) -> F,
    {
        if event_funcs.len() != self.specs.len() {
            return Err(IntegrateError::ValueError(
                "Number of event functions does not match number of event specifications"
                    .to_string(),
            ));
        }

        if self.last_state.is_none() {
            // Initialize if not done already
            self.initialize(t, y, event_funcs)?;
            return Ok(EventAction::Continue);
        }

        let (t_prev, y_prev) = self.last_state.as_ref().unwrap();

        // Check each event
        let mut action = EventAction::Continue;

        for (i, (func, spec)) in event_funcs.iter().zip(self.specs.iter()).enumerate() {
            // Skip if we've already reached the maximum count for this event
            if spec.max_count_reached(&spec.id, self.record.counts.get(&spec.id).cloned()) {
                continue;
            }

            // Compute current value
            let value = func(t, y.view());

            // Check if we have a previous value
            if let Some(prev_value) = self.last_values[i] {
                // Check if event occurred (zero-crossing)
                let rising = prev_value < F::zero() && value >= F::zero();
                let falling = prev_value > F::zero() && value <= F::zero();

                let triggered = match spec.direction {
                    EventDirection::Rising => rising,
                    EventDirection::Falling => falling,
                    EventDirection::Both => rising || falling,
                };

                if triggered {
                    // Refine the event time if requested and dense output is available
                    let (event_t, event_y, event_val, dir) =
                        if spec.precise_time && dense_output.is_some() {
                            self.refine_event_time(
                                *t_prev,
                                y_prev,
                                t,
                                y,
                                prev_value,
                                value,
                                func,
                                dense_output.unwrap(),
                            )?
                        } else {
                            // Use current time as event time (less accurate)
                            let dir = if rising { 1 } else { -1 };
                            (t, y.clone(), value, dir)
                        };

                    // Create event record
                    let event = Event {
                        id: spec.id.clone(),
                        time: event_t,
                        state: event_y,
                        value: event_val,
                        direction: dir,
                    };

                    // Add to record
                    self.record.add_event(event);

                    // If this event requires stopping, set the action
                    if spec.action == EventAction::Stop {
                        action = EventAction::Stop;
                    }
                }
            }

            // Update last value
            self.last_values[i] = Some(value);
        }

        // Update last state
        self.last_state = Some((t, y.clone()));

        Ok(action)
    }

    /// Refine the exact time of an event using bisection on the dense output
    #[allow(clippy::too_many_arguments)]
    fn refine_event_time<Func>(
        &self,
        t_prev: F,
        y_prev: &Array1<F>,
        t_curr: F,
        y_curr: &Array1<F>,
        value_prev: F,
        value_curr: F,
        event_func: &Func,
        dense_output: &DenseSolution<F>,
    ) -> IntegrateResult<(F, Array1<F>, F, i8)>
    where
        Func: Fn(F, ArrayView1<F>) -> F,
    {
        // Determine event direction
        let direction: i8 = if value_prev < F::zero() && value_curr >= F::zero() {
            1 // Rising
        } else {
            -1 // Falling
        };

        // Root-finding tolerance
        let tol = F::from_f64(1e-10).unwrap();
        let max_iter = 50;

        // Bisection search for zero-crossing
        let mut t_left = t_prev;
        let mut t_right = t_curr;
        let mut f_left = value_prev;
        let f_right = value_curr;

        // Handle the case where one endpoint is exactly zero
        if f_left.abs() < tol {
            return Ok((t_left, y_prev.clone(), f_left, direction));
        }

        if f_right.abs() < tol {
            return Ok((t_right, y_curr.clone(), f_right, direction));
        }

        // Bisection loop
        let mut t_mid = F::zero();
        let mut y_mid = Array1::<F>::zeros(y_prev.len());
        let mut f_mid = F::zero();

        for _ in 0..max_iter {
            // Compute midpoint time
            t_mid = (t_left + t_right) / F::from_f64(2.0).unwrap();

            // Get state at midpoint using dense output
            y_mid = dense_output.evaluate(t_mid)?;

            // Evaluate event function at midpoint
            f_mid = event_func(t_mid, y_mid.view());

            // Check convergence
            if f_mid.abs() < tol || (t_right - t_left).abs() < tol {
                break;
            }

            // Update interval
            if f_left * f_mid < F::zero() {
                t_right = t_mid;
                let _f_right = f_mid;
            } else {
                t_left = t_mid;
                f_left = f_mid;
            }
        }

        Ok((t_mid, y_mid, f_mid, direction))
    }

    /// Get the record of all detected events
    pub fn get_record(&self) -> &EventRecord<F> {
        &self.record
    }

    /// Check if an event occurred that requires stopping the integration
    pub fn should_stop(&self) -> bool {
        self.record.events.iter().any(|e| {
            let spec = self.specs.iter().find(|s| s.id == e.id).unwrap();
            spec.action == EventAction::Stop
        })
    }
}

/// Function to create a terminal event (one that stops integration when triggered)
pub fn terminal_event<F: IntegrateFloat>(id: &str, direction: EventDirection) -> EventSpec<F> {
    EventSpec {
        id: id.to_string(),
        direction,
        action: EventAction::Stop,
        threshold: F::from_f64(1e-6).unwrap(),
        max_count: Some(1),
        precise_time: true,
    }
}

/// Extension to ODEOptions to include event handling
#[derive(Debug, Clone)]
pub struct ODEOptionsWithEvents<F: IntegrateFloat> {
    /// Base ODE options
    pub base_options: super::super::types::ODEOptions<F>,
    /// Event specifications
    pub event_specs: Vec<EventSpec<F>>,
}

impl<F: IntegrateFloat> ODEOptionsWithEvents<F> {
    /// Create options with events from base options
    pub fn new(
        base_options: super::super::types::ODEOptions<F>,
        event_specs: Vec<EventSpec<F>>,
    ) -> Self {
        ODEOptionsWithEvents {
            base_options,
            event_specs,
        }
    }
}

/// Extended ODE result that includes event information
#[derive(Debug)]
pub struct ODEResultWithEvents<F: IntegrateFloat> {
    /// Base ODE result
    pub base_result: super::super::types::ODEResult<F>,
    /// Record of detected events
    pub events: EventRecord<F>,
    /// Dense output for the solution (if available)
    pub dense_output: Option<DenseSolution<F>>,
    /// Whether integration terminated due to an event
    pub event_termination: bool,
}

impl<F: IntegrateFloat> ODEResultWithEvents<F> {
    /// Create a new result with events
    pub fn new(
        base_result: super::super::types::ODEResult<F>,
        events: EventRecord<F>,
        dense_output: Option<DenseSolution<F>>,
        event_termination: bool,
    ) -> Self {
        ODEResultWithEvents {
            base_result,
            events,
            dense_output,
            event_termination,
        }
    }

    /// Get the solution at a specific time using dense output
    pub fn at_time(&self, t: F) -> IntegrateResult<Option<Array1<F>>> {
        if let Some(ref dense) = self.dense_output {
            Ok(Some(dense.evaluate(t)?))
        } else {
            // If no dense output available, check if we have exact time points
            for (i, &ti) in self.base_result.t.iter().enumerate() {
                if (ti - t).abs() < F::from_f64(1e-10).unwrap() {
                    return Ok(Some(self.base_result.y[i].clone()));
                }
            }
            Ok(None)
        }
    }

    /// Get events of a specific type
    pub fn get_events(&self, id: &str) -> Vec<&Event<F>> {
        self.events.get_events(id)
    }

    /// Get the first occurrence of a specific event
    pub fn first_event(&self, id: &str) -> Option<&Event<F>> {
        self.events.get_events(id).first().copied()
    }
}
