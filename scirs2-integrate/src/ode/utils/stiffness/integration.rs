//! Integration-related stiffness detection utilities

/// The current state of an adaptive method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveMethodState {
    /// Using non-stiff methods
    NonStiff,
    /// Using stiff methods
    Stiff,
    /// In transition between methods
    Transition,
}

/// Type of adaptive method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveMethodType {
    /// Adams methods for non-stiff problems
    Adams,
    /// BDF methods for stiff problems
    BDF,
    /// Runge-Kutta methods
    RungeKutta,
    /// Implicit methods (Radau, etc.)
    Implicit,
}