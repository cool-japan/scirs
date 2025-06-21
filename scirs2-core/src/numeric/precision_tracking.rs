//! # Automated Precision Tracking for Numerical Computations
//!
//! This module provides comprehensive precision tracking capabilities for numerical computations,
//! helping to monitor and control numerical accuracy throughout complex scientific calculations.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;
use std::fmt;
use std::sync::RwLock;
use std::time::Instant;

/// Precision context for tracking numerical accuracy
#[derive(Debug, Clone)]
pub struct PrecisionContext {
    /// Current precision estimate (in terms of significant digits)
    pub precision: f64,
    /// Error bounds (relative error)
    pub error_bounds: f64,
    /// Number of significant digits
    pub significant_digits: u32,
    /// Operations that contributed to precision loss
    pub precision_loss_sources: Vec<PrecisionLossSource>,
    /// Condition number estimate
    pub condition_number: Option<f64>,
    /// Whether the computation is numerically stable
    pub is_stable: bool,
    /// Timestamp of last update
    pub last_updated: Instant,
}

/// Source of precision loss in computations
#[derive(Debug, Clone)]
pub struct PrecisionLossSource {
    /// Operation that caused precision loss
    pub operation: String,
    /// Amount of precision lost (in digits)
    pub precision_lost: f64,
    /// Description of the loss
    pub description: String,
    /// Severity of the loss
    pub severity: PrecisionLossSeverity,
    /// Location in the computation
    pub location: Option<String>,
}

/// Severity of precision loss
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrecisionLossSeverity {
    /// Minimal precision loss (< 1 digit)
    Minimal,
    /// Moderate precision loss (1-3 digits)
    Moderate,
    /// Significant precision loss (3-6 digits)
    Significant,
    /// Severe precision loss (> 6 digits)
    Severe,
    /// Catastrophic precision loss (result unreliable)
    Catastrophic,
}

impl fmt::Display for PrecisionLossSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimal => write!(f, "Minimal"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Significant => write!(f, "Significant"),
            Self::Severe => write!(f, "Severe"),
            Self::Catastrophic => write!(f, "Catastrophic"),
        }
    }
}

impl Default for PrecisionContext {
    fn default() -> Self {
        Self {
            precision: 15.0, // Default for f64
            error_bounds: 0.0,
            significant_digits: 15,
            precision_loss_sources: Vec::new(),
            condition_number: None,
            is_stable: true,
            last_updated: Instant::now(),
        }
    }
}

impl PrecisionContext {
    /// Create a new precision context with given precision
    pub fn new(precision: f64) -> Self {
        Self {
            precision,
            significant_digits: precision as u32,
            ..Default::default()
        }
    }

    /// Create a precision context for single precision (f32)
    pub fn single_precision() -> Self {
        Self::new(7.0) // Typical for f32
    }

    /// Create a precision context for double precision (f64)
    pub fn double_precision() -> Self {
        Self::new(15.0) // Typical for f64
    }

    /// Create a precision context for extended precision
    pub fn extended_precision() -> Self {
        Self::new(18.0) // Extended precision
    }

    /// Update precision after an operation
    pub fn update_precision(&mut self, new_precision: f64, operation: &str) {
        if new_precision < self.precision {
            let loss = self.precision - new_precision;
            let severity = Self::classify_precision_loss(loss);

            self.precision_loss_sources.push(PrecisionLossSource {
                operation: operation.to_string(),
                precision_lost: loss,
                description: format!(
                    "Precision reduced from {:.2} to {:.2} digits",
                    self.precision, new_precision
                ),
                severity,
                location: None,
            });
        }

        self.precision = new_precision;
        self.significant_digits = new_precision as u32;
        self.is_stable = self.precision > 3.0; // Heuristic for stability
        self.last_updated = Instant::now();
    }

    /// Record precision loss from a specific operation
    pub fn record_precision_loss(
        &mut self,
        operation: &str,
        loss: f64,
        description: Option<String>,
        location: Option<String>,
    ) {
        let severity = Self::classify_precision_loss(loss);

        self.precision_loss_sources.push(PrecisionLossSource {
            operation: operation.to_string(),
            precision_lost: loss,
            description: description
                .unwrap_or_else(|| format!("Precision loss of {loss:.2} digits in {operation}")),
            severity,
            location,
        });

        self.precision = (self.precision - loss).max(0.0);
        self.significant_digits = self.precision as u32;
        self.is_stable = self.precision > 3.0;
        self.last_updated = Instant::now();
    }

    /// Set condition number estimate
    pub fn set_condition_number(&mut self, cond: f64) {
        self.condition_number = Some(cond);

        // Update stability based on condition number
        if cond > 1e12 {
            self.record_precision_loss(
                "ill_conditioning",
                (cond.log10() - 12.0).max(0.0),
                Some(format!("Ill-conditioned problem (κ = {cond:.2e})")),
                None,
            );
            // Force instability for very high condition numbers
            self.is_stable = false;
        } else if cond > 1e8 {
            // Moderately ill-conditioned
            self.record_precision_loss(
                "moderate_conditioning",
                (cond.log10() - 8.0).max(0.0),
                Some(format!("Moderately ill-conditioned (κ = {cond:.2e})")),
                None,
            );
        }
    }

    /// Get the total precision loss
    pub fn total_precision_loss(&self) -> f64 {
        self.precision_loss_sources
            .iter()
            .map(|source| source.precision_lost)
            .sum()
    }

    /// Get the worst precision loss severity
    pub fn worst_severity(&self) -> Option<PrecisionLossSeverity> {
        self.precision_loss_sources
            .iter()
            .map(|source| &source.severity)
            .max()
            .cloned()
    }

    /// Check if the computation has acceptable precision
    pub fn is_acceptable(&self, min_precision: f64) -> bool {
        self.precision >= min_precision && self.is_stable
    }

    /// Generate precision warning if necessary
    pub fn check_precision_warning(&self, min_acceptable: f64) -> Option<PrecisionWarning> {
        if self.precision < min_acceptable {
            Some(PrecisionWarning {
                current_precision: self.precision,
                required_precision: min_acceptable,
                severity: if self.precision < min_acceptable / 2.0 {
                    PrecisionLossSeverity::Severe
                } else {
                    PrecisionLossSeverity::Moderate
                },
                message: format!(
                    "Precision ({:.2} digits) below acceptable threshold ({:.2} digits)",
                    self.precision, min_acceptable
                ),
                suggestions: self.generate_suggestions(),
            })
        } else {
            None
        }
    }

    /// Classify precision loss severity
    fn classify_precision_loss(loss: f64) -> PrecisionLossSeverity {
        if loss < 1.0 {
            PrecisionLossSeverity::Minimal
        } else if loss < 3.0 {
            PrecisionLossSeverity::Moderate
        } else if loss < 6.0 {
            PrecisionLossSeverity::Significant
        } else if loss < 10.0 {
            PrecisionLossSeverity::Severe
        } else {
            PrecisionLossSeverity::Catastrophic
        }
    }

    /// Generate suggestions for improving precision
    fn generate_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if self.precision < 3.0 {
            suggestions.push("Consider using higher precision arithmetic".to_string());
            suggestions.push("Review algorithm for numerical stability".to_string());
        }

        if let Some(cond) = self.condition_number {
            if cond > 1e10 {
                suggestions.push("Problem is ill-conditioned; consider regularization".to_string());
                suggestions.push("Try alternative algorithms or preconditioning".to_string());
            }
        }

        for source in &self.precision_loss_sources {
            match source.severity {
                PrecisionLossSeverity::Significant
                | PrecisionLossSeverity::Severe
                | PrecisionLossSeverity::Catastrophic => {
                    suggestions.push(format!(
                        "Review {} operation for numerical stability",
                        source.operation
                    ));
                }
                _ => {}
            }
        }

        if suggestions.is_empty() {
            suggestions.push("Monitor precision throughout computation".to_string());
        }

        suggestions
    }
}

/// Warning about precision loss
#[derive(Debug, Clone)]
pub struct PrecisionWarning {
    /// Current precision level
    pub current_precision: f64,
    /// Required precision level
    pub required_precision: f64,
    /// Severity of the warning
    pub severity: PrecisionLossSeverity,
    /// Warning message
    pub message: String,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

impl fmt::Display for PrecisionWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Precision Warning ({}): {}", self.severity, self.message)?;
        if !self.suggestions.is_empty() {
            writeln!(f, "Suggestions:")?;
            for suggestion in &self.suggestions {
                writeln!(f, "  • {suggestion}")?;
            }
        }
        Ok(())
    }
}

/// Trait for numerical types that support precision tracking
pub trait PrecisionTracked {
    /// Get the current precision context
    fn precision_context(&self) -> &PrecisionContext;

    /// Get a mutable reference to the precision context
    fn precision_context_mut(&mut self) -> &mut PrecisionContext;

    /// Update precision after an operation
    fn update_precision_after_op(&mut self, operation: &str, result_precision: f64) {
        self.precision_context_mut()
            .update_precision(result_precision, operation);
    }

    /// Record precision loss
    fn record_loss(&mut self, operation: &str, loss: f64, description: Option<String>) {
        self.precision_context_mut()
            .record_precision_loss(operation, loss, description, None);
    }

    /// Check if precision is acceptable
    fn check_precision(&self, min_precision: f64) -> CoreResult<()> {
        if let Some(warning) = self
            .precision_context()
            .check_precision_warning(min_precision)
        {
            Err(CoreError::ValidationError(ErrorContext::new(
                warning.message,
            )))
        } else {
            Ok(())
        }
    }
}

/// Wrapper type for floating-point numbers with precision tracking
#[derive(Debug, Clone)]
pub struct TrackedFloat<T> {
    /// The underlying value
    pub value: T,
    /// Precision tracking context
    pub context: PrecisionContext,
}

impl<T> TrackedFloat<T>
where
    T: Copy + PartialOrd + fmt::Display,
{
    /// Create a new tracked float with default precision
    pub fn new(value: T) -> Self {
        Self {
            value,
            context: PrecisionContext::default(),
        }
    }

    /// Create a new tracked float with specified precision
    pub fn with_precision(value: T, precision: f64) -> Self {
        Self {
            value,
            context: PrecisionContext::new(precision),
        }
    }

    /// Get the underlying value
    pub fn value(&self) -> T {
        self.value
    }

    /// Check if the value is finite (for floating-point types)
    pub fn is_finite(&self) -> bool
    where
        T: num_traits::Float,
    {
        self.value.is_finite()
    }

    /// Check if precision is critically low
    pub fn is_precision_critical(&self) -> bool {
        self.context.precision < 2.0 || !self.context.is_stable
    }
}

impl<T> PrecisionTracked for TrackedFloat<T> {
    fn precision_context(&self) -> &PrecisionContext {
        &self.context
    }

    fn precision_context_mut(&mut self) -> &mut PrecisionContext {
        &mut self.context
    }
}

/// Implement arithmetic operations with precision tracking
impl TrackedFloat<f64> {
    /// Add two tracked floats
    pub fn add(&self, other: &Self) -> Self {
        let result_value = self.value + other.value;
        let mut result = Self::new(result_value);

        // Estimate precision loss from addition
        let min_precision = self.context.precision.min(other.context.precision);
        let relative_error = estimate_addition_error(self.value, other.value);
        let precision_loss = -relative_error.log10().max(0.0);

        result.context.precision = (min_precision - precision_loss).max(0.0);
        result.record_loss("addition", precision_loss, None);

        result
    }

    /// Subtract two tracked floats
    pub fn sub(&self, other: &Self) -> Self {
        let result_value = self.value - other.value;
        let mut result = Self::new(result_value);

        // Check for catastrophic cancellation
        let relative_magnitude =
            (self.value - other.value).abs() / self.value.abs().max(other.value.abs());
        if relative_magnitude < 1e-10 {
            result.record_loss(
                "subtraction",
                10.0,
                Some("Catastrophic cancellation detected".to_string()),
            );
        } else {
            let min_precision = self.context.precision.min(other.context.precision);
            let precision_loss = -relative_magnitude.log10().max(0.0);
            result.context.precision = (min_precision - precision_loss).max(0.0);
            result.record_loss("subtraction", precision_loss, None);
        }

        result
    }

    /// Multiply two tracked floats
    pub fn mul(&self, other: &Self) -> Self {
        let result_value = self.value * other.value;
        let mut result = Self::new(result_value);

        let min_precision = self.context.precision.min(other.context.precision);
        let relative_error = estimate_multiplication_error(self.value, other.value);
        let precision_loss = -relative_error.log10().max(0.0);

        result.context.precision = (min_precision - precision_loss).max(0.0);
        result.record_loss("multiplication", precision_loss, None);

        result
    }

    /// Divide two tracked floats
    pub fn div(&self, other: &Self) -> CoreResult<Self> {
        if other.value.abs() < f64::EPSILON {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Division by zero or near-zero value",
            )));
        }

        let result_value = self.value / other.value;
        let mut result = Self::new(result_value);

        let min_precision = self.context.precision.min(other.context.precision);
        let relative_error = estimate_division_error(self.value, other.value);
        let precision_loss = -relative_error.log10().max(0.0);

        // Additional precision loss for small divisors
        if other.value.abs() < 1e-10 {
            let extra_loss = -other.value.abs().log10() - 10.0;
            result.record_loss(
                "division",
                precision_loss + extra_loss,
                Some("Division by small number".to_string()),
            );
        } else {
            result.context.precision = (min_precision - precision_loss).max(0.0);
            result.record_loss("division", precision_loss, None);
        }

        Ok(result)
    }

    /// Take square root with precision tracking
    pub fn sqrt(&self) -> CoreResult<Self> {
        if self.value < 0.0 {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Square root of negative number",
            )));
        }

        let result_value = self.value.sqrt();
        let mut result = Self::new(result_value);

        // Square root generally preserves precision well
        let precision_loss = 0.5; // Typical loss for sqrt
        result.context.precision = (self.context.precision - precision_loss).max(0.0);
        result.record_loss("sqrt", precision_loss, None);

        Ok(result)
    }

    /// Natural logarithm with precision tracking
    pub fn ln(&self) -> CoreResult<Self> {
        if self.value <= 0.0 {
            return Err(CoreError::DomainError(ErrorContext::new(
                "Logarithm of non-positive number",
            )));
        }

        let result_value = self.value.ln();
        let mut result = Self::new(result_value);

        // Logarithm near 1 can cause precision loss
        if (self.value - 1.0).abs() < 1e-10 {
            let extra_loss = -(self.value - 1.0).abs().log10() - 10.0;
            result.record_loss(
                "logarithm",
                extra_loss,
                Some("Logarithm near 1".to_string()),
            );
        } else {
            let precision_loss = 1.0; // Typical loss for log
            result.context.precision = (self.context.precision - precision_loss).max(0.0);
            result.record_loss("logarithm", precision_loss, None);
        }

        Ok(result)
    }
}

/// Error estimation functions
fn estimate_addition_error(a: f64, b: f64) -> f64 {
    let result = a + b;
    if result == 0.0 {
        f64::EPSILON
    } else {
        (f64::EPSILON * (a.abs() + b.abs())) / result.abs()
    }
}

fn estimate_multiplication_error(_a: f64, _b: f64) -> f64 {
    2.0 * f64::EPSILON // Relative error for multiplication
}

fn estimate_division_error(a: f64, b: f64) -> f64 {
    let rel_error_a = f64::EPSILON;
    let rel_error_b = f64::EPSILON;
    (rel_error_a + rel_error_b) * (a / b).abs()
}

/// Global precision tracking registry
#[derive(Debug)]
pub struct PrecisionRegistry {
    /// Tracked computations
    computations: RwLock<HashMap<String, PrecisionContext>>,
    /// Global warnings
    warnings: RwLock<Vec<PrecisionWarning>>,
}

impl PrecisionRegistry {
    /// Create a new precision registry
    pub fn new() -> Self {
        Self {
            computations: RwLock::new(HashMap::new()),
            warnings: RwLock::new(Vec::new()),
        }
    }

    /// Register a computation for precision tracking
    pub fn register_computation(&self, name: &str, context: PrecisionContext) -> CoreResult<()> {
        let mut computations = self.computations.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;
        computations.insert(name.to_string(), context);
        Ok(())
    }

    /// Update precision for a computation
    pub fn update_computation_precision(
        &self,
        name: &str,
        precision: f64,
        operation: &str,
    ) -> CoreResult<()> {
        let mut computations = self.computations.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;

        if let Some(context) = computations.get_mut(name) {
            context.update_precision(precision, operation);
        } else {
            return Err(CoreError::ValidationError(ErrorContext::new(format!(
                "Computation '{name}' not found in registry"
            ))));
        }
        Ok(())
    }

    /// Get precision context for a computation
    pub fn get_computation_context(&self, name: &str) -> CoreResult<Option<PrecisionContext>> {
        let computations = self.computations.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire read lock"))
        })?;
        Ok(computations.get(name).cloned())
    }

    /// Add a precision warning
    pub fn add_warning(&self, warning: PrecisionWarning) -> CoreResult<()> {
        let mut warnings = self.warnings.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;
        warnings.push(warning);
        Ok(())
    }

    /// Get all warnings
    pub fn get_warnings(&self) -> CoreResult<Vec<PrecisionWarning>> {
        let warnings = self.warnings.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire read lock"))
        })?;
        Ok(warnings.clone())
    }

    /// Clear all warnings
    pub fn clear_warnings(&self) -> CoreResult<()> {
        let mut warnings = self.warnings.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire write lock"))
        })?;
        warnings.clear();
        Ok(())
    }

    /// Generate precision report
    pub fn generate_report(&self) -> CoreResult<String> {
        let computations = self.computations.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire read lock"))
        })?;
        let warnings = self.warnings.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire read lock"))
        })?;

        let mut report = String::new();
        report.push_str("=== Precision Tracking Report ===\n\n");

        if computations.is_empty() {
            report.push_str("No computations tracked.\n");
        } else {
            report.push_str("Tracked Computations:\n");
            for (name, context) in computations.iter() {
                report.push_str(&format!(
                    "  {}: {:.2} digits, {} sources of loss, stable: {}\n",
                    name,
                    context.precision,
                    context.precision_loss_sources.len(),
                    context.is_stable
                ));
            }
        }

        if !warnings.is_empty() {
            report.push_str(&format!("\nWarnings ({}):\n", warnings.len()));
            for warning in warnings.iter() {
                report.push_str(&format!("  {}\n", warning.message));
            }
        }

        Ok(report)
    }
}

impl Default for PrecisionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global precision registry instance
static GLOBAL_PRECISION_REGISTRY: std::sync::LazyLock<PrecisionRegistry> =
    std::sync::LazyLock::new(PrecisionRegistry::new);

/// Get the global precision registry
pub fn global_precision_registry() -> &'static PrecisionRegistry {
    &GLOBAL_PRECISION_REGISTRY
}

/// Convenience macros for precision tracking
#[macro_export]
macro_rules! track_precision {
    ($name:expr, $precision:expr) => {
        $crate::numeric::precision_tracking::global_precision_registry()
            .register_computation(
                $name,
                $crate::numeric::precision_tracking::PrecisionContext::new($precision),
            )
            .unwrap_or_else(|e| eprintln!("Failed to track precision: {:?}", e));
    };
}

#[macro_export]
macro_rules! update_precision {
    ($name:expr, $precision:expr, $operation:expr) => {
        $crate::numeric::precision_tracking::global_precision_registry()
            .update_computation_precision($name, $precision, $operation)
            .unwrap_or_else(|e| eprintln!("Failed to update precision: {:?}", e));
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_context() {
        let mut context = PrecisionContext::double_precision();
        assert_eq!(context.precision, 15.0);
        assert!(context.is_stable);

        context.update_precision(10.0, "test_operation");
        assert_eq!(context.precision, 10.0);
        assert_eq!(context.precision_loss_sources.len(), 1);
    }

    #[test]
    fn test_tracked_float_arithmetic() {
        let a = TrackedFloat::with_precision(1.0, 15.0);
        let b = TrackedFloat::with_precision(1e-15, 15.0);

        let result = a.add(&b);
        // Addition of very different magnitudes should show some precision loss or be tracked
        assert!(
            result.context.precision <= 15.0,
            "Precision should not increase, got: {}",
            result.context.precision
        );
        // Result should be close to 1.0 + 1e-15, which should be handled correctly
        let expected = 1.0 + 1e-15;
        assert!(
            (result.value - expected).abs() < 1e-14,
            "Expected {}, got {}",
            expected,
            result.value
        );
    }

    #[test]
    fn test_catastrophic_cancellation() {
        let a = TrackedFloat::with_precision(1.000_000_000_000_1, 15.0);
        let b = TrackedFloat::with_precision(1.0, 15.0);

        let result = a.sub(&b);
        // Subtraction of nearly equal numbers should show significant precision loss
        assert!(result.context.precision < 10.0);
        assert!(!result.context.precision_loss_sources.is_empty());
    }

    #[test]
    fn test_division_by_small_number() {
        let a = TrackedFloat::with_precision(1.0, 15.0);
        let b = TrackedFloat::with_precision(1e-12, 15.0);

        let result = a.div(&b).unwrap();
        // Division by small number should show precision loss
        assert!(result.context.precision < 15.0);
    }

    #[test]
    fn test_precision_warning() {
        let context = PrecisionContext::new(2.0);
        let warning = context.check_precision_warning(5.0);
        assert!(warning.is_some());

        let warning = warning.unwrap();
        assert_eq!(warning.current_precision, 2.0);
        assert_eq!(warning.required_precision, 5.0);
    }

    #[test]
    fn test_precision_registry() {
        let registry = PrecisionRegistry::new();
        let context = PrecisionContext::new(10.0);

        registry.register_computation("test", context).unwrap();

        let retrieved = registry.get_computation_context("test").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().precision, 10.0);
    }

    #[test]
    fn test_sqrt_precision() {
        let a = TrackedFloat::with_precision(4.0, 15.0);
        let result = a.sqrt().unwrap();
        assert_eq!(result.value, 2.0);
        assert!(result.context.precision < 15.0); // Some precision loss expected
    }

    #[test]
    fn test_ln_near_one() {
        let a = TrackedFloat::with_precision(1.0 + 1e-12, 15.0);
        let result = a.ln().unwrap();
        // Logarithm near 1 should show some precision loss
        assert!(
            result.context.precision < 15.0,
            "Expected precision loss for ln near 1, got precision: {}",
            result.context.precision
        );
        // Should have precision loss sources recorded
        assert!(
            !result.context.precision_loss_sources.is_empty(),
            "Should have recorded precision loss sources"
        );
    }

    #[test]
    fn test_condition_number() {
        let mut context = PrecisionContext::double_precision();
        // Verify initial state
        assert!(context.is_stable);
        assert!(context.precision_loss_sources.is_empty());

        context.set_condition_number(1e15);

        // After setting high condition number, context should be unstable
        // and should have precision loss sources
        assert!(
            !context.is_stable,
            "Context should be unstable with condition number 1e15"
        );
        assert!(
            !context.precision_loss_sources.is_empty(),
            "Should have precision loss sources after setting high condition number"
        );
        assert!(context.condition_number.is_some());
        assert_eq!(context.condition_number.unwrap(), 1e15);
    }

    #[test]
    fn test_precision_loss_severity() {
        assert_eq!(
            PrecisionContext::classify_precision_loss(0.5),
            PrecisionLossSeverity::Minimal
        );
        assert_eq!(
            PrecisionContext::classify_precision_loss(2.0),
            PrecisionLossSeverity::Moderate
        );
        assert_eq!(
            PrecisionContext::classify_precision_loss(5.0),
            PrecisionLossSeverity::Significant
        );
        assert_eq!(
            PrecisionContext::classify_precision_loss(8.0),
            PrecisionLossSeverity::Severe
        );
        assert_eq!(
            PrecisionContext::classify_precision_loss(12.0),
            PrecisionLossSeverity::Catastrophic
        );
    }
}
