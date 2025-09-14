//! Constraint types and validation logic
//!
//! This module provides various constraint types used for data validation,
//! including range constraints, pattern matching, and statistical constraints.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Validation constraints for data fields
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// Value must be within range (inclusive)
    Range { min: f64, max: f64 },
    /// String must match pattern
    Pattern(String),
    /// Value must be one of the allowed values
    AllowedValues(Vec<String>),
    /// String length constraints
    Length { min: usize, max: usize },
    /// Numeric precision constraints
    Precision { decimal_places: usize },
    /// Uniqueness constraint
    Unique,
    /// Non-null constraint
    NotNull,
    /// Custom validation rule
    Custom(String),
    /// Array element constraints
    ArrayElements(Box<Constraint>),
    /// Array size constraints
    ArraySize { min: usize, max: usize },
    /// Statistical constraints for numeric data
    Statistical(StatisticalConstraints),
    /// Time-based constraints
    Temporal(TimeConstraints),
    /// Matrix/array shape constraints
    Shape(ShapeConstraints),
    /// Logical AND of multiple constraints - all must pass
    And(Vec<Constraint>),
    /// Logical OR of multiple constraints - at least one must pass
    Or(Vec<Constraint>),
    /// Logical NOT of a constraint - must not pass
    Not(Box<Constraint>),
    /// Conditional constraint - if condition passes, then constraint must pass
    If {
        condition: Box<Constraint>,
        then_constraint: Box<Constraint>,
        else_constraint: Option<Box<Constraint>>,
    },
}

/// Statistical constraints for numeric data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticalConstraints {
    /// Minimum allowed mean value
    pub min_mean: Option<f64>,
    /// Maximum allowed mean value
    pub max_mean: Option<f64>,
    /// Minimum allowed standard deviation
    pub min_std: Option<f64>,
    /// Maximum allowed standard deviation
    pub max_std: Option<f64>,
    /// Expected statistical distribution
    pub expected_distribution: Option<String>,
}

/// Shape constraints for arrays and matrices
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShapeConstraints {
    /// Exact dimensions required (None = any size for that dimension)
    pub dimensions: Vec<Option<usize>>,
    /// Minimum number of elements
    pub min_elements: Option<usize>,
    /// Maximum number of elements
    pub max_elements: Option<usize>,
    /// Whether matrix must be square (for 2D only)
    pub require_square: bool,
    /// Whether to allow broadcasting-compatible shapes
    pub allow_broadcasting: bool,
}

/// Time series constraints
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeConstraints {
    /// Minimum time interval between samples
    pub min_interval: Option<Duration>,
    /// Maximum time interval between samples
    pub max_interval: Option<Duration>,
    /// Whether timestamps must be monotonic
    pub require_monotonic: bool,
    /// Whether to allow duplicate timestamps
    pub allow_duplicates: bool,
}

/// Sparse matrix formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SparseFormat {
    /// Compressed Sparse Row
    CSR,
    /// Compressed Sparse Column
    CSC,
    /// Coordinate format (COO)
    COO,
    /// Dictionary of Keys
    DOK,
}

/// Element validation function type
pub type ElementValidatorFn<T> = Box<dyn Fn(&T) -> bool + Send + Sync>;

/// Array validation constraints
pub struct ArrayValidationConstraints {
    /// Expected array shape
    pub expectedshape: Option<Vec<usize>>,
    /// Field name for error reporting
    pub fieldname: Option<String>,
    /// Check for NaN and infinity values
    pub check_numeric_quality: bool,
    /// Statistical constraints
    pub statistical_constraints: Option<StatisticalConstraints>,
    /// Check performance characteristics
    pub check_performance: bool,
    /// Element-wise validation function
    pub element_validator: Option<ElementValidatorFn<f64>>,
}

impl ArrayValidationConstraints {
    /// Create new array validation constraints
    pub fn new() -> Self {
        Self {
            expectedshape: None,
            fieldname: None,
            check_numeric_quality: false,
            statistical_constraints: None,
            check_performance: false,
            element_validator: None,
        }
    }

    /// Set expected shape
    pub fn withshape(mut self, shape: Vec<usize>) -> Self {
        self.expectedshape = Some(shape);
        self
    }

    /// Set field name
    pub fn with_fieldname(mut self, name: &str) -> Self {
        self.fieldname = Some(name.to_string());
        self
    }

    /// Enable numeric quality checks
    pub fn check_numeric_quality(mut self) -> Self {
        self.check_numeric_quality = true;
        self
    }

    /// Set statistical constraints
    pub fn with_statistical_constraints(mut self, constraints: StatisticalConstraints) -> Self {
        self.statistical_constraints = Some(constraints);
        self
    }

    /// Enable performance checks
    pub fn check_performance(mut self) -> Self {
        self.check_performance = true;
        self
    }
}

impl Default for ArrayValidationConstraints {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalConstraints {
    /// Create new statistical constraints
    pub fn new() -> Self {
        Self {
            min_mean: None,
            max_mean: None,
            min_std: None,
            max_std: None,
            expected_distribution: None,
        }
    }

    /// Set mean range
    pub fn with_mean_range(mut self, min: f64, max: f64) -> Self {
        self.min_mean = Some(min);
        self.max_mean = Some(max);
        self
    }

    /// Set standard deviation range
    pub fn with_std_range(mut self, min: f64, max: f64) -> Self {
        self.min_std = Some(min);
        self.max_std = Some(max);
        self
    }

    /// Set expected distribution
    pub fn with_distribution(mut self, distribution: &str) -> Self {
        self.expected_distribution = Some(distribution.to_string());
        self
    }
}

impl Default for StatisticalConstraints {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeConstraints {
    /// Create new shape constraints
    pub fn new() -> Self {
        Self {
            dimensions: Vec::new(),
            min_elements: None,
            max_elements: None,
            require_square: false,
            allow_broadcasting: false,
        }
    }

    /// Set exact dimensions
    pub fn with_dimensions(mut self, dimensions: Vec<Option<usize>>) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Set element count range
    pub fn with_element_range(mut self, min: usize, max: usize) -> Self {
        self.min_elements = Some(min);
        self.max_elements = Some(max);
        self
    }

    /// Require square matrix
    pub fn require_square(mut self) -> Self {
        self.require_square = true;
        self
    }

    /// Allow broadcasting
    pub fn allow_broadcasting(mut self) -> Self {
        self.allow_broadcasting = true;
        self
    }
}

impl Default for ShapeConstraints {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeConstraints {
    /// Create new time constraints
    pub fn new() -> Self {
        Self {
            min_interval: None,
            max_interval: None,
            require_monotonic: false,
            allow_duplicates: true,
        }
    }

    /// Set interval range
    pub fn with_interval_range(mut self, min: Duration, max: Duration) -> Self {
        self.min_interval = Some(min);
        self.max_interval = Some(max);
        self
    }

    /// Set minimum time interval
    pub fn with_min_interval(mut self, interval: Duration) -> Self {
        self.min_interval = Some(interval);
        self
    }

    /// Set maximum time interval
    pub fn with_max_interval(mut self, interval: Duration) -> Self {
        self.max_interval = Some(interval);
        self
    }

    /// Require monotonic timestamps
    pub fn require_monotonic(mut self) -> Self {
        self.require_monotonic = true;
        self
    }

    /// Disallow duplicate timestamps
    pub fn disallow_duplicates(mut self) -> Self {
        self.allow_duplicates = false;
        self
    }
}

impl Default for TimeConstraints {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for composing constraints
pub struct ConstraintBuilder {
    constraints: Vec<Constraint>,
}

impl ConstraintBuilder {
    /// Create a new constraint builder
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the builder
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add a range constraint
    pub fn range(self, min: f64, max: f64) -> Self {
        self.add(Constraint::Range { min, max })
    }

    /// Add a pattern constraint
    pub fn pattern(self, pattern: &str) -> Self {
        self.add(Constraint::Pattern(pattern.to_string()))
    }

    /// Add a length constraint
    pub fn length(self, min: usize, max: usize) -> Self {
        self.add(Constraint::Length { min, max })
    }

    /// Add a not-null constraint
    pub fn not_null(self) -> Self {
        self.add(Constraint::NotNull)
    }

    /// Build an AND constraint from all added constraints
    pub fn and(self) -> Constraint {
        match self.constraints.len() {
            0 => panic!("Cannot create AND constraint with no constraints"),
            1 => self.constraints.into_iter().next().unwrap(),
            _ => Constraint::And(self.constraints),
        }
    }

    /// Build an OR constraint from all added constraints
    pub fn or(self) -> Constraint {
        match self.constraints.len() {
            0 => panic!("Cannot create OR constraint with no constraints"),
            1 => self.constraints.into_iter().next().unwrap(),
            _ => Constraint::Or(self.constraints),
        }
    }
}

impl Default for ConstraintBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Constraint {
    /// Create a constraint that requires all of the given constraints to pass
    pub fn all_of(constraints: Vec<Constraint>) -> Self {
        Constraint::And(constraints)
    }

    /// Create a constraint that requires at least one of the given constraints to pass
    pub fn any_of(constraints: Vec<Constraint>) -> Self {
        Constraint::Or(constraints)
    }

    /// Create a constraint that requires the given constraint to not pass
    #[allow(clippy::should_implement_trait)]
    pub fn not(constraint: Constraint) -> Self {
        Constraint::Not(Box::new(constraint))
    }

    /// Create a conditional constraint
    pub fn if_then(
        condition: Constraint,
        then_constraint: Constraint,
        else_constraint: Option<Constraint>,
    ) -> Self {
        Constraint::If {
            condition: Box::new(condition),
            then_constraint: Box::new(then_constraint),
            else_constraint: else_constraint.map(Box::new),
        }
    }

    /// Chain this constraint with another using AND logic
    pub fn and(self, other: Constraint) -> Self {
        match self {
            Constraint::And(mut constraints) => {
                constraints.push(other);
                Constraint::And(constraints)
            }
            _ => Constraint::And(vec![self, other]),
        }
    }

    /// Chain this constraint with another using OR logic
    pub fn or(self, other: Constraint) -> Self {
        match self {
            Constraint::Or(mut constraints) => {
                constraints.push(other);
                Constraint::Or(constraints)
            }
            _ => Constraint::Or(vec![self, other]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_constraint() {
        let constraint = Constraint::Range {
            min: 0.0,
            max: 100.0,
        };
        match constraint {
            Constraint::Range { min, max } => {
                assert_eq!(min, 0.0);
                assert_eq!(max, 100.0);
            }
            _ => panic!("Expected Range constraint"),
        }
    }

    #[test]
    fn test_statistical_constraints() {
        let constraints = StatisticalConstraints::new()
            .with_mean_range(0.0, 10.0)
            .with_std_range(1.0, 5.0)
            .with_distribution("normal");

        assert_eq!(constraints.min_mean, Some(0.0));
        assert_eq!(constraints.max_mean, Some(10.0));
        assert_eq!(constraints.min_std, Some(1.0));
        assert_eq!(constraints.max_std, Some(5.0));
        assert_eq!(
            constraints.expected_distribution,
            Some("normal".to_string())
        );
    }

    #[test]
    fn testshape_constraints() {
        let constraints = ShapeConstraints::new()
            .with_dimensions(vec![Some(10), Some(20)])
            .with_element_range(100, 500)
            .require_square();

        assert_eq!(constraints.dimensions, vec![Some(10), Some(20)]);
        assert_eq!(constraints.min_elements, Some(100));
        assert_eq!(constraints.max_elements, Some(500));
        assert!(constraints.require_square);
    }

    #[test]
    fn test_constraint_builder() {
        // Test AND composition
        let constraint = ConstraintBuilder::new().range(0.0, 100.0).not_null().and();

        match constraint {
            Constraint::And(constraints) => {
                assert_eq!(constraints.len(), 2);
            }
            _ => panic!("Expected And constraint"),
        }

        // Test OR composition
        let constraint = ConstraintBuilder::new()
            .pattern("^[a-z]+$")
            .pattern("^[A-Z]+$")
            .or();

        match constraint {
            Constraint::Or(constraints) => {
                assert_eq!(constraints.len(), 2);
            }
            _ => panic!("Expected Or constraint"),
        }
    }

    #[test]
    fn test_constraint_chaining() {
        // Test AND chaining
        let constraint = Constraint::Range {
            min: 0.0,
            max: 100.0,
        }
        .and(Constraint::NotNull);

        match constraint {
            Constraint::And(constraints) => {
                assert_eq!(constraints.len(), 2);
            }
            _ => panic!("Expected And constraint"),
        }

        // Test OR chaining
        let constraint = Constraint::Pattern("^[a-z]+$".to_string())
            .or(Constraint::Pattern("^[A-Z]+$".to_string()));

        match constraint {
            Constraint::Or(constraints) => {
                assert_eq!(constraints.len(), 2);
            }
            _ => panic!("Expected Or constraint"),
        }
    }

    #[test]
    fn test_composite_constraints() {
        // Test NOT constraint
        let constraint = Constraint::not(Constraint::Pattern("forbidden".to_string()));
        match constraint {
            Constraint::Not(_) => {}
            _ => panic!("Expected Not constraint"),
        }

        // Test IF-THEN constraint
        let constraint = Constraint::if_then(
            Constraint::NotNull,
            Constraint::Range {
                min: 0.0,
                max: 100.0,
            },
            Some(Constraint::Pattern("N/A".to_string())),
        );

        match constraint {
            Constraint::If {
                condition: _,
                then_constraint: _,
                else_constraint,
            } => {
                assert!(else_constraint.is_some());
            }
            _ => panic!("Expected If constraint"),
        }
    }

    #[test]
    fn test_complex_composition() {
        // Test complex nested constraints
        let age_constraint = Constraint::all_of(vec![
            Constraint::Range {
                min: 0.0,
                max: 150.0,
            },
            Constraint::NotNull,
        ]);

        let name_constraint = Constraint::any_of(vec![
            Constraint::Pattern("^[A-Za-z ]+$".to_string()),
            Constraint::Pattern("^[\\p{L} ]+$".to_string()),
        ]);

        // Combine constraints
        let combined = Constraint::And(vec![age_constraint, name_constraint]);

        match combined {
            Constraint::And(constraints) => {
                assert_eq!(constraints.len(), 2);
            }
            _ => panic!("Expected And constraint"),
        }
    }

    #[test]
    fn test_time_constraints() {
        let constraints = TimeConstraints::new()
            .with_min_interval(Duration::from_secs(1))
            .with_max_interval(Duration::from_secs(60))
            .require_monotonic()
            .disallow_duplicates();

        assert_eq!(constraints.min_interval, Some(Duration::from_secs(1)));
        assert_eq!(constraints.max_interval, Some(Duration::from_secs(60)));
        assert!(constraints.require_monotonic);
        assert!(!constraints.allow_duplicates);
    }

    #[test]
    fn test_constraint_builder_edge_cases() {
        // Test single constraint with and()
        let constraint = ConstraintBuilder::new().range(0.0, 100.0).and();

        match constraint {
            Constraint::Range { min, max } => {
                assert_eq!(min, 0.0);
                assert_eq!(max, 100.0);
            }
            _ => panic!("Expected Range constraint, not And"),
        }

        // Test empty builder should panic
        let result = std::panic::catch_unwind(|| ConstraintBuilder::new().and());
        assert!(result.is_err());

        // Test builder with all constraint types
        let constraint = ConstraintBuilder::new()
            .range(0.0, 100.0)
            .pattern("^[A-Z]+$")
            .length(5, 10)
            .not_null()
            .and();

        match constraint {
            Constraint::And(constraints) => {
                assert_eq!(constraints.len(), 4);
            }
            _ => panic!("Expected And constraint"),
        }
    }

    #[test]
    fn test_nested_constraint_composition() {
        // Test deep nesting
        let inner = Constraint::Range {
            min: 0.0,
            max: 50.0,
        };
        let middle = Constraint::And(vec![inner, Constraint::NotNull]);
        let outer = Constraint::Or(vec![middle, Constraint::Pattern(special.to_string())]);
        let complex = Constraint::Not(Box::new(outer));

        match complex {
            Constraint::Not(inner) => match inner.as_ref() {
                Constraint::Or(constraints) => {
                    assert_eq!(constraints.len(), 2);
                }
                _ => panic!("Expected Or constraint"),
            },
            _ => panic!("Expected Not constraint"),
        }
    }

    #[test]
    fn test_constraint_equality() {
        let c1 = Constraint::Range {
            min: 0.0,
            max: 100.0,
        };
        let c2 = Constraint::Range {
            min: 0.0,
            max: 100.0,
        };
        let c3 = Constraint::Range {
            min: 0.0,
            max: 200.0,
        };

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);

        let and1 = Constraint::And(vec![c1.clone(), Constraint::NotNull]);
        let and2 = Constraint::And(vec![c2.clone(), Constraint::NotNull]);
        assert_eq!(and1, and2);
    }

    #[test]
    fn test_array_validation_constraints() {
        let constraints = ArrayValidationConstraints::new()
            .withshape(vec![10, 20])
            .with_fieldname(test_array)
            .check_numeric_quality()
            .check_performance();

        assert_eq!(constraints.expectedshape, Some(vec![10, 20]));
        assert_eq!(constraints.fieldname, Some(test_array.to_string()));
        assert!(constraints.check_numeric_quality);
        assert!(constraints.check_performance);
    }

    #[test]
    fn test_sparse_format() {
        let format = SparseFormat::CSR;
        assert_eq!(format, SparseFormat::CSR);
    }
}
