//! Symbolic expression tree.
//!
//! The [`Expr`] enum represents a symbolic mathematical expression as an immutable
//! tree of nodes. Arithmetic operators (`+`, `-`, `*`, `/`, unary `-`) are overloaded
//! so that expressions compose naturally.
//!
//! # Example
//! ```
//! use scirs2_symbolic::Expr;
//!
//! let x = Expr::var("x");
//! let f = x.clone() * x.clone() + Expr::from(3.0) * x.clone(); // x² + 3x
//! println!("{f}"); // "(((x * x) + (3 * x)))"
//! ```

/// A symbolic mathematical expression tree node.
///
/// Each variant is either a leaf (`Const`, `Var`) or an operator applied to one or
/// two sub-expressions. All nodes are heap-allocated via `Box` to keep the enum size
/// bounded.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// A numeric constant (f64 value).
    Const(f64),
    /// A named variable (e.g. `"x"`, `"theta"`).
    Var(String),
    /// Addition: `a + b`.
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction: `a - b`.
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication: `a * b`.
    Mul(Box<Expr>, Box<Expr>),
    /// Division: `a / b`.
    Div(Box<Expr>, Box<Expr>),
    /// Exponentiation: `base ^ exponent`.
    Pow(Box<Expr>, Box<Expr>),
    /// Negation: `-a`.
    Neg(Box<Expr>),
    /// Sine function: `sin(a)`.
    Sin(Box<Expr>),
    /// Cosine function: `cos(a)`.
    Cos(Box<Expr>),
    /// Tangent function: `tan(a)`.
    Tan(Box<Expr>),
    /// Natural exponential: `exp(a)` = eᵃ.
    Exp(Box<Expr>),
    /// Natural logarithm: `ln(a)`.
    Ln(Box<Expr>),
    /// Square root: `√a`.
    Sqrt(Box<Expr>),
    /// Absolute value: `|a|`.
    Abs(Box<Expr>),
}

impl Expr {
    /// Create a variable node.
    pub fn var(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    /// The additive identity (0).
    pub fn zero() -> Self {
        Expr::Const(0.0)
    }

    /// The multiplicative identity (1).
    pub fn one() -> Self {
        Expr::Const(1.0)
    }

    /// Returns `true` if this node is a `Const`.
    pub fn is_const(&self) -> bool {
        matches!(self, Expr::Const(_))
    }

    /// Returns `true` if this node is the constant `0.0`.
    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Const(v) if *v == 0.0)
    }

    /// Returns `true` if this node is the constant `1.0`.
    pub fn is_one(&self) -> bool {
        matches!(self, Expr::Const(v) if *v == 1.0)
    }

    /// Returns the constant value if this is a `Const` node.
    pub fn as_const(&self) -> Option<f64> {
        if let Expr::Const(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Build `sin(self)`.
    pub fn sin(self) -> Self {
        Expr::Sin(Box::new(self))
    }

    /// Build `cos(self)`.
    pub fn cos(self) -> Self {
        Expr::Cos(Box::new(self))
    }

    /// Build `tan(self)`.
    pub fn tan(self) -> Self {
        Expr::Tan(Box::new(self))
    }

    /// Build `exp(self)`.
    pub fn exp(self) -> Self {
        Expr::Exp(Box::new(self))
    }

    /// Build `ln(self)`.
    pub fn ln(self) -> Self {
        Expr::Ln(Box::new(self))
    }

    /// Build `sqrt(self)`.
    pub fn sqrt(self) -> Self {
        Expr::Sqrt(Box::new(self))
    }

    /// Build `abs(self)`.
    pub fn abs(self) -> Self {
        Expr::Abs(Box::new(self))
    }

    /// Build `self ^ exponent`.
    pub fn pow(self, exponent: Expr) -> Self {
        Expr::Pow(Box::new(self), Box::new(exponent))
    }

    /// Count the total number of nodes in the expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::Var(_) => 1,
            Expr::Neg(e)
            | Expr::Sin(e)
            | Expr::Cos(e)
            | Expr::Tan(e)
            | Expr::Exp(e)
            | Expr::Ln(e)
            | Expr::Sqrt(e)
            | Expr::Abs(e) => 1 + e.node_count(),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => 1 + a.node_count() + b.node_count(),
        }
    }

    /// Collect all variable names referenced in this expression.
    pub fn variables(&self) -> std::collections::BTreeSet<String> {
        let mut vars = std::collections::BTreeSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, acc: &mut std::collections::BTreeSet<String>) {
        match self {
            Expr::Var(name) => {
                acc.insert(name.clone());
            }
            Expr::Const(_) => {}
            Expr::Neg(e)
            | Expr::Sin(e)
            | Expr::Cos(e)
            | Expr::Tan(e)
            | Expr::Exp(e)
            | Expr::Ln(e)
            | Expr::Sqrt(e)
            | Expr::Abs(e) => {
                e.collect_vars(acc);
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                a.collect_vars(acc);
                b.collect_vars(acc);
            }
        }
    }

    /// Returns `true` if the expression contains a reference to `var`.
    pub fn contains_var(&self, var: &str) -> bool {
        match self {
            Expr::Var(name) => name == var,
            Expr::Const(_) => false,
            Expr::Neg(e)
            | Expr::Sin(e)
            | Expr::Cos(e)
            | Expr::Tan(e)
            | Expr::Exp(e)
            | Expr::Ln(e)
            | Expr::Sqrt(e)
            | Expr::Abs(e) => e.contains_var(var),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => a.contains_var(var) || b.contains_var(var),
        }
    }
}

// --- From conversions ---

impl From<f64> for Expr {
    fn from(v: f64) -> Self {
        Expr::Const(v)
    }
}

impl From<f32> for Expr {
    fn from(v: f32) -> Self {
        Expr::Const(v as f64)
    }
}

impl From<i32> for Expr {
    fn from(v: i32) -> Self {
        Expr::Const(v as f64)
    }
}

impl From<i64> for Expr {
    fn from(v: i64) -> Self {
        Expr::Const(v as f64)
    }
}

impl From<u32> for Expr {
    fn from(v: u32) -> Self {
        Expr::Const(v as f64)
    }
}

// --- Arithmetic operators ---

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::Div(Box::new(self), Box::new(rhs))
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}
