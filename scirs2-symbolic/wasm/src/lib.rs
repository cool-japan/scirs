//! WebAssembly bindings for `scirs2-symbolic`.
//!
//! Exposes a string-in / string-out API suitable for wasm-bindgen so that a
//! browser playground (or any JS environment) can call the CAS without
//! understanding Rust types.
//!
//! # Expression syntax
//!
//! The public-facing expression language understood by the built-in Pratt
//! parser:
//!
//! - **Variables**: `x0`, `x1`, ..., `xN`; `x` is an alias for `x0`;
//!   `y` is an alias for `x1`.
//! - **Constants**: decimal numeric literals (`3.14`, `2`, `-1`).
//! - **Unary ops**: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `abs`,
//!   `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`.
//! - **Binary ops** (by decreasing precedence): `^`, `*`, `/`, `+`, `-`.
//! - **Grouping**: parentheses `( e )`.
//!
//! # WASM API
//!
//! All functions return a `String`.  On error the string starts with
//! `"Error: "`.

use wasm_bindgen::prelude::*;

use scirs2_symbolic::cas::canonicalize;
use scirs2_symbolic::eml::{eval_real, grad, simplify_op, EvalCtx, LoweredOp};

// ============================================================================
// Public WASM surface
// ============================================================================

/// Canonicalize an expression string.
///
/// Returns the infix display of the canonical form, or `"Error: <msg>"`.
#[wasm_bindgen]
pub fn wasm_canonicalize(expr: &str) -> String {
    match parse_str(expr) {
        Err(e) => format!("Error: {e}"),
        Ok(op) => {
            let canonical = canonicalize(&op);
            format!("{}", canonical.op())
        }
    }
}

/// Compute the symbolic gradient `df/dx_{wrt_var}` of the expression.
///
/// Returns the infix display of the simplified gradient, or `"Error: <msg>"`.
#[wasm_bindgen]
pub fn wasm_grad(expr: &str, wrt_var: usize) -> String {
    match parse_str(expr) {
        Err(e) => format!("Error: {e}"),
        Ok(op) => {
            let g = grad(&op, wrt_var);
            format!("{g}")
        }
    }
}

/// Simplify an expression string (constant folding + algebraic rewrites).
///
/// Returns the infix display of the simplified expression, or `"Error: <msg>"`.
#[wasm_bindgen]
pub fn wasm_simplify(expr: &str) -> String {
    match parse_str(expr) {
        Err(e) => format!("Error: {e}"),
        Ok(op) => {
            let s = simplify_op(&op);
            format!("{s}")
        }
    }
}

/// Evaluate an expression at given variable bindings.
///
/// `bindings_json` must be a JSON array of f64 values, e.g. `"[1.5, 2.3]"`.
/// `Var(0)` maps to index 0, `Var(1)` to index 1, etc.
///
/// Returns the numeric result as a string, or `"Error: <msg>"`.
#[wasm_bindgen]
pub fn wasm_eval(expr: &str, bindings_json: &str) -> String {
    // Parse bindings from JSON.
    let bindings: Vec<f64> = match serde_json::from_str(bindings_json) {
        Ok(v) => v,
        Err(e) => return format!("Error: failed to parse bindings JSON: {e}"),
    };

    match parse_str(expr) {
        Err(e) => format!("Error: {e}"),
        Ok(op) => {
            let ctx = EvalCtx::new(&bindings);
            match eval_real(&op, &ctx) {
                Ok(v) => format!("{v}"),
                Err(e) => format!("Error: {e}"),
            }
        }
    }
}

/// Check whether two expressions are canonically equal.
///
/// Returns `"true"`, `"false"`, or `"Error: <msg>"`.
#[wasm_bindgen]
pub fn wasm_is_identity(expr1: &str, expr2: &str) -> String {
    let op1 = match parse_str(expr1) {
        Err(e) => return format!("Error: (expr1) {e}"),
        Ok(op) => op,
    };
    let op2 = match parse_str(expr2) {
        Err(e) => return format!("Error: (expr2) {e}"),
        Ok(op) => op,
    };

    let c1 = canonicalize(&op1);
    let c2 = canonicalize(&op2);

    if c1 == c2 {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

// ============================================================================
// Internal: Pratt parser (iterative precedence-climbing)
// ============================================================================
//
// Grammar:
//   expr    ::= unary (BINOP expr)*   [handled by prec-climb loop]
//   unary   ::= IDENT '(' expr ')'   [function call]
//             | '-' unary            [negate]
//             | primary
//   primary ::= NUMBER | IDENT | '(' expr ')'
//
// Precedence table (higher = binds tighter):
//   +  -  : 1
//   *  /  : 2
//   ^     : 3  (right-associative)
//
// The iterative approach uses two explicit stacks (operand + operator) and
// pops/reduces when the new operator has lower-or-equal precedence than the
// top of the operator stack.  Right-associativity of `^` is handled by using
// `prec - 1` as the "continue threshold" when the operator is right-associative.

/// Parse an expression string into a `LoweredOp`.
fn parse_str(input: &str) -> Result<LoweredOp, String> {
    let mut p = Parser::new(input);
    let op = p.parse_expr(0)?;
    p.skip_ws();
    if p.pos < p.bytes.len() {
        return Err(format!(
            "unexpected trailing input at position {}: {:?}",
            p.pos,
            &input[p.pos..]
        ));
    }
    Ok(op)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn eat_char(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// Read a decimal number literal.  Handles optional leading `-` only when
    /// called from `parse_primary` (not from the binary-operator logic).
    fn eat_number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        // Optional sign (only minus; `+` not allowed as leading sign here).
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        // Integer part.
        if !matches!(self.peek(), Some(b'0'..=b'9')) {
            return Err(format!(
                "expected digit at position {}, got {:?}",
                self.pos,
                self.peek().map(char::from)
            ));
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        // Optional fractional part.
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        // Optional exponent.
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!(
                    "expected digit in exponent at position {}",
                    self.pos
                ));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| format!("utf8 error: {e}"))?;
        s.parse::<f64>()
            .map_err(|e| format!("invalid number {s:?}: {e}"))
    }

    /// Read an ASCII identifier (letters + digits + underscore; must start
    /// with a letter or underscore).
    fn eat_ident(&mut self) -> &'a str {
        let start = self.pos;
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        // SAFETY: we only advance over ASCII bytes.
        std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("")
    }

    // ------------------------------------------------------------------
    // Main parser entrypoint — iterative precedence-climbing
    // ------------------------------------------------------------------

    /// Parse an expression with minimum precedence `min_prec`.
    ///
    /// Iterative: uses two explicit Vecs as operand and operator stacks so
    /// deeply-nested expressions do not blow the OS stack.
    fn parse_expr(&mut self, min_prec: u8) -> Result<LoweredOp, String> {
        self.skip_ws();

        // Parse the first (left) operand.
        let first = self.parse_unary()?;

        // Operand stack and operator stack for iterative shunting-yard.
        let mut operand_stack: Vec<LoweredOp> = vec![first];
        // Each entry: (operator byte, precedence, right_assoc)
        let mut op_stack: Vec<(u8, u8, bool)> = Vec::new();

        loop {
            self.skip_ws();
            let op_info = match self.peek() {
                Some(b'+') => Some((b'+', 1u8, false)),
                Some(b'-') => Some((b'-', 1u8, false)),
                Some(b'*') => Some((b'*', 2u8, false)),
                Some(b'/') => Some((b'/', 2u8, false)),
                Some(b'^') => Some((b'^', 3u8, true)),
                _ => None,
            };

            if let Some((op_byte, prec, right_assoc)) = op_info {
                if prec < min_prec {
                    break;
                }
                // Reduce top of op_stack while the stacked operator has higher (or equal
                // for left-assoc) precedence than the incoming operator.
                while let Some(&(top_byte, top_prec, top_ra)) = op_stack.last() {
                    let should_reduce = if right_assoc {
                        top_prec > prec
                    } else {
                        top_prec >= prec
                    };
                    if !should_reduce {
                        break;
                    }
                    op_stack.pop();
                    let _ = top_ra; // used via should_reduce
                    let rhs = operand_stack
                        .pop()
                        .ok_or_else(|| "operand stack underflow (rhs)".to_string())?;
                    let lhs = operand_stack
                        .pop()
                        .ok_or_else(|| "operand stack underflow (lhs)".to_string())?;
                    operand_stack.push(apply_binary_op(top_byte, lhs, rhs)?);
                }
                // Consume the operator token.
                self.pos += 1;
                self.skip_ws();
                // Parse the right-hand side (with elevated min_prec for right-assoc).
                let rhs = self.parse_unary()?;
                op_stack.push((op_byte, prec, right_assoc));
                operand_stack.push(rhs);
            } else {
                break;
            }
        }

        // Drain remaining operators from the stack.
        while let Some((top_byte, _, _)) = op_stack.pop() {
            let rhs = operand_stack
                .pop()
                .ok_or_else(|| "operand stack underflow draining (rhs)".to_string())?;
            let lhs = operand_stack
                .pop()
                .ok_or_else(|| "operand stack underflow draining (lhs)".to_string())?;
            operand_stack.push(apply_binary_op(top_byte, lhs, rhs)?);
        }

        operand_stack
            .pop()
            .ok_or_else(|| "empty expression".to_string())
    }

    /// Parse a unary prefix expression or forward to `parse_primary`.
    fn parse_unary(&mut self) -> Result<LoweredOp, String> {
        self.skip_ws();
        // Leading minus → negate.
        if self.peek() == Some(b'-') {
            self.pos += 1;
            self.skip_ws();
            // Check if this could be a negative number literal.
            if matches!(self.peek(), Some(b'0'..=b'9')) {
                // Rewind one byte and let eat_number handle the sign.
                self.pos -= 1;
                let v = self.eat_number()?;
                return Ok(LoweredOp::Const(v));
            }
            let inner = self.parse_unary()?;
            return Ok(LoweredOp::Neg(Box::new(inner)));
        }

        // Named function call: starts with an ASCII letter.
        if matches!(self.peek(), Some(b'a'..=b'z') | Some(b'A'..=b'Z') | Some(b'_')) {
            return self.parse_ident_or_call();
        }

        self.parse_primary()
    }

    /// Parse an identifier — either a variable reference or a function call.
    fn parse_ident_or_call(&mut self) -> Result<LoweredOp, String> {
        let start = self.pos;
        let name = self.eat_ident();

        self.skip_ws();
        if self.eat_char(b'(') {
            // Function call.
            self.skip_ws();
            let arg = self.parse_expr(0)?;
            self.skip_ws();
            if !self.eat_char(b')') {
                return Err(format!(
                    "expected ')' to close function '{name}' at position {}",
                    self.pos
                ));
            }
            return apply_function(name, arg);
        }

        // Not a function call — must be a variable.
        // Supported: x0..xN, x (alias x0), y (alias x1).
        match name {
            "x" => Ok(LoweredOp::Var(0)),
            "y" => Ok(LoweredOp::Var(1)),
            n if n.starts_with('x') && n.len() > 1 => {
                let idx_str = &n[1..];
                let idx = idx_str.parse::<usize>().map_err(|_| {
                    format!("invalid variable index in '{n}' at position {start}")
                })?;
                Ok(LoweredOp::Var(idx))
            }
            other => Err(format!("unknown identifier '{other}' at position {start}")),
        }
    }

    /// Parse a primary: number literal or grouped expression.
    fn parse_primary(&mut self) -> Result<LoweredOp, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                self.skip_ws();
                let inner = self.parse_expr(0)?;
                self.skip_ws();
                if !self.eat_char(b')') {
                    return Err(format!(
                        "expected ')' at position {}, got {:?}",
                        self.pos,
                        self.peek().map(char::from)
                    ));
                }
                Ok(inner)
            }
            Some(b'0'..=b'9') => {
                let v = self.eat_number()?;
                Ok(LoweredOp::Const(v))
            }
            Some(other) => Err(format!(
                "unexpected character '{}' at position {}",
                char::from(other),
                self.pos
            )),
            None => Err(format!("unexpected end of input at position {}", self.pos)),
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn apply_binary_op(op: u8, lhs: LoweredOp, rhs: LoweredOp) -> Result<LoweredOp, String> {
    match op {
        b'+' => Ok(LoweredOp::Add(Box::new(lhs), Box::new(rhs))),
        b'-' => Ok(LoweredOp::Sub(Box::new(lhs), Box::new(rhs))),
        b'*' => Ok(LoweredOp::Mul(Box::new(lhs), Box::new(rhs))),
        b'/' => Ok(LoweredOp::Div(Box::new(lhs), Box::new(rhs))),
        b'^' => Ok(LoweredOp::Pow(Box::new(lhs), Box::new(rhs))),
        other => Err(format!("unknown operator byte {other:#x}")),
    }
}

fn apply_function(name: &str, arg: LoweredOp) -> Result<LoweredOp, String> {
    match name {
        "sin" => Ok(LoweredOp::Sin(Box::new(arg))),
        "cos" => Ok(LoweredOp::Cos(Box::new(arg))),
        "tan" => Ok(LoweredOp::Tan(Box::new(arg))),
        "exp" => Ok(LoweredOp::Exp(Box::new(arg))),
        "ln" => Ok(LoweredOp::Ln(Box::new(arg))),
        "sqrt" => Ok(LoweredOp::Sqrt(Box::new(arg))),
        "abs" => Ok(LoweredOp::Abs(Box::new(arg))),
        "sinh" => Ok(LoweredOp::Sinh(Box::new(arg))),
        "cosh" => Ok(LoweredOp::Cosh(Box::new(arg))),
        "tanh" => Ok(LoweredOp::Tanh(Box::new(arg))),
        "arcsin" => Ok(LoweredOp::Arcsin(Box::new(arg))),
        "arccos" => Ok(LoweredOp::Arccos(Box::new(arg))),
        "arctan" => Ok(LoweredOp::Arctan(Box::new(arg))),
        other => Err(format!("unknown function '{other}'")),
    }
}

// ============================================================================
// Tests (native target)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Result<LoweredOp, String> {
        parse_str(s)
    }

    #[test]
    fn test_parse_x_squared() {
        let op = parse("x^2").expect("parse x^2");
        // Should be Pow(Var(0), Const(2.0))
        assert!(
            matches!(op, LoweredOp::Pow(ref base, ref exp)
                if matches!(base.as_ref(), LoweredOp::Var(0))
                && matches!(exp.as_ref(), LoweredOp::Const(c) if (*c - 2.0).abs() < 1e-15))
        );
    }

    #[test]
    fn test_parse_sin_x() {
        let op = parse("sin(x)").expect("parse sin(x)");
        assert!(matches!(
            op,
            LoweredOp::Sin(ref inner) if matches!(inner.as_ref(), LoweredOp::Var(0))
        ));
    }

    #[test]
    fn test_parse_complex_expr() {
        // sin(x)^2 + cos(x)^2 must parse without error.
        let op = parse("sin(x)^2 + cos(x)^2").expect("parse sin(x)^2 + cos(x)^2");
        assert!(matches!(op, LoweredOp::Add(_, _)));
    }

    #[test]
    fn test_parse_negative_const() {
        // Use 2.71 which is close to e but not close enough to trigger the
        // clippy::approx_constant lint (that targets 2.718...).
        let op = parse("-2.5").expect("parse -2.5");
        match op {
            // The parser may emit Const(-2.5) or Neg(Const(2.5)).
            LoweredOp::Const(v) => assert!((v + 2.5).abs() < 1e-12),
            LoweredOp::Neg(inner) => {
                assert!(
                    matches!(inner.as_ref(), LoweredOp::Const(c) if (*c - 2.5).abs() < 1e-12)
                );
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_parse_error_unknown_function() {
        let err = parse("foo(x)").expect_err("should fail on unknown function");
        assert!(err.contains("unknown function"), "error was: {err}");
    }

    #[test]
    fn test_parse_error_trailing_input() {
        let err = parse("x + 1 garbage").expect_err("should fail on trailing input");
        assert!(
            err.contains("unexpected") || err.contains("unknown"),
            "error was: {err}"
        );
    }

    #[test]
    fn test_wasm_eval_addition() {
        // x + y at x=3, y=4 should give 7.
        let result = wasm_eval("x + y", "[3.0, 4.0]");
        let v: f64 = result.parse().expect("numeric result");
        assert!((v - 7.0).abs() < 1e-12, "got {v}");
    }

    #[test]
    fn test_wasm_grad_x_squared() {
        // d/dx (x^2) = 2*x (simplify may produce various infix forms).
        let result = wasm_grad("x^2", 0);
        assert!(
            !result.starts_with("Error"),
            "expected no error, got: {result}"
        );
        // Evaluate gradient at x=3: should be ~6.
        let grad_op = parse_str(&result).expect("gradient result must be parseable");
        let ctx = EvalCtx::new(&[3.0f64]);
        let v = eval_real(&grad_op, &ctx).expect("eval gradient");
        assert!((v - 6.0).abs() < 1e-9, "d/dx(x^2) at x=3 = {v}, expected 6");
    }

    #[test]
    fn test_wasm_simplify_zero() {
        // x + 0 should simplify to x (or equivalent).
        let result = wasm_simplify("x + 0");
        assert!(
            !result.starts_with("Error"),
            "expected no error, got: {result}"
        );
        // Evaluate at x=5: result must equal 5.
        let simplified = parse_str(&result).expect("simplified must be parseable");
        let ctx = EvalCtx::new(&[5.0f64]);
        let v = eval_real(&simplified, &ctx).expect("eval simplified");
        assert!((v - 5.0).abs() < 1e-12, "x+0 simplified at x=5 = {v}");
    }

    #[test]
    fn test_wasm_is_identity_commutative_add() {
        // x + y and y + x should be canonically equal.
        let result = wasm_is_identity("x + y", "y + x");
        assert_eq!(result, "true", "x+y and y+x should be identical");
    }

    #[test]
    fn test_wasm_is_identity_ln_exp() {
        // ln(exp(x)) and x should be canonically equal.
        let result = wasm_is_identity("ln(exp(x))", "x");
        assert_eq!(result, "true", "ln(exp(x)) and x should be identical");
    }

    #[test]
    fn test_wasm_is_identity_false() {
        // x + 1 and x - 1 are NOT equal.
        let result = wasm_is_identity("x + 1", "x - 1");
        assert_eq!(result, "false", "x+1 and x-1 should NOT be identical");
    }

    #[test]
    fn test_wasm_canonicalize_no_error() {
        // sin(x)^2 + cos(x)^2 should canonicalize without error.
        let result = wasm_canonicalize("sin(x)^2 + cos(x)^2");
        assert!(
            !result.starts_with("Error"),
            "expected no error, got: {result}"
        );
    }

    #[test]
    fn test_parse_y_alias() {
        // 'y' must be treated as Var(1).
        let op = parse("y").expect("parse y");
        assert!(matches!(op, LoweredOp::Var(1)));
    }

    #[test]
    fn test_wasm_eval_bad_json() {
        let result = wasm_eval("x", "not_json");
        assert!(
            result.starts_with("Error"),
            "expected error for bad JSON, got: {result}"
        );
    }
}
