// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Just-In-Time (JIT) compilation support for array operations.
//!
//! This module provides functionality for JIT-compiling operations on arrays,
//! allowing for faster execution of custom operations.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, RwLock};

use crate::array_protocol::{
    ArrayFunction, ArrayProtocol, JITArray, JITFunction, JITFunctionFactory,
};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// JIT compilation backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JITBackend {
    /// LLVM backend
    LLVM,

    /// Cranelift backend
    Cranelift,

    /// WebAssembly backend
    WASM,

    /// Custom backend
    Custom(TypeId),
}

impl Default for JITBackend {
    fn default() -> Self {
        Self::LLVM
    }
}

/// Configuration for JIT compilation
#[derive(Debug, Clone)]
pub struct JITConfig {
    /// The JIT backend to use
    pub backend: JITBackend,

    /// Whether to optimize the generated code
    pub optimize: bool,

    /// Optimization level (0-3)
    pub opt_level: usize,

    /// Whether to cache compiled functions
    pub use_cache: bool,

    /// Additional backend-specific options
    pub backend_options: HashMap<String, String>,
}

impl Default for JITConfig {
    fn default() -> Self {
        Self {
            backend: JITBackend::default(),
            optimize: true,
            opt_level: 2,
            use_cache: true,
            backend_options: HashMap::new(),
        }
    }
}

/// Type alias for the complex function type
pub type JITFunctionType = dyn Fn(&[Box<dyn Any>]) -> CoreResult<Box<dyn Any>> + Send + Sync;

/// A compiled JIT function
pub struct JITFunctionImpl {
    /// The source code of the function
    source: String,

    /// The compiled function. `Arc` (rather than `Box`) so `clone_box` can
    /// produce a function that is byte-for-byte identical in behavior by
    /// sharing the same underlying closure, instead of trying to
    /// reconstruct an equivalent one from scratch.
    function: Arc<JITFunctionType>,

    /// Information about the compilation
    compile_info: HashMap<String, String>,
}

impl Debug for JITFunctionImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JITFunctionImpl")
            .field("source", &self.source)
            .field("compile_info", &self.compile_info)
            .finish_non_exhaustive()
    }
}

impl JITFunctionImpl {
    /// Create a new JIT function.
    #[must_use]
    pub fn new(
        source: String,
        function: Box<JITFunctionType>,
        compile_info: HashMap<String, String>,
    ) -> Self {
        Self {
            source,
            function: Arc::from(function),
            compile_info,
        }
    }
}

impl JITFunction for JITFunctionImpl {
    fn evaluate(&self, args: &[Box<dyn Any>]) -> CoreResult<Box<dyn Any>> {
        (self.function)(args)
    }

    fn source(&self) -> String {
        self.source.clone()
    }

    fn compile_info(&self) -> HashMap<String, String> {
        self.compile_info.clone()
    }

    fn clone_box(&self) -> Box<dyn JITFunction> {
        Box::new(Self {
            source: self.source.clone(),
            function: Arc::clone(&self.function),
            compile_info: self.compile_info.clone(),
        })
    }
}

// ============================================================================
// A minimal expression parser + interpreter.
//
// Neither an LLVM nor a Cranelift toolchain is wired into this crate (Pure
// Rust policy, and pulling either in as a hard dependency of `scirs2-core`
// would be a large, separate undertaking), so `LLVMFunctionFactory` and
// `CraneliftFunctionFactory` below both fall back to interpreting the
// requested expression directly rather than JIT-compiling native code for
// it. This is a real (if unoptimized) implementation of "evaluate this
// expression for the given arguments" — not a stand-in that ignores its
// input — for a small but useful arithmetic grammar:
//
//   expr   := term (('+' | '-') term)*
//   term   := unary (('*' | '/') unary)*
//   unary  := '-' unary | power
//   power  := primary ('^' unary)?          (right-associative)
//   primary:= number | ident ('(' expr (',' expr)* ')')? | '(' expr ')'
//
// with `sin/cos/tan/exp/ln/sqrt/abs` (1 argument) and `min/max` (2
// arguments) as built-in function calls.
//
// `JITFunction::evaluate`'s `args: &[Box<dyn Any>]` are `f64` values bound
// positionally to the expression's free variables in order of first
// occurrence (left to right) — e.g. for `"x + y"`, `args[0]` is `x` and
// `args[1]` is `y`.
// ============================================================================

/// An expression AST node.
#[derive(Debug, Clone)]
enum JITExpr {
    Number(f64),
    Var(String),
    Neg(Box<JITExpr>),
    Add(Box<JITExpr>, Box<JITExpr>),
    Sub(Box<JITExpr>, Box<JITExpr>),
    Mul(Box<JITExpr>, Box<JITExpr>),
    Div(Box<JITExpr>, Box<JITExpr>),
    Pow(Box<JITExpr>, Box<JITExpr>),
    Call(String, Vec<JITExpr>),
}

#[derive(Debug, Clone, PartialEq)]
enum JITToken {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
}

fn jit_tokenize(input: &str) -> Result<Vec<JITToken>, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit()
            || (c == '.' && chars.get(i + 1).is_some_and(char::is_ascii_digit))
        {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // Support a trailing exponent, e.g. "1e-3".
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    i = j;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let text: String = chars[start..i].iter().collect();
            let value: f64 = text
                .parse()
                .map_err(|_| format!("invalid number literal: '{text}'"))?;
            tokens.push(JITToken::Number(value));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(JITToken::Ident(chars[start..i].iter().collect()));
        } else {
            let token = match c {
                '+' => JITToken::Plus,
                '-' => JITToken::Minus,
                '*' => JITToken::Star,
                '/' => JITToken::Slash,
                '^' => JITToken::Caret,
                '(' => JITToken::LParen,
                ')' => JITToken::RParen,
                ',' => JITToken::Comma,
                other => return Err(format!("unexpected character: '{other}'")),
            };
            tokens.push(token);
            i += 1;
        }
    }
    Ok(tokens)
}

struct JITParser {
    tokens: Vec<JITToken>,
    pos: usize,
}

impl JITParser {
    fn peek(&self) -> Option<&JITToken> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<JITToken> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &JITToken) -> Result<(), String> {
        match self.advance() {
            Some(ref tok) if tok == expected => Ok(()),
            Some(other) => Err(format!("expected {expected:?}, found {other:?}")),
            None => Err(format!("expected {expected:?}, found end of expression")),
        }
    }

    fn parse_expr(&mut self) -> Result<JITExpr, String> {
        let mut node = self.parse_term()?;
        loop {
            match self.peek() {
                Some(JITToken::Plus) => {
                    self.advance();
                    node = JITExpr::Add(Box::new(node), Box::new(self.parse_term()?));
                }
                Some(JITToken::Minus) => {
                    self.advance();
                    node = JITExpr::Sub(Box::new(node), Box::new(self.parse_term()?));
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_term(&mut self) -> Result<JITExpr, String> {
        let mut node = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(JITToken::Star) => {
                    self.advance();
                    node = JITExpr::Mul(Box::new(node), Box::new(self.parse_unary()?));
                }
                Some(JITToken::Slash) => {
                    self.advance();
                    node = JITExpr::Div(Box::new(node), Box::new(self.parse_unary()?));
                }
                _ => break,
            }
        }
        Ok(node)
    }

    fn parse_unary(&mut self) -> Result<JITExpr, String> {
        if matches!(self.peek(), Some(JITToken::Minus)) {
            self.advance();
            Ok(JITExpr::Neg(Box::new(self.parse_unary()?)))
        } else {
            self.parse_power()
        }
    }

    fn parse_power(&mut self) -> Result<JITExpr, String> {
        let base = self.parse_primary()?;
        if matches!(self.peek(), Some(JITToken::Caret)) {
            self.advance();
            let exponent = self.parse_unary()?;
            Ok(JITExpr::Pow(Box::new(base), Box::new(exponent)))
        } else {
            Ok(base)
        }
    }

    fn parse_primary(&mut self) -> Result<JITExpr, String> {
        match self.advance() {
            Some(JITToken::Number(n)) => Ok(JITExpr::Number(n)),
            Some(JITToken::Ident(name)) => {
                if matches!(self.peek(), Some(JITToken::LParen)) {
                    self.advance();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(JITToken::RParen)) {
                        args.push(self.parse_expr()?);
                        while matches!(self.peek(), Some(JITToken::Comma)) {
                            self.advance();
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(&JITToken::RParen)?;
                    Ok(JITExpr::Call(name, args))
                } else {
                    Ok(JITExpr::Var(name))
                }
            }
            Some(JITToken::LParen) => {
                let inner = self.parse_expr()?;
                self.expect(&JITToken::RParen)?;
                Ok(inner)
            }
            Some(other) => Err(format!("unexpected token: {other:?}")),
            None => Err("unexpected end of expression".to_string()),
        }
    }
}

/// Parses `expression` into an AST, returning an error string describing
/// the syntax problem on failure (never panics on malformed input).
fn jit_parse_expression(expression: &str) -> Result<JITExpr, String> {
    let tokens = jit_tokenize(expression)?;
    let mut parser = JITParser { tokens, pos: 0 };
    let expr = parser.parse_expr()?;
    if parser.pos != parser.tokens.len() {
        return Err(format!(
            "unexpected trailing token: {:?}",
            parser.tokens[parser.pos]
        ));
    }
    Ok(expr)
}

/// Collects the expression's free variables in order of first (left-to-right)
/// occurrence — this fixes the positional order `evaluate`'s `args` are
/// bound in.
fn jit_collect_vars(
    expr: &JITExpr,
    order: &mut Vec<String>,
    seen: &mut std::collections::HashSet<String>,
) {
    match expr {
        JITExpr::Number(_) => {}
        JITExpr::Var(name) => {
            if seen.insert(name.clone()) {
                order.push(name.clone());
            }
        }
        JITExpr::Neg(inner) => jit_collect_vars(inner, order, seen),
        JITExpr::Add(a, b)
        | JITExpr::Sub(a, b)
        | JITExpr::Mul(a, b)
        | JITExpr::Div(a, b)
        | JITExpr::Pow(a, b) => {
            jit_collect_vars(a, order, seen);
            jit_collect_vars(b, order, seen);
        }
        JITExpr::Call(_, args) => {
            for a in args {
                jit_collect_vars(a, order, seen);
            }
        }
    }
}

/// Evaluates `expr` given a binding of variable name to value.
fn jit_eval(expr: &JITExpr, vars: &HashMap<String, f64>) -> Result<f64, String> {
    match expr {
        JITExpr::Number(n) => Ok(*n),
        JITExpr::Var(name) => vars
            .get(name)
            .copied()
            .ok_or_else(|| format!("undefined variable: '{name}'")),
        JITExpr::Neg(inner) => Ok(-jit_eval(inner, vars)?),
        JITExpr::Add(a, b) => Ok(jit_eval(a, vars)? + jit_eval(b, vars)?),
        JITExpr::Sub(a, b) => Ok(jit_eval(a, vars)? - jit_eval(b, vars)?),
        JITExpr::Mul(a, b) => Ok(jit_eval(a, vars)? * jit_eval(b, vars)?),
        JITExpr::Div(a, b) => Ok(jit_eval(a, vars)? / jit_eval(b, vars)?),
        JITExpr::Pow(a, b) => Ok(jit_eval(a, vars)?.powf(jit_eval(b, vars)?)),
        JITExpr::Call(name, argexprs) => {
            let values = argexprs
                .iter()
                .map(|e| jit_eval(e, vars))
                .collect::<Result<Vec<f64>, String>>()?;
            match (name.as_str(), values.as_slice()) {
                ("sin", [x]) => Ok(x.sin()),
                ("cos", [x]) => Ok(x.cos()),
                ("tan", [x]) => Ok(x.tan()),
                ("exp", [x]) => Ok(x.exp()),
                ("ln", [x]) => Ok(x.ln()),
                ("sqrt", [x]) => Ok(x.sqrt()),
                ("abs", [x]) => Ok(x.abs()),
                ("min", [a, b]) => Ok(a.min(*b)),
                ("max", [a, b]) => Ok(a.max(*b)),
                (other, vals) => Err(format!(
                    "unknown function or wrong number of arguments: {other}({n} args)",
                    n = vals.len()
                )),
            }
        }
    }
}

/// Parses `expression` and builds a real (interpreted) [`JITFunctionType`]
/// closure for it, used by both [`LLVMFunctionFactory`] and
/// [`CraneliftFunctionFactory`] as their fallback "JIT" backend (see the
/// module-level comment above). Returns a descriptive [`CoreError::JITError`]
/// for anything the small grammar doesn't cover, rather than silently
/// returning a placeholder value.
fn jit_make_interpreted_function(expression: &str) -> CoreResult<Box<JITFunctionType>> {
    let expr = jit_parse_expression(expression).map_err(|e| {
        CoreError::JITError(ErrorContext::new(format!(
            "failed to parse expression '{expression}': {e}"
        )))
    })?;

    let mut order = Vec::new();
    let mut seen = std::collections::HashSet::new();
    jit_collect_vars(&expr, &mut order, &mut seen);

    let function: Box<JITFunctionType> = Box::new(move |args: &[Box<dyn Any>]| {
        if args.len() < order.len() {
            return Err(CoreError::JITError(ErrorContext::new(format!(
                "expression requires {need} argument(s) ({vars}) but only {got} were provided",
                need = order.len(),
                vars = order.join(", "),
                got = args.len()
            ))));
        }
        let mut vars = HashMap::with_capacity(order.len());
        for (name, arg) in order.iter().zip(args.iter()) {
            let value = arg.downcast_ref::<f64>().copied().ok_or_else(|| {
                CoreError::JITError(ErrorContext::new(format!(
                    "argument for variable '{name}' must be an f64"
                )))
            })?;
            vars.insert(name.clone(), value);
        }
        let result = jit_eval(&expr, &vars).map_err(|e| {
            CoreError::JITError(ErrorContext::new(format!("evaluation error: {e}")))
        })?;
        Ok(Box::new(result) as Box<dyn Any>)
    });

    Ok(function)
}

/// A factory for creating JIT functions using the LLVM backend
pub struct LLVMFunctionFactory {
    /// Configuration for JIT compilation
    config: JITConfig,

    /// Cache of compiled functions
    cache: HashMap<String, Arc<dyn JITFunction>>,
}

impl LLVMFunctionFactory {
    /// Create a new LLVM function factory.
    pub fn new(config: JITConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    /// Compile a function for the given expression.
    ///
    /// No LLVM toolchain is linked into this crate (Pure Rust policy), so
    /// this interprets the expression directly instead — a real
    /// implementation of "evaluate this expression" for the grammar
    /// documented on [`jit_make_interpreted_function`], not a placeholder.
    fn compile(&self, expression: &str, array_typeid: TypeId) -> CoreResult<Arc<dyn JITFunction>> {
        let mut compile_info = HashMap::new();
        compile_info.insert("backend".to_string(), "LLVM".to_string());
        compile_info.insert("opt_level".to_string(), self.config.opt_level.to_string());
        compile_info.insert("array_type".to_string(), format!("{array_typeid:?}"));

        let source = expression.to_string();
        let function = jit_make_interpreted_function(expression)?;

        let jit_function = JITFunctionImpl::new(source, function, compile_info);

        Ok(Arc::new(jit_function))
    }
}

impl JITFunctionFactory for LLVMFunctionFactory {
    fn create_jit_function(
        &self,
        expression: &str,
        array_typeid: TypeId,
    ) -> CoreResult<Box<dyn JITFunction>> {
        // Check if the function is already in the cache
        if self.config.use_cache {
            let cache_key = format!("{expression}-{array_typeid:?}");
            if let Some(cached_fn) = self.cache.get(&cache_key) {
                return Ok(cached_fn.as_ref().clone_box());
            }
        }

        // Compile the function
        let jit_function = self.compile(expression, array_typeid)?;

        if self.config.use_cache {
            // Add the function to the cache
            let cache_key = format!("{expression}-{array_typeid:?}");
            // In a real implementation, we'd need to handle this in a thread-safe way
            // For now, we'll just clone the function
            let mut cache = self.cache.clone();
            cache.insert(cache_key, jit_function.clone());
        }

        // Clone the function and return it
        Ok(jit_function.as_ref().clone_box())
    }

    fn supports_array_type(&self, _array_typeid: TypeId) -> bool {
        // For simplicity, we'll say this factory supports all array types
        true
    }
}

/// A factory for creating JIT functions using the Cranelift backend
pub struct CraneliftFunctionFactory {
    /// Configuration for JIT compilation
    config: JITConfig,

    /// Cache of compiled functions
    cache: HashMap<String, Arc<dyn JITFunction>>,
}

impl CraneliftFunctionFactory {
    /// Create a new Cranelift function factory.
    pub fn new(config: JITConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    /// Compile a function for the given expression.
    ///
    /// No Cranelift toolchain is linked into this crate (Pure Rust policy),
    /// so this interprets the expression directly instead — a real
    /// implementation of "evaluate this expression" for the grammar
    /// documented on [`jit_make_interpreted_function`], not a placeholder.
    fn compile(&self, expression: &str, array_typeid: TypeId) -> CoreResult<Arc<dyn JITFunction>> {
        let mut compile_info = HashMap::new();
        compile_info.insert("backend".to_string(), "Cranelift".to_string());
        compile_info.insert("opt_level".to_string(), self.config.opt_level.to_string());
        compile_info.insert("array_type".to_string(), format!("{array_typeid:?}"));

        let source = expression.to_string();
        let function = jit_make_interpreted_function(expression)?;

        let jit_function = JITFunctionImpl::new(source, function, compile_info);

        Ok(Arc::new(jit_function))
    }
}

impl JITFunctionFactory for CraneliftFunctionFactory {
    fn create_jit_function(
        &self,
        expression: &str,
        array_typeid: TypeId,
    ) -> CoreResult<Box<dyn JITFunction>> {
        // Check if the function is already in the cache
        if self.config.use_cache {
            let cache_key = format!("{expression}-{array_typeid:?}");
            if let Some(cached_fn) = self.cache.get(&cache_key) {
                return Ok(cached_fn.as_ref().clone_box());
            }
        }

        // Compile the function
        let jit_function = self.compile(expression, array_typeid)?;

        if self.config.use_cache {
            // Add the function to the cache
            let cache_key = format!("{expression}-{array_typeid:?}");
            // In a real implementation, we'd need to handle this in a thread-safe way
            // For now, we'll just clone the function
            let mut cache = self.cache.clone();
            cache.insert(cache_key, jit_function.clone());
        }

        // Clone the function and return it
        Ok(jit_function.as_ref().clone_box())
    }

    fn supports_array_type(&self, _array_typeid: TypeId) -> bool {
        // For simplicity, we'll say this factory supports all array types
        true
    }
}

/// A JIT manager that selects the appropriate factory for a given array type
pub struct JITManager {
    /// The available JIT function factories
    factories: Vec<Box<dyn JITFunctionFactory>>,

    /// Default configuration for JIT compilation
    defaultconfig: JITConfig,
}

impl JITManager {
    /// Create a new JIT manager.
    pub fn new(defaultconfig: JITConfig) -> Self {
        Self {
            factories: Vec::new(),
            defaultconfig,
        }
    }

    /// Register a JIT function factory.
    pub fn register_factory(&mut self, factory: Box<dyn JITFunctionFactory>) {
        self.factories.push(factory);
    }

    /// Get a JIT function factory that supports the given array type.
    pub fn get_factory_for_array_type(
        &self,
        array_typeid: TypeId,
    ) -> Option<&dyn JITFunctionFactory> {
        for factory in &self.factories {
            if factory.supports_array_type(array_typeid) {
                return Some(&**factory);
            }
        }
        None
    }

    /// Compile a JIT function for the given expression and array type.
    pub fn compile(
        &self,
        expression: &str,
        array_typeid: TypeId,
    ) -> CoreResult<Box<dyn JITFunction>> {
        // Find a factory that supports the array type
        if let Some(factory) = self.get_factory_for_array_type(array_typeid) {
            factory.create_jit_function(expression, array_typeid)
        } else {
            Err(CoreError::JITError(ErrorContext::new(format!(
                "No JIT factory supports array type: {array_typeid:?}"
            ))))
        }
    }

    /// Initialize the JIT manager with default factories.
    pub fn initialize(&mut self) {
        // Create and register the default factories
        let llvm_config = JITConfig {
            backend: JITBackend::LLVM,
            ..self.defaultconfig.clone()
        };
        let llvm_factory = Box::new(LLVMFunctionFactory::new(llvm_config));

        let cranelift_config = JITConfig {
            backend: JITBackend::Cranelift,
            ..self.defaultconfig.clone()
        };
        let cranelift_factory = Box::new(CraneliftFunctionFactory::new(cranelift_config));

        self.register_factory(llvm_factory);
        self.register_factory(cranelift_factory);
    }

    /// Get the global JIT manager instance.
    #[must_use]
    pub fn global() -> &'static RwLock<Self> {
        static INSTANCE: LazyLock<RwLock<JITManager>> = LazyLock::new(|| {
            RwLock::new(JITManager {
                factories: Vec::new(),
                defaultconfig: JITConfig {
                    backend: JITBackend::LLVM,
                    optimize: true,
                    opt_level: 2,
                    use_cache: true,
                    backend_options: HashMap::new(),
                },
            })
        });
        &INSTANCE
    }
}

/// An array that supports JIT compilation
pub struct JITEnabledArray<T, A> {
    /// The underlying array
    inner: A,

    /// Phantom data for the element type
    phantom: PhantomData<T>,
}

impl<T, A> JITEnabledArray<T, A> {
    /// Create a new JIT-enabled array.
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }

    /// Get a reference to the inner array.
    pub const fn inner(&self) -> &A {
        &self.inner
    }
}

impl<T, A: Clone> Clone for JITEnabledArray<T, A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            phantom: PhantomData::<T>,
        }
    }
}

impl<T, A> JITArray for JITEnabledArray<T, A>
where
    T: Send + Sync + 'static,
    A: ArrayProtocol + Clone + Send + Sync + 'static,
{
    fn compile(&self, expression: &str) -> CoreResult<Box<dyn JITFunction>> {
        // Get the JIT manager
        let jit_manager = JITManager::global();
        let jit_manager = jit_manager.read().expect("Operation failed");

        // Compile the function
        (*jit_manager).compile(expression, TypeId::of::<A>())
    }

    fn supports_jit(&self) -> bool {
        // Check if there's a factory that supports this array type
        let jit_manager = JITManager::global();
        let jit_manager = jit_manager.read().expect("Operation failed");

        jit_manager
            .get_factory_for_array_type(TypeId::of::<A>())
            .is_some()
    }

    fn jit_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();

        // Check if JIT is supported
        let supported = self.supports_jit();
        info.insert("supports_jit".to_string(), supported.to_string());

        if supported {
            // Get the JIT manager
            let jit_manager = JITManager::global();
            let jit_manager = jit_manager.read().expect("Operation failed");

            // Get the factory
            if jit_manager
                .get_factory_for_array_type(TypeId::of::<A>())
                .is_some()
            {
                // Get the factory's info
                info.insert("factory".to_string(), "JIT factory available".to_string());
            }
        }

        info
    }
}

impl<T, A> ArrayProtocol for JITEnabledArray<T, A>
where
    T: Send + Sync + 'static,
    A: ArrayProtocol + Clone + Send + Sync + 'static,
{
    fn array_function(
        &self,
        func: &ArrayFunction,
        types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, crate::array_protocol::NotImplemented> {
        // For now, just delegate to the inner array
        self.inner.array_function(func, types, args, kwargs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn dtype(&self) -> TypeId {
        self.inner.dtype()
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        // Clone the inner array directly
        let inner_clone = self.inner.clone();
        Box::new(Self {
            inner: inner_clone,
            phantom: PhantomData::<T>,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_protocol::NdarrayWrapper;
    use ::ndarray::Array2;

    #[test]
    fn test_jit_function_creation() {
        // Create a JIT function factory
        let config = JITConfig {
            backend: JITBackend::LLVM,
            ..Default::default()
        };
        let factory = LLVMFunctionFactory::new(config);

        // Create a simple expression
        let expression = "x + y";

        // Compile the function
        let array_typeid = TypeId::of::<NdarrayWrapper<f64, crate::ndarray::Ix2>>();
        let jit_function = factory
            .create_jit_function(expression, array_typeid)
            .expect("Operation failed");

        // Check the function's properties
        assert_eq!(jit_function.source(), expression);
        let compile_info = jit_function.compile_info();
        assert_eq!(
            compile_info.get("backend").expect("Operation failed"),
            "LLVM"
        );
    }

    #[test]
    fn test_jit_manager() {
        // Initialize the JIT manager
        let mut jit_manager = JITManager::new(JITConfig::default());
        jit_manager.initialize();

        // Check that the factories were registered
        let array_typeid = TypeId::of::<NdarrayWrapper<f64, crate::ndarray::Ix2>>();
        assert!(jit_manager
            .get_factory_for_array_type(array_typeid)
            .is_some());

        // Compile a function
        let expression = "x + y";
        let jit_function = jit_manager
            .compile(expression, array_typeid)
            .expect("Operation failed");

        // Check the function's properties
        assert_eq!(jit_function.source(), expression);
    }

    #[test]
    fn test_jit_enabled_array() {
        // Create an ndarray
        let array = Array2::<f64>::ones((10, 5));
        let wrapped = NdarrayWrapper::new(array);

        // Create a JIT-enabled array
        let jit_array: JITEnabledArray<f64, _> = JITEnabledArray::new(wrapped);

        // Initialize the JIT manager
        {
            let mut jit_manager = JITManager::global().write().expect("Operation failed");
            jit_manager.initialize();
        }

        // Check if JIT is supported
        assert!(jit_array.supports_jit());

        // Compile a function
        let expression = "x + y";
        let jit_function = jit_array.compile(expression).expect("Operation failed");

        // Check the function's properties
        assert_eq!(jit_function.source(), expression);
    }

    /// Regression test: both backends used to always return the constant
    /// 42.0 for any expression/arguments. Uses non-constant, non-42 data
    /// specifically so a lingering hardcoded-42.0 fallback would fail.
    #[test]
    fn test_llvm_and_cranelift_backends_evaluate_for_real() {
        let array_typeid = TypeId::of::<NdarrayWrapper<f64, crate::ndarray::Ix2>>();

        for factory in [
            Box::new(LLVMFunctionFactory::new(JITConfig::default())) as Box<dyn JITFunctionFactory>,
            Box::new(CraneliftFunctionFactory::new(JITConfig::default()))
                as Box<dyn JITFunctionFactory>,
        ] {
            let jit_function = factory
                .create_jit_function("x * y + 1", array_typeid)
                .expect("compile should succeed");

            let args: Vec<Box<dyn Any>> = vec![Box::new(3.0_f64), Box::new(4.0_f64)];
            let result = jit_function
                .evaluate(&args)
                .expect("evaluate should succeed");
            let value = *result.downcast_ref::<f64>().expect("result should be f64");
            assert_eq!(value, 3.0 * 4.0 + 1.0);

            // Different (non-constant) arguments must give a different
            // result — this is what a hardcoded-42.0 stub could never do.
            let args2: Vec<Box<dyn Any>> = vec![Box::new(10.0_f64), Box::new(-2.0_f64)];
            let result2 = jit_function
                .evaluate(&args2)
                .expect("evaluate should succeed");
            let value2 = *result2.downcast_ref::<f64>().expect("result should be f64");
            assert_eq!(value2, 10.0 * -2.0 + 1.0);
            assert_ne!(value, value2);
        }
    }

    #[test]
    fn test_jit_expression_grammar_operators_and_functions() {
        let factory = LLVMFunctionFactory::new(JITConfig::default());
        let array_typeid = TypeId::of::<NdarrayWrapper<f64, crate::ndarray::Ix2>>();

        let cases: &[(&str, &[f64], f64)] = &[
            ("2 + 3 * 4", &[], 14.0),
            ("(2 + 3) * 4", &[], 20.0),
            ("2 ^ 10", &[], 1024.0),
            ("-x + 1", &[5.0], -4.0),
            ("sqrt(x)", &[16.0], 4.0),
            ("max(x, y)", &[3.0, 7.0], 7.0),
            ("min(x, y)", &[3.0, 7.0], 3.0),
            ("abs(x)", &[-9.5], 9.5),
        ];

        for (expr, args, expected) in cases {
            let jit_function = factory
                .create_jit_function(expr, array_typeid)
                .unwrap_or_else(|e| panic!("compiling '{expr}' should succeed: {e}"));
            let boxed_args: Vec<Box<dyn Any>> =
                args.iter().map(|&v| Box::new(v) as Box<dyn Any>).collect();
            let result = jit_function
                .evaluate(&boxed_args)
                .unwrap_or_else(|e| panic!("evaluating '{expr}' should succeed: {e}"));
            let value = *result
                .downcast_ref::<f64>()
                .unwrap_or_else(|| panic!("'{expr}' result should be f64"));
            assert!(
                (value - expected).abs() < 1e-9,
                "'{expr}' evaluated to {value}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_jit_invalid_expression_is_honest_error() {
        let factory = LLVMFunctionFactory::new(JITConfig::default());
        let array_typeid = TypeId::of::<NdarrayWrapper<f64, crate::ndarray::Ix2>>();

        let result = factory.create_jit_function("x + * y", array_typeid);
        assert!(
            result.is_err(),
            "a syntactically invalid expression must be rejected, not silently compiled"
        );
    }

    #[test]
    fn test_jit_clone_box_preserves_behavior() {
        let factory = LLVMFunctionFactory::new(JITConfig::default());
        let array_typeid = TypeId::of::<NdarrayWrapper<f64, crate::ndarray::Ix2>>();
        let jit_function = factory
            .create_jit_function("x * x", array_typeid)
            .expect("compile should succeed");

        let cloned = jit_function.clone_box();
        assert_eq!(cloned.source(), jit_function.source());

        let args: Vec<Box<dyn Any>> = vec![Box::new(6.0_f64)];
        let original_result = *jit_function
            .evaluate(&args)
            .expect("evaluate should succeed")
            .downcast_ref::<f64>()
            .expect("result should be f64");
        let cloned_result = *cloned
            .evaluate(&args)
            .expect("evaluate should succeed")
            .downcast_ref::<f64>()
            .expect("result should be f64");

        assert_eq!(original_result, 36.0);
        assert_eq!(cloned_result, 36.0);
    }
}
