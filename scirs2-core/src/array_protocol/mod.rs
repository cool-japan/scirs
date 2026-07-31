// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Implementation of Array Protocol (similar to ``NumPy``'s `__array_function__` protocol)
//!
//! This module provides a mechanism for third-party array implementations to
//! override ``SciRS2`` functions. It is inspired by ``NumPy``'s `__array_function__`
//! protocol defined in NEP-18.
//!
//! The protocol enables third-party arrays to implement ``SciRS2`` functions in a way
//! that is recognized by the ``SciRS2`` library. This allows for seamless integration with
//! distributed arrays, GPU arrays, and other custom array implementations.
//!
//! ## Core Components
//!
//! The Array Protocol system includes:
//!
//! * Specialized array implementations (GPU, distributed, JIT)
//! * Automatic device placement with `AutoDevice`
//! * Mixed-precision operations
//! * Neural network layers and models
//! * Gradient computation and training capabilities
//! * Distributed training and model serialization

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, RwLock};
use std::time::{Duration, Instant};

use crate::error::{CoreError, CoreResult, ErrorContext};

// Internal submodules
mod distributed_impl;
mod gpu_impl;
#[cfg(feature = "array_protocol_wgpu")]
pub mod gpu_ndarray;
#[cfg(feature = "array_protocol_wgpu")]
mod gpu_ndarray_shaders;
mod jit_impl;
mod operations;

// Re-export the array_function_dispatch macro
pub use crate::array_function_dispatch;

// Public submodules
pub mod auto_device;
pub mod distributed_training;
pub mod grad;
pub mod mixed_precision;
pub mod ml_ops;
pub mod neural;
#[cfg(feature = "serialization")]
pub mod serialization;
pub mod training;

/// Trait for objects that can handle the array protocol.
///
/// This is similar to `NumPy`'s `__array_function__` protocol.
pub trait ArrayProtocol: Any + Send + Sync {
    /// Implementation of the array protocol.
    ///
    /// * `func` - The function being called
    /// * `types` - The types of all arguments that implement `ArrayProtocol`
    /// * `args` - The arguments to the function
    /// * `kwargs` - Named arguments to the function
    ///
    /// Returns `Ok(result)` if the operation is successful, or `Err(NotImplemented)`
    /// if the operation is not implemented for this type.
    ///
    /// # Errors
    ///
    /// Returns `Err(NotImplemented)` if the operation is not supported by this array type.
    fn array_function(
        &self,
        func: &ArrayFunction,
        types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented>;

    /// Get the array as Any for downcasting
    #[must_use]
    fn as_any(&self) -> &dyn Any;

    /// Get the shape of the array (default implementation returns empty slice)
    #[must_use]
    fn shape(&self) -> &[usize] {
        &[]
    }

    /// Get the data type of the array (default implementation returns f64)
    #[must_use]
    fn dtype(&self) -> TypeId {
        TypeId::of::<f64>()
    }

    /// Whether this concrete array type provides its own [`array_function`](Self::array_function)
    /// handling for the named operation.
    ///
    /// Dispatch uses this to decide whether it's worth delegating to this
    /// argument's own `array_function` at all. Types that only implement a
    /// subset of operations (like the plain [`NdarrayWrapper`], whose
    /// `array_function` only matches a handful of operation names) should
    /// override this to report `false` for anything outside that subset, so
    /// callers correctly fall through to another candidate (or to a
    /// plain-ndarray fallback implementation) instead of receiving a
    /// spurious `NotImplemented` from this type's dispatch and giving up
    /// immediately.
    ///
    /// The default assumes the type may handle any operation:
    /// `array_function` itself remains the ultimate authority (returning
    /// `Err(NotImplemented)` when it truly doesn't apply), so types that
    /// already do that correctly (or that implement most/all operations)
    /// don't need to override this.
    #[must_use]
    fn supports_op(&self, _opname: &str) -> bool {
        true
    }

    /// Clone this array protocol object.
    #[must_use]
    fn box_clone(&self) -> Box<dyn ArrayProtocol>;
}

/// Make `Box<dyn ArrayProtocol>` cloneable via `box_clone`
impl Clone for Box<dyn ArrayProtocol> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// Marker for functions not implemented by a specific type.
///
/// This is part of the Array Protocol API design and is used as a marker to indicate
/// that a function is not implemented by a specific array type. It's different from
/// the `CoreError::NotImplementedError` enum variant, which is used for error reporting.
///
/// When an error is propagated up the call chain, `NotImplemented` is converted
/// to `OperationError::NotImplemented` and then to `CoreError::NotImplementedError`
/// for consistent error handling.
#[derive(Debug, Clone, Copy)]
pub struct NotImplemented;

impl Display for NotImplemented {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NotImplemented")
    }
}

/// Type alias for the complex function implementation type
pub type ArrayFunctionImpl = dyn Fn(&[Box<dyn Any>], &HashMap<String, Box<dyn Any>>) -> CoreResult<Box<dyn Any>>
    + Send
    + Sync;

/// A wrapper for functions that can be overridden by the array protocol.
#[derive(Clone)]
pub struct ArrayFunction {
    /// The name of the function, including its module path
    pub name: &'static str,

    /// The function implementation
    pub implementation: Arc<ArrayFunctionImpl>,
}

impl Debug for ArrayFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayFunction")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl PartialEq for ArrayFunction {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for ArrayFunction {}

impl std::hash::Hash for ArrayFunction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl ArrayFunction {
    /// Create a new array function with the given name
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            // Default implementation that returns NotImplemented
            implementation: Arc::new(|_args, _kwargs| {
                Err(CoreError::NotImplementedError(ErrorContext::new(
                    "Function not implemented".to_string(),
                )))
            }),
        }
    }
}

/// Cache entry for function dispatch optimization
#[derive(Debug, Clone)]
pub struct DispatchCacheEntry {
    /// Type signature for the cached result
    #[allow(dead_code)]
    type_signature: Vec<TypeId>,
    /// Which implementation type to try first
    #[allow(dead_code)]
    preferred_impl_type: TypeId,
    /// Cache timestamp for TTL management
    timestamp: Instant,
    /// Number of cache hits
    hit_count: u64,
}

/// Registry of all array functions with dispatch caching.
#[derive(Debug)]
pub struct ArrayFunctionRegistry {
    /// Map of function names to array functions
    functions: HashMap<&'static str, ArrayFunction>,
    /// Dispatch cache for performance optimization
    dispatch_cache: HashMap<(&'static str, Vec<TypeId>), DispatchCacheEntry>,
    /// Maximum cache size to prevent unbounded growth
    max_cache_size: usize,
    /// Cache TTL for entries (prevents stale cache)
    cache_ttl: Duration,
}

impl Default for ArrayFunctionRegistry {
    fn default() -> Self {
        Self {
            functions: HashMap::new(),
            dispatch_cache: HashMap::new(),
            max_cache_size: 1000,                // Reasonable default cache size
            cache_ttl: Duration::from_secs(300), // 5 minutes TTL
        }
    }
}

impl ArrayFunctionRegistry {
    /// Get the global registry.
    #[must_use]
    pub fn global() -> &'static RwLock<Self> {
        static REGISTRY: LazyLock<RwLock<ArrayFunctionRegistry>> =
            LazyLock::new(|| RwLock::new(ArrayFunctionRegistry::default()));
        &REGISTRY
    }

    /// Register a new array function.
    pub fn register(&mut self, func: ArrayFunction) {
        self.functions.insert(func.name, func);
    }

    /// Get an array function by name.
    #[must_use]
    #[allow(dead_code)]
    pub fn get(&self, name: &str) -> Option<&ArrayFunction> {
        self.functions.get(name)
    }

    /// Get all registered functions.
    #[must_use]
    pub fn all_functions(&self) -> Vec<&ArrayFunction> {
        self.functions.values().collect()
    }

    /// Get cached dispatch entry for optimization
    #[must_use]
    pub fn get_cached_dispatch(
        &self,
        funcname: &'static str,
        types: &[TypeId],
    ) -> Option<&DispatchCacheEntry> {
        let key = (funcname, types.to_vec());
        if let Some(entry) = self.dispatch_cache.get(&key) {
            // Check if cache entry is still valid (TTL check)
            if entry.timestamp.elapsed() < self.cache_ttl {
                return Some(entry);
            }
        }
        None
    }

    /// Cache dispatch result for future optimization
    pub fn cache_dispatch(
        &mut self,
        funcname: &'static str,
        types: Vec<TypeId>,
        impl_type: TypeId,
    ) {
        // Clean cache if it's getting too large
        if self.dispatch_cache.len() >= self.max_cache_size {
            self.cleanup_cache();
        }

        let key = (funcname, types.clone());
        let entry = DispatchCacheEntry {
            type_signature: types,
            preferred_impl_type: impl_type,
            timestamp: Instant::now(),
            hit_count: 0,
        };
        self.dispatch_cache.insert(key, entry);
    }

    /// Update cache hit count for an entry
    pub fn update_cache_hit(&mut self, funcname: &'static str, types: &[TypeId]) {
        let key = (funcname, types.to_vec());
        if let Some(entry) = self.dispatch_cache.get_mut(&key) {
            entry.hit_count += 1;
        }
    }

    /// Clean up expired cache entries
    fn cleanup_cache(&mut self) {
        let now = Instant::now();
        self.dispatch_cache
            .retain(|_, entry| now.duration_since(entry.timestamp) < self.cache_ttl);

        // If still too large, remove least recently used entries
        if self.dispatch_cache.len() >= self.max_cache_size {
            let mut entries: Vec<_> = self
                .dispatch_cache
                .iter()
                .map(|(k, v)| (k.clone(), v.hit_count))
                .collect();
            entries.sort_by_key(|(_, hit_count)| *hit_count);

            // Remove bottom 25% of entries by hit count
            let to_remove = self.dispatch_cache.len() / 4;
            let keys_to_remove: Vec<_> = entries
                .iter()
                .take(to_remove)
                .map(|(key, _)| key.clone())
                .collect();
            for key in keys_to_remove {
                self.dispatch_cache.remove(&key);
            }
        }
    }

    /// Get cache statistics for monitoring
    #[must_use]
    pub fn cache_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("cache_size".to_string(), self.dispatch_cache.len() as u64);
        stats.insert("max_cache_size".to_string(), self.max_cache_size as u64);

        let total_hits: u64 = self.dispatch_cache.values().map(|e| e.hit_count).sum();
        stats.insert("total_hits".to_string(), total_hits);

        stats
    }
}

/// Helper function to extract all arguments implementing the `ArrayProtocol` trait
/// that also claim (via [`ArrayProtocol::supports_op`]) to support the named operation.
///
/// This is similar to `NumPy`'s `_get_implementing_args` function.
/// Optimized version with pre-allocated capacity and fast-path for common cases.
///
/// `func_name` should be the same operation name later passed to
/// `array_function`/`ArrayFunction::new` at the call site (e.g.
/// `"scirs2::array_protocol::operations::subtract"`). Filtering on it here
/// means an argument whose concrete type doesn't implement this specific
/// operation is excluded from the candidate list, so callers correctly fall
/// through to a different implementation (or a plain-ndarray fallback)
/// instead of delegating to a candidate that's guaranteed to return
/// `NotImplemented` for this particular `func_name`.
pub fn get_implementing_args<'a>(
    func_name: &str,
    args: &'a [Box<dyn Any>],
) -> Vec<(TypeId, &'a dyn ArrayProtocol)> {
    if args.is_empty() {
        return Vec::new();
    }

    // Pre-allocate with capacity to avoid reallocation
    let mut implementing_args = Vec::with_capacity(args.len());

    for arg in args {
        if let Some(array_protocol_obj) = arg.downcast_ref::<Box<dyn ArrayProtocol>>() {
            if !array_protocol_obj.supports_op(func_name) {
                continue;
            }
            let type_id = (**array_protocol_obj).type_id();
            implementing_args.push((type_id, &**array_protocol_obj));
        }
    }

    implementing_args
}

/// Calls the array protocol with the given function and arguments.
///
/// * `func` - The array function to call
/// * `args` - The arguments to the function
/// * `kwargs` - Named arguments to the function
///
/// Returns the result of the function call, or an error if the function
/// cannot be dispatched to any of the array protocol implementations.
///
/// Optimized version with caching and fast-path optimizations.
#[allow(dead_code)]
pub fn array_function_dispatch(
    func: &ArrayFunction,
    args: &[Box<dyn Any>],
    kwargs: &HashMap<String, Box<dyn Any>>,
) -> CoreResult<Box<dyn Any>> {
    // Fast path for empty args
    if args.is_empty() {
        return (func.implementation)(args, kwargs);
    }

    // Find all arguments implementing ArrayProtocol
    let implementing_args = get_implementing_args(func.name, args);

    if implementing_args.is_empty() {
        // No arguments implement ArrayProtocol, use default implementation
        return (func.implementation)(args, kwargs);
    }

    // Fast path for single implementing argument
    if implementing_args.len() == 1 {
        let (type_id, array_protocol_obj) = implementing_args[0];
        let types = [type_id];
        match array_protocol_obj.array_function(func, &types, args, kwargs) {
            Ok(result) => return Ok(result),
            Err(NotImplemented) => {
                return Err(CoreError::DispatchError(ErrorContext::new(format!(
                    "No implementation found for {} with type {:?}",
                    func.name, type_id
                ))));
            }
        }
    }

    // Extract all unique types that implement ArrayProtocol (optimized)
    let mut unique_types = Vec::with_capacity(implementing_args.len());
    let mut seen_types = std::collections::HashSet::with_capacity(implementing_args.len());

    for &(type_id, _) in &implementing_args {
        if seen_types.insert(type_id) {
            unique_types.push(type_id);
        }
    }

    // Try dispatching to each implementation in priority order
    for (_, array_protocol_obj) in implementing_args {
        if let Ok(result) = array_protocol_obj.array_function(func, &unique_types, args, kwargs) {
            return Ok(result);
        }
    }

    // If we get here, no implementation was found
    Err(CoreError::DispatchError(ErrorContext::new(format!(
        "No implementation found for {} with {} argument types: {:?}",
        func.name,
        unique_types.len(),
        unique_types
    ))))
}

/// Decorator for adding array function dispatch capabilities to a function.
///
/// This is similar to `NumPy`'s `array_function_dispatch` decorator.
pub struct ArrayFunctionDecorator<F> {
    function: F,
    name: &'static str,
}

impl<F> ArrayFunctionDecorator<F>
where
    F: Send + Sync + 'static,
{
    /// Create a new array function decorator.
    #[must_use]
    pub fn new(function: F, name: &'static str) -> Self {
        Self { function, name }
    }

    /// Register the function with the global registry.
    pub fn register(self) -> F {
        let implementation = Arc::new(
            move |_args: &[Box<dyn Any>], _kwargs: &HashMap<String, Box<dyn Any>>| {
                // Implementation that converts generic arguments to specific types
                // and calls the original function
                // This is a simplified version - in practice, we would need more complex
                // type conversion
                Err(CoreError::NotImplementedError(ErrorContext::new(
                    "ArrayFunctionDecorator: Type conversion in array_function_dispatch is not implemented yet".to_string()
                )))
            },
        );

        let func = ArrayFunction {
            name: self.name,
            implementation,
        };

        // Register the function with the global registry
        let registry = ArrayFunctionRegistry::global();
        if let Ok(mut registry) = registry.write() {
            registry.register(func);
        } else {
            eprintln!("Warning: Failed to acquire write lock on ArrayFunctionRegistry, skipping function registration");
            // Continue without registration - this may result in reduced functionality but avoids crash
        }

        self.function
    }
}

/// Trait for arrays that can support GPU operations.
pub trait GPUArray: ArrayProtocol {
    /// Move the array to GPU.
    fn to_gpu(&self) -> CoreResult<Box<dyn GPUArray>>;

    /// Move the array from GPU to CPU.
    fn to_cpu(&self) -> CoreResult<Box<dyn ArrayProtocol>>;

    /// Check if the array is on GPU.
    #[must_use]
    fn is_on_gpu(&self) -> bool;

    /// Get information about the GPU device that holds this array.
    #[must_use]
    fn device_info(&self) -> HashMap<String, String>;
}

/// Trait for distributed arrays that can span multiple machines.
pub trait DistributedArray: ArrayProtocol {
    /// Get information about the distribution of this array.
    #[must_use]
    fn distribution_info(&self) -> HashMap<String, String>;

    /// Gather the distributed array to a single node.
    fn gather(&self) -> CoreResult<Box<dyn ArrayProtocol>>;

    /// Scatter a regular array to a distributed array.
    fn scatter(&self, chunks: usize) -> CoreResult<Box<dyn DistributedArray>>;

    /// Check if this array is distributed.
    #[must_use]
    fn is_distributed(&self) -> bool;
}

/// JIT (Just-In-Time) compilation support for arrays.
pub trait JITArray: ArrayProtocol {
    /// Compile an expression to be evaluated on this array.
    fn compile(&self, expression: &str) -> CoreResult<Box<dyn JITFunction>>;

    /// Check if JIT compilation is supported for this array type.
    #[must_use]
    fn supports_jit(&self) -> bool;

    /// Get information about the JIT compiler being used.
    #[must_use]
    fn jit_info(&self) -> HashMap<String, String>;
}

/// A JIT-compiled function that can be evaluated on arrays.
pub trait JITFunction: Send + Sync {
    /// Evaluate the function with the given arguments.
    fn evaluate(&self, args: &[Box<dyn Any>]) -> CoreResult<Box<dyn Any>>;

    /// Get the source code of the compiled function.
    #[must_use]
    fn source(&self) -> String;

    /// Get information about how the function was compiled.
    #[must_use]
    fn compile_info(&self) -> HashMap<String, String>;

    /// Clone this JIT function into a `Box<dyn JITFunction>`.
    #[must_use]
    fn clone_box(&self) -> Box<dyn JITFunction>;
}

/// A factory for creating JIT functions for specific array implementations.
pub trait JITFunctionFactory: Send + Sync {
    /// Create a new JIT function for the given expression and array type.
    fn create_jit_function(
        &self,
        expression: &str,
        array_typeid: TypeId,
    ) -> CoreResult<Box<dyn JITFunction>>;

    /// Check if this factory supports the given array type.
    #[must_use]
    fn supports_array_type(&self, array_typeid: TypeId) -> bool;
}

/// Registry of JIT function factories.
#[derive(Default)]
pub struct JITFactoryRegistry {
    factories: Vec<Box<dyn JITFunctionFactory>>,
}

impl std::fmt::Debug for JITFactoryRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "JITFactoryRegistry {{ factories: {} }}",
            self.factories.len()
        )
    }
}

impl JITFactoryRegistry {
    /// Get the global registry.
    #[must_use]
    pub fn global() -> &'static RwLock<Self> {
        static REGISTRY: LazyLock<RwLock<JITFactoryRegistry>> = LazyLock::new(|| {
            RwLock::new(JITFactoryRegistry {
                factories: Vec::new(),
            })
        });
        &REGISTRY
    }

    /// Register a new JIT function factory.
    pub fn register(&mut self, factory: Box<dyn JITFunctionFactory>) {
        self.factories.push(factory);
    }

    /// Get a JIT function factory that supports the given array type.
    #[must_use]
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
}

/// Downcast an `array_function` argument to a concrete `ArrayProtocol` impl,
/// tolerating either boxing convention seen across this module's callers.
///
/// Most callers (the dispatch functions in [`crate::array_protocol::operations`],
/// which go through [`get_implementing_args`]) erase an opaque `&dyn
/// ArrayProtocol` via `Box::new(x.box_clone())`, so `arg`'s concrete type is
/// `Box<dyn ArrayProtocol>` wrapping the real value — there is no other way to
/// produce a `Box<dyn Any>` from a trait object without already knowing its
/// concrete type. A few callers instead build `args` themselves from an
/// already-concrete value (for example
/// [`mixed_precision`](super::mixed_precision)'s `MixedPrecisionArray`
/// delegating straight to `NdarrayWrapper::array_function`), boxing the
/// concrete type directly with no extra indirection. Trying the
/// double-boxed shape first and falling back to a direct downcast handles
/// both without requiring every internal caller to agree on one convention.
fn downcast_array_function_arg<C: 'static>(arg: &dyn Any) -> Option<&C> {
    if let Some(boxed) = arg.downcast_ref::<Box<dyn ArrayProtocol>>() {
        return boxed.as_any().downcast_ref::<C>();
    }
    arg.downcast_ref::<C>()
}

/// Downcast an `array_function` argument to an owned, dynamically-dimensioned
/// array of element type `T`, regardless of whether the concrete wrapper is
/// `NdarrayWrapper<T, IxDyn>`, `NdarrayWrapper<T, Ix2>`, or `NdarrayWrapper<T,
/// Ix1>` (tried in that order), and regardless of which boxing convention
/// wraps it (see `downcast_array_function_arg`). Used as a fallback when
/// combining two
/// `NdarrayWrapper` operands whose *concrete* dimension types differ but
/// whose runtime shapes are still combinable — e.g. a gradient tensor
/// produced by [`super::grad`]'s `backward()` as `IxDyn` combined with an
/// `Ix2`-typed value from the forward pass.
fn downcast_arg_to_ixdyn<T: Clone + Send + Sync + 'static>(
    arg: &dyn Any,
) -> Option<crate::ndarray::Array<T, crate::ndarray::IxDyn>> {
    if let Some(w) = downcast_array_function_arg::<NdarrayWrapper<T, crate::ndarray::IxDyn>>(arg) {
        return Some(w.as_array().clone());
    }
    if let Some(w) = downcast_array_function_arg::<NdarrayWrapper<T, crate::ndarray::Ix2>>(arg) {
        return Some(w.as_array().clone().into_dyn());
    }
    if let Some(w) = downcast_array_function_arg::<NdarrayWrapper<T, crate::ndarray::Ix1>>(arg) {
        return Some(w.as_array().clone().into_dyn());
    }
    None
}

/// A wrapper for ndarray to implement the ArrayProtocol trait.
#[derive(Debug, Clone)]
pub struct NdarrayWrapper<T, D: crate::ndarray::Dimension> {
    array: crate::ndarray::Array<T, D>,
    phantom: PhantomData<(T, D)>,
}

impl<T, D> NdarrayWrapper<T, D>
where
    T: Clone + 'static,
    D: crate::ndarray::Dimension + 'static,
{
    /// Create a new ndarray wrapper.
    #[must_use]
    pub fn new(array: crate::ndarray::Array<T, D>) -> Self {
        Self {
            array,
            phantom: PhantomData,
        }
    }

    /// Get the underlying ndarray.
    #[must_use]
    pub const fn as_array(&self) -> &crate::ndarray::Array<T, D> {
        &self.array
    }

    /// Convert into the underlying ndarray.
    #[must_use]
    pub fn into_array(self) -> crate::ndarray::Array<T, D> {
        self.array
    }

    /// Update the underlying array with a new one.
    pub fn array_2(&mut self, newarray: crate::ndarray::Array<T, D>) {
        self.array = newarray;
    }
}

/// Operation names that [`NdarrayWrapper::array_function`] actually matches
/// against `func.name` below. Kept as a single source of truth so
/// `supports_op` can't silently drift out of sync with the `match` arms.
const NDARRAY_WRAPPER_SUPPORTED_OPS: &[&str] = &[
    "scirs2::array_protocol::operations::add",
    "scirs2::array_protocol::operations::multiply",
    "scirs2::array_protocol::operations::matmul",
    "scirs2::array_protocol::operations::transpose",
    "scirs2::array_protocol::operations::sum",
    "scirs2::array_protocol::operations::reshape",
];

impl<T, D> ArrayProtocol for NdarrayWrapper<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: crate::ndarray::Dimension + Send + Sync + 'static,
{
    fn supports_op(&self, opname: &str) -> bool {
        NDARRAY_WRAPPER_SUPPORTED_OPS.contains(&opname)
    }

    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        match func.name {
            "scirs2::array_protocol::operations::add" => {
                // Addition operation for NdarrayWrapper
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                // See `downcast_array_function_arg`: callers disagree on whether
                // `args[1]` is boxed as `Box<dyn ArrayProtocol>` (the dispatcher
                // in `operations.rs`) or as the concrete type directly
                // (`mixed_precision`'s internal delegation), so try both shapes.
                let other = downcast_array_function_arg::<NdarrayWrapper<T, D>>(args[1].as_ref());

                if let Some(other) = other {
                    if let (Some(a), Some(b)) = (
                        self.as_any().downcast_ref::<NdarrayWrapper<T, D>>(),
                        other.as_any().downcast_ref::<NdarrayWrapper<T, D>>(),
                    ) {
                        // Need to make sure T supports addition
                        if TypeId::of::<T>() == TypeId::of::<f64>() {
                            let a_f64 =
                                unsafe { &*(a as *const _ as *const NdarrayWrapper<f64, D>) };
                            let b_f64 =
                                unsafe { &*(b as *const _ as *const NdarrayWrapper<f64, D>) };
                            let result = a_f64.as_array() + b_f64.as_array();
                            return Ok(Box::new(
                                Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                            ));
                        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
                            let a_f32 =
                                unsafe { &*(a as *const _ as *const NdarrayWrapper<f32, D>) };
                            let b_f32 =
                                unsafe { &*(b as *const _ as *const NdarrayWrapper<f32, D>) };
                            let result = a_f32.as_array() + b_f32.as_array();
                            return Ok(Box::new(
                                Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                            ));
                        }
                    }
                }

                // Fallback: same element type T, but `other` has a different
                // concrete dimension type than `self` (e.g. combining a
                // gradient tensor created as IxDyn with an Ix2/Ix1 value from
                // the forward pass) — the fast path above requires an exact D
                // match, so normalize both operands to IxDyn and combine there.
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    if let Some(other_arr) = downcast_arg_to_ixdyn::<f64>(args[1].as_ref()) {
                        let self_f64 =
                            unsafe { &*(self as *const _ as *const NdarrayWrapper<f64, D>) };
                        let result = self_f64.as_array().to_owned().into_dyn() + other_arr;
                        return Ok(Box::new(
                            Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                        ));
                    }
                } else if TypeId::of::<T>() == TypeId::of::<f32>() {
                    if let Some(other_arr) = downcast_arg_to_ixdyn::<f32>(args[1].as_ref()) {
                        let self_f32 =
                            unsafe { &*(self as *const _ as *const NdarrayWrapper<f32, D>) };
                        let result = self_f32.as_array().to_owned().into_dyn() + other_arr;
                        return Ok(Box::new(
                            Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                        ));
                    }
                }
                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::multiply" => {
                // Element-wise multiplication for NdarrayWrapper
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                let other = downcast_array_function_arg::<NdarrayWrapper<T, D>>(args[1].as_ref());

                if let Some(other) = other {
                    if let (Some(a), Some(b)) = (
                        self.as_any().downcast_ref::<NdarrayWrapper<T, D>>(),
                        other.as_any().downcast_ref::<NdarrayWrapper<T, D>>(),
                    ) {
                        if TypeId::of::<T>() == TypeId::of::<f64>() {
                            let a_f64 =
                                unsafe { &*(a as *const _ as *const NdarrayWrapper<f64, D>) };
                            let b_f64 =
                                unsafe { &*(b as *const _ as *const NdarrayWrapper<f64, D>) };
                            let result = a_f64.as_array() * b_f64.as_array();
                            return Ok(Box::new(
                                Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                            ));
                        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
                            let a_f32 =
                                unsafe { &*(a as *const _ as *const NdarrayWrapper<f32, D>) };
                            let b_f32 =
                                unsafe { &*(b as *const _ as *const NdarrayWrapper<f32, D>) };
                            let result = a_f32.as_array() * b_f32.as_array();
                            return Ok(Box::new(
                                Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                            ));
                        }
                    }
                }

                // Fallback: same element type T, different concrete D — see
                // the identical fallback in the "add" arm above.
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    if let Some(other_arr) = downcast_arg_to_ixdyn::<f64>(args[1].as_ref()) {
                        let self_f64 =
                            unsafe { &*(self as *const _ as *const NdarrayWrapper<f64, D>) };
                        let result = self_f64.as_array().to_owned().into_dyn() * other_arr;
                        return Ok(Box::new(
                            Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                        ));
                    }
                } else if TypeId::of::<T>() == TypeId::of::<f32>() {
                    if let Some(other_arr) = downcast_arg_to_ixdyn::<f32>(args[1].as_ref()) {
                        let self_f32 =
                            unsafe { &*(self as *const _ as *const NdarrayWrapper<f32, D>) };
                        let result = self_f32.as_array().to_owned().into_dyn() * other_arr;
                        return Ok(Box::new(
                            Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                        ));
                    }
                }
                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::matmul" => {
                // Matrix multiplication for NdarrayWrapper
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                // We can only handle matrix multiplication for 2D arrays
                // Check for 2D array using TypeId
                if TypeId::of::<D>() != TypeId::of::<crate::ndarray::Ix2>() {
                    return Err(NotImplemented);
                }

                let other = downcast_array_function_arg::<NdarrayWrapper<T, D>>(args[1].as_ref());

                if let Some(other) = other {
                    // Since we've already checked TypeId::of::<D>() == TypeId::of::<crate::ndarray::Ix2>()
                    // We can safely specialize for Ix2 matrices

                    // Handle the case for f64 matrices
                    if TypeId::of::<T>() == TypeId::of::<f64>() {
                        // Cast to concrete _types we know how to handle
                        let a_f64 = unsafe {
                            &*(self as *const _ as *const NdarrayWrapper<f64, crate::ndarray::Ix2>)
                        };
                        let b_f64 = unsafe {
                            &*(other as *const _ as *const NdarrayWrapper<f64, crate::ndarray::Ix2>)
                        };

                        // Get dimensions
                        let ashape = a_f64.as_array().shape();
                        let bshape = b_f64.as_array().shape();

                        if ashape.len() != 2 || bshape.len() != 2 || ashape[1] != bshape[0] {
                            return Err(NotImplemented);
                        }

                        // Use the higher-level dot operation which will be more efficient
                        // than our manual implementation
                        let result = a_f64.as_array().dot(b_f64.as_array());
                        return Ok(Box::new(
                            Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                        ));
                    }
                    // Handle the case for f32 matrices
                    else if TypeId::of::<T>() == TypeId::of::<f32>() {
                        // Cast to concrete _types we know how to handle
                        let a_f32 = unsafe {
                            &*(self as *const _ as *const NdarrayWrapper<f32, crate::ndarray::Ix2>)
                        };
                        let b_f32 = unsafe {
                            &*(other as *const _ as *const NdarrayWrapper<f32, crate::ndarray::Ix2>)
                        };

                        // Get dimensions
                        let ashape = a_f32.as_array().shape();
                        let bshape = b_f32.as_array().shape();

                        if ashape.len() != 2 || bshape.len() != 2 || ashape[1] != bshape[0] {
                            return Err(NotImplemented);
                        }

                        // Use the higher-level dot operation which will be more efficient
                        // than our manual implementation
                        let result = a_f32.as_array().dot(b_f32.as_array());
                        return Ok(Box::new(
                            Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                        ));
                    }
                }
                // If we get here, we don't know how to handle this case
                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::transpose" => {
                // Transpose operation for NdarrayWrapper
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    let a_f64 = unsafe { &*(self as *const _ as *const NdarrayWrapper<f64, D>) };
                    let result = a_f64.as_array().t().to_owned();
                    return Ok(Box::new(
                        Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                    ));
                } else if TypeId::of::<T>() == TypeId::of::<f32>() {
                    let a_f32 = unsafe { &*(self as *const _ as *const NdarrayWrapper<f32, D>) };
                    let result = a_f32.as_array().t().to_owned();
                    return Ok(Box::new(
                        Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                    ));
                }
                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::sum" => {
                // Sum operation for NdarrayWrapper
                let axis_ref = kwargs.get("axis").and_then(|a| a.downcast_ref::<usize>());

                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    let a_f64 = unsafe { &*(self as *const _ as *const NdarrayWrapper<f64, D>) };
                    match axis_ref {
                        Some(&_ax) => {
                            // Can't use sum_axis without RemoveAxis trait
                            // Just return the full sum for now
                            let result = a_f64.as_array().sum();
                            return Ok(Box::new(result));
                        }
                        None => {
                            let result = a_f64.as_array().sum();
                            return Ok(Box::new(result));
                        }
                    }
                } else if TypeId::of::<T>() == TypeId::of::<f32>() {
                    let a_f32 = unsafe { &*(self as *const _ as *const NdarrayWrapper<f32, D>) };
                    match axis_ref {
                        Some(&_ax) => {
                            // Can't use sum_axis without RemoveAxis trait
                            // Just return the full sum for now
                            let result = a_f32.as_array().sum();
                            return Ok(Box::new(result));
                        }
                        None => {
                            let result = a_f32.as_array().sum();
                            return Ok(Box::new(result));
                        }
                    }
                }
                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::reshape" => {
                // Reshape operation for NdarrayWrapper
                if let Some(shape) = kwargs
                    .get("shape")
                    .and_then(|s| s.downcast_ref::<Vec<usize>>())
                {
                    if TypeId::of::<T>() == TypeId::of::<f64>() {
                        let a_f64 =
                            unsafe { &*(self as *const _ as *const NdarrayWrapper<f64, D>) };
                        match a_f64
                            .as_array()
                            .clone()
                            .into_shape_with_order(shape.clone())
                        {
                            Ok(result) => {
                                return Ok(Box::new(
                                    Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                                ))
                            }
                            Err(_) => return Err(NotImplemented),
                        }
                    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
                        let a_f32 =
                            unsafe { &*(self as *const _ as *const NdarrayWrapper<f32, D>) };
                        match a_f32
                            .as_array()
                            .clone()
                            .into_shape_with_order(shape.clone())
                        {
                            Ok(result) => {
                                return Ok(Box::new(
                                    Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>
                                ))
                            }
                            Err(_) => return Err(NotImplemented),
                        }
                    }
                }
                Err(NotImplemented)
            }
            _ => Err(NotImplemented),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    fn dtype(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

// Example implementation for a third-party array library:

/// A mock distributed array implementation.
#[derive(Debug, Clone)]
pub struct MockDistributedArray<T: Clone + 'static> {
    chunks: Vec<T>,
    shape: Vec<usize>,
}

impl<T: Clone + Send + Sync + 'static> MockDistributedArray<T> {
    /// Create a new mock distributed array.
    #[must_use]
    pub fn new(chunks: Vec<T>, shape: Vec<usize>) -> Self {
        Self { chunks, shape }
    }
}

impl<T: Clone + Send + Sync + 'static> ArrayProtocol for MockDistributedArray<T> {
    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        _args: &[Box<dyn Any>],
        _kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        match func.name {
            "scirs2::mean" => {
                // Example: Implement a mean function for distributed arrays
                // In a real implementation, this would use distributed computation

                // For simplicity, we'll just return a dummy result
                let result = T::clone(&self.chunks[0]);
                Ok(Box::new(result))
            }
            _ => Err(NotImplemented),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

impl<T: Clone + Send + Sync + 'static> DistributedArray for MockDistributedArray<T> {
    fn distribution_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("type".to_string(), "mock_distributed".to_string());
        info.insert("chunks".to_string(), self.chunks.len().to_string());
        info
    }

    fn gather(&self) -> CoreResult<Box<dyn ArrayProtocol>> {
        // In a real implementation, this would gather data from all nodes
        // For now, we just return self boxed as ArrayProtocol
        Ok(Box::new(self.clone()) as Box<dyn ArrayProtocol>)
    }

    fn scatter(&self, _numchunks: usize) -> CoreResult<Box<dyn DistributedArray>> {
        // In a real implementation, this would scatter data to multiple nodes
        // For now, we just return self boxed as DistributedArray
        Ok(Box::new(self.clone()) as Box<dyn DistributedArray>)
    }

    fn is_distributed(&self) -> bool {
        true
    }
}

/// A mock GPU array implementation.
#[derive(Debug, Clone)]
pub struct MockGPUArray<T: Clone + 'static> {
    data: Vec<T>,
    shape: Vec<usize>,
    device: String,
}

impl<T: Clone + Send + Sync + 'static> MockGPUArray<T> {
    /// Create a new mock GPU array.
    #[must_use]
    pub fn new(data: Vec<T>, shape: Vec<usize>, device: String) -> Self {
        Self {
            data,
            shape,
            device,
        }
    }
}

impl<T: Clone + Send + Sync + 'static> ArrayProtocol for MockGPUArray<T> {
    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        _args: &[Box<dyn Any>],
        _kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        match func.name {
            "scirs2::matmul" => {
                // Example: Implement a GPU-accelerated matrix multiplication
                // In a real implementation, this would use GPU computation

                // For simplicity, we'll just return a dummy result
                let result =
                    MockGPUArray::new(self.data.clone(), self.shape.clone(), self.device.clone());
                Ok(Box::new(result))
            }
            _ => Err(NotImplemented),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

impl<T: Clone + Send + Sync + 'static> GPUArray for MockGPUArray<T> {
    fn to_gpu(&self) -> CoreResult<Box<dyn GPUArray>> {
        // Already on GPU
        Ok(Box::new(self.clone()) as Box<dyn GPUArray>)
    }

    fn to_cpu(&self) -> CoreResult<Box<dyn ArrayProtocol>> {
        // In a real implementation, this would transfer data from GPU to CPU
        // For now, we just return self boxed as ArrayProtocol
        Ok(Box::new(self.clone()) as Box<dyn ArrayProtocol>)
    }

    fn is_on_gpu(&self) -> bool {
        true
    }

    fn device_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("device".to_string(), self.device.clone());
        info.insert("type".to_string(), "mock_gpu".to_string());
        info
    }
}

/// A factory for creating and registering array protocol enabled functions.
///
/// This provides a convenient way to create functions that can be overridden
/// by third-party array implementations.
#[derive(Debug)]
pub struct ArrayProtocolFunction<F> {
    func: F,
    name: &'static str,
}

impl<F> ArrayProtocolFunction<F> {
    /// Create a new array protocol function.
    #[must_use]
    pub fn new(func: F, name: &'static str) -> Self {
        Self { func, name }
    }
}

impl<F> ArrayProtocolFunction<F>
where
    F: Clone + Send + Sync + 'static,
{
    /// Register this function with the array protocol system.
    pub fn register(self) -> F {
        let implementation = Arc::new(
            move |_args: &[Box<dyn Any>], _kwargs: &HashMap<String, Box<dyn Any>>| {
                // This is a placeholder for actual implementation that would:
                // 1. Convert generic args to specific types needed by the function
                // 2. Call the function with the converted args
                // 3. Return the result as a Box<dyn Any>
                Err(CoreError::NotImplementedError(ErrorContext::new(
                    "ArrayProtocolFunction: Implementation for array protocol functions is not complete".to_string()
                )))
            },
        );

        let array_func = ArrayFunction {
            name: self.name,
            implementation,
        };

        // Register the function
        if let Ok(mut registry) = ArrayFunctionRegistry::global().write() {
            registry.register(array_func);
        } else {
            eprintln!("Warning: Failed to acquire write lock on ArrayFunctionRegistry during array protocol building, skipping function registration");
            // Continue without registration - this may result in reduced functionality but avoids crash
        }

        self.func
    }
}

/// Convenience macro for defining array protocol functions.
///
/// This macro creates a function and registers it with the array protocol system.
/// The function can then be overridden by array types that implement the ArrayProtocol trait.
///
/// Example usage:
/// ```ignore
/// use scirs2_core::array_protocol::{ArrayFunction, ArrayFunctionRegistry};
/// use std::sync::Arc;
/// use std::collections::HashMap;
/// use std::any::Any;
///
/// // Define and register a sum function
/// fn register_sum_function() {
///     let implementation = Arc::new(
///         move |args: &[Box<dyn Any>], kwargs: &HashMap<String, Box<dyn Any>>| {
///             if let Some(array) = args.get(0)
///                 .and_then(|arg| arg.downcast_ref::<crate::ndarray::Array<f64, crate::ndarray::Ix2>>()) {
///                 let sum = array.sum();
///                 Ok(Box::new(sum) as Box<dyn Any>)
///             } else {
///                 Err(scirs2_core::error::CoreError::InvalidArgument(
///                     scirs2_core::error::ErrorContext::new(
///                         "Expected Array2<f64> as first argument".to_string()
///                     )
///                 ))
///             }
///         }
///     );
///     
///     let func = ArrayFunction {
///         name: "scirs2::sum",
///         implementation,
///     };
///     
///     // Register the function
///     if let Ok(mut registry) = ArrayFunctionRegistry::global().write() {
///         registry.register(func);
///     }
/// }
/// ```
#[macro_export]
macro_rules! array_function_def {
    (fn $name:ident $(<$($gen:ident),*>)? ($($arg:ident : $arg_ty:ty),*) -> $ret:ty $body:block, $funcname:expr) => {
        {
            // Define the function
            fn $name $(<$($gen),*>)? ($($arg : $arg_ty),*) -> $ret $body

            // Return the function so it can be used
            $name
        }
    };
}

// Re-export distributed array implementation
pub use self::distributed_impl::{
    ArrayChunk, DistributedBackend, DistributedConfig, DistributedNdarray, DistributionStrategy,
};

// Re-export GPU array implementation
pub use self::gpu_impl::{
    kernels as gpu_kernels, GPUArrayBuilder, GPUBackend, GPUConfig, GPUNdarray,
};

// Re-export JIT compilation implementation
pub use self::jit_impl::{
    CraneliftFunctionFactory, JITBackend, JITConfig, JITEnabledArray, JITFunctionImpl, JITManager,
    LLVMFunctionFactory,
};

// Re-export array operations
pub use self::operations::{
    add, apply_elementwise, concatenate, inverse, matmul, multiply, reshape, subtract, sum, svd,
    transpose, OperationError,
};

// Re-export ml_ops
pub use self::ml_ops::{
    activation, batch_norm, conv2d, cross_entropy, dropout, max_pool2d, self_attention,
    ActivationFunc,
};

/// Initializes the array protocol system.
///
/// This function initializes the array protocol system by registering the
/// default JIT function factories and other components. It should be called
/// before using any of the array protocol features.
#[allow(dead_code)]
pub fn init() {
    // Initialize the JIT manager
    let mut jit_manager = JITManager::global().write().expect("Operation failed");
    jit_manager.initialize();
}

/// Extra traits for third-party array implementations.
pub mod traits {
    use super::*;

    /// Trait for arrays that support strided access.
    pub trait StridedArray: ArrayProtocol {
        /// Get the strides of this array.
        #[must_use]
        fn strides(&self) -> Vec<usize>;

        /// Check if this array is contiguous.
        #[must_use]
        fn is_contiguous(&self) -> bool;

        /// Check if this array is Fortran-contiguous (column-major).
        #[must_use]
        fn is_fortran_contiguous(&self) -> bool;
    }

    /// Trait for arrays that support zero-copy operations.
    pub trait ZeroCopyArray: ArrayProtocol {
        /// Create a view of this array.
        #[must_use]
        fn view(&self) -> Box<dyn ZeroCopyArray>;

        /// Create a mutable view of this array.
        #[must_use]
        fn view_mut(&mut self) -> Box<dyn ZeroCopyArray>;

        /// Check if this array is a view.
        #[must_use]
        fn is_view(&self) -> bool;
    }

    /// Trait for arrays that support automatic differentiation.
    pub trait DifferentiableArray: ArrayProtocol {
        /// Compute the gradient of this array with respect to some variables.
        fn gradient(
            &self,
            variables: &[Box<dyn DifferentiableArray>],
        ) -> Vec<Box<dyn DifferentiableArray>>;

        /// Set whether to record operations for automatic differentiation.
        fn set_requiresgrad(&mut self, requiresgrad: bool);

        /// Check if this array requires gradient computation.
        #[must_use]
        fn requiresgrad(&self) -> bool;

        /// Get the gradient of this array.
        #[must_use]
        fn grad(&self) -> Option<Box<dyn DifferentiableArray>>;
    }

    /// Trait for arrays that support asynchronous operations.
    pub trait AsyncArray: ArrayProtocol {
        /// Perform an asynchronous operation on this array.
        fn async_op<F, R>(&self, op: F) -> impl std::future::Future<Output = CoreResult<R>>
        where
            F: FnOnce(&Self) -> CoreResult<R> + Send + 'static,
            R: Send + 'static;

        /// Check if this array supports asynchronous operations.
        #[must_use]
        fn supports_async(&self) -> bool;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_protocol_registry() {
        // Create a function and register it
        let implementation = Arc::new(
            move |_args: &[Box<dyn Any>], _kwargs: &HashMap<String, Box<dyn Any>>| {
                Ok(Box::new(42.0) as Box<dyn Any>)
            },
        );

        let func = ArrayFunction {
            name: "scirs2::test::test_func",
            implementation,
        };

        let registry = ArrayFunctionRegistry::global();
        {
            let mut reg = registry.write().expect("Operation failed");
            reg.register(func.clone());
        }

        // Verify the function was registered
        {
            let reg = registry.read().expect("Operation failed");
            let registered_func = reg
                .get("scirs2::test::test_func")
                .expect("Operation failed");
            assert_eq!(registered_func.name, "scirs2::test::test_func");
        }
    }

    #[test]
    fn test_mock_distributed_array() {
        let array = MockDistributedArray::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(array.is_distributed());

        let info = array.distribution_info();
        assert_eq!(
            info.get("type").expect("Operation failed"),
            "mock_distributed"
        );
        assert_eq!(info.get("chunks").expect("Operation failed"), "3");
    }

    #[test]
    fn test_mock_gpu_array() {
        let array = MockGPUArray::new(vec![1.0, 2.0, 3.0], vec![3], "cuda:0".to_string());
        assert!(array.is_on_gpu());

        let info = array.device_info();
        assert_eq!(info.get("device").expect("Operation failed"), "cuda:0");
        assert_eq!(info.get("type").expect("Operation failed"), "mock_gpu");
    }

    #[test]
    fn test_box_clone() {
        // Test Box<dyn ArrayProtocol> cloning for NdarrayWrapper
        let array = crate::ndarray::Array2::<f64>::ones((3, 3));
        let wrapped = NdarrayWrapper::new(array);
        let boxed: Box<dyn ArrayProtocol> = Box::new(wrapped);
        let cloned = boxed.clone();

        // Verify the clone is correct
        assert_eq!(cloned.shape(), &[3, 3]);

        // Test Box<dyn ArrayProtocol> cloning for MockDistributedArray
        let array = MockDistributedArray::new(vec![1.0, 2.0, 3.0], vec![3]);
        let boxed: Box<dyn ArrayProtocol> = Box::new(array);
        let cloned = boxed.clone();

        // Verify the clone is correct
        assert_eq!(cloned.shape(), &[3]);
    }
}

/// Examples of using the array protocol.
#[cfg(test)]
mod examples {
    use super::*;
    use ::ndarray::Array2;
    use std::any::Any;
    use std::collections::HashMap;

    /// Example: Create and use a distributed array.
    #[test]
    fn example_distributed_array() {
        // Non-constant data: an all-ones array wouldn't distinguish real
        // chunk reconstruction from a stub that fabricates a plausible
        // all-ones placeholder (as `to_array` used to, before it was fixed
        // to actually reassemble the chunks — see distributed_impl.rs).
        let array = Array2::from_shape_fn((10, 5), |(i, j)| (i * 5 + j) as f64);

        // Create a distributed array configuration
        let config = DistributedConfig {
            chunks: 3,
            balance: true,
            strategy: DistributionStrategy::RowWise,
            backend: DistributedBackend::Threaded,
        };

        // Create a distributed array
        let dist_array = DistributedNdarray::from_array(&array, config);

        // Check that the array was split correctly
        assert_eq!(dist_array.num_chunks(), 3);
        assert_eq!(dist_array.shape(), &[10, 5]);

        // Convert back to a regular array
        let result = dist_array.to_array().expect("Operation failed");

        // Check that the result matches the original array. `to_array()`
        // returns `Array<f64, IxDyn>` while `array` is `Array2<f64>`, so a
        // direct `assert_eq!` doesn't type-check — compare via `into_dyn()`.
        assert_eq!(result.shape(), array.shape());
        assert_eq!(result, array.into_dyn());
    }

    /// Example: Create and use a GPU array.
    #[test]
    fn example_gpu_array() {
        // Create a regular array
        let array = Array2::<f64>::ones((10, 5));

        // Create a GPU array configuration
        let config = GPUConfig {
            backend: GPUBackend::CUDA,
            device_id: 0,
            async_ops: true,
            mixed_precision: false,
            memory_fraction: 0.9,
        };

        // Create a GPU array
        let gpu_array = GPUNdarray::new(array.clone(), config);

        // Check that the array was created correctly
        assert_eq!(gpu_array.shape(), &[10, 5]);
        assert!(gpu_array.is_on_gpu());

        // Get device information
        let info = gpu_array.device_info();
        assert_eq!(info.get("backend").expect("Operation failed"), "CUDA");

        // Test box_clone for GPU array
        let gpu_box: Box<dyn ArrayProtocol> = Box::new(gpu_array);
        let gpu_clone = gpu_box.clone();

        // Check the cloned GPU array
        assert_eq!(gpu_clone.shape(), &[10, 5]);
    }

    /// Example: Create and use a JIT-enabled array.
    #[test]
    fn example_jit_array() {
        // Initialize the JIT manager
        init();

        // Create a regular array
        let array = Array2::<f64>::ones((10, 5));
        let wrapped = NdarrayWrapper::new(array);

        // Create a JIT-enabled array
        let jitarray: JITEnabledArray<f64, NdarrayWrapper<f64, crate::ndarray::Ix2>> =
            JITEnabledArray::new(wrapped);

        // Check if JIT is supported
        assert!(jitarray.supports_jit());

        // Compile a function
        let expression = "x + y";
        let jit_function = jitarray.compile(expression).expect("Operation failed");

        // Check the function's properties
        assert_eq!(jit_function.source(), expression);

        // Get JIT information
        let info = jitarray.jit_info();
        assert_eq!(info.get("supports_jit").expect("Operation failed"), "true");

        // Test box_clone for JIT-enabled array
        let jit_box: Box<dyn ArrayProtocol> = Box::new(jitarray);
        let jit_clone = jit_box.clone();

        // Check the cloned JIT array
        assert_eq!(jit_clone.shape(), &[10, 5]);
    }

    /// Example: Test cloning Box<dyn ArrayProtocol>
    #[test]
    fn example_cloning_array_protocol_objects() {
        // Create a GPU array with box_clone support
        let array = Array2::<f64>::ones((10, 5));
        let config = GPUConfig::default();
        let gpu_array = GPUNdarray::new(array.clone(), config);

        // Box the array as ArrayProtocol and clone it
        let boxed: Box<dyn ArrayProtocol> = Box::new(gpu_array);
        let cloned = boxed.clone();

        // Verify the clone works correctly
        assert_eq!(cloned.shape(), &[10, 5]);

        // Create a distributed array and test box_clone
        let config = DistributedConfig {
            chunks: 3,
            balance: true,
            strategy: DistributionStrategy::RowWise,
            backend: DistributedBackend::Threaded,
        };
        let dist_array = DistributedNdarray::from_array(&array, config);

        // Box the array as ArrayProtocol and clone it
        let boxed: Box<dyn ArrayProtocol> = Box::new(dist_array);
        let cloned = boxed.clone();

        // Verify the clone works correctly
        assert_eq!(cloned.shape(), &[10, 5]);
    }

    /*
    // Commented out examples using macros - we'll fix these later

    /// Example: Define an array function using the macro.
    /// Example: Register and use an array function.
    #[test]
    fn example_array_function() {
        // Create a simple array function (without using macros)
        let funcname = "scirs2::example::sum";

        // Create an ArrayFunction manually
        let implementation = Arc::new(move |args: &[Box<dyn Any>], kwargs: &HashMap<String, Box<dyn Any>>| {
            if let Some(array) = args.get(0)
                .and_then(|arg| arg.downcast_ref::<Array2<f64>>()) {
                let sum = array.sum();
                Ok(Box::new(sum))
            } else {
                Err(CoreError::InvalidArgument(ErrorContext::new(
                    "Expected Array2<f64> as first argument".to_string()
                )))
            }
        });

        let func = ArrayFunction {
            name: funcname,
            implementation,
        };

        // Register the function
        let registry = ArrayFunctionRegistry::global();
        {
            let mut reg = registry.write().expect("Operation failed");
            reg.register(func.clone());
        }

        // Verify the function was registered
        {
            let reg = registry.read().expect("Operation failed");
            let registered_func = reg.get(funcname).expect("Operation failed");
            assert_eq!(registered_func.name, funcname);
        }
    }
    */

    /// Example: Interoperability between different array types
    #[test]
    fn example_array_interoperability() {
        // Initialize the system
        init();

        // Create arrays of different types
        let cpu_array = Array2::<f64>::ones((5, 5));

        // Create a GPU array
        let gpu_config = GPUConfig {
            backend: GPUBackend::CUDA,
            device_id: 0,
            async_ops: false,
            mixed_precision: false,
            memory_fraction: 0.9,
        };
        let gpu_array = GPUNdarray::new(cpu_array.clone(), gpu_config);

        // Create a distributed array
        let dist_config = DistributedConfig {
            chunks: 2,
            balance: true,
            strategy: DistributionStrategy::RowWise,
            backend: DistributedBackend::Threaded,
        };
        let dist_array = DistributedNdarray::from_array(&cpu_array, dist_config);

        // Simple test of interoperability: convert both to Box<dyn ArrayProtocol>
        let gpu_wrapper: Box<dyn ArrayProtocol> = Box::new(gpu_array);
        let dist_wrapper: Box<dyn ArrayProtocol> = Box::new(dist_array);

        // Verify the clones work correctly
        let gpu_clone = gpu_wrapper.clone();
        let dist_clone = dist_wrapper.clone();

        assert_eq!(gpu_clone.shape(), &[5, 5]);
        assert_eq!(dist_clone.shape(), &[5, 5]);
    }

    /// Example: Advanced usage with custom array type
    #[test]
    fn example_custom_array_type() {
        use std::sync::Arc;

        // Define a custom array type
        struct MyCustomArray<T> {
            data: Vec<T>,
            shape: Vec<usize>,
        }

        impl<T: Clone + 'static> MyCustomArray<T> {
            fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
                Self { data, shape }
            }

            // Commented out since it's unused but may be needed in the future
            // fn shape(&self) -> &[usize] {
            //     &self.shape
            // }
        }

        // Implement ArrayProtocol for the custom array type
        impl<T: Clone + Send + Sync + 'static> ArrayProtocol for MyCustomArray<T> {
            fn array_function(
                &self,
                func: &ArrayFunction,
                _types: &[TypeId],
                _args: &[Box<dyn Any>],
                _kwargs: &HashMap<String, Box<dyn Any>>,
            ) -> Result<Box<dyn Any>, NotImplemented> {
                if func.name == "scirs2::example::custom_sum" {
                    // Implement custom sum for our array type
                    match std::any::TypeId::of::<T>() {
                        tid if tid == std::any::TypeId::of::<f64>() => {
                            // For f64 arrays, cast to f64 slice
                            let f64_data = unsafe {
                                std::slice::from_raw_parts(
                                    self.data.as_ptr() as *const f64,
                                    self.data.len(),
                                )
                            };
                            let sum = f64_data.iter().sum::<f64>();
                            Ok(Box::new(sum))
                        }
                        tid if tid == std::any::TypeId::of::<f32>() => {
                            // For f32 arrays, cast to f32 slice
                            let f32_data = unsafe {
                                std::slice::from_raw_parts(
                                    self.data.as_ptr() as *const f32,
                                    self.data.len(),
                                )
                            };
                            let sum = f32_data.iter().sum::<f32>();
                            Ok(Box::new(sum))
                        }
                        _ => Err(NotImplemented),
                    }
                } else {
                    Err(NotImplemented)
                }
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn shape(&self) -> &[usize] {
                &self.shape
            }

            fn box_clone(&self) -> Box<dyn ArrayProtocol> {
                Box::new(MyCustomArray {
                    data: self.data.clone(),
                    shape: self.shape.clone(),
                })
            }
        }

        // Create an instance of the custom array type
        let custom_array = MyCustomArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        // Test box_clone functionality
        let boxed: Box<dyn ArrayProtocol> = Box::new(custom_array);
        let cloned = boxed.clone();

        // Verify the clone has the correct shape
        assert_eq!(cloned.shape(), &[2, 2]);

        // Create an ArrayFunction for testing
        let func = ArrayFunction {
            name: "scirs2::example::custom_sum",
            implementation: Arc::new(move |_args, _kwargs| {
                // Dummy implementation
                Ok(Box::new(42.0) as Box<dyn Any>)
            }),
        };

        // Test array_function directly
        let result = cloned.array_function(
            &func,
            &[std::any::TypeId::of::<f64>()],
            &[],
            &HashMap::new(),
        );

        // Verify we get a result (the sum of 1+2+3+4 = 10)
        assert!(result.is_ok());
        if let Ok(value) = result {
            let sum = *value.downcast_ref::<f64>().expect("Operation failed");
            assert_eq!(sum, 10.0);
        }
    }
}
/// Make `Box<dyn JITFunction>` cloneable via clone_box
impl Clone for Box<dyn JITFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
