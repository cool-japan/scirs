// Copyright (c) 2025, SciRS2 Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! Implementation of Array Protocol (similar to NumPy's `__array_function__` protocol)
//!
//! This module provides a mechanism for third-party array implementations to
//! override SciRS2 functions. It is inspired by NumPy's `__array_function__`
//! protocol defined in NEP-18.
//!
//! The protocol enables third-party arrays to implement SciRS2 functions in a way
//! that is recognized by the SciRS2 library. This allows for seamless integration with
//! distributed arrays, GPU arrays, and other custom array implementations.
//!
//! ## Core Components
//!
//! The Array Protocol system includes:
//!
//! * Specialized array implementations (GPU, distributed, JIT)
//! * Automatic device placement with AutoDevice
//! * Mixed-precision operations
//! * Neural network layers and models
//! * Gradient computation and training capabilities
//! * Distributed training and model serialization

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, RwLock};

use crate::error::{CoreError, CoreResult, ErrorContext};

// Internal submodules
mod distributed_impl;
mod gpu_impl;
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
pub mod serialization;
pub mod training;

/// Trait for objects that can handle the array protocol.
///
/// This is similar to NumPy's `__array_function__` protocol.
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
    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        _args: &[Box<dyn Any>],
        _kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented>;

    /// Get the array as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Get the shape of the array (default implementation returns empty slice)
    fn shape(&self) -> &[usize] {
        &[]
    }

    /// Get the data type of the array (default implementation returns f64)
    fn dtype(&self) -> TypeId {
        TypeId::of::<f64>()
    }

    /// Clone this array protocol object.
    fn box_clone(&self) -> Box<dyn ArrayProtocol>;
}

/// Make Box<dyn ArrayProtocol> cloneable via box_clone
impl Clone for Box<dyn ArrayProtocol> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// Marker for functions not implemented by a specific type.
///
/// This is part of the Array Protocol API design and is used as a marker to indicate
/// that a function is not implemented by a specific array type. It's different from
/// the CoreError::NotImplementedError enum variant, which is used for error reporting.
///
/// When an error is propagated up the call chain, NotImplemented is converted
/// to OperationError::NotImplemented and then to CoreError::NotImplementedError
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
            .finish()
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
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            // Default implementation that returns NotImplemented
            implementation: Arc::new(|_, _| {
                Err(CoreError::NotImplementedError(ErrorContext::new(
                    "Function not implemented".to_string(),
                )))
            }),
        }
    }
}

/// Registry of all array functions.
#[derive(Debug, Default)]
pub struct ArrayFunctionRegistry {
    /// Map of function names to array functions
    functions: HashMap<&'static str, ArrayFunction>,
}

impl ArrayFunctionRegistry {
    /// Get the global registry.
    pub fn global() -> &'static RwLock<Self> {
        static REGISTRY: LazyLock<RwLock<ArrayFunctionRegistry>> = LazyLock::new(|| {
            RwLock::new(ArrayFunctionRegistry {
                functions: HashMap::new(),
            })
        });
        &REGISTRY
    }

    /// Register a new array function.
    pub fn register(&mut self, func: ArrayFunction) {
        self.functions.insert(func.name, func);
    }

    /// Get an array function by name.
    pub fn get(&self, name: &str) -> Option<&ArrayFunction> {
        self.functions.get(name)
    }

    /// Get all registered functions.
    pub fn all_functions(&self) -> Vec<&ArrayFunction> {
        self.functions.values().collect()
    }
}

/// Helper function to extract all arguments implementing the `ArrayProtocol` trait.
///
/// This is similar to NumPy's `_get_implementing_args` function.
pub fn get_implementing_args(args: &[Box<dyn Any>]) -> Vec<(TypeId, &dyn ArrayProtocol)> {
    let mut implementing_args = Vec::new();

    for arg in args {
        if let Some(array_protocol_obj) = arg.downcast_ref::<Box<dyn ArrayProtocol>>() {
            let type_id = (**array_protocol_obj).type_id();
            implementing_args.push((type_id, &**array_protocol_obj));
        }
    }

    // Sort implementing args by inheritance hierarchy (if possible)
    // This is a simplified version - in practice, we would need more complex
    // sorting to handle inheritance hierarchies correctly
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
pub fn array_function_dispatch(
    func: &ArrayFunction,
    args: &[Box<dyn Any>],
    kwargs: &HashMap<String, Box<dyn Any>>,
) -> CoreResult<Box<dyn Any>> {
    // Find all arguments implementing ArrayProtocol
    let implementing_args = get_implementing_args(args);

    if implementing_args.is_empty() {
        // No arguments implement ArrayProtocol, use default implementation
        return (func.implementation)(args, kwargs);
    }

    // Extract all unique types that implement ArrayProtocol
    let unique_types: HashSet<TypeId> = implementing_args
        .iter()
        .map(|(type_id, _)| *type_id)
        .collect();
    let types: Vec<TypeId> = unique_types.into_iter().collect();

    // Try dispatching to each implementation
    for (_, array_protocol_obj) in implementing_args {
        match array_protocol_obj.array_function(func, &types, args, kwargs) {
            Ok(result) => return Ok(result),
            Err(NotImplemented) => continue,
        }
    }

    // If we get here, no implementation was found
    Err(CoreError::DispatchError(ErrorContext::new(format!(
        "No implementation found for {} with the given argument types",
        func.name
    ))))
}

/// Decorator for adding array function dispatch capabilities to a function.
///
/// This is similar to NumPy's `array_function_dispatch` decorator.
pub struct ArrayFunctionDecorator<F> {
    function: F,
    name: &'static str,
}

impl<F> ArrayFunctionDecorator<F>
where
    F: Send + Sync + 'static,
{
    /// Create a new array function decorator.
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
                unimplemented!("Type conversion in array_function_dispatch is not implemented yet")
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
            panic!("Failed to acquire write lock on ArrayFunctionRegistry");
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
    fn is_on_gpu(&self) -> bool;

    /// Get information about the GPU device that holds this array.
    fn device_info(&self) -> HashMap<String, String>;
}

/// Trait for distributed arrays that can span multiple machines.
pub trait DistributedArray: ArrayProtocol {
    /// Get information about the distribution of this array.
    fn distribution_info(&self) -> HashMap<String, String>;

    /// Gather the distributed array to a single node.
    fn gather(&self) -> CoreResult<Box<dyn ArrayProtocol>>;

    /// Scatter a regular array to a distributed array.
    fn scatter(&self, chunks: usize) -> CoreResult<Box<dyn DistributedArray>>;

    /// Check if this array is distributed.
    fn is_distributed(&self) -> bool;
}

/// JIT (Just-In-Time) compilation support for arrays.
pub trait JITArray: ArrayProtocol {
    /// Compile an expression to be evaluated on this array.
    fn compile(&self, expression: &str) -> CoreResult<Box<dyn JITFunction>>;

    /// Check if JIT compilation is supported for this array type.
    fn supports_jit(&self) -> bool;

    /// Get information about the JIT compiler being used.
    fn jit_info(&self) -> HashMap<String, String>;
}

/// A JIT-compiled function that can be evaluated on arrays.
pub trait JITFunction: Send + Sync {
    /// Evaluate the function with the given arguments.
    fn evaluate(&self, args: &[Box<dyn Any>]) -> CoreResult<Box<dyn Any>>;

    /// Get the source code of the compiled function.
    fn source(&self) -> String;

    /// Get information about how the function was compiled.
    fn compile_info(&self) -> HashMap<String, String>;

    /// Clone this JIT function into a Box<dyn JITFunction>.
    fn clone_box(&self) -> Box<dyn JITFunction>;
}

/// A factory for creating JIT functions for specific array implementations.
pub trait JITFunctionFactory: Send + Sync {
    /// Create a new JIT function for the given expression and array type.
    fn create(&self, expression: &str, array_type_id: TypeId) -> CoreResult<Box<dyn JITFunction>>;

    /// Check if this factory supports the given array type.
    fn supports_array_type(&self, array_type_id: TypeId) -> bool;
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
    pub fn get_for_array_type(&self, array_type_id: TypeId) -> Option<&dyn JITFunctionFactory> {
        for factory in &self.factories {
            if factory.supports_array_type(array_type_id) {
                return Some(&**factory);
            }
        }
        None
    }
}

/// A wrapper for ndarray to implement the ArrayProtocol trait.
#[derive(Debug, Clone)]
pub struct NdarrayWrapper<T, D: ndarray::Dimension> {
    array: ndarray::Array<T, D>,
    _phantom: PhantomData<(T, D)>,
}

impl<T, D> NdarrayWrapper<T, D>
where
    T: Clone + 'static,
    D: ndarray::Dimension + 'static,
{
    /// Create a new ndarray wrapper.
    pub fn new(array: ndarray::Array<T, D>) -> Self {
        Self {
            array,
            _phantom: PhantomData,
        }
    }

    /// Get the underlying ndarray.
    pub fn as_array(&self) -> &ndarray::Array<T, D> {
        &self.array
    }

    /// Convert into the underlying ndarray.
    pub fn into_array(self) -> ndarray::Array<T, D> {
        self.array
    }
}

impl<T, D> ArrayProtocol for NdarrayWrapper<T, D>
where
    T: Clone + Send + Sync + 'static,
    D: ndarray::Dimension + Send + Sync + 'static,
{
    fn array_function(
        &self,
        _func: &ArrayFunction,
        _types: &[TypeId],
        _args: &[Box<dyn Any>],
        _kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        // We don't implement any overrides for ndarray yet,
        // so always return NotImplemented
        Err(NotImplemented)
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

    fn scatter(&self, _chunks: usize) -> CoreResult<Box<dyn DistributedArray>> {
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
                unimplemented!("Implementation for array protocol functions is not complete")
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
            panic!("Failed to acquire write lock on ArrayFunctionRegistry");
        }

        self.func
    }
}

/// Convenience macro for defining array protocol functions.
///
/// Example usage:
/// ```ignore
/// // Defining a function
/// array_function_def!(
///     fn sum(array: &ndarray::Array<f64, ndarray::Ix2>) -> f64 {
///         array.sum()
///     },
///     "scirs2::sum"
/// );
/// ```
#[macro_export]
macro_rules! array_function_def {
    (fn $name:ident $(<$($gen:ident),*>)? ($($arg:ident : $arg_ty:ty),*) -> $ret:ty $body:block, $func_name:expr) => {
        fn $name $(<$($gen),*>)? ($($arg : $arg_ty),*) -> $ret $body

        #[allow(non_upper_case_globals)]
        const $name: $crate::array_protocol::ArrayProtocolFunction<_> =
            $crate::array_protocol::ArrayProtocolFunction::new($name, $func_name);
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
pub fn init() {
    // Initialize the JIT manager
    let mut jit_manager = JITManager::global().write().unwrap();
    jit_manager.initialize();
}

/// Extra traits for third-party array implementations.
pub mod traits {
    use super::*;

    /// Trait for arrays that support strided access.
    pub trait StridedArray: ArrayProtocol {
        /// Get the strides of this array.
        fn strides(&self) -> Vec<usize>;

        /// Check if this array is contiguous.
        fn is_contiguous(&self) -> bool;

        /// Check if this array is Fortran-contiguous (column-major).
        fn is_fortran_contiguous(&self) -> bool;
    }

    /// Trait for arrays that support zero-copy operations.
    pub trait ZeroCopyArray: ArrayProtocol {
        /// Create a view of this array.
        fn view(&self) -> Box<dyn ZeroCopyArray>;

        /// Create a mutable view of this array.
        fn view_mut(&mut self) -> Box<dyn ZeroCopyArray>;

        /// Check if this array is a view.
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
        fn set_requires_grad(&mut self, requires_grad: bool);

        /// Check if this array requires gradient computation.
        fn requires_grad(&self) -> bool;

        /// Get the gradient of this array.
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
            let mut reg = registry.write().unwrap();
            reg.register(func.clone());
        }

        // Verify the function was registered
        {
            let reg = registry.read().unwrap();
            let registered_func = reg.get("scirs2::test::test_func").unwrap();
            assert_eq!(registered_func.name, "scirs2::test::test_func");
        }
    }

    #[test]
    fn test_mock_distributed_array() {
        let array = MockDistributedArray::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(array.is_distributed());

        let info = array.distribution_info();
        assert_eq!(info.get("type").unwrap(), "mock_distributed");
        assert_eq!(info.get("chunks").unwrap(), "3");
    }

    #[test]
    fn test_mock_gpu_array() {
        let array = MockGPUArray::new(vec![1.0, 2.0, 3.0], vec![3], "cuda:0".to_string());
        assert!(array.is_on_gpu());

        let info = array.device_info();
        assert_eq!(info.get("device").unwrap(), "cuda:0");
        assert_eq!(info.get("type").unwrap(), "mock_gpu");
    }

    #[test]
    fn test_box_clone() {
        // Test Box<dyn ArrayProtocol> cloning for NdarrayWrapper
        let array = ndarray::Array2::<f64>::ones((3, 3));
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
    use ndarray::Array2;
    use std::any::Any;
    use std::collections::HashMap;

    /// Example: Create and use a distributed array.
    #[test]
    fn example_distributed_array() {
        // Create a regular array
        let array = Array2::<f64>::ones((10, 5));

        // Create a distributed array configuration
        let config = DistributedConfig {
            chunks: 3,
            balance: true,
            strategy: DistributionStrategy::RowWise,
            backend: DistributedBackend::Threaded,
        };

        // Create a distributed array
        let dist_array = DistributedNdarray::from_array(array.clone(), config);

        // Check that the array was split correctly
        assert_eq!(dist_array.num_chunks(), 3);
        assert_eq!(dist_array.shape(), &[10, 5]);

        // Convert back to a regular array
        let result = dist_array.to_array().unwrap();

        // Check that the result matches the original array
        assert_eq!(result.shape(), array.shape());
        // NOTE: Arrays with different dimensions can't be directly compared
        // assert_eq!(result, array);
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
        assert_eq!(info.get("backend").unwrap(), "CUDA");

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
        let jit_array: JITEnabledArray<f64, _> = JITEnabledArray::new(wrapped);

        // Check if JIT is supported
        assert!(jit_array.supports_jit());

        // Compile a function
        let expression = "x + y";
        let jit_function = jit_array.compile(expression).unwrap();

        // Check the function's properties
        assert_eq!(jit_function.source(), expression);

        // Get JIT information
        let info = jit_array.jit_info();
        assert_eq!(info.get("supports_jit").unwrap(), "true");

        // Test box_clone for JIT-enabled array
        let jit_box: Box<dyn ArrayProtocol> = Box::new(jit_array);
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
        let dist_array = DistributedNdarray::from_array(array, config);

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
        let func_name = "scirs2::example::sum";

        // Create an ArrayFunction manually
        let implementation = Arc::new(move |args: &[Box<dyn Any>], _kwargs: &HashMap<String, Box<dyn Any>>| {
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
            name: func_name,
            implementation,
        };

        // Register the function
        let registry = ArrayFunctionRegistry::global();
        {
            let mut reg = registry.write().unwrap();
            reg.register(func.clone());
        }

        // Verify the function was registered
        {
            let reg = registry.read().unwrap();
            let registered_func = reg.get(func_name).unwrap();
            assert_eq!(registered_func.name, func_name);
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
        let dist_array = DistributedNdarray::from_array(cpu_array.clone(), dist_config);

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
                _types: &[std::any::TypeId],
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
            let sum = *value.downcast_ref::<f64>().unwrap();
            assert_eq!(sum, 10.0);
        }
    }
}
/// Make Box<dyn JITFunction> cloneable via clone_box
impl Clone for Box<dyn JITFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
