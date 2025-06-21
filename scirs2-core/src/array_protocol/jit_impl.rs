// Copyright (c) 2025, `SciRS2` Team
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

//! Just-In-Time (JIT) compilation support for array operations.
//!
//! This module provides functionality for JIT-compiling operations on arrays,
//! allowing for faster execution of custom operations.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock, RwLock};

use crate::array_protocol::{ArrayProtocol, JITArray, JITFunction, JITFunctionFactory};
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

    /// The compiled function
    function: Box<JITFunctionType>,

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
            function,
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
        // Create a new JITFunctionImpl with a fresh function that behaves the same way
        let source = self.source.clone();
        let compile_info = self.compile_info.clone();

        // Create a dummy function that returns a constant value
        // In a real implementation, this would properly clone the behavior
        let cloned_function: Box<JITFunctionType> = Box::new(move |_args| {
            // Return a dummy result (42.0) as an example
            Ok(Box::new(42.0))
        });

        Box::new(Self {
            source,
            function: cloned_function,
            compile_info,
        })
    }
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

    /// Compile a function using LLVM.
    fn compile(&self, expression: &str, array_type_id: TypeId) -> CoreResult<Arc<dyn JITFunction>> {
        // In a real implementation, this would use LLVM to compile the function
        // For now, we'll just create a placeholder function

        // Create some compile info
        let mut compile_info = HashMap::new();
        compile_info.insert("backend".to_string(), "LLVM".to_string());
        compile_info.insert("opt_level".to_string(), self.config.opt_level.to_string());
        compile_info.insert("array_type".to_string(), format!("{array_type_id:?}"));

        // Create a function that just returns a constant value
        // In a real implementation, this would be a compiled function
        let source = expression.to_string();
        let function: Box<JITFunctionType> = Box::new(move |_args| {
            // Mock function just returns a constant value
            Ok(Box::new(42.0))
        });

        // Create the JIT function
        let jit_function = JITFunctionImpl::new(source, function, compile_info);

        Ok(Arc::new(jit_function))
    }
}

impl JITFunctionFactory for LLVMFunctionFactory {
    fn create(&self, expression: &str, array_type_id: TypeId) -> CoreResult<Box<dyn JITFunction>> {
        // Check if the function is already in the cache
        if self.config.use_cache {
            let cache_key = format!("{expression}-{array_type_id:?}");
            if let Some(cached_fn) = self.cache.get(&cache_key) {
                return Ok(cached_fn.as_ref().clone_box());
            }
        }

        // Compile the function
        let jit_function = self.compile(expression, array_type_id)?;

        if self.config.use_cache {
            // Add the function to the cache
            let cache_key = format!("{expression}-{array_type_id:?}");
            // In a real implementation, we'd need to handle this in a thread-safe way
            // For now, we'll just clone the function
            let mut cache = self.cache.clone();
            cache.insert(cache_key, jit_function.clone());
        }

        // Clone the function and return it
        Ok(jit_function.as_ref().clone_box())
    }

    fn supports_array_type(&self, _array_type_id: TypeId) -> bool {
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

    /// Compile a function using Cranelift.
    fn compile(&self, expression: &str, array_type_id: TypeId) -> CoreResult<Arc<dyn JITFunction>> {
        // In a real implementation, this would use Cranelift to compile the function
        // For now, we'll just create a placeholder function

        // Create some compile info
        let mut compile_info = HashMap::new();
        compile_info.insert("backend".to_string(), "Cranelift".to_string());
        compile_info.insert("opt_level".to_string(), self.config.opt_level.to_string());
        compile_info.insert("array_type".to_string(), format!("{array_type_id:?}"));

        // Create a function that just returns a constant value
        // In a real implementation, this would be a compiled function
        let source = expression.to_string();
        let function: Box<JITFunctionType> = Box::new(move |_args| {
            // Mock function just returns a constant value
            Ok(Box::new(42.0))
        });

        // Create the JIT function
        let jit_function = JITFunctionImpl::new(source, function, compile_info);

        Ok(Arc::new(jit_function))
    }
}

impl JITFunctionFactory for CraneliftFunctionFactory {
    fn create(&self, expression: &str, array_type_id: TypeId) -> CoreResult<Box<dyn JITFunction>> {
        // Check if the function is already in the cache
        if self.config.use_cache {
            let cache_key = format!("{expression}-{array_type_id:?}");
            if let Some(cached_fn) = self.cache.get(&cache_key) {
                return Ok(cached_fn.as_ref().clone_box());
            }
        }

        // Compile the function
        let jit_function = self.compile(expression, array_type_id)?;

        if self.config.use_cache {
            // Add the function to the cache
            let cache_key = format!("{expression}-{array_type_id:?}");
            // In a real implementation, we'd need to handle this in a thread-safe way
            // For now, we'll just clone the function
            let mut cache = self.cache.clone();
            cache.insert(cache_key, jit_function.clone());
        }

        // Clone the function and return it
        Ok(jit_function.as_ref().clone_box())
    }

    fn supports_array_type(&self, _array_type_id: TypeId) -> bool {
        // For simplicity, we'll say this factory supports all array types
        true
    }
}

/// A JIT manager that selects the appropriate factory for a given array type
pub struct JITManager {
    /// The available JIT function factories
    factories: Vec<Box<dyn JITFunctionFactory>>,

    /// Default configuration for JIT compilation
    default_config: JITConfig,
}

impl JITManager {
    /// Create a new JIT manager.
    pub fn new(default_config: JITConfig) -> Self {
        Self {
            factories: Vec::new(),
            default_config,
        }
    }

    /// Register a JIT function factory.
    pub fn register_factory(&mut self, factory: Box<dyn JITFunctionFactory>) {
        self.factories.push(factory);
    }

    /// Get a JIT function factory that supports the given array type.
    pub fn get_factory_for_array_type(
        &self,
        array_type_id: TypeId,
    ) -> Option<&dyn JITFunctionFactory> {
        for factory in &self.factories {
            if factory.supports_array_type(array_type_id) {
                return Some(&**factory);
            }
        }
        None
    }

    /// Compile a JIT function for the given expression and array type.
    pub fn compile(
        &self,
        expression: &str,
        array_type_id: TypeId,
    ) -> CoreResult<Box<dyn JITFunction>> {
        // Find a factory that supports the array type
        if let Some(factory) = self.get_factory_for_array_type(array_type_id) {
            factory.create(expression, array_type_id)
        } else {
            Err(CoreError::JITError(ErrorContext::new(format!(
                "No JIT factory supports array type: {:?}",
                array_type_id
            ))))
        }
    }

    /// Initialize the JIT manager with default factories.
    pub fn initialize(&mut self) {
        // Create and register the default factories
        let llvm_config = JITConfig {
            backend: JITBackend::LLVM,
            ..self.default_config.clone()
        };
        let llvm_factory = Box::new(LLVMFunctionFactory::new(llvm_config));

        let cranelift_config = JITConfig {
            backend: JITBackend::Cranelift,
            ..self.default_config.clone()
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
                default_config: JITConfig {
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
    _phantom: PhantomData<T>,
}

impl<T, A> JITEnabledArray<T, A> {
    /// Create a new JIT-enabled array.
    pub fn new(inner: A) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
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
            _phantom: PhantomData::<T>,
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
        let jit_manager = jit_manager.read().unwrap();

        // Compile the function
        jit_manager.compile(expression, TypeId::of::<A>())
    }

    fn supports_jit(&self) -> bool {
        // Check if there's a factory that supports this array type
        let jit_manager = JITManager::global();
        let jit_manager = jit_manager.read().unwrap();

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
            let jit_manager = jit_manager.read().unwrap();

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
        func: &crate::array_protocol::ArrayFunction,
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
            _phantom: PhantomData::<T>,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_protocol::NdarrayWrapper;
    use ndarray::Array2;

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
        let array_type_id = TypeId::of::<NdarrayWrapper<f64, ndarray::Ix2>>();
        let jit_function = factory.create(expression, array_type_id).unwrap();

        // Check the function's properties
        assert_eq!(jit_function.source(), expression);
        let compile_info = jit_function.compile_info();
        assert_eq!(compile_info.get("backend").unwrap(), "LLVM");
    }

    #[test]
    fn test_jit_manager() {
        // Initialize the JIT manager
        let mut jit_manager = JITManager::new(JITConfig::default());
        jit_manager.initialize();

        // Check that the factories were registered
        let array_type_id = TypeId::of::<NdarrayWrapper<f64, ndarray::Ix2>>();
        assert!(jit_manager
            .get_factory_for_array_type(array_type_id)
            .is_some());

        // Compile a function
        let expression = "x + y";
        let jit_function = jit_manager.compile(expression, array_type_id).unwrap();

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
            let mut jit_manager = JITManager::global().write().unwrap();
            jit_manager.initialize();
        }

        // Check if JIT is supported
        assert!(jit_array.supports_jit());

        // Compile a function
        let expression = "x + y";
        let jit_function = jit_array.compile(expression).unwrap();

        // Check the function's properties
        assert_eq!(jit_function.source(), expression);
    }
}
