// Copyright (c) 2025, SciRS2 Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Tests for the Array Protocol implementation.

use scirs2_core::array_protocol::{
    self,
    ArrayFunction,
    ArrayProtocol,
    DistributedBackend,
    DistributedConfig,
    DistributedNdarray,
    DistributionStrategy,
    GPUArray,
    GPUBackend,
    GPUConfig,
    GPUNdarray,
    JITArray,
    // Remove unused imports:
    // JITConfig, JITBackend
    JITEnabledArray,
    NdarrayWrapper,
    NotImplemented,
};

// Define a simpler version of the array_function macro for tests
macro_rules! array_function {
    (fn $name:ident($($arg:ident: $arg_ty:ty),* $(,)?) -> $ret:ty $body:block, $funcname:expr) => {
        // Define the function
        fn $name($($arg: $arg_ty),*) -> $ret $body
    };
}
use scirs2_core::ndarray_ext::{arr2, Array2};
use std::any::{Any, TypeId};
use std::collections::HashMap;

#[test]
#[allow(dead_code)]
fn test_ndarray_wrapper() {
    // Create a regular ndarray
    let arr = Array2::<f64>::ones((3, 3));

    // Wrap it in the NdarrayWrapper
    let wrapped = NdarrayWrapper::new(arr.clone());

    // Check that it implements the ArrayProtocol trait
    let proto: &dyn ArrayProtocol = &wrapped;

    // Check that we can get the original array back
    let unwrapped = wrapped.as_array();
    assert_eq!(unwrapped.shape(), arr.shape());
    assert_eq!(unwrapped, &arr);
}

#[test]
#[allow(dead_code)]
fn test_gpu_array() {
    // Create a regular ndarray
    let arr = Array2::<f64>::ones((3, 3));

    // Create a GPU array configuration
    let config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: false,
        mixed_precision: false,
        memory_fraction: 0.9,
    };

    // Create a GPU array
    let gpu_array = GPUNdarray::new(arr.clone(), config);

    // Check properties
    assert_eq!(gpu_array.shape(), &[3, 3]);
    assert!(gpu_array.is_on_gpu());

    // Check device info
    let info = gpu_array.device_info();
    assert!(info.contains_key("backend"));
    assert_eq!(info.get("backend").unwrap_or(&"".to_string()), "CUDA");

    // Convert back to CPU
    match gpu_array.to_cpu() {
        Ok(cpu_array) => {
            // First check if we can downcast to IxDyn
            if let Some(wrapped) = cpu_array
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::IxDyn>>()
            {
                assert_eq!(wrapped.as_array().shape(), arr.shape());
            }
            // If not, try to downcast to Ix2 which might be used instead
            else if let Some(wrapped) = cpu_array
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>()
            {
                assert_eq!(wrapped.as_array().shape(), arr.shape());
            } else {
                // If downcast failed, at least check the shape through the ArrayProtocol trait
                assert_eq!(cpu_array.shape(), arr.shape());
            }
        }
        Err(e) => panic!("Failed to convert GPU array to CPU: {e}"),
    }
}

#[test]
#[allow(dead_code)]
fn test_distributed_array() {
    // Create a regular ndarray
    let arr = Array2::<f64>::ones((10, 5));

    // Create a distributed array configuration
    let config = DistributedConfig {
        chunks: 3,
        balance: true,
        strategy: DistributionStrategy::RowWise,
        backend: DistributedBackend::Threaded,
    };

    // Create a distributed array
    let dist_array = DistributedNdarray::from_array(&arr, config);

    // Check properties
    assert_eq!(dist_array.shape(), &[10, 5]);
    assert_eq!(dist_array.num_chunks(), 3);

    // Convert back to a regular array
    let result = dist_array.to_array().expect("Test: operation failed");
    assert_eq!(result.shape(), arr.shape());

    // Convert both arrays to IxDyn for comparison
    let result_dyn = result.into_dyn();
    let arr_dyn = arr.into_dyn();
    assert_eq!(result_dyn, arr_dyn);
}

#[test]
#[allow(dead_code)]
fn test_jit_array() {
    // Initialize the array protocol system
    array_protocol::init();

    // Create a regular ndarray
    let arr = Array2::<f64>::ones((3, 3));
    let wrapped = NdarrayWrapper::new(arr);

    // Create a JIT-enabled array
    let jitarray = JITEnabledArray::<f64, _>::new(wrapped);

    // Check properties
    assert!(jitarray.supports_jit());

    // Compile a function
    let expression = "x + y";
    let jit_function = jitarray
        .compile(expression)
        .expect("Test: operation failed");

    // Check function properties
    assert_eq!(jit_function.source(), expression);

    // Get JIT info
    let info = jitarray.jit_info();
    assert_eq!(
        info.get("supports_jit").expect("Test: operation failed"),
        "true"
    );
}

#[test]
#[allow(dead_code)]
fn test_array_function_dispatch() {
    // Initialize the array protocol system
    array_protocol::init();

    // Define a custom function with a more specific name
    let test_function_name = "scirs2::test::sum_array";

    // Manually create and register the function with an implementation
    let implementation = std::sync::Arc::new(
        move |_args: &[Box<dyn std::any::Any>],
              kwargs: &std::collections::HashMap<String, Box<dyn std::any::Any>>| {
            // In a real implementation, we would extract the arguments properly
            // For this test, we just return a fixed result
            Ok(Box::new(10.0f64) as Box<dyn std::any::Any>)
        },
    );

    let func = array_protocol::ArrayFunction {
        name: test_function_name,
        implementation,
    };

    // Register the function with the global registry
    let registry = array_protocol::ArrayFunctionRegistry::global();
    {
        let mut registry_write = registry.write().expect("Test: operation failed");
        registry_write.register(func);
    }

    // Now, define the test function using the macro
    array_function!(
        fn sum_array(arr: &Array2<f64>) -> f64 {
            arr.sum()
        },
        "test::sum_array"
    );

    // Use the function directly
    let registered_sum = sum_array;

    // Create an array and test the function
    let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let sum = registered_sum(&array);
    assert_eq!(sum, 10.0);

    // Check that the function was registered with the global registry
    let registry = array_protocol::ArrayFunctionRegistry::global();
    let registry = registry.read().expect("Test: operation failed");

    // Check for our custom function first
    if let Some(func) = registry.get(test_function_name) {
        assert_eq!(func.name, test_function_name);
    } else {
        panic!("Custom function was not registered correctly");
    }

    // In case the test::sum_array is registered separately
    if let Some(func) = registry.get("test::sum_array") {
        assert_eq!(func.name, "test::sum_array");
    }
}

#[test]
#[allow(dead_code)]
fn test_array_interoperability() {
    // Initialize the array protocol system
    array_protocol::init();

    // Create arrays of different types
    let cpu_array = Array2::<f64>::ones((3, 3));

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

    // Define an operation that works with any array type
    array_function!(
        fn dot_product(
            a: &dyn ArrayProtocol,
            b: &dyn ArrayProtocol,
        ) -> Result<Box<dyn ArrayProtocol>, NotImplemented> {
            // In a real implementation, this would dispatch to the appropriate implementation
            // based on the array types. For this test, we'll use a simplified implementation.
            // `cpu_array` below is `Array2<f64>` (Ix2), so `a_wrapped`/`b_wrapped` are
            // `NdarrayWrapper<f64, Ix2>` — matching against `IxDyn` here would never
            // downcast successfully, silently reporting every call as unimplemented.
            let a_array = a
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>();
            let b_array = b
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>();

            if let (Some(a), Some(b)) = (a_array, b_array) {
                let result = a.as_array().dot(b.as_array());
                Ok(Box::new(NdarrayWrapper::new(result)))
            } else {
                // In a real implementation, we would try other combinations here
                Err(NotImplemented)
            }
        },
        "test::dot_product"
    );

    // The macro already defined the function above

    // Register a handler for the dot_product function in the global registry
    let dot_product_name = "test::dot_product";
    let implementation = std::sync::Arc::new(
        move |_args: &[Box<dyn std::any::Any>],
              kwargs: &std::collections::HashMap<String, Box<dyn std::any::Any>>| {
            // In a real implementation, we would extract the arguments properly
            // For this test, we just return a fixed result - a dummy NdarrayWrapper
            let dummy_array = scirs2_core::ndarray::Array2::<f64>::eye(3);
            let wrapped = NdarrayWrapper::new(dummy_array);
            Ok(Box::new(wrapped) as Box<dyn std::any::Any>)
        },
    );

    let func = array_protocol::ArrayFunction {
        name: dot_product_name,
        implementation,
    };

    // Register the function with the global registry
    let registry = array_protocol::ArrayFunctionRegistry::global();
    {
        let mut registry_write = registry.write().expect("Test: operation failed");
        registry_write.register(func);
    }

    // Use the function with the CPU array
    let a_wrapped = NdarrayWrapper::new(cpu_array.clone());
    let b_wrapped = NdarrayWrapper::new(cpu_array.clone());

    let result = dot_product(&a_wrapped, &b_wrapped)
        .expect("dot product of two NdarrayWrapper<f64, Ix2> operands should succeed");
    let result_array = result
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>()
        .expect("dot_product should return a NdarrayWrapper<f64, Ix2>");
    // ones((3,3)).dot(ones((3,3))) == a 3x3 matrix filled with 3.0 (each entry
    // sums 3 products of 1.0 * 1.0).
    assert_eq!(
        result_array.as_array(),
        &scirs2_core::ndarray::Array2::<f64>::from_elem((3, 3), 3.0)
    );

    // `dot_product` only recognizes NdarrayWrapper<f64, Ix2> pairs (per its own
    // `else` branch) — confirm it correctly reports a foreign array type as
    // unimplemented rather than silently miscomputing something for it.
    assert!(
        dot_product(&a_wrapped, &gpu_array).is_err(),
        "dot_product should report NotImplemented for a GPUNdarray operand"
    );
    assert!(
        dot_product(&a_wrapped, &dist_array).is_err(),
        "dot_product should report NotImplemented for a DistributedNdarray operand"
    );
}

#[test]
#[allow(dead_code)]
fn test_array_operations() {
    // Initialize the array protocol system
    array_protocol::init();

    // Create regular arrays
    let a = Array2::<f64>::eye(3);
    let b = Array2::<f64>::ones((3, 3));

    // Wrap them in NdarrayWrapper
    let wrapped_a = NdarrayWrapper::new(a.clone());
    let wrapped_b = NdarrayWrapper::new(b.clone());

    // Test array operations from the operations module

    // Matrix multiplication
    let matmul_result =
        array_protocol::matmul(&wrapped_a, &wrapped_b).expect("matmul should succeed");
    let matmul_array = matmul_result
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>()
        .expect("matmul should return a NdarrayWrapper<f64, Ix2>");
    assert_eq!(matmul_array.as_array(), &a.dot(&b));

    // Addition
    let add_result = array_protocol::add(&wrapped_a, &wrapped_b).expect("add should succeed");
    let add_array = add_result
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>()
        .expect("add should return a NdarrayWrapper<f64, Ix2>");
    assert_eq!(add_array.as_array(), &(a.clone() + b.clone()));

    // Multiplication
    let multiply_result =
        array_protocol::multiply(&wrapped_a, &wrapped_b).expect("multiply should succeed");
    let multiply_array = multiply_result
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>()
        .expect("multiply should return a NdarrayWrapper<f64, Ix2>");
    assert_eq!(multiply_array.as_array(), &(a.clone() * b.clone()));

    // Sum
    let sum_result = array_protocol::sum(&wrapped_a, None).expect("sum should succeed");
    let sum_value = sum_result
        .downcast_ref::<f64>()
        .expect("sum should return an f64");
    assert_eq!(*sum_value, a.sum());

    // Transpose
    let transpose_result = array_protocol::transpose(&wrapped_a).expect("transpose should succeed");
    let transpose_array = transpose_result
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>()
        .expect("transpose should return a NdarrayWrapper<f64, Ix2>");
    assert_eq!(transpose_array.as_array(), &a.t().to_owned());

    // Test with GPU arrays
    let gpu_config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: false,
        mixed_precision: false,
        memory_fraction: 0.9,
    };

    let gpu_a = GPUNdarray::new(a.clone(), gpu_config.clone());
    let gpu_b = GPUNdarray::new(b.clone(), gpu_config);

    // Matrix multiplication with GPU arrays
    let gpu_matmul_result =
        array_protocol::matmul(&gpu_a, &gpu_b).expect("GPU matmul should succeed");
    assert!(
        gpu_matmul_result
            .as_any()
            .downcast_ref::<GPUNdarray<f64, scirs2_core::ndarray::IxDyn>>()
            .is_some()
            || gpu_matmul_result
                .as_any()
                .downcast_ref::<GPUNdarray<f64, scirs2_core::ndarray::Ix2>>()
                .is_some(),
        "GPU matmul should return a GPUNdarray<f64, _>"
    );

    // Addition with GPU arrays
    let gpu_add_result = array_protocol::add(&gpu_a, &gpu_b).expect("GPU add should succeed");
    assert!(
        gpu_add_result
            .as_any()
            .downcast_ref::<GPUNdarray<f64, scirs2_core::ndarray::IxDyn>>()
            .is_some()
            || gpu_add_result
                .as_any()
                .downcast_ref::<GPUNdarray<f64, scirs2_core::ndarray::Ix2>>()
                .is_some(),
        "GPU add should return a GPUNdarray<f64, _>"
    );
}

#[test]
#[allow(dead_code)]
#[ignore = "not-implemented: cross-backend operand-pair dispatch (Regular+GPU, \
            GPU+Distributed, Regular+Distributed) is architecturally unimplemented — \
            every ArrayProtocol::array_function impl in this crate only recognizes \
            Self in args[1] and returns NotImplemented for any other concrete type, \
            and the ArrayFunctionRegistry handler registered by this test is never \
            consulted by operations::add's dispatch path (see get_implementing_args \
            + array_ref.array_function in operations.rs, which never queries the \
            registry). Naively trying the next implementing_args candidate would \
            silently compute a self-op on the wrong operand (e.g. gpu_a + gpu_a \
            instead of wrapped_a + gpu_a) rather than a real cross-backend result."]
fn test_mixed_array_types() {
    // Initialize the array protocol system
    array_protocol::init();

    // Create arrays of different types
    let a = Array2::<f64>::eye(3);
    let wrapped_a = NdarrayWrapper::new(a.clone());

    let gpu_config = GPUConfig {
        backend: GPUBackend::CUDA,
        device_id: 0,
        async_ops: false,
        mixed_precision: false,
        memory_fraction: 0.9,
    };
    let gpu_a = GPUNdarray::new(a.clone(), gpu_config);

    let dist_config = DistributedConfig {
        chunks: 2,
        balance: true,
        strategy: DistributionStrategy::RowWise,
        backend: DistributedBackend::Threaded,
    };
    let dist_a = DistributedNdarray::from_array(&a, dist_config);

    // Test operations between different array types
    // Register array operations for mixed arrays in the global registry
    // These registrations ensure that we provide proper fallbacks for mixed array operations

    // First, let's create a wrapper for mixed array addition
    let add_op_name = "scirs2::array_protocol::operations::add";
    let add_implementation = std::sync::Arc::new(
        move |_args: &[Box<dyn std::any::Any>],
              kwargs: &std::collections::HashMap<String, Box<dyn std::any::Any>>| {
            // In a real implementation, we would extract and handle arguments properly
            // For this test, we just return a fixed result
            let dummy_array = scirs2_core::ndarray::Array2::<f64>::ones((3, 3));
            let wrapped = NdarrayWrapper::new(dummy_array);
            Ok(Box::new(wrapped) as Box<dyn std::any::Any>)
        },
    );

    let add_func = array_protocol::ArrayFunction {
        name: add_op_name,
        implementation: add_implementation,
    };

    // Register the function with the global registry
    let registry = array_protocol::ArrayFunctionRegistry::global();
    {
        let mut registry_write = registry.write().expect("Test: operation failed");
        registry_write.register(add_func);
    }

    // Regular + GPU
    match array_protocol::add(&wrapped_a, &gpu_a) {
        Ok(result) => {
            // Check for several possible result types
            let is_valid_type = result
                .as_any()
                .downcast_ref::<GPUNdarray<f64, scirs2_core::ndarray::IxDyn>>()
                .is_some()
                || result
                    .as_any()
                    .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::IxDyn>>()
                    .is_some()
                || result
                    .as_any()
                    .downcast_ref::<GPUNdarray<f64, scirs2_core::ndarray::Ix2>>()
                    .is_some()
                || result
                    .as_any()
                    .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>()
                    .is_some();

            assert!(
                is_valid_type,
                "Result not of expected type for Regular + GPU operation"
            );
        }
        Err(e) => {
            // This ignored test documents a real, still-open gap (see the
            // #[ignore] reason above): silently skipping here would let the
            // test pass vacuously regardless of whether cross-backend
            // dispatch actually works. Fail loudly instead so that running
            // it with --ignored gives an honest signal.
            panic!("Regular + GPU add: cross-backend dispatch not reached, got error: {e}");
        }
    }

    // GPU + Distributed
    match array_protocol::add(&gpu_a, &dist_a) {
        Ok(result) => {
            // Check for several possible result types
            let is_valid_type = result
                .as_any()
                .downcast_ref::<GPUNdarray<f64, scirs2_core::ndarray::IxDyn>>()
                .is_some()
                || result
                    .as_any()
                    .downcast_ref::<DistributedNdarray<f64, scirs2_core::ndarray::IxDyn>>()
                    .is_some()
                || result
                    .as_any()
                    .downcast_ref::<GPUNdarray<f64, scirs2_core::ndarray::Ix2>>()
                    .is_some()
                || result
                    .as_any()
                    .downcast_ref::<DistributedNdarray<f64, scirs2_core::ndarray::Ix2>>()
                    .is_some();

            assert!(
                is_valid_type,
                "Result not of expected type for GPU + Distributed operation"
            );
        }
        Err(e) => {
            panic!("GPU + Distributed add: cross-backend dispatch not reached, got error: {e}");
        }
    }

    // Regular + Distributed
    match array_protocol::add(&wrapped_a, &dist_a) {
        Ok(result) => {
            // Check for several possible result types
            let is_valid_type = result
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::IxDyn>>()
                .is_some()
                || result
                    .as_any()
                    .downcast_ref::<DistributedNdarray<f64, scirs2_core::ndarray::IxDyn>>()
                    .is_some()
                || result
                    .as_any()
                    .downcast_ref::<NdarrayWrapper<f64, scirs2_core::ndarray::Ix2>>()
                    .is_some()
                || result
                    .as_any()
                    .downcast_ref::<DistributedNdarray<f64, scirs2_core::ndarray::Ix2>>()
                    .is_some();

            assert!(
                is_valid_type,
                "Result not of expected type for Regular + Distributed operation"
            );
        }
        Err(e) => {
            panic!("Regular + Distributed add: cross-backend dispatch not reached, got error: {e}");
        }
    }
}

// Define a custom array type for testing
struct CustomArray<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T: Clone + 'static> CustomArray<T> {
    fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    // This method is commented out to avoid "never used" warnings
    // It's kept here for documentation purposes
    // fn shape(&self) -> &[usize] {
    //    &self.shape
    // }
}

// Implement ArrayProtocol for the custom array type
impl<T: Clone + Send + Sync + 'static> ArrayProtocol for CustomArray<T> {
    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        _args: &[Box<dyn Any>],
        _kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        if func.name == "test::custom_sum" {
            // For testing purposes, just return a fixed value
            Ok(Box::new(42.0f64))
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
        Box::new(CustomArray {
            data: self.data.clone(),
            shape: self.shape.clone(),
        })
    }
}

#[test]
#[allow(dead_code)]
fn test_custom_array_type() {
    // Initialize the array protocol system
    array_protocol::init();

    // Create a custom array
    let custom_array = CustomArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // Define a function that works with the custom array type
    array_function!(
        fn custom_sum(arr: &dyn ArrayProtocol) -> Result<f64, NotImplemented> {
            match arr.array_function(
                &ArrayFunction::new("test::custom_sum"),
                &[TypeId::of::<f64>()],
                &[],
                &HashMap::new(),
            ) {
                Ok(result) => Ok(*result
                    .downcast_ref::<f64>()
                    .expect("Test: operation failed")),
                Err(_) => Err(NotImplemented),
            }
        },
        "test::custom_sum"
    );

    // Use the function directly
    let sum_func = custom_sum;

    // Use the function with the custom array type
    let custom_array_ref: &dyn ArrayProtocol = &custom_array;
    let sum = sum_func(custom_array_ref);

    assert!(sum.is_ok());
    assert_eq!(sum.expect("Test: operation failed"), 42.0);
}
