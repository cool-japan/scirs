// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Tests for the grad module — split out to keep grad.rs under 2000 lines.

use super::*;
use ::ndarray::{array, Array2, ArrayD, Ix2, IxDyn};

#[test]
fn test_gradient_tensor_creation() {
    // Create a gradient tensor
    let array = Array2::<f64>::ones((2, 2));
    let tensor = GradientTensor::from_array(array, true);

    // Check properties
    assert!(tensor.requiresgrad());
    assert!(tensor.is_leaf());
    assert!(tensor.grad_2().is_none());
}

#[test]
fn test_gradient_computation_add() {
    // Create gradient tensors
    let a_array = Array2::<f64>::ones((2, 2));
    let b_array = Array2::<f64>::ones((2, 2)) * 2.0;

    let a = GradientTensor::from_array(a_array, true);
    let b = GradientTensor::from_array(b_array, true);

    // Forward: `a` and `b` are both `Ix2`, so `grad_add` takes the exact-D
    // fast path in `NdarrayWrapper::array_function` and the result stays Ix2.
    let c = grad_add(&a, &b).expect("grad_add should succeed for two Ix2 f64 tensors");
    let c_value = c.value();
    let c_array = c_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        .expect("grad_add result should be a NdarrayWrapper<f64, Ix2>");
    assert_eq!(c_array.as_array(), &array![[3.0, 3.0], [3.0, 3.0]]);

    // Backward: `backward()` seeds the gradient as `NdarrayWrapper<f64,
    // IxDyn>` (matching the output's shape via `ArrayProtocol::shape()`, not
    // its concrete dimension type), so the leaf gradients that flow straight
    // through the "add" backward rule inherit that IxDyn typing rather than
    // the forward value's Ix2.
    c.backward().expect("backward should succeed");

    let a_grad = a.grad_2().expect("gradient for a should be set");
    let a_grad_array = a_grad
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("a_grad should be a NdarrayWrapper<f64, IxDyn>");
    assert_eq!(
        a_grad_array.as_array(),
        &array![[1.0, 1.0], [1.0, 1.0]].into_dyn()
    );

    let b_grad = b.grad_2().expect("gradient for b should be set");
    let b_grad_array = b_grad
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("b_grad should be a NdarrayWrapper<f64, IxDyn>");
    assert_eq!(
        b_grad_array.as_array(),
        &array![[1.0, 1.0], [1.0, 1.0]].into_dyn()
    );
}

#[test]
fn test_gradient_computation_multiply() {
    // Create gradient tensors
    let a_array = Array2::<f64>::ones((2, 2)) * 2.0;
    let b_array = Array2::<f64>::ones((2, 2)) * 3.0;

    let a = GradientTensor::from_array(a_array, true);
    let b = GradientTensor::from_array(b_array, true);

    // Forward: both operands are Ix2, so the exact-D fast path applies and
    // the result stays Ix2 (same reasoning as test_gradient_computation_add).
    let c = grad_multiply(&a, &b).expect("grad_multiply should succeed for two Ix2 f64 tensors");
    let c_value = c.value();
    let c_array = c_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        .expect("grad_multiply result should be a NdarrayWrapper<f64, Ix2>");
    assert_eq!(c_array.as_array(), &array![[6.0, 6.0], [6.0, 6.0]]);

    // Backward: multiply's backward rule combines the (IxDyn-seeded) output
    // gradient with the *other* operand's Ix2 value via the mixed-dimension
    // fallback in `NdarrayWrapper::array_function`, which always normalizes
    // its result to IxDyn.
    c.backward().expect("backward should succeed");

    let a_grad = a.grad_2().expect("gradient for a should be set");
    let a_grad_array = a_grad
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("a_grad should be a NdarrayWrapper<f64, IxDyn>");
    assert_eq!(
        a_grad_array.as_array(),
        &array![[3.0, 3.0], [3.0, 3.0]].into_dyn()
    );

    let b_grad = b.grad_2().expect("gradient for b should be set");
    let b_grad_array = b_grad
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("b_grad should be a NdarrayWrapper<f64, IxDyn>");
    assert_eq!(
        b_grad_array.as_array(),
        &array![[2.0, 2.0], [2.0, 2.0]].into_dyn()
    );
}

#[test]
fn test_sgd_optimizer() {
    // Create variables
    let weight_array = Array2::<f64>::ones((2, 2));
    let weight = Variable::new("weight", weight_array);

    let bias_array = Array2::<f64>::zeros((2, 2));
    let bias = Variable::new("bias", bias_array);

    // Create optimizer
    let mut optimizer = SGD::new(0.1, Some(0.9));
    optimizer.add_variable(weight);
    optimizer.add_variable(bias);

    // Manually set gradients for testing
    let weight_grad_array = Array2::<f64>::ones((2, 2));
    let weight_grad = NdarrayWrapper::new(weight_grad_array);
    optimizer.variables()[0].tensor.node.borrow_mut().grad = Some(Rc::new(weight_grad));

    let bias_grad_array = Array2::<f64>::ones((2, 2)) * 2.0;
    let bias_grad = NdarrayWrapper::new(bias_grad_array);
    optimizer.variables()[1].tensor.node.borrow_mut().grad = Some(Rc::new(bias_grad));

    // Take an optimization step. With no prior velocity, `SGD::step` uses
    // `update = lr * grad`, so: weight = 1.0 - 0.1*1.0 = 0.9,
    // bias = 0.0 - 0.1*2.0 = -0.2.
    optimizer.step().expect("SGD step should succeed");

    let weight_value = optimizer.variables()[0].value();
    let weight_array = weight_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("updated weight should be a NdarrayWrapper<f64, IxDyn>");
    for &v in weight_array.as_array().iter() {
        assert!((v - 0.9).abs() < 1e-12, "expected weight 0.9, got {v}");
    }

    let bias_value = optimizer.variables()[1].value();
    let bias_array = bias_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("updated bias should be a NdarrayWrapper<f64, IxDyn>");
    for &v in bias_array.as_array().iter() {
        assert!((v - (-0.2)).abs() < 1e-12, "expected bias -0.2, got {v}");
    }

    // Zero gradients
    optimizer.zero_grad();

    // Check that gradients are zeroed
    assert!(optimizer.variables()[0].grad_2().is_none());
    assert!(optimizer.variables()[1].grad_2().is_none());
}

// ---- integer dtype coverage tests ----

fn make_i32_tensor(values: &[i32]) -> GradientTensor {
    let arr =
        ArrayD::<i32>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    GradientTensor::from_array(arr, true)
}

fn make_i64_tensor(values: &[i64]) -> GradientTensor {
    let arr =
        ArrayD::<i64>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    GradientTensor::from_array(arr, true)
}

fn make_u8_tensor(values: &[u8]) -> GradientTensor {
    let arr =
        ArrayD::<u8>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    GradientTensor::from_array(arr, true)
}

fn make_u16_tensor(values: &[u16]) -> GradientTensor {
    let arr =
        ArrayD::<u16>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    GradientTensor::from_array(arr, true)
}

fn make_u32_tensor(values: &[u32]) -> GradientTensor {
    let arr =
        ArrayD::<u32>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    GradientTensor::from_array(arr, true)
}

fn make_u64_tensor(values: &[u64]) -> GradientTensor {
    let arr =
        ArrayD::<u64>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    GradientTensor::from_array(arr, true)
}

fn make_i32_protocol(values: &[i32]) -> Box<dyn ArrayProtocol> {
    let arr =
        ArrayD::<i32>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    Box::new(NdarrayWrapper::new(arr))
}

fn make_u32_protocol(values: &[u32]) -> Box<dyn ArrayProtocol> {
    let arr =
        ArrayD::<u32>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    Box::new(NdarrayWrapper::new(arr))
}

fn make_u8_protocol(values: &[u8]) -> Box<dyn ArrayProtocol> {
    let arr =
        ArrayD::<u8>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    Box::new(NdarrayWrapper::new(arr))
}

fn make_u16_protocol(values: &[u16]) -> Box<dyn ArrayProtocol> {
    let arr =
        ArrayD::<u16>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    Box::new(NdarrayWrapper::new(arr))
}

fn make_u64_protocol(values: &[u64]) -> Box<dyn ArrayProtocol> {
    let arr =
        ArrayD::<u64>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    Box::new(NdarrayWrapper::new(arr))
}

fn make_i64_protocol(values: &[i64]) -> Box<dyn ArrayProtocol> {
    let arr =
        ArrayD::<i64>::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("shape vec");
    Box::new(NdarrayWrapper::new(arr))
}

fn extract_f64_result(proto: &dyn ArrayProtocol) -> Vec<f64> {
    proto
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("expected f64 result")
        .as_array()
        .iter()
        .copied()
        .collect()
}

fn extract_i32_result(proto: &dyn ArrayProtocol) -> Vec<i32> {
    proto
        .as_any()
        .downcast_ref::<NdarrayWrapper<i32, IxDyn>>()
        .expect("expected i32 result")
        .as_array()
        .iter()
        .copied()
        .collect()
}

fn extract_u32_result(proto: &dyn ArrayProtocol) -> Vec<u32> {
    proto
        .as_any()
        .downcast_ref::<NdarrayWrapper<u32, IxDyn>>()
        .expect("expected u32 result")
        .as_array()
        .iter()
        .copied()
        .collect()
}

#[test]
fn test_grad_mean_i32() {
    // mean([1,2,3]) == 2.0
    let t = make_i32_tensor(&[1, 2, 3]);
    let result = grad_mean(&t).expect("grad_mean i32");
    let val = result.value();
    let out = extract_f64_result(val.as_ref());
    assert_eq!(out.len(), 1);
    assert!((out[0] - 2.0).abs() < 1e-12, "expected 2.0 got {}", out[0]);
}

#[test]
fn test_grad_mean_i64() {
    // mean([10,20,30]) == 20.0
    let t = make_i64_tensor(&[10, 20, 30]);
    let result = grad_mean(&t).expect("grad_mean i64");
    let val = result.value();
    let out = extract_f64_result(val.as_ref());
    assert!(
        (out[0] - 20.0).abs() < 1e-12,
        "expected 20.0 got {}",
        out[0]
    );
}

#[test]
fn test_grad_mean_u8() {
    // mean([2,4,6]) == 4.0
    let t = make_u8_tensor(&[2, 4, 6]);
    let result = grad_mean(&t).expect("grad_mean u8");
    let val = result.value();
    let out = extract_f64_result(val.as_ref());
    assert!((out[0] - 4.0).abs() < 1e-12, "expected 4.0 got {}", out[0]);
}

#[test]
fn test_grad_mean_u16() {
    // mean([100,200,300]) == 200.0
    let t = make_u16_tensor(&[100, 200, 300]);
    let result = grad_mean(&t).expect("grad_mean u16");
    let val = result.value();
    let out = extract_f64_result(val.as_ref());
    assert!(
        (out[0] - 200.0).abs() < 1e-12,
        "expected 200.0 got {}",
        out[0]
    );
}

#[test]
fn test_grad_mean_u32() {
    // mean([3,6,9]) == 6.0
    let t = make_u32_tensor(&[3, 6, 9]);
    let result = grad_mean(&t).expect("grad_mean u32");
    let val = result.value();
    let out = extract_f64_result(val.as_ref());
    assert!((out[0] - 6.0).abs() < 1e-12, "expected 6.0 got {}", out[0]);
}

#[test]
fn test_grad_mean_u64() {
    // mean([1000,2000,3000]) == 2000.0
    let t = make_u64_tensor(&[1000, 2000, 3000]);
    let result = grad_mean(&t).expect("grad_mean u64");
    let val = result.value();
    let out = extract_f64_result(val.as_ref());
    assert!(
        (out[0] - 2000.0).abs() < 1e-12,
        "expected 2000.0 got {}",
        out[0]
    );
}

#[test]
fn test_multiply_by_scalar_u32() {
    // [2,4,6] * 3.0 -> [6,12,18]
    let a = make_u32_protocol(&[2, 4, 6]);
    let result = multiply_by_scalar(a.as_ref(), 3.0).expect("multiply_by_scalar u32");
    let out = extract_u32_result(result.as_ref());
    assert_eq!(out, vec![6u32, 12, 18]);
}

#[test]
fn test_multiply_by_scalar_u8() {
    // [1,2,3] * 4.0 -> [4,8,12]
    let a = make_u8_protocol(&[1, 2, 3]);
    let result = multiply_by_scalar(a.as_ref(), 4.0).expect("multiply_by_scalar u8");
    let out = result
        .as_any()
        .downcast_ref::<NdarrayWrapper<u8, IxDyn>>()
        .expect("u8 result")
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<u8>>();
    assert_eq!(out, vec![4u8, 8, 12]);
}

#[test]
fn test_multiply_by_scalar_u16() {
    // [10,20,30] * 2.0 -> [20,40,60]
    let a = make_u16_protocol(&[10, 20, 30]);
    let result = multiply_by_scalar(a.as_ref(), 2.0).expect("multiply_by_scalar u16");
    let out = result
        .as_any()
        .downcast_ref::<NdarrayWrapper<u16, IxDyn>>()
        .expect("u16 result")
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<u16>>();
    assert_eq!(out, vec![20u16, 40, 60]);
}

#[test]
fn test_multiply_by_scalar_u64() {
    // [100,200,300] * 5.0 -> [500,1000,1500]
    let a = make_u64_protocol(&[100, 200, 300]);
    let result = multiply_by_scalar(a.as_ref(), 5.0).expect("multiply_by_scalar u64");
    let out = result
        .as_any()
        .downcast_ref::<NdarrayWrapper<u64, IxDyn>>()
        .expect("u64 result")
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<u64>>();
    assert_eq!(out, vec![500u64, 1000, 1500]);
}

#[test]
fn test_subtract_arrays_int_types() {
    // i32: [5,6,7] - [1,2,3] == [4,4,4]
    let a = make_i32_protocol(&[5, 6, 7]);
    let b = make_i32_protocol(&[1, 2, 3]);
    let result = subtract_arrays(a.as_ref(), b.as_ref()).expect("subtract i32");
    let out = extract_i32_result(result.as_ref());
    assert_eq!(out, vec![4i32, 4, 4]);

    // u8: [10,20,30] - [1,2,3] == [9,18,27]
    let ua = make_u8_protocol(&[10, 20, 30]);
    let ub = make_u8_protocol(&[1, 2, 3]);
    let ures = subtract_arrays(ua.as_ref(), ub.as_ref()).expect("subtract u8");
    let uout = ures
        .as_any()
        .downcast_ref::<NdarrayWrapper<u8, IxDyn>>()
        .expect("u8")
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<u8>>();
    assert_eq!(uout, vec![9u8, 18, 27]);

    // u32: [100,200,300] - [10,20,30] == [90,180,270]
    let u32a = make_u32_protocol(&[100, 200, 300]);
    let u32b = make_u32_protocol(&[10, 20, 30]);
    let u32res = subtract_arrays(u32a.as_ref(), u32b.as_ref()).expect("subtract u32");
    let u32out = extract_u32_result(u32res.as_ref());
    assert_eq!(u32out, vec![90u32, 180, 270]);
}

#[test]
fn test_sqrt_i32() {
    // sqrt([4, 9, 16]) -> [2.0, 3.0, 4.0]
    let a = make_i32_protocol(&[4, 9, 16]);
    let result = sqrt(a.as_ref()).expect("sqrt i32");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 2.0).abs() < 1e-12);
    assert!((out[1] - 3.0).abs() < 1e-12);
    assert!((out[2] - 4.0).abs() < 1e-12);
}

#[test]
fn test_sqrt_i64() {
    // sqrt([25, 36, 49]) -> [5.0, 6.0, 7.0]
    let a = make_i64_protocol(&[25, 36, 49]);
    let result = sqrt(a.as_ref()).expect("sqrt i64");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 5.0).abs() < 1e-12);
    assert!((out[1] - 6.0).abs() < 1e-12);
    assert!((out[2] - 7.0).abs() < 1e-12);
}

#[test]
fn test_sqrt_u8() {
    // sqrt([1, 4, 9]) -> [1.0, 2.0, 3.0]
    let a = make_u8_protocol(&[1, 4, 9]);
    let result = sqrt(a.as_ref()).expect("sqrt u8");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 1.0).abs() < 1e-12);
    assert!((out[1] - 2.0).abs() < 1e-12);
    assert!((out[2] - 3.0).abs() < 1e-12);
}

#[test]
fn test_sqrt_u16() {
    // sqrt([100, 225, 400]) -> [10.0, 15.0, 20.0]
    let a = make_u16_protocol(&[100, 225, 400]);
    let result = sqrt(a.as_ref()).expect("sqrt u16");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 10.0).abs() < 1e-12);
    assert!((out[1] - 15.0).abs() < 1e-12);
    assert!((out[2] - 20.0).abs() < 1e-12);
}

#[test]
fn test_sqrt_u32() {
    // sqrt([1, 4, 9]) -> [1.0, 2.0, 3.0]
    let a = make_u32_protocol(&[1, 4, 9]);
    let result = sqrt(a.as_ref()).expect("sqrt u32");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 1.0).abs() < 1e-12);
    assert!((out[1] - 2.0).abs() < 1e-12);
    assert!((out[2] - 3.0).abs() < 1e-12);
}

#[test]
fn test_sqrt_u64() {
    // sqrt([4, 9, 16]) -> [2.0, 3.0, 4.0]
    let a = make_u64_protocol(&[4, 9, 16]);
    let result = sqrt(a.as_ref()).expect("sqrt u64");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 2.0).abs() < 1e-12);
    assert!((out[1] - 3.0).abs() < 1e-12);
    assert!((out[2] - 4.0).abs() < 1e-12);
}

#[test]
fn test_add_scalar_u8() {
    // [1,2,3] + 10 -> [11,12,13]
    let a = make_u8_protocol(&[1, 2, 3]);
    let result = add_scalar(a.as_ref(), 10.0).expect("add_scalar u8");
    let out = result
        .as_any()
        .downcast_ref::<NdarrayWrapper<u8, IxDyn>>()
        .expect("u8")
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<u8>>();
    assert_eq!(out, vec![11u8, 12, 13]);
}

#[test]
fn test_add_scalar_u16() {
    // [10,20,30] + 5 -> [15,25,35]
    let a = make_u16_protocol(&[10, 20, 30]);
    let result = add_scalar(a.as_ref(), 5.0).expect("add_scalar u16");
    let out = result
        .as_any()
        .downcast_ref::<NdarrayWrapper<u16, IxDyn>>()
        .expect("u16")
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<u16>>();
    assert_eq!(out, vec![15u16, 25, 35]);
}

#[test]
fn test_add_scalar_u32() {
    // [100,200,300] + 50 -> [150,250,350]
    let a = make_u32_protocol(&[100, 200, 300]);
    let result = add_scalar(a.as_ref(), 50.0).expect("add_scalar u32");
    let out = extract_u32_result(result.as_ref());
    assert_eq!(out, vec![150u32, 250, 350]);
}

#[test]
fn test_add_scalar_u64() {
    // [1000,2000,3000] + 500 -> [1500,2500,3500]
    let a = make_u64_protocol(&[1000, 2000, 3000]);
    let result = add_scalar(a.as_ref(), 500.0).expect("add_scalar u64");
    let out = result
        .as_any()
        .downcast_ref::<NdarrayWrapper<u64, IxDyn>>()
        .expect("u64")
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<u64>>();
    assert_eq!(out, vec![1500u64, 2500, 3500]);
}

#[test]
fn test_divide_i32() {
    // [6,8,9] / [2,4,3] -> [3.0, 2.0, 3.0]
    let a = make_i32_protocol(&[6, 8, 9]);
    let b = make_i32_protocol(&[2, 4, 3]);
    let result = divide(a.as_ref(), b.as_ref()).expect("divide i32");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 3.0).abs() < 1e-12);
    assert!((out[1] - 2.0).abs() < 1e-12);
    assert!((out[2] - 3.0).abs() < 1e-12);
}

#[test]
fn test_divide_i64() {
    // [100,200,300] / [10,20,30] -> [10.0, 10.0, 10.0]
    let a = make_i64_protocol(&[100, 200, 300]);
    let b = make_i64_protocol(&[10, 20, 30]);
    let result = divide(a.as_ref(), b.as_ref()).expect("divide i64");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 10.0).abs() < 1e-12);
    assert!((out[1] - 10.0).abs() < 1e-12);
    assert!((out[2] - 10.0).abs() < 1e-12);
}

#[test]
fn test_divide_u8() {
    // [4,8,12] / [2,4,3] -> [2.0, 2.0, 4.0]
    let a = make_u8_protocol(&[4, 8, 12]);
    let b = make_u8_protocol(&[2, 4, 3]);
    let result = divide(a.as_ref(), b.as_ref()).expect("divide u8");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 2.0).abs() < 1e-12);
    assert!((out[1] - 2.0).abs() < 1e-12);
    assert!((out[2] - 4.0).abs() < 1e-12);
}

#[test]
fn test_divide_u16() {
    // [100,200,300] / [10,20,30] -> [10.0, 10.0, 10.0]
    let a = make_u16_protocol(&[100, 200, 300]);
    let b = make_u16_protocol(&[10, 20, 30]);
    let result = divide(a.as_ref(), b.as_ref()).expect("divide u16");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 10.0).abs() < 1e-12);
    assert!((out[1] - 10.0).abs() < 1e-12);
    assert!((out[2] - 10.0).abs() < 1e-12);
}

#[test]
fn test_divide_u32() {
    // [6,8,10] / [2,4,5] -> [3.0, 2.0, 2.0]
    let a = make_u32_protocol(&[6, 8, 10]);
    let b = make_u32_protocol(&[2, 4, 5]);
    let result = divide(a.as_ref(), b.as_ref()).expect("divide u32");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 3.0).abs() < 1e-12);
    assert!((out[1] - 2.0).abs() < 1e-12);
    assert!((out[2] - 2.0).abs() < 1e-12);
}

#[test]
fn test_divide_u64() {
    // [1000,2000,3000] / [100,200,300] -> [10.0, 10.0, 10.0]
    let a = make_u64_protocol(&[1000, 2000, 3000]);
    let b = make_u64_protocol(&[100, 200, 300]);
    let result = divide(a.as_ref(), b.as_ref()).expect("divide u64");
    let out = extract_f64_result(result.as_ref());
    assert!((out[0] - 10.0).abs() < 1e-12);
    assert!((out[1] - 10.0).abs() < 1e-12);
    assert!((out[2] - 10.0).abs() < 1e-12);
}
