//! Automatic differentiation example for linear algebra operations
//!
//! This example demonstrates the use of automatic differentiation
//! with various linear algebra operations in scirs2-linalg.

#[cfg(feature = "autograd")]
use scirs2_linalg::autograd::{
    det, dot, eig, expm, inv, matmul, matvec, norm, svd, trace, transpose,
    variable::{
        var_det, var_dot, var_eig, var_expm, var_inv, var_matmul, var_matvec, var_norm, var_svd,
        var_trace, var_transpose,
    },
};

#[cfg(feature = "autograd")]
use scirs2_autograd::{
    error::Result as AutogradResult, 
    tensor::Tensor,
    variable::Variable,
};

#[cfg(feature = "autograd")]
use ndarray::{array, arr1, arr2, s, Array1, Array2};
#[cfg(feature = "autograd")]
use num_traits::Float;

#[cfg(not(feature = "autograd"))]
fn main() {
    println!("This example requires the 'autograd' feature. Run with:");
    println!("cargo run --example autograd_example --features=autograd");
}

#[cfg(feature = "autograd")]
fn main() -> AutogradResult<()> {
    println!("SciRS2 Linear Algebra Automatic Differentiation Example");
    println!("====================================================\n");

    // Example 1: Simple matrix multiplication with gradient computation
    demo_matrix_multiplication()?;

    // Example 2: Matrix determinant and trace
    demo_det_and_trace()?;

    // Example 3: Matrix inverse
    demo_matrix_inverse()?;

    // Example 4: Matrix-vector operations
    demo_matrix_vector_ops()?;

    // Example 5: Matrix norms
    demo_matrix_norms()?;

    // Example 6: Composite operations
    demo_composite_operations()?;

    // Example 7: Matrix exponential
    demo_matrix_exponential()?;

    Ok(())
}

#[cfg(feature = "autograd")]
fn demo_matrix_multiplication() -> AutogradResult<()> {
    println!("1. Matrix Multiplication with Gradients");
    println!("-------------------------------------");

    // Create two matrices with gradient tracking enabled
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

    // Create tensors from the arrays
    let a_tensor = Tensor::new(a.into_dyn(), true);
    let b_tensor = Tensor::new(b.into_dyn(), true);

    // Perform matrix multiplication
    let c_tensor = matmul(&a_tensor, &b_tensor)?;

    // Compute loss (sum of all elements)
    let loss = sum_all_elements(&c_tensor)?;

    // Backward pass
    loss.backward()?;

    println!("Matrix A:");
    if let Some(grad) = &a_tensor.grad {
        println!("Gradient: {:?}", grad);
    }

    println!("\nMatrix B:");
    if let Some(grad) = &b_tensor.grad {
        println!("Gradient: {:?}", grad);
    }

    // Using Variables for easier manipulation
    let a_var = Variable::new(a_tensor.clone());
    let b_var = Variable::new(b_tensor.clone());
    let c_var = var_matmul(&a_var, &b_var)?;

    println!("\nUsing Variables:");
    println!("Result C: {:?}", c_var.tensor.data);

    println!();
    Ok(())
}

#[cfg(feature = "autograd")]
fn demo_det_and_trace() -> AutogradResult<()> {
    println!("2. Determinant and Trace with Gradients");
    println!("-------------------------------------");

    // Create a 2x2 matrix
    let a = arr2(&[[3.0, 1.0], [2.0, 4.0]]);
    let a_tensor = Tensor::new(a.into_dyn(), true);

    // Compute determinant
    let det_tensor = det(&a_tensor)?;
    println!("Determinant: {:?}", det_tensor.data);

    // Compute trace
    let trace_tensor = trace(&a_tensor)?;
    println!("Trace: {:?}", trace_tensor.data);

    // Backward pass on determinant
    det_tensor.backward()?;
    if let Some(grad) = &a_tensor.grad {
        println!("Gradient from determinant: {:?}", grad);
    }

    // Clear gradients and compute for trace
    a_tensor.zero_grad();
    trace_tensor.backward()?;
    if let Some(grad) = &a_tensor.grad {
        println!("Gradient from trace: {:?}", grad);
    }

    println!();
    Ok(())
}

#[cfg(feature = "autograd")]
fn demo_matrix_inverse() -> AutogradResult<()> {
    println!("3. Matrix Inverse with Gradients");
    println!("-------------------------------");

    // Create a simple 2x2 matrix
    let a = arr2(&[[2.0, 1.0], [1.0, 3.0]]);
    let a_tensor = Tensor::new(a.into_dyn(), true);

    // Compute inverse
    let inv_tensor = inv(&a_tensor)?;
    println!("Inverse: {:?}", inv_tensor.data);

    // Verify A * A^(-1) = I
    let identity = matmul(&a_tensor, &inv_tensor)?;
    println!("A * A^(-1): {:?}", identity.data);

    // Compute gradient
    let loss = sum_all_elements(&inv_tensor)?;
    loss.backward()?;

    if let Some(grad) = &a_tensor.grad {
        println!("Gradient: {:?}", grad);
    }

    println!();
    Ok(())
}

#[cfg(feature = "autograd")]
fn demo_matrix_vector_ops() -> AutogradResult<()> {
    println!("4. Matrix-Vector Operations with Gradients");
    println!("----------------------------------------");

    // Create a matrix and vector
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let x = arr1(&[5.0, 6.0]);

    let a_tensor = Tensor::new(a.into_dyn(), true);
    let x_tensor = Tensor::new(x.into_dyn(), true);

    // Matrix-vector multiplication
    let y_tensor = matvec(&a_tensor, &x_tensor)?;
    println!("Matrix-vector product: {:?}", y_tensor.data);

    // Backward pass
    y_tensor.backward()?;

    if let Some(grad) = &a_tensor.grad {
        println!("Gradient w.r.t. matrix: {:?}", grad);
    }
    if let Some(grad) = &x_tensor.grad {
        println!("Gradient w.r.t. vector: {:?}", grad);
    }

    // Dot product
    let v1 = arr1(&[1.0, 2.0, 3.0]);
    let v2 = arr1(&[4.0, 5.0, 6.0]);

    let v1_tensor = Tensor::new(v1.into_dyn(), true);
    let v2_tensor = Tensor::new(v2.into_dyn(), true);

    let dot_product = dot(&v1_tensor, &v2_tensor)?;
    println!("\nDot product: {:?}", dot_product.data);

    dot_product.backward()?;
    if let Some(grad) = &v1_tensor.grad {
        println!("Gradient w.r.t. v1: {:?}", grad);
    }
    if let Some(grad) = &v2_tensor.grad {
        println!("Gradient w.r.t. v2: {:?}", grad);
    }

    println!();
    Ok(())
}

#[cfg(feature = "autograd")]
fn demo_matrix_norms() -> AutogradResult<()> {
    println!("5. Matrix Norms with Gradients");
    println!("----------------------------");

    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let a_tensor = Tensor::new(a.into_dyn(), true);

    // Frobenius norm
    let frobenius_norm = norm(&a_tensor, "fro")?;
    println!("Frobenius norm: {:?}", frobenius_norm.data);

    // Backward pass
    frobenius_norm.backward()?;
    if let Some(grad) = &a_tensor.grad {
        println!("Gradient: {:?}", grad);
    }

    println!();
    Ok(())
}

#[cfg(feature = "autograd")]
fn demo_composite_operations() -> AutogradResult<()> {
    println!("6. Composite Operations with Gradients");
    println!("------------------------------------");

    // Create matrices
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[2.0, 0.0], [1.0, 2.0]]);

    let a_tensor = Tensor::new(a.into_dyn(), true);
    let b_tensor = Tensor::new(b.into_dyn(), true);

    // Composite operation: trace(A * B^T)
    let b_t = transpose(&b_tensor)?;
    let ab_t = matmul(&a_tensor, &b_t)?;
    let result = trace(&ab_t)?;

    println!("trace(A * B^T): {:?}", result.data);

    // Backward pass
    result.backward()?;

    if let Some(grad) = &a_tensor.grad {
        println!("Gradient w.r.t. A: {:?}", grad);
    }
    if let Some(grad) = &b_tensor.grad {
        println!("Gradient w.r.t. B: {:?}", grad);
    }

    // Another composite operation: ||A * B||_F^2
    a_tensor.zero_grad();
    b_tensor.zero_grad();

    let ab = matmul(&a_tensor, &b_tensor)?;
    let norm_ab = norm(&ab, "fro")?;
    
    // Square the norm by multiplying with itself
    let norm_squared = Tensor::new(
        ndarray::Array::from_elem(ndarray::IxDyn(&[1]), 
            norm_ab.data.iter().next().unwrap() * norm_ab.data.iter().next().unwrap()),
        true
    );

    println!("\n||A * B||_F^2: {:?}", norm_squared.data);

    norm_squared.backward()?;

    if let Some(grad) = &a_tensor.grad {
        println!("Gradient w.r.t. A: {:?}", grad);
    }
    if let Some(grad) = &b_tensor.grad {
        println!("Gradient w.r.t. B: {:?}", grad);
    }

    println!();
    Ok(())
}

#[cfg(feature = "autograd")]
fn demo_matrix_exponential() -> AutogradResult<()> {
    println!("7. Matrix Exponential with Gradients");
    println!("----------------------------------");

    // Create a small matrix
    let a = arr2(&[[0.0, 1.0], [-1.0, 0.0]]);
    let a_tensor = Tensor::new(a.into_dyn(), true);

    // Compute matrix exponential
    let exp_a = expm(&a_tensor)?;
    println!("exp(A): {:?}", exp_a.data);

    // This should approximate cos(1) and sin(1) for this rotation matrix
    // Backward pass
    let loss = sum_all_elements(&exp_a)?;
    loss.backward()?;

    if let Some(grad) = &a_tensor.grad {
        println!("Gradient: {:?}", grad);
    }

    println!();
    Ok(())
}

// Helper function to sum all elements of a tensor (for creating scalar loss)
#[cfg(feature = "autograd")]
fn sum_all_elements<F: Float + Send + Sync + 'static>(tensor: &Tensor<F>) -> AutogradResult<Tensor<F>> {
    let sum = tensor.data.iter().fold(F::zero(), |acc, &x| acc + x);
    let result = Tensor::new(ndarray::Array::from_elem(ndarray::IxDyn(&[1]), sum), tensor.requires_grad);
    
    if tensor.requires_grad {
        let shape = tensor.shape().to_vec();
        let backward = Box::new(move |grad: ndarray::Array<F, ndarray::IxDyn>| -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> {
            let grad_scalar = grad.iter().next().unwrap().clone();
            Ok(ndarray::Array::from_elem(ndarray::IxDyn(&shape), grad_scalar))
        }) as Box<dyn Fn(ndarray::Array<F, ndarray::IxDyn>) -> AutogradResult<ndarray::Array<F, ndarray::IxDyn>> + Send + Sync>;
        
        let node = scirs2_autograd::graph::Node::new(
            scirs2_autograd::graph::OpType::Activation("sum".to_string()),
            vec![tensor],
            vec![Some(backward)],
        );
        
        let mut result_with_grad = result;
        result_with_grad.node = Some(node);
        Ok(result_with_grad)
    } else {
        Ok(result)
    }
}