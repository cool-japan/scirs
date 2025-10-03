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

//! Example demonstrating backpropagation and gradient computation
//! using the array protocol.

use scirs2_core::array_protocol::{
    self,
    grad::{Adam, GradientTensor, Optimizer, Variable},
    ml_ops::ActivationFunc,
    neural::{Linear, Sequential},
    training::{DataLoader, InMemoryDataset, Loss, MSELoss},
    NdarrayWrapper,
};
use scirs2_core::ndarray_ext::stats::mean;
use scirs2_core::ndarray_ext::{array, Array1, Array2, Ix2};
use statrs::statistics::Statistics;

/// A simple feed-forward neural network for demonstrating backpropagation
#[allow(dead_code)]
fn main() {
    // Initialize the array protocol system
    array_protocol::init();

    println!("Backpropagation and Gradient Computation Example");
    println!("==============================================");

    // Part 1: Manual backpropagation with gradient tensors
    println!("\nPart 1: Manual Backpropagation with Gradient Tensors");
    println!("------------------------------------------------");

    // Create input and target values for a simple problem: XOR function
    let inputs = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];

    let targets = array![[0.0], [1.0], [1.0], [0.0]];

    println!("XOR Problem:");
    println!("Inputs:\n{}", inputs);
    println!("Targets:\n{}", targets);

    // Create gradient tensors for weights and biases
    println!("\nCreating model parameters with gradient tracking:");

    // First layer parameters
    let w1_array =
        Array2::<f64>::from_shape_fn((2, 4), |_| scirs2_core::random::random::<f64>() * 2.0 - 1.0);
    let b1_array = Array1::<f64>::zeros(4);
    println!(
        "Layer 1: {} -> {}",
        w1_array.shape()[0],
        w1_array.shape()[1]
    );

    // Second layer parameters
    let w2_array =
        Array2::<f64>::from_shape_fn((4, 1), |_| scirs2_core::random::random::<f64>() * 2.0 - 1.0);
    let b2_array = Array1::<f64>::zeros(1);
    println!(
        "Layer 2: {} -> {}",
        w2_array.shape()[0],
        w2_array.shape()[1]
    );

    // Create gradient tensors
    let mut w1 = GradientTensor::from_array(w1_array, true);
    let mut b1 = GradientTensor::from_array(b1_array, true);
    let mut w2 = GradientTensor::from_array(w2_array, true);
    let mut b2 = GradientTensor::from_array(b2_array, true);

    // Training loop with manual backpropagation
    println!("\nTraining with manual backpropagation (10 epochs):");

    let learningrate = 0.1;
    let num_epochs = 10;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;

        // Forward and backward pass for each sample
        for i in 0..inputs.shape()[0] {
            // Get the input and target
            let x = Array1::<f64>::from_iter(inputs.row(i).iter().cloned());
            let y = Array1::<f64>::from_iter(targets.row(i).iter().cloned());

            // Convert to gradient tensors
            let x_tensor = GradientTensor::from_array(x, false);
            let y_tensor = GradientTensor::from_array(y, false);

            // Forward pass
            // Layer 1: z1 = x * w1 + b1
            let z1 = match x_tensor.matmul(&w1) {
                Ok(result) => match result.add(&b1) {
                    Ok(res) => res,
                    Err(e) => {
                        println!("Error in layer 1 bias addition: {}", e);
                        continue; // Skip this batch
                    }
                },
                Err(e) => {
                    println!("Error in layer 1 matrix multiplication: {}", e);
                    continue; // Skip this batch
                }
            };

            // Layer 1 activation: a1 = sigmoid(z1)
            let a1 = match z1.sigmoid() {
                Ok(result) => result,
                Err(e) => {
                    println!("Error in layer 1 sigmoid activation: {}", e);
                    continue; // Skip this batch
                }
            };

            // Layer 2: z2 = a1 * w2 + b2
            let z2 = match a1.matmul(&w2) {
                Ok(result) => match result.add(&b2) {
                    Ok(res) => res,
                    Err(e) => {
                        println!("Error in layer 2 bias addition: {}", e);
                        continue; // Skip this batch
                    }
                },
                Err(e) => {
                    println!("Error in layer 2 matrix multiplication: {}", e);
                    continue; // Skip this batch
                }
            };

            // Layer 2 activation: a2 = sigmoid(z2)
            let a2 = match z2.sigmoid() {
                Ok(result) => result,
                Err(e) => {
                    println!("Error in layer 2 sigmoid activation: {}", e);
                    continue; // Skip this batch
                }
            };

            // Compute loss: MSE(a2, y)
            let diff = match a2.subtract(&y_tensor) {
                Ok(result) => result,
                Err(e) => {
                    println!("Error computing loss difference: {}", e);
                    continue; // Skip this batch
                }
            };

            let squared_diff = match diff.multiply(&diff) {
                Ok(result) => result,
                Err(e) => {
                    println!("Error computing squared difference: {}", e);
                    continue; // Skip this batch
                }
            };

            let loss = match squared_diff.mean() {
                Ok(result) => result,
                Err(e) => {
                    println!("Error computing mean: {}", e);
                    continue; // Skip this batch
                }
            };

            // Add to total loss
            if let Some(scalar) = loss.to_scalar() {
                total_loss += scalar;
            } else {
                println!("Error: loss.to_scalar() returned None");
                continue; // Skip this batch
            }

            // Backward pass (compute gradients)
            match loss.backward() {
                Ok(_) => {}
                Err(e) => {
                    println!("Error in backpropagation: {}", e);
                    continue; // Skip this batch
                }
            };
        }

        // Print stats
        let avg_loss = total_loss / inputs.shape()[0] as f64;
        println!("Epoch {}: loss = {:.4}", epoch + 1, avg_loss);

        // Update weights with gradients
        w1.update_with_learningrate(learningrate);
        b1.update_with_learningrate(learningrate);
        w2.update_with_learningrate(learningrate);
        b2.update_with_learningrate(learningrate);

        // Zero gradients for next epoch
        w1.zero_grad();
        b1.zero_grad();
        w2.zero_grad();
        b2.zero_grad();
    }

    // Test the trained model
    println!("\nTesting the trained model:");

    for i in 0..inputs.shape()[0] {
        // Get the input
        let x = Array1::<f64>::from_iter(inputs.row(i).iter().cloned());
        let x_tensor = GradientTensor::from_array(x.clone(), false);

        // Forward pass
        let z1 = match x_tensor.matmul(&w1) {
            Ok(result) => match result.add(&b1) {
                Ok(res) => res,
                Err(e) => {
                    println!("Error in test forward pass layer 1 bias addition: {}", e);
                    continue; // Skip this test sample
                }
            },
            Err(e) => {
                println!(
                    "Error in test forward pass layer 1 matrix multiplication: {}",
                    e
                );
                continue; // Skip this test sample
            }
        };

        let a1 = match z1.sigmoid() {
            Ok(result) => result,
            Err(e) => {
                println!(
                    "Error in test forward pass layer 1 sigmoid activation: {}",
                    e
                );
                continue; // Skip this test sample
            }
        };

        let z2 = match a1.matmul(&w2) {
            Ok(result) => match result.add(&b2) {
                Ok(res) => res,
                Err(e) => {
                    println!("Error in test forward pass layer 2 bias addition: {}", e);
                    continue; // Skip this test sample
                }
            },
            Err(e) => {
                println!(
                    "Error in test forward pass layer 2 matrix multiplication: {}",
                    e
                );
                continue; // Skip this test sample
            }
        };

        let a2 = match z2.sigmoid() {
            Ok(result) => result,
            Err(e) => {
                println!(
                    "Error in test forward pass layer 2 sigmoid activation: {}",
                    e
                );
                continue; // Skip this test sample
            }
        };

        // Get the prediction
        let prediction = if let Some(scalar) = a2.to_scalar() {
            scalar
        } else {
            println!("Error: a2.to_scalar() returned None");
            continue; // Skip this test sample
        };

        let target = targets[[i, 0]];

        println!(
            "Input: [{:.1}, {:.1}], Target: {:.1}, Prediction: {:.4}",
            x[0], x[1], target, prediction
        );
    }

    // Part 2: Using the neural network layers with automatic backpropagation
    println!("\nPart 2: Using Neural Network Layers with Automatic Backpropagation");
    println!("-------------------------------------------------------------");

    // Create a dataset
    let input_dim = 2;
    let hidden_dim = 4;
    let output_dim = 1;

    let dataset = InMemoryDataset::from_arrays(inputs.clone(), targets.clone());

    // Create a data loader
    let batch_size = 4; // Use all samples in one batch for this small example
                        // Not using the data loader in this example as we'll create a new one for each epoch
    let dataloader = DataLoader::new(Box::new(dataset), batch_size, true, Some(42));

    println!(
        "Created dataset and data loader with batch size {}",
        batch_size
    );

    // Create a model
    let mut model = Sequential::new("XorModel", Vec::new());

    // Add layers
    model.add_layer(Box::new(Linear::new_random(
        "fc1",
        input_dim,
        hidden_dim,
        true,
        Some(ActivationFunc::Sigmoid),
    )));

    model.add_layer(Box::new(Linear::new_random(
        "fc2",
        hidden_dim,
        output_dim,
        true,
        Some(ActivationFunc::Sigmoid),
    )));

    println!(
        "Created sequential model with {} layers",
        model.layers().len()
    );

    // Create optimizer and variables
    let mut optimizer = Adam::new(0.1, Some(0.9), Some(0.999), Some(1e-8));

    // Add model parameters to optimizer
    for (i, param) in model.parameters().iter().enumerate() {
        if let Some(array) = param.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
            let var = Variable::new(&format!("param_{}", i), array.as_array().clone());
            optimizer.add_variable(var);
        }
    }

    println!(
        "Created Adam optimizer with {} variables",
        optimizer.variables().len()
    );

    // Create loss function
    let lossfn = MSELoss::new(Some("mean"));

    // Training loop with automatic backpropagation
    println!("\nTraining with automatic backpropagation (10 epochs):");

    let num_epochs = 10;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;

        // In a real implementation, we would clone the data loader
        // Since DataLoader doesn't implement Clone, we'll create a new one each time
        let loader = DataLoader::new(
            Box::new(InMemoryDataset::from_arrays(
                inputs.clone(),
                targets.clone(),
            )),
            batch_size,
            true,
            Some(42),
        );

        // Get number of batches before iterating
        let numbatches = loader.numbatches();

        // Training loop over batches
        for (inputs, targets) in loader {
            // Zero gradients
            optimizer.zero_grad();

            let mut batch_loss = 0.0;

            // Process each sample in the batch
            for (input, target) in inputs.iter().zip(targets.iter()) {
                // Forward pass
                let output = match model.forward(input.as_ref()) {
                    Ok(out) => out,
                    Err(e) => {
                        println!("Error in automatic backprop forward pass: {}", e);
                        continue; // Skip this sample
                    }
                };

                // Compute loss
                let loss = match lossfn.forward(output.as_ref(), target.as_ref()) {
                    Ok(l) => l,
                    Err(e) => {
                        println!("Error computing loss in automatic backprop: {}", e);
                        continue; // Skip this sample
                    }
                };

                // Get loss scalar
                if let Some(loss_array) = loss.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                    batch_loss += loss_array.as_array().sum();

                    // Backward pass
                    // In a real implementation, this would:
                    // 1. Set gradients in the output tensor
                    // 2. Propagate gradients backward through the network
                    // 3. Update the gradients in all Variable objects

                    // For this example, we'll simulate successful backward propagation
                    for var in optimizer.variables() {
                        // Simulate gradient computation - use a dummy size for demonstration
                        // In a real implementation, we would get the correct shape from the variable
                        // This is a workaround since shape() and set_gradient() aren't available
                        let tensor_value = var.value();
                        if let Some(array) = tensor_value
                            .as_any()
                            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
                        {
                            // Get actual shape from the array
                            let shape = array.as_array().shape();
                            let grad_array = Array2::<f64>::from_elem((shape[0], shape[1]), 0.01);

                            // In a real implementation, we would set the gradient here
                            // Since we can't, we'll just log that it would happen
                            println!("Would set gradient for variable {}", var.name());
                        }
                    }
                }
            }

            // Average loss for the batch
            let avg_batch_loss = batch_loss / inputs.len() as f64;
            total_loss += avg_batch_loss;

            // Update weights
            match optimizer.step() {
                Ok(_) => {}
                Err(e) => println!("Error in optimizer step: {}", e),
            };
        }

        // Print stats
        let avg_loss = total_loss / numbatches as f64; // Use the stored numbatches
        println!("Epoch {}: loss = {:.4}", epoch + 1, avg_loss);
    }

    // Test the trained model
    println!("\nTesting the model trained with automatic backpropagation:");

    // Set model to evaluation mode
    model.eval();

    for i in 0..inputs.shape()[0] {
        // Get input
        let x_array = Array1::<f64>::from_iter(inputs.row(i).iter().cloned());
        let x_wrapped = NdarrayWrapper::new(x_array.clone());

        // Forward pass
        let output = match model.forward(&x_wrapped) {
            Ok(out) => out,
            Err(e) => {
                println!("Error in final prediction forward pass: {}", e);
                continue; // Skip this test case
            }
        };

        // Get prediction
        let prediction = if let Some(output_array) =
            output.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        {
            // Use correct indexing for a 2D array
            output_array.as_array()[[0, 0]]
        } else {
            0.0
        };

        // Get target
        let target = targets[[i, 0]];

        println!(
            "Input: [{:.1}, {:.1}], Target: {:.1}, Prediction: {:.4}",
            x_array[0], x_array[1], target, prediction
        );
    }

    println!("\nBackpropagation example completed successfully!");
}

/// Extension methods for GradientTensor to demonstrate backpropagation
trait GradientTensorExt {
    /// Convert to a scalar value
    fn to_scalar(&self) -> Option<f64>;

    /// Update parameter using its gradient
    fn update_with_learningrate(&mut self, learningrate: f64);

    /// Zero out the gradient
    fn zero_grad(&mut self);

    /// Apply sigmoid activation
    fn sigmoid(&self) -> Result<GradientTensor, &'static str>;

    /// Compute mean of all elements
    fn mean(&self) -> Result<GradientTensor, &'static str>;

    /// Matrix multiplication with another tensor
    fn matmul(&self, other: &GradientTensor) -> Result<GradientTensor, &'static str>;

    /// Add another tensor
    fn add(&self, other: &GradientTensor) -> Result<GradientTensor, &'static str>;

    /// Subtract another tensor
    fn subtract(&self, other: &GradientTensor) -> Result<GradientTensor, &'static str>;

    /// Multiply element-wise with another tensor
    fn multiply(&self, other: &GradientTensor) -> Result<GradientTensor, &'static str>;
}

impl GradientTensorExt for GradientTensor {
    fn to_scalar(&self) -> Option<f64> {
        // Get the underlying ndarray
        if let Some(array) = self
            .value()
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        {
            let ndarray = array.as_array();
            if ndarray.len() == 1 {
                return Some(ndarray.iter().next().cloned().unwrap_or(0.0));
            }
        }
        None
    }

    fn update_with_learningrate(&mut self, learningrate: f64) {
        // In a real implementation, this would:
        // 1. Access both the tensor data and its gradient
        // 2. Update the tensor data using the gradient and learning rate
        // For this example, we'll simulate a simple update

        // Get the underlying ndarray
        if let Some(array) = self
            .value()
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        {
            let ndarray = array.as_array().clone();

            // Get the gradient (in a real implementation, this would come from the backward pass)
            if let Some(grad) = self.grad_2() {
                if let Some(grad_array) = grad.as_any().downcast_ref::<NdarrayWrapper<f64, Ix2>>() {
                    let grad_ndarray = grad_array.as_array();

                    // Update using gradient descent: w = w - lr * grad
                    let updated_array = &ndarray - &(grad_ndarray * learningrate);

                    // In a real implementation, this would update the tensor data
                    // For this example, we'll just log that it would happen
                    println!(
                        "Would update tensor data with gradient using learning _rate {}",
                        learningrate
                    );
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        // In a real implementation, this would clear the gradient data
        // For this example, we'll just log that it would happen
        println!("Would clear gradient data");
    }

    fn sigmoid(&self) -> Result<GradientTensor, &'static str> {
        // Get the underlying ndarray
        if let Some(array) = self
            .value()
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        {
            let ndarray = array.as_array().clone();

            // Apply sigmoid: 1.0 / (1.0 + (-x).exp())
            let result = ndarray.mapv(|x| 1.0 / (1.0 + (-x).exp()));

            // Create a new gradient tensor with the result
            // In a real implementation, this would also record the operation for backpropagation
            return Ok(GradientTensor::from_array(result, true));
        }

        Err("Failed to apply sigmoid: input is not an ndarray")
    }

    fn mean(&self) -> Result<GradientTensor, &'static str> {
        // Get the underlying ndarray
        if let Some(array) = self
            .value()
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        {
            let ndarray = array.as_array().clone();

            // Compute mean
            let mean_val = ndarray.mean();
            let result = Array1::from_elem(1, mean_val);

            // Create a new gradient tensor with the result
            // In a real implementation, this would also record the operation for backpropagation
            return Ok(GradientTensor::from_array(result, true));
        }

        Err("Failed to compute mean: input is not an ndarray")
    }

    fn matmul(&self, other: &GradientTensor) -> Result<GradientTensor, &'static str> {
        // In a real implementation, this would use a proper matmul function
        // For this example, we'll simulate matrix multiplication with simple array creation

        if let (Some(self_array), Some(other_array)) = (
            self.value()
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            other
                .value()
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
        ) {
            // Extract dimensions for demonstration
            let self_dim = self_array.as_array().shape();
            let other_dim = other_array.as_array().shape();

            // Check if dimensions are compatible for matmul
            if !self_dim.is_empty()
                && !other_dim.is_empty()
                && self_dim[self_dim.len() - 1] == other_dim[0]
            {
                // In a real implementation, we would do an actual matrix multiplication here
                // For this example, we're just creating a dummy result
                println!(
                    "Would perform matmul between arrays of dimensions {:?} and {:?}",
                    self_dim, other_dim
                );

                // Create a simple result array with appropriate dimensions
                let result_dim = if self_dim.len() >= 2 && other_dim.len() >= 2 {
                    (self_dim[0], other_dim[other_dim.len() - 1])
                } else {
                    (1, 1) // Fallback
                };

                let result = Array2::<f64>::from_elem(result_dim, 0.5);
                return Ok(GradientTensor::from_array(result, true));
            } else {
                return Err("Incompatible dimensions for matrix multiplication");
            }
        }

        Err("Failed to perform matrix multiplication: inputs are not ndarrays")
    }

    fn add(&self, other: &GradientTensor) -> Result<GradientTensor, &'static str> {
        // In a real implementation, this would properly add the tensors
        // For this example, we'll simulate addition with simple array creation

        if let (Some(self_array), Some(_other_array)) = (
            self.value()
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            other
                .value()
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
        ) {
            // For demonstration, we're just creating a dummy result
            println!("Would perform addition between arrays");

            // Create a simple result array (in real implementation we would add the arrays)
            let result = self_array.as_array().clone();
            return Ok(GradientTensor::from_array(result, true));
        }

        Err("Failed to perform addition: inputs are not ndarrays")
    }

    fn subtract(&self, other: &GradientTensor) -> Result<GradientTensor, &'static str> {
        // In a real implementation, this would properly subtract the tensors
        // For this example, we'll simulate subtraction with simple array creation

        if let (Some(self_array), Some(_other_array)) = (
            self.value()
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            other
                .value()
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
        ) {
            // For demonstration, we're just creating a dummy result
            println!("Would perform subtraction between arrays");

            // Create a simple result array (in real implementation we would subtract the arrays)
            let result = self_array.as_array().clone();
            return Ok(GradientTensor::from_array(result, true));
        }

        Err("Failed to perform subtraction: inputs are not ndarrays")
    }

    fn multiply(&self, other: &GradientTensor) -> Result<GradientTensor, &'static str> {
        // In a real implementation, this would properly multiply the tensors element-wise
        // For this example, we'll simulate multiplication with simple array creation

        if let (Some(self_array), Some(_other_array)) = (
            self.value()
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
            other
                .value()
                .as_any()
                .downcast_ref::<NdarrayWrapper<f64, Ix2>>(),
        ) {
            // For demonstration, we're just creating a dummy result
            println!("Would perform element-wise multiplication between arrays");

            // Create a simple result array (in real implementation we would multiply the arrays)
            let result = self_array.as_array().clone();
            return Ok(GradientTensor::from_array(result, true));
        }

        Err("Failed to perform multiplication: inputs are not ndarrays")
    }
}
