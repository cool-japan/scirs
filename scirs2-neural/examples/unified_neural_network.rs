#![allow(dead_code)]

use ndarray::{s, Array, Array1, Array2, Array3, Array4, Axis};
use rand::Rng;
use std::f32;
use std::fmt::Debug;

// Trait for all neural network layers
#[allow(dead_code)]
trait Layer: Debug {
    // Forward pass for training mode
    fn forward(&mut self, input: &Tensor, is_training: bool) -> Tensor;

    // Backward pass returning gradients for inputs
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;

    // Update layer parameters (weights, biases, etc.)
    fn update_parameters(&mut self, learning_rate: f32);

    // Return a descriptive name for the layer
    fn name(&self) -> String;
}

// Enum for tensor dimensions
#[derive(Debug, Clone, PartialEq)]
enum TensorDim {
    Dim1(usize),                      // [size]
    Dim2(usize, usize),               // [batch_size, features]
    Dim3(usize, usize, usize),        // [batch_size, seq_len, features]
    Dim4(usize, usize, usize, usize), // [batch_size, channels, height, width]
}

// Tensor wrapper to handle different dimensions
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Tensor {
    // Actual data is stored in the most general case
    data: Array4<f32>,

    // Dimension information for correct interpretation
    dim: TensorDim,
}

impl Tensor {
    // Create a 1D tensor
    fn new_1d(data: Array1<f32>) -> Self {
        let shape = data.shape();
        let mut tensor_data = Array4::<f32>::zeros((1, 1, 1, shape[0]));

        for i in 0..shape[0] {
            tensor_data[[0, 0, 0, i]] = data[i];
        }

        Tensor {
            data: tensor_data,
            dim: TensorDim::Dim1(shape[0]),
        }
    }

    // Create a 2D tensor
    fn new_2d(data: Array2<f32>) -> Self {
        let shape = data.shape();
        let mut tensor_data = Array4::<f32>::zeros((shape[0], 1, 1, shape[1]));

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                tensor_data[[i, 0, 0, j]] = data[[i, j]];
            }
        }

        Tensor {
            data: tensor_data,
            dim: TensorDim::Dim2(shape[0], shape[1]),
        }
    }

    // Create a 3D tensor
    fn new_3d(data: Array3<f32>) -> Self {
        let shape = data.shape();
        let mut tensor_data = Array4::<f32>::zeros((shape[0], 1, shape[1], shape[2]));

        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    tensor_data[[i, 0, j, k]] = data[[i, j, k]];
                }
            }
        }

        Tensor {
            data: tensor_data,
            dim: TensorDim::Dim3(shape[0], shape[1], shape[2]),
        }
    }

    // Create a 4D tensor
    fn new_4d(data: Array4<f32>) -> Self {
        let shape = data.shape().to_vec();
        Tensor {
            data,
            dim: TensorDim::Dim4(shape[0], shape[1], shape[2], shape[3]),
        }
    }

    // Get the underlying data as a view appropriate for the dimension
    fn as_1d(&self) -> Array1<f32> {
        match self.dim {
            TensorDim::Dim1(size) => {
                let mut result = Array1::<f32>::zeros(size);
                for i in 0..size {
                    result[i] = self.data[[0, 0, 0, i]];
                }
                result
            }
            _ => panic!("Cannot convert tensor to 1D: incompatible dimensions"),
        }
    }

    fn as_2d(&self) -> Array2<f32> {
        match self.dim {
            TensorDim::Dim2(dim1, dim2) => {
                let mut result = Array2::<f32>::zeros((dim1, dim2));
                for i in 0..dim1 {
                    for j in 0..dim2 {
                        result[[i, j]] = self.data[[i, 0, 0, j]];
                    }
                }
                result
            }
            _ => panic!("Cannot convert tensor to 2D: incompatible dimensions"),
        }
    }

    fn as_3d(&self) -> Array3<f32> {
        match self.dim {
            TensorDim::Dim3(dim1, dim2, dim3) => {
                let mut result = Array3::<f32>::zeros((dim1, dim2, dim3));
                for i in 0..dim1 {
                    for j in 0..dim2 {
                        for k in 0..dim3 {
                            result[[i, j, k]] = self.data[[i, 0, j, k]];
                        }
                    }
                }
                result
            }
            _ => panic!("Cannot convert tensor to 3D: incompatible dimensions"),
        }
    }

    fn as_4d(&self) -> Array4<f32> {
        match self.dim {
            TensorDim::Dim4(_, _, _, _) => self.data.clone(),
            _ => panic!("Cannot convert tensor to 4D: incompatible dimensions"),
        }
    }

    // Get shape based on dimensions
    fn shape(&self) -> Vec<usize> {
        match self.dim {
            TensorDim::Dim1(d1) => vec![d1],
            TensorDim::Dim2(d1, d2) => vec![d1, d2],
            TensorDim::Dim3(d1, d2, d3) => vec![d1, d2, d3],
            TensorDim::Dim4(d1, d2, d3, d4) => vec![d1, d2, d3, d4],
        }
    }

    // Reshape tensor to new dimensions
    fn reshape(&self, new_dim: TensorDim) -> Self {
        // Check if total size is compatible
        let orig_size: usize = match self.dim {
            TensorDim::Dim1(d1) => d1,
            TensorDim::Dim2(d1, d2) => d1 * d2,
            TensorDim::Dim3(d1, d2, d3) => d1 * d2 * d3,
            TensorDim::Dim4(d1, d2, d3, d4) => d1 * d2 * d3 * d4,
        };

        let new_size: usize = match new_dim {
            TensorDim::Dim1(d1) => d1,
            TensorDim::Dim2(d1, d2) => d1 * d2,
            TensorDim::Dim3(d1, d2, d3) => d1 * d2 * d3,
            TensorDim::Dim4(d1, d2, d3, d4) => d1 * d2 * d3 * d4,
        };

        assert_eq!(orig_size, new_size, "Incompatible sizes for reshape");

        // Create a new tensor with proper dimensions
        match new_dim {
            TensorDim::Dim1(d1) => {
                let mut arr = Array1::<f32>::zeros(d1);
                let mut idx = 0;

                // Linearize data
                match self.dim {
                    TensorDim::Dim1(_) => {
                        for i in 0..d1 {
                            arr[i] = self.data[[0, 0, 0, i]];
                        }
                    }
                    TensorDim::Dim2(old_d1, old_d2) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                arr[idx] = self.data[[i, 0, 0, j]];
                                idx += 1;
                            }
                        }
                    }
                    TensorDim::Dim3(old_d1, old_d2, old_d3) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                for k in 0..old_d3 {
                                    arr[idx] = self.data[[i, 0, j, k]];
                                    idx += 1;
                                }
                            }
                        }
                    }
                    TensorDim::Dim4(old_d1, old_d2, old_d3, old_d4) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                for k in 0..old_d3 {
                                    for l in 0..old_d4 {
                                        arr[idx] = self.data[[i, j, k, l]];
                                        idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                Self::new_1d(arr)
            }
            TensorDim::Dim2(d1, d2) => {
                let mut arr = Array2::<f32>::zeros((d1, d2));
                let mut idx = 0;

                // Linearize and reshape
                let total = d1 * d2;
                match self.dim {
                    TensorDim::Dim1(old_d1) => {
                        for i in 0..old_d1 {
                            let row = idx / d2;
                            let col = idx % d2;
                            arr[[row, col]] = self.data[[0, 0, 0, i]];
                            idx += 1;
                        }
                    }
                    TensorDim::Dim2(old_d1, old_d2) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                let row = idx / d2;
                                let col = idx % d2;
                                arr[[row, col]] = self.data[[i, 0, 0, j]];
                                idx += 1;
                                if idx >= total {
                                    break;
                                }
                            }
                        }
                    }
                    TensorDim::Dim3(old_d1, old_d2, old_d3) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                for k in 0..old_d3 {
                                    let row = idx / d2;
                                    let col = idx % d2;
                                    arr[[row, col]] = self.data[[i, 0, j, k]];
                                    idx += 1;
                                    if idx >= total {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    TensorDim::Dim4(old_d1, old_d2, old_d3, old_d4) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                for k in 0..old_d3 {
                                    for l in 0..old_d4 {
                                        let row = idx / d2;
                                        let col = idx % d2;
                                        arr[[row, col]] = self.data[[i, j, k, l]];
                                        idx += 1;
                                        if idx >= total {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Self::new_2d(arr)
            }
            TensorDim::Dim3(d1, d2, d3) => {
                // Implementation for 3D reshaping
                let mut arr = Array3::<f32>::zeros((d1, d2, d3));
                let mut idx = 0;
                let total = d1 * d2 * d3;

                // Linearize and reshape
                match self.dim {
                    // Implementation for all source dimensions
                    TensorDim::Dim4(old_d1, old_d2, old_d3, old_d4) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                for k in 0..old_d3 {
                                    for l in 0..old_d4 {
                                        let i_new = idx / (d2 * d3);
                                        let rem = idx % (d2 * d3);
                                        let j_new = rem / d3;
                                        let k_new = rem % d3;

                                        arr[[i_new, j_new, k_new]] = self.data[[i, j, k, l]];
                                        idx += 1;
                                        if idx >= total {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Add implementations for other source dimensions as needed
                    _ => panic!("Reshape from this dimension to 3D not implemented"),
                }

                Self::new_3d(arr)
            }
            TensorDim::Dim4(d1, d2, d3, d4) => {
                // Allocate new 4D tensor
                let mut new_data = Array4::<f32>::zeros((d1, d2, d3, d4));
                let mut idx = 0;
                let total = d1 * d2 * d3 * d4;

                // Collect all values in linear order
                let mut values = Vec::with_capacity(total);

                match self.dim {
                    TensorDim::Dim1(old_d1) => {
                        for i in 0..old_d1 {
                            values.push(self.data[[0, 0, 0, i]]);
                        }
                    }
                    TensorDim::Dim2(old_d1, old_d2) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                values.push(self.data[[i, 0, 0, j]]);
                            }
                        }
                    }
                    TensorDim::Dim3(old_d1, old_d2, old_d3) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                for k in 0..old_d3 {
                                    values.push(self.data[[i, 0, j, k]]);
                                }
                            }
                        }
                    }
                    TensorDim::Dim4(old_d1, old_d2, old_d3, old_d4) => {
                        for i in 0..old_d1 {
                            for j in 0..old_d2 {
                                for k in 0..old_d3 {
                                    for l in 0..old_d4 {
                                        values.push(self.data[[i, j, k, l]]);
                                    }
                                }
                            }
                        }
                    }
                }

                // Fill the new tensor
                for i in 0..d1 {
                    for j in 0..d2 {
                        for k in 0..d3 {
                            for l in 0..d4 {
                                if idx < values.len() {
                                    new_data[[i, j, k, l]] = values[idx];
                                }
                                idx += 1;
                            }
                        }
                    }
                }

                Tensor {
                    data: new_data,
                    dim: TensorDim::Dim4(d1, d2, d3, d4),
                }
            }
        }
    }
}

// Linear Layer
#[allow(dead_code)]
#[derive(Debug)]
struct Linear {
    name_str: String,
    in_features: usize,
    out_features: usize,
    weight: Array2<f32>,
    bias: Array1<f32>,

    // Gradients
    dweight: Option<Array2<f32>>,
    dbias: Option<Array1<f32>>,

    // Cache for backward pass
    input: Option<Tensor>,
}

impl Linear {
    fn new(in_features: usize, out_features: usize, name: Option<String>) -> Self {
        // Xavier/Glorot initialization
        let bound = (6.0 / (in_features + out_features) as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Initialize weight matrix with random values
        let mut weight = Array2::<f32>::zeros((out_features, in_features));
        for elem in weight.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        let bias = Array::zeros(out_features);

        Linear {
            name_str: name.unwrap_or_else(|| format!("Linear_{}_{}", in_features, out_features)),
            in_features,
            out_features,
            weight,
            bias,
            dweight: None,
            dbias: None,
            input: None,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Tensor, is_training: bool) -> Tensor {
        // Input should be 2D tensor [batch_size, in_features]
        let x = input.as_2d();

        if is_training {
            self.input = Some(input.clone());
        }

        // Compute output
        let result = x.dot(&self.weight.t());

        // Add bias
        let mut output = result.clone();
        for i in 0..output.shape()[0] {
            for j in 0..output.shape()[1] {
                output[[i, j]] += self.bias[j];
            }
        }

        Tensor::new_2d(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Gradient from next layer
        let dout = grad_output.as_2d();
        let _batch_size = dout.shape()[0];

        // Get cached input
        let x = self
            .input
            .as_ref()
            .expect("No cached input for backward pass")
            .as_2d();

        // Compute gradients
        let dweight = dout.t().dot(&x);
        let dbias = dout.sum_axis(Axis(0));

        // Store gradients for parameter update
        self.dweight = Some(dweight);
        self.dbias = Some(dbias);

        // Compute gradient for input
        let dx = dout.dot(&self.weight);

        Tensor::new_2d(dx)
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        if let Some(dw) = &self.dweight {
            self.weight = &self.weight - &(dw * learning_rate);
        }

        if let Some(db) = &self.dbias {
            self.bias = &self.bias - &(db * learning_rate);
        }
    }

    fn name(&self) -> String {
        self.name_str.clone()
    }
}

// ReLU Activation Layer
#[allow(dead_code)]
#[derive(Debug)]
struct ReLU {
    name_str: String,
    input: Option<Tensor>,
}

impl ReLU {
    fn new(name: Option<String>) -> Self {
        ReLU {
            name_str: name.unwrap_or_else(|| "ReLU".to_string()),
            input: None,
        }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor, is_training: bool) -> Tensor {
        if is_training {
            self.input = Some(input.clone());
        }

        // Apply ReLU based on tensor dimension
        match input.dim {
            TensorDim::Dim1(_) => {
                let x = input.as_1d();
                let output = x.mapv(|v| v.max(0.0));
                Tensor::new_1d(output)
            }
            TensorDim::Dim2(_, _) => {
                let x = input.as_2d();
                let output = x.mapv(|v| v.max(0.0));
                Tensor::new_2d(output)
            }
            TensorDim::Dim3(_, _, _) => {
                let x = input.as_3d();
                let output = x.mapv(|v| v.max(0.0));
                Tensor::new_3d(output)
            }
            TensorDim::Dim4(_, _, _, _) => {
                let x = input.as_4d();
                let output = x.mapv(|v| v.max(0.0));
                Tensor::new_4d(output)
            }
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Get cached input
        let input = self
            .input
            .as_ref()
            .expect("No cached input for backward pass");

        // ReLU derivative: 1 if input > 0, 0 otherwise
        match input.dim {
            TensorDim::Dim1(_) => {
                let x = input.as_1d();
                let dout = grad_output.as_1d();
                let mut dx = dout.clone();

                for i in 0..x.len() {
                    if x[i] <= 0.0 {
                        dx[i] = 0.0;
                    }
                }

                Tensor::new_1d(dx)
            }
            TensorDim::Dim2(_, _) => {
                let x = input.as_2d();
                let dout = grad_output.as_2d();
                let mut dx = dout.clone();

                for i in 0..x.shape()[0] {
                    for j in 0..x.shape()[1] {
                        if x[[i, j]] <= 0.0 {
                            dx[[i, j]] = 0.0;
                        }
                    }
                }

                Tensor::new_2d(dx)
            }
            TensorDim::Dim3(_, _, _) => {
                let x = input.as_3d();
                let dout = grad_output.as_3d();
                let mut dx = dout.clone();

                for i in 0..x.shape()[0] {
                    for j in 0..x.shape()[1] {
                        for k in 0..x.shape()[2] {
                            if x[[i, j, k]] <= 0.0 {
                                dx[[i, j, k]] = 0.0;
                            }
                        }
                    }
                }

                Tensor::new_3d(dx)
            }
            TensorDim::Dim4(_, _, _, _) => {
                let x = input.as_4d();
                let dout = grad_output.as_4d();
                let mut dx = dout.clone();

                for i in 0..x.shape()[0] {
                    for j in 0..x.shape()[1] {
                        for k in 0..x.shape()[2] {
                            for l in 0..x.shape()[3] {
                                if x[[i, j, k, l]] <= 0.0 {
                                    dx[[i, j, k, l]] = 0.0;
                                }
                            }
                        }
                    }
                }

                Tensor::new_4d(dx)
            }
        }
    }

    fn update_parameters(&mut self, _learning_rate: f32) {
        // ReLU has no parameters to update
    }

    fn name(&self) -> String {
        self.name_str.clone()
    }
}

// Sigmoid Activation Layer
#[allow(dead_code)]
#[derive(Debug)]
struct Sigmoid {
    name_str: String,
    output: Option<Tensor>,
}

impl Sigmoid {
    fn new(name: Option<String>) -> Self {
        Sigmoid {
            name_str: name.unwrap_or_else(|| "Sigmoid".to_string()),
            output: None,
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: &Tensor, is_training: bool) -> Tensor {
        // Apply sigmoid based on tensor dimension
        let output = match input.dim {
            TensorDim::Dim1(_) => {
                let x = input.as_1d();
                let output = x.mapv(|v| Self::sigmoid(v));
                Tensor::new_1d(output)
            }
            TensorDim::Dim2(_, _) => {
                let x = input.as_2d();
                let output = x.mapv(|v| Self::sigmoid(v));
                Tensor::new_2d(output)
            }
            TensorDim::Dim3(_, _, _) => {
                let x = input.as_3d();
                let output = x.mapv(|v| Self::sigmoid(v));
                Tensor::new_3d(output)
            }
            TensorDim::Dim4(_, _, _, _) => {
                let x = input.as_4d();
                let output = x.mapv(|v| Self::sigmoid(v));
                Tensor::new_4d(output)
            }
        };

        if is_training {
            self.output = Some(output.clone());
        }

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Get cached output
        let output = self
            .output
            .as_ref()
            .expect("No cached output for backward pass");

        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        match output.dim {
            TensorDim::Dim1(_) => {
                let y = output.as_1d();
                let dout = grad_output.as_1d();
                let mut dx = dout.clone();

                for i in 0..y.len() {
                    dx[i] *= y[i] * (1.0 - y[i]);
                }

                Tensor::new_1d(dx)
            }
            TensorDim::Dim2(_, _) => {
                let y = output.as_2d();
                let dout = grad_output.as_2d();
                let mut dx = dout.clone();

                for i in 0..y.shape()[0] {
                    for j in 0..y.shape()[1] {
                        dx[[i, j]] *= y[[i, j]] * (1.0 - y[[i, j]]);
                    }
                }

                Tensor::new_2d(dx)
            }
            TensorDim::Dim3(_, _, _) => {
                let y = output.as_3d();
                let dout = grad_output.as_3d();
                let mut dx = dout.clone();

                for i in 0..y.shape()[0] {
                    for j in 0..y.shape()[1] {
                        for k in 0..y.shape()[2] {
                            dx[[i, j, k]] *= y[[i, j, k]] * (1.0 - y[[i, j, k]]);
                        }
                    }
                }

                Tensor::new_3d(dx)
            }
            TensorDim::Dim4(_, _, _, _) => {
                let y = output.as_4d();
                let dout = grad_output.as_4d();
                let mut dx = dout.clone();

                for i in 0..y.shape()[0] {
                    for j in 0..y.shape()[1] {
                        for k in 0..y.shape()[2] {
                            for l in 0..y.shape()[3] {
                                dx[[i, j, k, l]] *= y[[i, j, k, l]] * (1.0 - y[[i, j, k, l]]);
                            }
                        }
                    }
                }

                Tensor::new_4d(dx)
            }
        }
    }

    fn update_parameters(&mut self, _learning_rate: f32) {
        // Sigmoid has no parameters to update
    }

    fn name(&self) -> String {
        self.name_str.clone()
    }
}

// Softmax Layer for classification
#[allow(dead_code)]
#[derive(Debug)]
struct Softmax {
    name_str: String,
    output: Option<Tensor>,
}

impl Softmax {
    fn new(name: Option<String>) -> Self {
        Softmax {
            name_str: name.unwrap_or_else(|| "Softmax".to_string()),
            output: None,
        }
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor, is_training: bool) -> Tensor {
        // Input should be 2D tensor [batch_size, features]
        let x = input.as_2d();

        let mut output = Array2::<f32>::zeros(x.raw_dim());

        for (i, row) in x.outer_iter().enumerate() {
            // Find max for numerical stability
            let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Calculate exp and sum
            let mut sum = 0.0;
            let mut exp_vals = vec![0.0; row.len()];

            for (j, &val) in row.iter().enumerate() {
                let exp_val = (val - max_val).exp();
                exp_vals[j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for (j, &exp_val) in exp_vals.iter().enumerate() {
                output[[i, j]] = exp_val / sum;
            }
        }

        let result = Tensor::new_2d(output);

        if is_training {
            self.output = Some(result.clone());
        }

        result
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // This is a simplified backward pass for softmax
        // In practice, softmax and cross-entropy loss are often combined
        // for better numerical stability

        // Get cached output
        let output = self
            .output
            .as_ref()
            .expect("No cached output for backward pass");
        let y = output.as_2d();
        let dout = grad_output.as_2d();

        let batch_size = y.shape()[0];
        let num_classes = y.shape()[1];

        let mut dx = Array2::<f32>::zeros(y.raw_dim());

        // For each item in the batch
        for i in 0..batch_size {
            // Compute Jacobian matrix of softmax
            let mut jacobian = Array2::<f32>::zeros((num_classes, num_classes));

            for j in 0..num_classes {
                for k in 0..num_classes {
                    if j == k {
                        jacobian[[j, k]] = y[[i, j]] * (1.0 - y[[i, j]]);
                    } else {
                        jacobian[[j, k]] = -y[[i, j]] * y[[i, k]];
                    }
                }
            }

            // Apply Jacobian to gradient
            let grad_i = Array1::<f32>::from_iter(dout.slice(s![i, ..]).iter().cloned());
            let dx_i = jacobian.dot(&grad_i);

            for j in 0..num_classes {
                dx[[i, j]] = dx_i[j];
            }
        }

        Tensor::new_2d(dx)
    }

    fn update_parameters(&mut self, _learning_rate: f32) {
        // Softmax has no parameters to update
    }

    fn name(&self) -> String {
        self.name_str.clone()
    }
}

// LSTM Cell for sequence processing
#[allow(dead_code)]
#[derive(Debug)]
struct LSTMCell {
    name_str: String,
    input_size: usize,
    hidden_size: usize,

    // Parameters
    w_ii: Array2<f32>, // Input to input gate
    w_hi: Array2<f32>, // Hidden to input gate
    b_i: Array1<f32>,  // Input gate bias

    w_if: Array2<f32>, // Input to forget gate
    w_hf: Array2<f32>, // Hidden to forget gate
    b_f: Array1<f32>,  // Forget gate bias

    w_ig: Array2<f32>, // Input to cell gate
    w_hg: Array2<f32>, // Hidden to cell gate
    b_g: Array1<f32>,  // Cell gate bias

    w_io: Array2<f32>, // Input to output gate
    w_ho: Array2<f32>, // Hidden to output gate
    b_o: Array1<f32>,  // Output gate bias

    // Gradients
    dw_ii: Option<Array2<f32>>,
    dw_hi: Option<Array2<f32>>,
    db_i: Option<Array1<f32>>,

    dw_if: Option<Array2<f32>>,
    dw_hf: Option<Array2<f32>>,
    db_f: Option<Array1<f32>>,

    dw_ig: Option<Array2<f32>>,
    dw_hg: Option<Array2<f32>>,
    db_g: Option<Array1<f32>>,

    dw_io: Option<Array2<f32>>,
    dw_ho: Option<Array2<f32>>,
    db_o: Option<Array1<f32>>,

    // Cache for backward pass
    input: Option<Tensor>,
    h_prev: Option<Tensor>,
    c_prev: Option<Tensor>,
    i_t: Option<Array2<f32>>,
    f_t: Option<Array2<f32>>,
    g_t: Option<Array2<f32>>,
    o_t: Option<Array2<f32>>,
    c_t: Option<Array2<f32>>,
    h_t: Option<Array2<f32>>,
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize, name: Option<String>) -> Self {
        // Xavier/Glorot initialization
        let bound = (6.0 / (input_size + hidden_size) as f32).sqrt();

        // Create a random number generator
        let mut rng = rand::rng();

        // Input gate weights
        let mut w_ii = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hi = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_i = Array1::zeros(hidden_size);

        // Initialize with random values
        for elem in w_ii.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hi.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        // Forget gate weights (initialize forget gate bias to 1 to avoid vanishing gradients early in training)
        let mut w_if = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hf = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_f = Array1::ones(hidden_size);

        // Initialize with random values
        for elem in w_if.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hf.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        // Cell gate weights
        let mut w_ig = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_hg = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_g = Array1::zeros(hidden_size);

        // Initialize with random values
        for elem in w_ig.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_hg.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        // Output gate weights
        let mut w_io = Array2::<f32>::zeros((hidden_size, input_size));
        let mut w_ho = Array2::<f32>::zeros((hidden_size, hidden_size));
        let b_o = Array1::zeros(hidden_size);

        // Initialize with random values
        for elem in w_io.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }
        for elem in w_ho.iter_mut() {
            *elem = rng.random_range(-bound..bound);
        }

        LSTMCell {
            name_str: name.unwrap_or_else(|| format!("LSTMCell_{}_{}", input_size, hidden_size)),
            input_size,
            hidden_size,
            w_ii,
            w_hi,
            b_i,
            w_if,
            w_hf,
            b_f,
            w_ig,
            w_hg,
            b_g,
            w_io,
            w_ho,
            b_o,
            dw_ii: None,
            dw_hi: None,
            db_i: None,
            dw_if: None,
            dw_hf: None,
            db_f: None,
            dw_ig: None,
            dw_hg: None,
            db_g: None,
            dw_io: None,
            dw_ho: None,
            db_o: None,
            input: None,
            h_prev: None,
            c_prev: None,
            i_t: None,
            f_t: None,
            g_t: None,
            o_t: None,
            c_t: None,
            h_t: None,
        }
    }

    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn tanh(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.tanh())
    }
}

// Sequential model that contains a list of layers
#[allow(dead_code)]
#[derive(Debug)]
struct Sequential {
    name_str: String,
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    fn new(name: Option<String>) -> Self {
        Sequential {
            name_str: name.unwrap_or_else(|| "Sequential".to_string()),
            layers: Vec::new(),
        }
    }

    fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    fn forward(&mut self, input: &Tensor, is_training: bool) -> Tensor {
        let mut output = input.clone();

        for layer in &mut self.layers {
            output = layer.forward(&output, is_training);
        }

        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut grad = grad_output.clone();

        // Backpropagate through layers in reverse order
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }

        grad
    }

    fn update_parameters(&mut self, learning_rate: f32) {
        for layer in &mut self.layers {
            layer.update_parameters(learning_rate);
        }
    }

    fn name(&self) -> String {
        self.name_str.clone()
    }
}

// Loss function trait
trait Loss {
    fn forward(&mut self, predictions: &Tensor, targets: &Tensor) -> f32;
    fn backward(&self) -> Tensor;
}

// Mean Squared Error Loss
#[derive(Debug)]
struct MSELoss {
    predictions: Option<Tensor>,
    targets: Option<Tensor>,
}

impl MSELoss {
    fn new() -> Self {
        MSELoss {
            predictions: None,
            targets: None,
        }
    }
}

impl Loss for MSELoss {
    fn forward(&mut self, predictions: &Tensor, targets: &Tensor) -> f32 {
        self.predictions = Some(predictions.clone());
        self.targets = Some(targets.clone());

        // Calculate MSE based on tensor dimensions
        match predictions.dim {
            TensorDim::Dim2(_, _) => {
                let pred = predictions.as_2d();
                let targ = targets.as_2d();

                let mut total_loss = 0.0;
                let mut count = 0;

                for i in 0..pred.shape()[0] {
                    for j in 0..pred.shape()[1] {
                        let error = pred[[i, j]] - targ[[i, j]];
                        total_loss += error * error;
                        count += 1;
                    }
                }

                total_loss / count as f32
            }
            _ => panic!("MSELoss only supports 2D tensors"),
        }
    }

    fn backward(&self) -> Tensor {
        let predictions = self
            .predictions
            .as_ref()
            .expect("No predictions for backward pass");
        let targets = self.targets.as_ref().expect("No targets for backward pass");

        match predictions.dim {
            TensorDim::Dim2(_, _) => {
                let pred = predictions.as_2d();
                let targ = targets.as_2d();

                let batch_size = pred.shape()[0];
                let features = pred.shape()[1];

                let mut grad = Array2::<f32>::zeros((batch_size, features));

                for i in 0..batch_size {
                    for j in 0..features {
                        // dL/dpred = 2 * (pred - target) / (batch_size * features)
                        grad[[i, j]] =
                            2.0 * (pred[[i, j]] - targ[[i, j]]) / (batch_size * features) as f32;
                    }
                }

                Tensor::new_2d(grad)
            }
            _ => panic!("MSELoss only supports 2D tensors"),
        }
    }
}

// Cross Entropy Loss
#[derive(Debug)]
struct CrossEntropyLoss {
    predictions: Option<Tensor>,
    targets: Option<Tensor>,
}

impl CrossEntropyLoss {
    fn new() -> Self {
        CrossEntropyLoss {
            predictions: None,
            targets: None,
        }
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&mut self, predictions: &Tensor, targets: &Tensor) -> f32 {
        self.predictions = Some(predictions.clone());
        self.targets = Some(targets.clone());

        // Calculate cross entropy loss for 2D predictions [batch_size, classes]
        // and targets can be either one-hot encoded or class indices
        match predictions.dim {
            TensorDim::Dim2(_, _) => {
                let pred = predictions.as_2d();
                let batch_size = pred.shape()[0];
                let num_classes = pred.shape()[1];

                let mut total_loss = 0.0;

                match targets.dim {
                    // Targets are class indices [batch_size]
                    TensorDim::Dim1(_) => {
                        let targ = targets.as_1d();

                        for i in 0..batch_size {
                            let target_idx = targ[i] as usize;
                            if target_idx < num_classes {
                                // Add small epsilon for numerical stability
                                let epsilon = 1e-10;
                                total_loss -= (pred[[i, target_idx]] + epsilon).ln();
                            }
                        }
                    }
                    // Targets are one-hot encoded [batch_size, classes]
                    TensorDim::Dim2(_, _) => {
                        let targ = targets.as_2d();

                        for i in 0..batch_size {
                            for j in 0..num_classes {
                                if targ[[i, j]] > 0.0 {
                                    // Add small epsilon for numerical stability
                                    let epsilon = 1e-10;
                                    total_loss -= targ[[i, j]] * (pred[[i, j]] + epsilon).ln();
                                }
                            }
                        }
                    }
                    _ => panic!("Unsupported target tensor dimensions for CrossEntropyLoss"),
                }

                total_loss / batch_size as f32
            }
            _ => panic!("CrossEntropyLoss only supports 2D predictions"),
        }
    }

    fn backward(&self) -> Tensor {
        let predictions = self
            .predictions
            .as_ref()
            .expect("No predictions for backward pass");
        let targets = self.targets.as_ref().expect("No targets for backward pass");

        // Gradient of cross entropy loss
        match predictions.dim {
            TensorDim::Dim2(_, _) => {
                let pred = predictions.as_2d();
                let batch_size = pred.shape()[0];
                let num_classes = pred.shape()[1];

                let mut grad = Array2::<f32>::zeros((batch_size, num_classes));

                match targets.dim {
                    // Targets are class indices [batch_size]
                    TensorDim::Dim1(_) => {
                        let targ = targets.as_1d();

                        for i in 0..batch_size {
                            let target_idx = targ[i] as usize;
                            if target_idx < num_classes {
                                // Copy predictions to gradient
                                for j in 0..num_classes {
                                    grad[[i, j]] = pred[[i, j]];
                                }

                                // Subtract 1 from the target class
                                grad[[i, target_idx]] -= 1.0;

                                // Normalize by batch size
                                for j in 0..num_classes {
                                    grad[[i, j]] /= batch_size as f32;
                                }
                            }
                        }
                    }
                    // Targets are one-hot encoded [batch_size, classes]
                    TensorDim::Dim2(_, _) => {
                        let targ = targets.as_2d();

                        for i in 0..batch_size {
                            for j in 0..num_classes {
                                grad[[i, j]] = pred[[i, j]] - targ[[i, j]];
                                grad[[i, j]] /= batch_size as f32;
                            }
                        }
                    }
                    _ => {
                        panic!("Unsupported target tensor dimensions for CrossEntropyLoss backward")
                    }
                }

                Tensor::new_2d(grad)
            }
            _ => panic!("CrossEntropyLoss backward only supports 2D predictions"),
        }
    }
}

// Optimizer trait
#[allow(dead_code)]
trait Optimizer {
    fn step(&mut self, model: &mut Sequential, learning_rate: f32);
    fn zero_grad(&mut self);
}

// Simple SGD Optimizer
#[allow(dead_code)]
#[derive(Debug)]
struct SGD {
    model: Option<Box<Sequential>>,
}

impl SGD {
    fn new() -> Self {
        SGD { model: None }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, model: &mut Sequential, learning_rate: f32) {
        model.update_parameters(learning_rate);
    }

    fn zero_grad(&mut self) {
        // No additional state to reset in basic SGD
    }
}

// Trainer for coordinating training
struct Trainer {
    model: Box<Sequential>,
    loss_fn: Box<dyn Loss>,
    optimizer: Box<dyn Optimizer>,
    learning_rate: f32,
}

impl Trainer {
    fn new(
        model: Sequential,
        loss_fn: Box<dyn Loss>,
        optimizer: Box<dyn Optimizer>,
        learning_rate: f32,
    ) -> Self {
        Trainer {
            model: Box::new(model),
            loss_fn,
            optimizer,
            learning_rate,
        }
    }

    fn train_step(&mut self, x: &Tensor, y: &Tensor) -> f32 {
        // Forward pass
        let predictions = self.model.forward(x, true);

        // Calculate loss
        let loss = self.loss_fn.forward(&predictions, y);

        // Backward pass
        let grad = self.loss_fn.backward();
        self.model.backward(&grad);

        // Update parameters
        self.optimizer.step(&mut self.model, self.learning_rate);

        loss
    }

    fn evaluate(&mut self, x: &Tensor, y: &Tensor) -> (f32, Tensor) {
        // Forward pass in evaluation mode
        let predictions = self.model.forward(x, false);

        // Calculate loss
        let loss = self.loss_fn.forward(&predictions, y);

        (loss, predictions)
    }

    fn train_epoch(&mut self, x_batches: &[Tensor], y_batches: &[Tensor]) -> f32 {
        assert_eq!(
            x_batches.len(),
            y_batches.len(),
            "Number of input and target batches must match"
        );

        let mut total_loss = 0.0;

        for (x_batch, y_batch) in x_batches.iter().zip(y_batches.iter()) {
            let batch_loss = self.train_step(x_batch, y_batch);
            total_loss += batch_loss;
        }

        total_loss / x_batches.len() as f32
    }
}

// Helper functions for dataset creation and manipulation

// Create simple classification dataset
fn create_classification_dataset(
    num_samples: usize,
    num_features: usize,
    num_classes: usize,
) -> (Tensor, Tensor) {
    // Create random features
    let mut x_data = Array2::<f32>::zeros((num_samples, num_features));

    // Assign random values
    for i in 0..num_samples {
        for j in 0..num_features {
            x_data[[i, j]] = rand::random::<f32>() * 2.0 - 1.0;
        }
    }

    // Create class labels based on simple rule
    let mut y_data = Array1::<f32>::zeros(num_samples);

    for i in 0..num_samples {
        // Assign class based on first feature's sign and second feature's magnitude
        let sum = x_data[[i, 0]] + x_data[[i, 1]].abs();
        let class =
            (sum * num_classes as f32 / 2.0 + num_classes as f32 / 2.0) as usize % num_classes;
        y_data[i] = class as f32;
    }

    // Convert to tensors
    let x_tensor = Tensor::new_2d(x_data);
    let y_tensor = Tensor::new_1d(y_data);

    (x_tensor, y_tensor)
}

// Create regression dataset with non-linear relationship
fn create_regression_dataset(num_samples: usize, num_features: usize) -> (Tensor, Tensor) {
    // Create random features
    let mut x_data = Array2::<f32>::zeros((num_samples, num_features));

    // Assign random values
    for i in 0..num_samples {
        for j in 0..num_features {
            x_data[[i, j]] = rand::random::<f32>() * 2.0 - 1.0;
        }
    }

    // Create target values based on non-linear function
    let mut y_data = Array2::<f32>::zeros((num_samples, 1));

    for i in 0..num_samples {
        let x1 = x_data[[i, 0]];
        let x2 = if num_features > 1 {
            x_data[[i, 1]]
        } else {
            0.0
        };

        // Non-linear function: y = sin(x1) + x2^2 + noise
        let noise = rand::random::<f32>() * 0.1 - 0.05;
        y_data[[i, 0]] = x1.sin() + x2 * x2 + noise;
    }

    // Convert to tensors
    let x_tensor = Tensor::new_2d(x_data);
    let y_tensor = Tensor::new_2d(y_data);

    (x_tensor, y_tensor)
}

// Split data into batches
fn create_batches(x: &Tensor, y: &Tensor, batch_size: usize) -> (Vec<Tensor>, Vec<Tensor>) {
    match (x.dim.clone(), y.dim.clone()) {
        (TensorDim::Dim2(samples, features), _) => {
            let num_batches = (samples + batch_size - 1) / batch_size;

            let mut x_batches = Vec::with_capacity(num_batches);
            let mut y_batches = Vec::with_capacity(num_batches);

            for i in 0..num_batches {
                let start_idx = i * batch_size;
                let end_idx = (start_idx + batch_size).min(samples);

                // Create X batch
                let x_data = x.as_2d();
                let mut x_batch = Array2::<f32>::zeros((end_idx - start_idx, features));

                for j in start_idx..end_idx {
                    for k in 0..features {
                        x_batch[[j - start_idx, k]] = x_data[[j, k]];
                    }
                }

                // Create Y batch
                match y.dim {
                    TensorDim::Dim1(_) => {
                        let y_data = y.as_1d();
                        let mut y_batch = Array1::<f32>::zeros(end_idx - start_idx);

                        for j in start_idx..end_idx {
                            y_batch[j - start_idx] = y_data[j];
                        }

                        x_batches.push(Tensor::new_2d(x_batch));
                        y_batches.push(Tensor::new_1d(y_batch));
                    }
                    TensorDim::Dim2(_, y_features) => {
                        let y_data = y.as_2d();
                        let mut y_batch = Array2::<f32>::zeros((end_idx - start_idx, y_features));

                        for j in start_idx..end_idx {
                            for k in 0..y_features {
                                y_batch[[j - start_idx, k]] = y_data[[j, k]];
                            }
                        }

                        x_batches.push(Tensor::new_2d(x_batch));
                        y_batches.push(Tensor::new_2d(y_batch));
                    }
                    _ => panic!("Unsupported target tensor dimensions for batching"),
                }
            }

            (x_batches, y_batches)
        }
        _ => panic!("Batching only supported for 2D input tensors"),
    }
}

// Calculate accuracy for classification
fn calculate_accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
    match (predictions.dim.clone(), targets.dim.clone()) {
        (TensorDim::Dim2(batch_size, _), TensorDim::Dim1(_)) => {
            let pred = predictions.as_2d();
            let targ = targets.as_1d();

            let mut correct = 0;

            for i in 0..batch_size {
                // Find predicted class (argmax)
                let mut max_idx = 0;
                let mut max_val = pred[[i, 0]];

                for j in 1..pred.shape()[1] {
                    if pred[[i, j]] > max_val {
                        max_idx = j;
                        max_val = pred[[i, j]];
                    }
                }

                // Compare with target
                if max_idx as f32 == targ[i] {
                    correct += 1;
                }
            }

            correct as f32 / batch_size as f32
        }
        _ => panic!("Accuracy calculation only supported for classification problems"),
    }
}

// Example of building and training a simple classification model
fn classification_example() {
    println!("Classification Example");
    println!("=====================");

    // Create dataset
    let num_samples = 1000;
    let num_features = 2;
    let num_classes = 3;
    let (x, y) = create_classification_dataset(num_samples, num_features, num_classes);

    // Split into train and test sets (80% train, 20% test)
    let num_train = (num_samples as f32 * 0.8) as usize;

    let x_train = match x.dim.clone() {
        TensorDim::Dim2(_, features) => {
            let x_data = x.as_2d();
            let mut x_train_data = Array2::<f32>::zeros((num_train, features));

            for i in 0..num_train {
                for j in 0..features {
                    x_train_data[[i, j]] = x_data[[i, j]];
                }
            }

            Tensor::new_2d(x_train_data)
        }
        _ => panic!("Expected 2D tensor for x"),
    };

    let y_train = match y.dim.clone() {
        TensorDim::Dim1(_) => {
            let y_data = y.as_1d();
            let mut y_train_data = Array1::<f32>::zeros(num_train);

            for i in 0..num_train {
                y_train_data[i] = y_data[i];
            }

            Tensor::new_1d(y_train_data)
        }
        _ => panic!("Expected 1D tensor for y"),
    };

    let x_test = match x.dim.clone() {
        TensorDim::Dim2(_, features) => {
            let x_data = x.as_2d();
            let mut x_test_data = Array2::<f32>::zeros((num_samples - num_train, features));

            for i in num_train..num_samples {
                for j in 0..features {
                    x_test_data[[i - num_train, j]] = x_data[[i, j]];
                }
            }

            Tensor::new_2d(x_test_data)
        }
        _ => panic!("Expected 2D tensor for x"),
    };

    let y_test = match y.dim.clone() {
        TensorDim::Dim1(_) => {
            let y_data = y.as_1d();
            let mut y_test_data = Array1::<f32>::zeros(num_samples - num_train);

            for i in num_train..num_samples {
                y_test_data[i - num_train] = y_data[i];
            }

            Tensor::new_1d(y_test_data)
        }
        _ => panic!("Expected 1D tensor for y"),
    };

    // Create batches
    let batch_size = 32;
    let (x_batches, y_batches) = create_batches(&x_train, &y_train, batch_size);

    // Create model
    let mut model = Sequential::new(Some("ClassificationModel".to_string()));
    model.add(Box::new(Linear::new(
        num_features,
        16,
        Some("hidden1".to_string()),
    )));
    model.add(Box::new(ReLU::new(Some("relu1".to_string()))));
    model.add(Box::new(Linear::new(16, 8, Some("hidden2".to_string()))));
    model.add(Box::new(ReLU::new(Some("relu2".to_string()))));
    model.add(Box::new(Linear::new(
        8,
        num_classes,
        Some("output".to_string()),
    )));
    model.add(Box::new(Softmax::new(Some("softmax".to_string()))));

    // Create loss function and optimizer
    let loss_fn = Box::new(CrossEntropyLoss::new());
    let optimizer = Box::new(SGD::new());

    // Create trainer
    let mut trainer = Trainer::new(model, loss_fn, optimizer, 0.01);

    // Training loop
    let num_epochs = 10;
    println!("Training for {} epochs...", num_epochs);

    for epoch in 0..num_epochs {
        let epoch_loss = trainer.train_epoch(&x_batches, &y_batches);

        // Evaluate on test set
        let (test_loss, test_predictions) = trainer.evaluate(&x_test, &y_test);
        let test_accuracy = calculate_accuracy(&test_predictions, &y_test);

        println!(
            "Epoch {}/{}: Train Loss: {:.4}, Test Loss: {:.4}, Test Accuracy: {:.2}%",
            epoch + 1,
            num_epochs,
            epoch_loss,
            test_loss,
            test_accuracy * 100.0
        );
    }

    println!("Classification training completed!");
}

// Example of building and training a regression model
fn regression_example() {
    println!("\nRegression Example");
    println!("=================");

    // Create dataset
    let num_samples = 1000;
    let num_features = 2;
    let (x, y) = create_regression_dataset(num_samples, num_features);

    // Split into train and test sets (80% train, 20% test)
    let num_train = (num_samples as f32 * 0.8) as usize;

    let x_train = match x.dim.clone() {
        TensorDim::Dim2(_, features) => {
            let x_data = x.as_2d();
            let mut x_train_data = Array2::<f32>::zeros((num_train, features));

            for i in 0..num_train {
                for j in 0..features {
                    x_train_data[[i, j]] = x_data[[i, j]];
                }
            }

            Tensor::new_2d(x_train_data)
        }
        _ => panic!("Expected 2D tensor for x"),
    };

    let y_train = match y.dim.clone() {
        TensorDim::Dim2(_, y_features) => {
            let y_data = y.as_2d();
            let mut y_train_data = Array2::<f32>::zeros((num_train, y_features));

            for i in 0..num_train {
                for j in 0..y_features {
                    y_train_data[[i, j]] = y_data[[i, j]];
                }
            }

            Tensor::new_2d(y_train_data)
        }
        _ => panic!("Expected 2D tensor for y"),
    };

    let x_test = match x.dim.clone() {
        TensorDim::Dim2(_, features) => {
            let x_data = x.as_2d();
            let mut x_test_data = Array2::<f32>::zeros((num_samples - num_train, features));

            for i in num_train..num_samples {
                for j in 0..features {
                    x_test_data[[i - num_train, j]] = x_data[[i, j]];
                }
            }

            Tensor::new_2d(x_test_data)
        }
        _ => panic!("Expected 2D tensor for x"),
    };

    let y_test = match y.dim.clone() {
        TensorDim::Dim2(_, y_features) => {
            let y_data = y.as_2d();
            let mut y_test_data = Array2::<f32>::zeros((num_samples - num_train, y_features));

            for i in num_train..num_samples {
                for j in 0..y_features {
                    y_test_data[[i - num_train, j]] = y_data[[i, j]];
                }
            }

            Tensor::new_2d(y_test_data)
        }
        _ => panic!("Expected 2D tensor for y"),
    };

    // Create batches
    let batch_size = 32;
    let (x_batches, y_batches) = create_batches(&x_train, &y_train, batch_size);

    // Create model
    let mut model = Sequential::new(Some("RegressionModel".to_string()));
    model.add(Box::new(Linear::new(
        num_features,
        32,
        Some("hidden1".to_string()),
    )));
    model.add(Box::new(ReLU::new(Some("relu1".to_string()))));
    model.add(Box::new(Linear::new(32, 16, Some("hidden2".to_string()))));
    model.add(Box::new(ReLU::new(Some("relu2".to_string()))));
    model.add(Box::new(Linear::new(16, 1, Some("output".to_string()))));

    // Create loss function and optimizer
    let loss_fn = Box::new(MSELoss::new());
    let optimizer = Box::new(SGD::new());

    // Create trainer
    let mut trainer = Trainer::new(model, loss_fn, optimizer, 0.01);

    // Training loop
    let num_epochs = 10;
    println!("Training for {} epochs...", num_epochs);

    for epoch in 0..num_epochs {
        let epoch_loss = trainer.train_epoch(&x_batches, &y_batches);

        // Evaluate on test set
        let (test_loss, _) = trainer.evaluate(&x_test, &y_test);

        println!(
            "Epoch {}/{}: Train Loss: {:.4}, Test Loss: {:.4}",
            epoch + 1,
            num_epochs,
            epoch_loss,
            test_loss
        );
    }

    println!("Regression training completed!");
}

fn main() {
    println!("Unified Neural Network Framework Example");
    println!("=======================================");

    // Run classification example
    classification_example();

    // Run regression example
    regression_example();
}
