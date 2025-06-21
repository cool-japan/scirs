// OpenCL kernels for GPU-accelerated activation functions
// Optimized for neural network forward and backward passes

// ReLU activation function
__kernel void relu_forward(
    __global const float* input,
    __global float* output,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        output[idx] = fmax(0.0f, input[idx]);
    }
}

// ReLU derivative for backpropagation
__kernel void relu_backward(
    __global const float* input,
    __global const float* grad_output,
    __global float* grad_input,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// Sigmoid activation function
__kernel void sigmoid_forward(
    __global const float* input,
    __global float* output,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + exp(-input[idx]));
    }
}

// Sigmoid derivative
__kernel void sigmoid_backward(
    __global const float* output,
    __global const float* grad_output,
    __global float* grad_input,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        const float sigmoid_out = output[idx];
        grad_input[idx] = grad_output[idx] * sigmoid_out * (1.0f - sigmoid_out);
    }
}

// Tanh activation function
__kernel void tanh_forward(
    __global const float* input,
    __global float* output,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        output[idx] = tanh(input[idx]);
    }
}

// Tanh derivative
__kernel void tanh_backward(
    __global const float* output,
    __global const float* grad_output,
    __global float* grad_input,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        const float tanh_out = output[idx];
        grad_input[idx] = grad_output[idx] * (1.0f - tanh_out * tanh_out);
    }
}

// Leaky ReLU activation function
__kernel void leaky_relu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        const float x = input[idx];
        output[idx] = (x > 0.0f) ? x : alpha * x;
    }
}

// Leaky ReLU derivative
__kernel void leaky_relu_backward(
    __global const float* input,
    __global const float* grad_output,
    __global float* grad_input,
    const float alpha,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : alpha * grad_output[idx];
    }
}

// ELU (Exponential Linear Unit) activation function
__kernel void elu_forward(
    __global const float* input,
    __global float* output,
    const float alpha,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        const float x = input[idx];
        output[idx] = (x > 0.0f) ? x : alpha * (exp(x) - 1.0f);
    }
}

// ELU derivative
__kernel void elu_backward(
    __global const float* input,
    __global const float* output,
    __global const float* grad_output,
    __global float* grad_input,
    const float alpha,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        const float x = input[idx];
        if (x > 0.0f) {
            grad_input[idx] = grad_output[idx];
        } else {
            grad_input[idx] = grad_output[idx] * (output[idx] + alpha);
        }
    }
}

// Softmax activation function (requires two passes)
__kernel void softmax_forward_pass1(
    __global const float* input,
    __global float* max_vals,
    const unsigned int batch_size,
    const unsigned int num_classes
) {
    const int batch_idx = get_global_id(0);
    
    if (batch_idx < batch_size) {
        const int offset = batch_idx * num_classes;
        float max_val = input[offset];
        
        for (int i = 1; i < num_classes; i++) {
            max_val = fmax(max_val, input[offset + i]);
        }
        
        max_vals[batch_idx] = max_val;
    }
}

__kernel void softmax_forward_pass2(
    __global const float* input,
    __global const float* max_vals,
    __global float* output,
    const unsigned int batch_size,
    const unsigned int num_classes
) {
    const int batch_idx = get_global_id(0);
    const int class_idx = get_global_id(1);
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        const int offset = batch_idx * num_classes;
        const float max_val = max_vals[batch_idx];
        
        // Compute exp(x - max) for numerical stability
        const float exp_val = exp(input[offset + class_idx] - max_val);
        
        // Compute sum of all exp values for this batch
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum += exp(input[offset + i] - max_val);
        }
        
        output[offset + class_idx] = exp_val / sum;
    }
}

// GELU (Gaussian Error Linear Unit) activation function
__kernel void gelu_forward(
    __global const float* input,
    __global float* output,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        const float x = input[idx];
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        const float tanh_arg = sqrt_2_over_pi * (x + coeff * x * x * x);
        output[idx] = 0.5f * x * (1.0f + tanh(tanh_arg));
    }
}

// Swish activation function (x * sigmoid(x))
__kernel void swish_forward(
    __global const float* input,
    __global float* output,
    const unsigned int size
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        const float x = input[idx];
        const float sigmoid_x = 1.0f / (1.0f + exp(-x));
        output[idx] = x * sigmoid_x;
    }
}