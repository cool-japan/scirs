// OpenCL kernels for GPU-accelerated batch normalization
// Supports both training and inference modes

// Batch normalization forward pass (inference mode)
__kernel void batch_normalize(
    __global const float* input,       // Input tensor
    __global const float* mean,        // Running mean per channel
    __global const float* variance,    // Running variance per channel
    __global const float* gamma,       // Scale parameter per channel
    __global const float* beta,        // Shift parameter per channel
    __global float* output,            // Output tensor
    const unsigned int size,           // Total number of elements
    const unsigned int num_channels,   // Number of channels
    const float epsilon               // Small constant for numerical stability
) {
    const int idx = get_global_id(0);
    
    if (idx < size) {
        // Determine which channel this element belongs to
        // Assuming channel-last format (NHWC)
        const int channel = idx % num_channels;
        
        // Normalize: (x - mean) / sqrt(var + epsilon)
        const float normalized = (input[idx] - mean[channel]) / sqrt(variance[channel] + epsilon);
        
        // Scale and shift: gamma * normalized + beta
        output[idx] = gamma[channel] * normalized + beta[channel];
    }
}

// Batch normalization forward pass (training mode) - Step 1: Compute statistics
__kernel void batch_norm_compute_mean(
    __global const float* input,       // Input tensor (NCHW format)
    __global float* mean,              // Output: mean per channel
    const unsigned int batch_size,
    const unsigned int num_channels,
    const unsigned int spatial_size    // H * W for 2D, or total spatial dimensions
) {
    const int channel = get_global_id(0);
    
    if (channel < num_channels) {
        float sum = 0.0f;
        const int elements_per_channel = batch_size * spatial_size;
        
        // Sum all elements for this channel across batch and spatial dimensions
        for (int n = 0; n < batch_size; n++) {
            for (int s = 0; s < spatial_size; s++) {
                const int idx = n * num_channels * spatial_size + channel * spatial_size + s;
                sum += input[idx];
            }
        }
        
        mean[channel] = sum / (float)elements_per_channel;
    }
}

__kernel void batch_norm_compute_variance(
    __global const float* input,       // Input tensor (NCHW format)
    __global const float* mean,        // Mean per channel
    __global float* variance,          // Output: variance per channel
    const unsigned int batch_size,
    const unsigned int num_channels,
    const unsigned int spatial_size
) {
    const int channel = get_global_id(0);
    
    if (channel < num_channels) {
        float sum_sq_diff = 0.0f;
        const float channel_mean = mean[channel];
        const int elements_per_channel = batch_size * spatial_size;
        
        // Sum squared differences from mean
        for (int n = 0; n < batch_size; n++) {
            for (int s = 0; s < spatial_size; s++) {
                const int idx = n * num_channels * spatial_size + channel * spatial_size + s;
                const float diff = input[idx] - channel_mean;
                sum_sq_diff += diff * diff;
            }
        }
        
        variance[channel] = sum_sq_diff / (float)elements_per_channel;
    }
}

// Batch normalization forward pass (training mode) - Step 2: Normalize
__kernel void batch_norm_forward_training(
    __global const float* input,       // Input tensor
    __global const float* mean,        // Computed mean per channel
    __global const float* variance,    // Computed variance per channel
    __global const float* gamma,       // Scale parameter per channel
    __global const float* beta,        // Shift parameter per channel
    __global float* output,            // Normalized output
    __global float* normalized,        // Store normalized values for backward pass
    const unsigned int batch_size,
    const unsigned int num_channels,
    const unsigned int spatial_size,
    const float epsilon
) {
    const int idx = get_global_id(0);
    const int total_size = batch_size * num_channels * spatial_size;
    
    if (idx < total_size) {
        // Extract channel index (assuming NCHW format)
        const int remaining = idx % (num_channels * spatial_size);
        const int channel = remaining / spatial_size;
        
        // Normalize
        const float norm_val = (input[idx] - mean[channel]) / sqrt(variance[channel] + epsilon);
        normalized[idx] = norm_val;
        
        // Scale and shift
        output[idx] = gamma[channel] * norm_val + beta[channel];
    }
}

// Batch normalization backward pass - compute gradients
__kernel void batch_norm_backward_gamma_beta(
    __global const float* grad_output,    // Gradient from next layer
    __global const float* normalized,     // Normalized values from forward pass
    __global float* grad_gamma,           // Gradient w.r.t. gamma
    __global float* grad_beta,            // Gradient w.r.t. beta
    const unsigned int batch_size,
    const unsigned int num_channels,
    const unsigned int spatial_size
) {
    const int channel = get_global_id(0);
    
    if (channel < num_channels) {
        float sum_grad_output = 0.0f;
        float sum_grad_norm = 0.0f;
        
        // Sum gradients for this channel
        for (int n = 0; n < batch_size; n++) {
            for (int s = 0; s < spatial_size; s++) {
                const int idx = n * num_channels * spatial_size + channel * spatial_size + s;
                sum_grad_output += grad_output[idx];
                sum_grad_norm += grad_output[idx] * normalized[idx];
            }
        }
        
        grad_beta[channel] = sum_grad_output;
        grad_gamma[channel] = sum_grad_norm;
    }
}

__kernel void batch_norm_backward_input(
    __global const float* grad_output,    // Gradient from next layer
    __global const float* input,          // Original input
    __global const float* mean,           // Mean per channel
    __global const float* variance,       // Variance per channel
    __global const float* gamma,          // Scale parameter
    __global const float* grad_gamma,     // Gradient w.r.t. gamma
    __global const float* grad_beta,      // Gradient w.r.t. beta
    __global float* grad_input,           // Gradient w.r.t. input
    const unsigned int batch_size,
    const unsigned int num_channels,
    const unsigned int spatial_size,
    const float epsilon
) {
    const int idx = get_global_id(0);
    const int total_size = batch_size * num_channels * spatial_size;
    
    if (idx < total_size) {
        // Extract channel index
        const int remaining = idx % (num_channels * spatial_size);
        const int channel = remaining / spatial_size;
        
        const float N = (float)(batch_size * spatial_size);  // Number of elements per channel
        const float var_eps = variance[channel] + epsilon;
        const float inv_std = 1.0f / sqrt(var_eps);
        
        // Compute gradient w.r.t. input
        const float grad_norm = (grad_output[idx] * gamma[channel] - 
                                grad_beta[channel] / N - 
                                (input[idx] - mean[channel]) * grad_gamma[channel] / (N * var_eps)) * inv_std;
        
        grad_input[idx] = grad_norm;
    }
}

// Layer normalization (normalize across features instead of batch)
__kernel void layer_normalize(
    __global const float* input,       // Input tensor
    __global const float* gamma,       // Scale parameter
    __global const float* beta,        // Shift parameter
    __global float* output,            // Output tensor
    const unsigned int batch_size,
    const unsigned int feature_size,   // Number of features to normalize over
    const float epsilon
) {
    const int batch_idx = get_global_id(0);
    
    if (batch_idx < batch_size) {
        const int offset = batch_idx * feature_size;
        
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            sum += input[offset + i];
        }
        const float mean = sum / (float)feature_size;
        
        // Compute variance
        float sum_sq_diff = 0.0f;
        for (int i = 0; i < feature_size; i++) {
            const float diff = input[offset + i] - mean;
            sum_sq_diff += diff * diff;
        }
        const float variance = sum_sq_diff / (float)feature_size;
        
        // Normalize
        const float inv_std = 1.0f / sqrt(variance + epsilon);
        for (int i = 0; i < feature_size; i++) {
            const float normalized = (input[offset + i] - mean) * inv_std;
            output[offset + i] = gamma[i] * normalized + beta[i];
        }
    }
}