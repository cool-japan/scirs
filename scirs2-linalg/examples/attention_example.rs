use scirs2_core::ndarray::{Array2, Array3};
use scirs2_linalg::attention::{
    causal_attention, flash_attention, linear_attention, multi_head_attention, rotary_embedding,
    scaled_dot_product_attention, sparse_attention, AttentionConfig,
};

#[allow(dead_code)]
fn main() {
    println!("Attention Mechanism Examples");
    println!("============================\n");

    // Common setup
    let batchsize = 2;
    let seq_len = 6;
    let d_model = 8;
    let scale = 1.0 / (d_model as f32).sqrt();

    // Create query, key, value tensors (in a real scenario, these would contain meaningful data)
    let query = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);
    let key = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);
    let value = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);

    // Example 1: Basic Scaled Dot-Product Attention
    println!("Example 1: Basic Scaled Dot-Product Attention");
    let output =
        scaled_dot_product_attention(&query.view(), &key.view(), &value.view(), None, scale)
            .unwrap();
    println!("Output shape: {:?}", output.shape());
    println!("First value: {:.6}", output[[0, 0, 0]]);
    println!();

    // Example 2: Causal Attention (for autoregressive models)
    println!("Example 2: Causal Attention");
    let causal_output = causal_attention(&query.view(), &key.view(), &value.view(), scale).unwrap();
    println!("Output shape: {:?}", causal_output.shape());
    println!("First value: {:.6}", causal_output[[0, 0, 0]]);
    println!();

    // Example 3: Flash Attention (memory-efficient)
    println!("Example 3: Flash Attention (memory-efficient)");
    let blocksize = 2;
    let flash_output = flash_attention(
        &query.view(),
        &key.view(),
        &value.view(),
        None,
        scale,
        blocksize,
    )
    .unwrap();
    println!("Output shape: {:?}", flash_output.shape());
    println!("First value: {:.6}", flash_output[[0, 0, 0]]);
    println!();

    // Example 4: Sparse Attention
    println!("Example 4: Sparse Attention");
    // Create a sparse pattern - each token attends to itself and adjacent positions
    let mut pattern = Array2::<bool>::from_elem((seq_len, seq_len), false);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if (i as isize - j as isize).abs() <= 1 {
                pattern[[i, j]] = true;
            }
        }
    }
    let sparse_output = sparse_attention(
        &query.view(),
        &key.view(),
        &value.view(),
        &pattern.view(),
        scale,
    )
    .unwrap();
    println!("Output shape: {:?}", sparse_output.shape());
    println!("First value: {:.6}", sparse_output[[0, 0, 0]]);
    println!();

    // Example 5: Linear Attention (O(n) complexity)
    println!("Example 5: Linear Attention (O(n) complexity)");
    let linear_output = linear_attention(&query.view(), &key.view(), &value.view(), scale).unwrap();
    println!("Output shape: {:?}", linear_output.shape());
    println!("First value: {:.6}", linear_output[[0, 0, 0]]);
    println!();

    // Example 6: Multi-Head Attention
    println!("Example 6: Multi-Head Attention");
    // Setup for multi-head attention
    let num_heads = 2;
    let head_dim = d_model / num_heads;

    // Create projection weights (in a real scenario, these would be learned parameters)
    let wq = Array2::<f32>::from_elem((d_model, d_model), 0.1);
    let wk = Array2::<f32>::from_elem((d_model, d_model), 0.1);
    let wv = Array2::<f32>::from_elem((d_model, d_model), 0.1);
    let wo = Array2::<f32>::from_elem((d_model, d_model), 0.1);

    // Configure attention
    let config = AttentionConfig {
        num_heads,
        head_dim,
        dropout_prob: 0.0,
        causal: false,
        scale: Some(1.0 / (head_dim as f32).sqrt()),
    };

    let mha_output = multi_head_attention(
        &query.view(),
        &key.view(),
        &value.view(),
        &wq.view(),
        &wk.view(),
        &wv.view(),
        &wo.view(),
        None,
        &config,
    )
    .unwrap();
    println!("Output shape: {:?}", mha_output.shape());
    println!("First value: {:.6}", mha_output[[0, 0, 0]]);
    println!();

    // Example 7: Rotary Position Embeddings (RoPE)
    println!("Example 7: Rotary Position Embeddings (RoPE)");
    let freq_base = 10000.0_f32;
    let rope_output = rotary_embedding(&query.view(), freq_base).unwrap();
    println!("Output shape: {:?}", rope_output.shape());
    println!("First value: {:.6}", rope_output[[0, 0, 0]]);

    // Performance comparison
    // In a real-world scenario, we'd measure performance more carefully
    println!("\nPerformance Notes:");
    println!("- Flash Attention: Most memory-efficient for long sequences");
    println!("- Linear Attention: Fastest for very long sequences (O(n) vs O(n²))");
    println!("- Sparse Attention: Good compromise for structured sparsity patterns");
    println!("- Multi-Head Attention: Standard approach for most transformer models");
    println!("- Causal Attention: Required for autoregressive generation");
}
