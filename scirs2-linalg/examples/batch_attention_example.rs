use scirs2_core::ndarray::{Array2, Array3};
use scirs2_linalg::attention::{AttentionConfig, AttentionMask};
use scirs2_linalg::batch::attention::{
    batch_flash_attention, batch_multi_head_attention, batch_multi_query_attention,
};

#[allow(dead_code)]
fn main() {
    println!("Batch Attention Mechanism Examples");
    println!("==================================\n");

    // Common setup
    let batchsize = 4;
    let seq_len = 8;
    let d_model = 16;
    let scale = 1.0 / (d_model as f32).sqrt();

    // Example 1: Batch Multi-Query Attention
    // --------------------------------------
    // This is useful for decoder-only architectures where each sequence has its own
    // query matrix but shares the same key and value matrices
    println!("Example 1: Batch Multi-Query Attention");

    // Create query batch (each batch element has its own query matrix)
    let batch_query = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);

    // Create shared key and value matrices
    let key = Array2::<f32>::from_elem((seq_len, d_model), 0.1);
    let value = Array2::<f32>::from_elem((seq_len, d_model), 0.1);

    // Compute batch multi-query attention
    let output =
        batch_multi_query_attention(&batch_query.view(), &key.view(), &value.view(), None, scale)
            .unwrap();

    println!("Input batch query shape: {:?}", batch_query.shape());
    println!("Shared key shape: {:?}", key.shape());
    println!("Shared value shape: {:?}", value.shape());
    println!("Output shape: {:?}", output.shape());
    println!("First value: {:.6}", output[[0, 0, 0]]);
    println!();

    // Example 2: Batch Multi-Head Attention with Causal Masking
    // --------------------------------------------------------
    // This is the standard transformer attention processed in parallel across batches
    println!("Example 2: Batch Multi-Head Attention with Causal Masking");

    // Create query, key, value batches
    let batch_query = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);
    let batch_key = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);
    let batch_value = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);

    // Setup for multi-head attention
    let num_heads = 4;
    let head_dim = d_model / num_heads;

    // Create projection weights
    let wq = Array2::<f32>::from_elem((d_model, d_model), 0.1);
    let wk = Array2::<f32>::from_elem((d_model, d_model), 0.1);
    let wv = Array2::<f32>::from_elem((d_model, d_model), 0.1);
    let wo = Array2::<f32>::from_elem((d_model, d_model), 0.1);

    // Create causal mask for autoregressive models
    let mask = AttentionMask::Causal;

    // Configure attention
    let config = AttentionConfig {
        num_heads,
        head_dim,
        dropout_prob: 0.0,
        causal: true,
        scale: Some(1.0 / (head_dim as f32).sqrt()),
    };

    // Compute batch multi-head attention with causal masking
    let output = batch_multi_head_attention(
        &batch_query.view(),
        &batch_key.view(),
        &batch_value.view(),
        &wq.view(),
        &wk.view(),
        &wv.view(),
        &wo.view(),
        Some(&mask),
        &config,
    )
    .unwrap();

    println!(
        "Input shapes - batchsize: {}, seq_len: {}, d_model: {}",
        batchsize, seq_len, d_model
    );
    println!(
        "Multi-head config - heads: {}, head_dim: {}",
        num_heads, head_dim
    );
    println!("Output shape: {:?}", output.shape());
    println!("First value: {:.6}", output[[0, 0, 0]]);
    println!();

    // Example 3: Batch Flash Attention
    // -------------------------------
    // Memory-efficient implementation that avoids materializing the full attention matrix
    println!("Example 3: Batch Flash Attention");

    // Create query, key, value batches
    let batch_query = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);
    let batch_key = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);
    let batch_value = Array3::<f32>::from_elem((batchsize, seq_len, d_model), 0.1);

    // Block size for memory efficiency (smaller blocks use less memory but might be slower)
    let blocksize = 4;

    // Compute batch flash attention
    let output = batch_flash_attention(
        &batch_query.view(),
        &batch_key.view(),
        &batch_value.view(),
        None,
        scale,
        blocksize,
    )
    .unwrap();

    println!(
        "Input shapes - batchsize: {}, seq_len: {}, d_model: {}",
        batchsize, seq_len, d_model
    );
    println!("Block size: {}", blocksize);
    println!("Output shape: {:?}", output.shape());
    println!("First value: {:.6}", output[[0, 0, 0]]);
    println!();

    // Performance comparison and use cases
    println!("Performance Comparison and Use Cases");
    println!("-----------------------------------");
    println!("1. Batch Multi-Query Attention:");
    println!("   - Best for: Decoder-only architectures (e.g., GPT models)");
    println!("   - Memory efficiency: Good (shared key/value matrices)");
    println!("   - Use when: Different sequences attend to the same context");
    println!();
    println!("2. Batch Multi-Head Attention:");
    println!("   - Best for: Standard transformer architectures");
    println!("   - Flexibility: Highest (supports all attention patterns)");
    println!("   - Use when: Full transformer functionality is needed");
    println!();
    println!("3. Batch Flash Attention:");
    println!("   - Best for: Very long sequences");
    println!("   - Memory efficiency: Excellent (O(N) memory usage)");
    println!("   - Use when: Memory is constrained or sequences are long");
    println!();
    println!("All batch operations support parallel processing across multiple sequences,");
    println!("making them ideal for high-throughput machine learning workloads.");
}
