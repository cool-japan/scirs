//! Transformer model example
//!
//! This example demonstrates how to create and use a transformer model
//! with the scirs2-neural crate.

use ndarray::Array3;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::layers::Layer;
use scirs2_neural::transformer::{Transformer, TransformerConfig};
use scirs2_neural::utils::PositionalEncodingType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Transformer Model Example");
    println!("========================");

    // Create a seeded RNG for reproducibility
    let mut rng = SmallRng::seed_from_u64(42);

    // Create a small transformer configuration for demonstration
    let config = TransformerConfig {
        d_model: 64,                                           // Embedding dimension
        n_encoder_layers: 2,                                   // Number of encoder layers
        n_decoder_layers: 2,                                   // Number of decoder layers
        n_heads: 4,                                            // Number of attention heads
        d_ff: 128,       // Feed-forward network hidden dimension
        max_seq_len: 50, // Maximum sequence length
        dropout: 0.1,    // Dropout rate
        pos_encoding_type: PositionalEncodingType::Sinusoidal, // Positional encoding type
        epsilon: 1e-5,   // Small constant for layer normalization
    };

    println!("Creating transformer model with config:");
    println!("  - d_model: {}", config.d_model);
    println!("  - n_encoder_layers: {}", config.n_encoder_layers);
    println!("  - n_decoder_layers: {}", config.n_decoder_layers);
    println!("  - n_heads: {}", config.n_heads);
    println!("  - d_ff: {}", config.d_ff);
    println!("  - max_seq_len: {}", config.max_seq_len);

    // Create the transformer model
    let transformer = Transformer::<f64>::new(config, &mut rng)?;

    // Create sample inputs
    // In a real application, these would be token embeddings
    let batch_size = 2;
    let src_seq_len = 10;
    let tgt_seq_len = 8;
    let d_model = 64;

    println!("\nSample dimensions:");
    println!("  - Batch size: {}", batch_size);
    println!("  - Source sequence length: {}", src_seq_len);
    println!("  - Target sequence length: {}", tgt_seq_len);

    // Create source and target sequence embeddings
    // In practice, these would come from embedding layers or tokenizers
    let src_embeddings = Array3::<f64>::from_elem((batch_size, src_seq_len, d_model), 0.1);
    let tgt_embeddings = Array3::<f64>::from_elem((batch_size, tgt_seq_len, d_model), 0.1);

    // Convert to dyn format once and reuse
    let src_embeddings_dyn = src_embeddings.clone().into_dyn();
    let tgt_embeddings_dyn = tgt_embeddings.clone().into_dyn();

    println!("\nRunning encoder-only inference...");
    // Run encoder-only inference (useful for tasks like classification)
    let encoder_output = transformer.forward(&src_embeddings_dyn)?;
    println!("Encoder output shape: {:?}", encoder_output.shape());

    println!("\nRunning full transformer inference (training mode)...");
    // Run full transformer training (teacher forcing)
    let output_train = transformer.forward_train(&src_embeddings_dyn, &tgt_embeddings_dyn)?;
    println!("Training output shape: {:?}", output_train.shape());

    println!("\nRunning autoregressive inference (one step)...");
    // Simulate autoregressive generation (one step)
    // In practice, we would use a loop to generate tokens one by one
    let first_token = Array3::<f64>::from_elem((batch_size, 1, d_model), 0.1);
    let first_token_dyn = first_token.clone().into_dyn();
    let output_inference = transformer.forward_inference(&src_embeddings_dyn, &first_token_dyn)?;
    println!("Inference output shape: {:?}", output_inference.shape());

    println!("\nExample completed successfully");
    Ok(())
}
