//! Transformer model example
//!
//! This example demonstrates how to create and use a transformer model
//! with the scirs2-neural crate. Dimensions are kept small so the example
//! runs quickly.

use scirs2_core::ndarray::Array3;
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::SeedableRng;
use scirs2_neural::error::Result;
use scirs2_neural::layers::Layer;
use scirs2_neural::transformer::{PositionalEncodingType, Transformer, TransformerConfig};

fn main() -> Result<()> {
    println!("Transformer Model Example");
    println!("=========================");

    // Seeded RNG for reproducibility.
    let mut rng = SmallRng::seed_from_u64(42);

    // Small transformer configuration for demonstration.
    let config = TransformerConfig {
        d_model: 64,                                           // Embedding dimension
        n_encoder_layers: 2,                                   // Number of encoder layers
        n_decoder_layers: 2,                                   // Number of decoder layers
        n_heads: 4,                                            // Number of attention heads
        d_ff: 128,                                             // Feed-forward hidden dimension
        max_seq_len: 50,                                       // Maximum sequence length
        dropout: 0.1,                                          // Dropout rate
        pos_encoding_type: PositionalEncodingType::Sinusoidal, // Positional encoding type
        epsilon: 1e-5,                                         // Layer-norm epsilon
    };
    println!("Creating transformer model with config:");
    println!("  - d_model: {}", config.d_model);
    println!("  - n_encoder_layers: {}", config.n_encoder_layers);
    println!("  - n_decoder_layers: {}", config.n_decoder_layers);
    println!("  - n_heads: {}", config.n_heads);
    println!("  - d_ff: {}", config.d_ff);
    println!("  - max_seq_len: {}", config.max_seq_len);

    let d_model = config.d_model;

    // Create the transformer model.
    let transformer = Transformer::<f64>::new(config, &mut rng)?;

    // Sample inputs. In a real application these would be token embeddings.
    let batch_size = 2;
    let src_seq_len = 10;
    let tgt_seq_len = 8;
    println!("\nSample dimensions:");
    println!("  - Batch size: {}", batch_size);
    println!("  - Source sequence length: {}", src_seq_len);
    println!("  - Target sequence length: {}", tgt_seq_len);

    let src_embeddings =
        Array3::<f64>::from_elem((batch_size, src_seq_len, d_model), 0.1).into_dyn();
    let tgt_embeddings =
        Array3::<f64>::from_elem((batch_size, tgt_seq_len, d_model), 0.1).into_dyn();

    // Encoder-only inference (useful for tasks like classification).
    // The full `Transformer` has no bare `forward`; the encoder stack is
    // exposed via `encoder()` and implements the `Layer` trait.
    println!("\nRunning encoder-only inference...");
    let encoder_output = transformer.encoder().forward(&src_embeddings)?;
    println!("Encoder output shape: {:?}", encoder_output.shape());

    // Full transformer training pass (teacher forcing).
    println!("\nRunning full transformer inference (training mode)...");
    let output_train = transformer.forward_train(&src_embeddings, &tgt_embeddings)?;
    println!("Training output shape: {:?}", output_train.shape());

    // Autoregressive inference (single step).
    println!("\nRunning autoregressive inference (one step)...");
    let first_token = Array3::<f64>::from_elem((batch_size, 1, d_model), 0.1).into_dyn();
    let output_inference = transformer.forward_inference(&src_embeddings, &first_token)?;
    println!("Inference output shape: {:?}", output_inference.shape());

    println!("\nTransformer example completed successfully!");
    Ok(())
}
