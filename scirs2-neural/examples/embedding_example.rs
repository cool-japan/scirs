use ndarray::{Array, IxDyn};
use scirs2_neural::layers::{
    Embedding, EmbeddingConfig, Layer, PatchEmbedding, PositionalEmbedding,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running embedding examples...");

    // Example 1: Basic Embedding
    println!("\n--- Basic Embedding Example ---");
    let config = EmbeddingConfig {
        num_embeddings: 10,
        embedding_dim: 5,
        padding_idx: Some(0),
        max_norm: None,
        norm_type: 2.0,
        scale_grad_by_freq: false,
        sparse: false,
    };

    let embedding = Embedding::<f32>::new(config)?;

    // Create input indices
    let indices = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1, 2, 0, 3, 0, 4])?;

    // Forward pass
    let output = embedding.forward(&indices.mapv(|x| x as f32))?;

    println!("Input indices shape: {:?}", indices.shape());
    println!("Output embeddings shape: {:?}", output.shape());
    println!(
        "First embedding vector: {:?}",
        output.slice(ndarray::s![0, 0, ..]).to_owned()
    );

    // Example 2: Positional Embedding
    println!("\n--- Positional Embedding Example ---");

    // Create fixed sinusoidal positional embeddings
    let pos_embedding = PositionalEmbedding::<f32>::new(10, 8, false)?;

    // Create dummy input (like token embeddings)
    let token_embeddings = Array::from_shape_fn(IxDyn(&[2, 5, 8]), |_| 1.0f32);

    // Add positional information
    let output = pos_embedding.forward(&token_embeddings)?;

    println!("Token embeddings shape: {:?}", token_embeddings.shape());
    println!(
        "Output with positional encoding shape: {:?}",
        output.shape()
    );
    println!(
        "First token before positional encoding: {:?}",
        token_embeddings.slice(ndarray::s![0, 0, ..]).to_owned()
    );
    println!(
        "First token after positional encoding: {:?}",
        output.slice(ndarray::s![0, 0, ..]).to_owned()
    );

    // Example 3: Patch Embedding (for Vision Transformers)
    println!("\n--- Patch Embedding Example ---");

    // Create patch embedding for a vision transformer
    let patch_embedding = PatchEmbedding::<f32>::new((32, 32), (8, 8), 3, 96, true)?;

    // Create random image input
    let image_input = Array::from_shape_fn(IxDyn(&[1, 3, 32, 32]), |_| rand::random::<f32>());

    // Extract patch embeddings
    let output = patch_embedding.forward(&image_input)?;

    println!("Input image shape: {:?}", image_input.shape());
    println!("Patch embeddings shape: {:?}", output.shape());
    println!("Number of patches: {}", patch_embedding.num_patches());
    println!("Embedding dimension: {}", patch_embedding.embedding_dim);

    // Print first patch embedding
    println!(
        "First patch embedding (first 5 values): {:?}",
        output.slice(ndarray::s![0, 0, ..5]).to_owned()
    );

    println!("\nAll embedding examples completed successfully!");

    Ok(())
}
