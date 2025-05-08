use ndarray::{Array, IxDyn};
use scirs2_neural::layers::Layer;
use scirs2_neural::models::{GPTConfig, GPTModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPT Model Example");

    // Create a small GPT model for demonstration
    println!("Creating a small GPT model...");

    let config = GPTConfig::custom(
        10000, // vocab_size
        128,   // hidden_size
        2,     // num_hidden_layers
        2,     // num_attention_heads
    );

    let model = GPTModel::<f32>::new(config)?;

    // Create dummy input (batch_size=2, seq_len=16)
    // Input tensor contains token IDs
    let input = Array::from_shape_fn(
        IxDyn(&[2, 16]),
        |_| rand::random::<f32>() * 100.0, // Random token IDs between 0 and 100
    );

    println!("Input shape: {:?}", input.shape());

    // Forward pass to get hidden states
    let hidden_states = model.forward(&input)?;

    println!("Hidden states shape: {:?}", hidden_states.shape());

    // Calculate logits for next-token prediction
    let logits = model.logits(&input)?;

    println!("Logits shape: {:?}", logits.shape());
    println!("Vocabulary size: {}", logits.shape()[2]);

    // Let's create a GPT-2 Small model
    println!("\nCreating a GPT-2 Small model...");

    let gpt2_small = GPTModel::<f32>::gpt2_small()?;

    // Create dummy input for a longer sequence
    let small_input = Array::from_shape_fn(
        IxDyn(&[1, 32]),
        |_| rand::random::<f32>() * 1000.0, // Random token IDs
    );

    println!("GPT-2 Small input shape: {:?}", small_input.shape());

    // Forward pass
    let small_hidden_states = gpt2_small.forward(&small_input)?;

    println!(
        "GPT-2 Small hidden states shape: {:?}",
        small_hidden_states.shape()
    );
    println!(
        "GPT-2 Small hidden dimension: {}",
        small_hidden_states.shape()[2]
    );

    // For text generation (logits for next token prediction)
    let small_logits = gpt2_small.logits(&small_input)?;
    println!("GPT-2 Small logits shape: {:?}", small_logits.shape());
    println!("GPT-2 Small vocabulary size: {}", small_logits.shape()[2]);

    println!("\nGPT example completed successfully!");

    Ok(())
}
