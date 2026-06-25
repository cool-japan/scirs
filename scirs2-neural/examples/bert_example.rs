//! BERT model example
//!
//! Demonstrates building a small BERT model and running a forward pass to get
//! both the sequence output and the pooled output. Dimensions are kept small
//! so the example runs quickly.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Layer;
use scirs2_neural::models::{BertConfig, BertModel};

fn main() -> Result<()> {
    println!("BERT Model Example");
    println!("==================");

    // Create a small BERT model for demonstration.
    println!("Creating a small BERT model...");
    let config = BertConfig::custom(
        1000, // vocab_size
        32,   // hidden_size
        2,    // num_hidden_layers
        4,    // num_attention_heads
    );
    let model = BertModel::<f32>::new(config)?;

    // Dummy input (batch_size=2, seq_len=8). Values represent token IDs.
    let input = Array::from_shape_fn(IxDyn(&[2, 8]), |_| {
        (scirs2_core::random::random::<f32>() * 100.0).floor()
    });
    println!("Input shape: {:?}", input.shape());

    // Sequence output (hidden states) via the Layer `forward` method.
    let sequence_output = model.forward(&input)?;
    println!("Sequence output shape: {:?}", sequence_output.shape());

    // Pooled output (for classification tasks).
    let pooled_output = model.get_pooled_output(&input)?;
    println!("Pooled output shape: {:?}", pooled_output.shape());

    // Demonstrate the BERT-Base convenience constructor. BERT-Base is large (12
    // transformer layers at hidden size 768), so we construct it and report its
    // configuration rather than running a full forward pass, keeping the example
    // fast. (The forward-pass demonstration above uses the small model.)
    println!("\nCreating a BERT-Base model...");
    let _bert_base = BertModel::<f32>::bert_base_uncased()?;
    let base_config = BertConfig::bert_base_uncased();
    println!("BERT-Base model created successfully.");
    println!("  - hidden size: {}", base_config.hidden_size);
    println!("  - hidden layers: {}", base_config.num_hidden_layers);
    println!("  - attention heads: {}", base_config.num_attention_heads);

    println!("\nBERT example completed successfully!");
    Ok(())
}
