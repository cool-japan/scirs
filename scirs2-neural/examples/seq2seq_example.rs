use ndarray::Array;
use rand::Rng;
use scirs2_neural::error::Result;
use scirs2_neural::{
    models::architectures::{RNNCellType, Seq2Seq, Seq2SeqConfig},
    prelude::*,
};

fn main() -> Result<()> {
    println!("Sequence-to-Sequence (Seq2Seq) Model Example");
    println!("--------------------------------------------");

    // Define vocabulary sizes
    let src_vocab_size = 10000; // Source language vocabulary size
    let tgt_vocab_size = 8000; // Target language vocabulary size

    // Create random input sequences (batch_size=2, sequence_length=10)
    let input_shape = [2, 10];
    let mut input_seq = Array::<f32, _>::zeros(input_shape).into_dyn();

    // Fill with random token IDs (between 0 and src_vocab_size-1)
    let mut rng = rand::rng();
    for elem in input_seq.iter_mut() {
        *elem = (rng.random_range(0.0..1.0) * (src_vocab_size as f32 - 1.0)).floor();
    }

    // Create random target sequences for teacher forcing (batch_size=2, sequence_length=8)
    let target_shape = [2, 8];
    let mut target_seq = Array::<f32, _>::zeros(target_shape).into_dyn();

    // Fill with random token IDs (between 0 and tgt_vocab_size-1)
    for elem in target_seq.iter_mut() {
        *elem = (rng.random_range(0.0..1.0) * (tgt_vocab_size as f32 - 1.0)).floor();
    }

    // 1. Create a basic translation model
    println!("\nCreating Basic Translation Model...");
    let mut translation_model = Seq2Seq::create_translation_model(
        src_vocab_size,
        tgt_vocab_size,
        256, // Hidden dimension
    )?;

    // Run forward pass with teacher forcing
    println!("Running forward pass with teacher forcing...");
    let train_output = translation_model.forward_train(&input_seq, &target_seq)?;
    println!("Training output shape: {:?}", train_output.shape());

    // Generate sequences
    println!("\nGenerating sequences...");
    let generated = translation_model.generate(
        &input_seq,
        Some(15), // Maximum length
        1,        // Start token ID (usually 1 for <START>)
        Some(2),  // End token ID (usually 2 for <END>)
    )?;
    println!("Generated sequence shape: {:?}", generated.shape());

    // Print generated sequences (token IDs)
    println!("Generated sequences (token IDs):");
    for b in 0..generated.shape()[0] {
        print!("  Sequence {}: ", b);
        for t in 0..generated.shape()[1] {
            if generated[[b, t]] > 0.0 {
                print!("{} ", generated[[b, t]]);
            }
        }
        println!();
    }

    // 2. Create a custom Seq2Seq model with different configuration
    println!("\nCreating Custom Seq2Seq Model...");
    let custom_config = Seq2SeqConfig {
        input_vocab_size: src_vocab_size,
        output_vocab_size: tgt_vocab_size,
        embedding_dim: 128,
        hidden_dim: 256,
        num_layers: 2,
        encoder_cell_type: RNNCellType::GRU,
        decoder_cell_type: RNNCellType::LSTM, // Mixing cell types
        bidirectional_encoder: true,
        use_attention: true,
        dropout_rate: 0.2,
        max_seq_len: 50,
    };

    let custom_model = Seq2Seq::<f32>::new(custom_config)?;
    println!("Custom model created successfully.");

    // 3. Creating a small and fast model for quick experimentation
    println!("\nCreating Small Seq2Seq Model...");
    let small_model = Seq2Seq::create_small_model(src_vocab_size, tgt_vocab_size)?;

    let small_generated = small_model.generate(&input_seq, Some(10), 1, Some(2))?;
    println!(
        "Small model generated sequence shape: {:?}",
        small_generated.shape()
    );

    // 4. Demonstrate switching between training and inference modes
    println!("\nDemonstrating Training/Inference Mode Switching:");

    // Set to training mode
    translation_model.set_training(true);
    println!("Is in training mode: {}", translation_model.is_training());

    // Set to inference mode
    translation_model.set_training(false);
    println!(
        "Is in training mode after switching: {}",
        translation_model.is_training()
    );

    // 5. Example of model parameter count
    println!("\nModel Parameter Counts:");
    println!(
        "Translation model parameters: {}",
        translation_model.params().len()
    );
    println!("Custom model parameters: {}", custom_model.params().len());
    println!("Small model parameters: {}", small_model.params().len());

    println!("\nSeq2Seq Example Completed Successfully!");

    Ok(())
}
