use ndarray::Array2;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::layers::{Dense, Dropout, LayerNorm};
use scirs2_neural::models::{Model, Sequential};
use scirs2_neural::serialization::{self, SerializationFormat};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Model Serialization Example");

    // Initialize random number generator
    let mut rng = SmallRng::seed_from_u64(42);

    // 1. Create a simple neural network model
    let mut model = Sequential::new();

    // Add layers
    let input_dim = 784; // MNIST image size: 28x28 = 784
    let hidden_dim_1 = 256;
    let hidden_dim_2 = 128;
    let output_dim = 10; // 10 classes for digits 0-9

    // Input layer to first hidden layer
    let dense1 = Dense::new(input_dim, hidden_dim_1, Some("relu"), &mut rng)?;
    model.add_layer(dense1);

    // Dropout for regularization
    let dropout1 = Dropout::new(0.2, &mut rng)?;
    model.add_layer(dropout1);

    // First hidden layer to second hidden layer
    let dense2 = Dense::new(hidden_dim_1, hidden_dim_2, Some("relu"), &mut rng)?;
    model.add_layer(dense2);

    // Layer normalization
    let layer_norm = LayerNorm::new(hidden_dim_2, 1e-5, &mut rng)?;
    model.add_layer(layer_norm);

    // Second hidden layer to output layer
    let dense3 = Dense::new(hidden_dim_2, output_dim, Some("softmax"), &mut rng)?;
    model.add_layer(dense3);

    println!(
        "Created a neural network with {} layers",
        model.num_layers()
    );

    // 2. Test the model with some dummy input
    let batch_size = 2;
    let input = Array2::<f32>::from_elem((batch_size, input_dim), 0.1);
    let output = model.forward(&input.clone().into_dyn())?;

    println!("Model output shape: {:?}", output.shape());
    println!("First few output values:");
    for i in 0..batch_size {
        print!("Sample {}: [ ", i);
        for j in 0..5 {
            // Print first 5 values
            print!("{:.6} ", output[[i, j]]);
        }
        println!("... ]");
    }

    // 3. Save the model to a file
    let model_path = Path::new("mnist_model.json");
    serialization::save_model(&model, model_path, SerializationFormat::JSON)?;

    println!("\nModel saved to {}", model_path.display());

    // 4. Load the model from the file
    let loaded_model = serialization::load_model::<f32, _>(model_path, SerializationFormat::JSON)?;

    println!(
        "Model loaded successfully with {} layers",
        loaded_model.num_layers()
    );

    // 5. Test the loaded model with the same input
    let loaded_output = loaded_model.forward(&input.into_dyn())?;

    println!("\nLoaded model output shape: {:?}", loaded_output.shape());
    println!("First few output values:");
    for i in 0..batch_size {
        print!("Sample {}: [ ", i);
        for j in 0..5 {
            // Print first 5 values
            print!("{:.6} ", loaded_output[[i, j]]);
        }
        println!("... ]");
    }

    // 6. Compare original and loaded model outputs
    let mut max_diff = 0.0;
    for i in 0..batch_size {
        for j in 0..output_dim {
            let diff = (output[[i, j]] - loaded_output[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    println!(
        "\nMaximum difference between original and loaded model outputs: {:.6}",
        max_diff
    );

    if max_diff < 1e-6 {
        println!("Models are identical! Serialization and deserialization worked correctly.");
    } else {
        println!("Warning: There are differences between the original and loaded models.");
        println!(
            "This might be due to numerical precision issues or a problem with serialization."
        );
    }

    Ok(())
}
