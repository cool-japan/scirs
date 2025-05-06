use ndarray::{Array, Ix2, IxDyn};
use scirs2_core::error::Result;
use scirs2_neural::{
    models::architectures::{CLIPConfig, CLIPTextConfig, ViTConfig, VisionTransformer, CLIP},
    prelude::*,
};

fn main() -> Result<()> {
    println!("CLIP Example");
    println!("------------");

    // Create a random image input tensor (batch_size=2, channels=3, height=224, width=224)
    let image_shape = [2, 3, 224, 224];
    let mut image_input = Array::<f32, _>::zeros(image_shape).into_dyn();

    // Fill with random values between 0 and 1
    for elem in image_input.iter_mut() {
        *elem = rand::random::<f32>();
    }

    // Create a random text input tensor (batch_size=2, sequence_length=77)
    // In a real scenario, these would be token IDs from a tokenizer
    let text_shape = [2, 77];
    let mut text_input = Array::<f32, _>::zeros(text_shape).into_dyn();

    // Fill with random token IDs (between 0 and vocab_size-1)
    let vocab_size = 49408;
    for elem in text_input.iter_mut() {
        *elem = (rand::random::<f32>() * (vocab_size as f32)).floor();
    }

    // Create CLIP base model
    println!("\nCreating CLIP base model...");
    let clip_model = CLIP::clip_base(1000, false)?;

    // Demonstrate forward pass for contrastive learning
    println!("Running contrastive forward pass...");
    let (image_features, text_features, similarity) =
        clip_model.forward_contrastive(&image_input, &text_input)?;

    println!("Image features shape: {:?}", image_features.shape());
    println!("Text features shape: {:?}", text_features.shape());
    println!("Similarity matrix shape: {:?}", similarity.shape());

    // Demonstrate zero-shot classification
    println!("\nZero-shot classification example:");

    // Create a smaller batch for this example
    let single_image = image_input
        .slice_axis(ndarray::Axis(0), ndarray::Slice::from(0..1))
        .to_owned();

    // In real applications, these would be text embeddings for class names
    // For example, embeddings for "a photo of a dog", "a photo of a cat", etc.
    let num_classes = 5;
    let class_embeddings =
        Array::<f32, _>::zeros((num_classes, clip_model.config.projection_dim)).into_dyn();

    let predictions = clip_model.forward_classification(&single_image, &class_embeddings)?;
    println!(
        "Classification predictions shape: {:?}",
        predictions.shape()
    );

    // Print the probabilities (would be meaningful with real class embeddings)
    let probs = softmax(&predictions.into_dimensionality::<Ix2>()?)?;
    println!("Probabilities for first image:");
    for (i, &prob) in probs.iter().enumerate() {
        println!("  Class {}: {:.4}", i, prob);
    }

    // Demonstrate using CLIP with custom configuration
    println!("\nCreating CLIP with custom configuration...");

    let vision_config = ViTConfig {
        image_size: 224,
        patch_size: 16,
        in_channels: 3,
        hidden_size: 384,
        num_layers: 6,
        num_heads: 6,
        mlp_dim: 1536,
        dropout_rate: 0.1,
        attention_dropout_rate: 0.1,
        classifier: "token".to_string(),
        num_classes: 1000,
        include_top: false,
    };

    let text_config = CLIPTextConfig {
        vocab_size: 49408,
        hidden_size: 384,
        intermediate_size: 1536,
        num_layers: 6,
        num_heads: 6,
        max_position_embeddings: 77,
        dropout_rate: 0.1,
        layer_norm_eps: 1e-5,
    };

    let clip_config = CLIPConfig {
        text_config,
        vision_config,
        projection_dim: 256,
        include_head: true,
        num_classes: 10,
    };

    let custom_clip = CLIP::new(clip_config)?;
    println!("Custom CLIP model created successfully.");

    // Example feature extraction
    let features = custom_clip.forward(&image_input)?;
    println!("Feature extraction shape: {:?}", features.shape());

    Ok(())
}

// Simple softmax implementation for demonstration
fn softmax<F: num_traits::Float>(x: &Array<F, Ix2>) -> Result<Array<F, Ix2>> {
    let mut result = x.clone();

    for mut row in result.axis_iter_mut(ndarray::Axis(0)) {
        // Find max for numerical stability
        let max = row.fold(F::neg_infinity(), |a, &b| a.max(b));

        // Subtract max and compute exp
        for v in row.iter_mut() {
            *v = (*v - max).exp();
        }

        // Normalize
        let sum = row.sum();
        if sum > F::zero() {
            for v in row.iter_mut() {
                *v = *v / sum;
            }
        }
    }

    Ok(result)
}
