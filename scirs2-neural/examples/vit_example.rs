//! Vision Transformer (ViT) example
//!
//! Demonstrates building a Vision Transformer and running a forward pass on a
//! small dummy image batch. Dimensions are kept intentionally small so the
//! example runs quickly.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Layer;
use scirs2_neural::models::{ViTConfig, VisionTransformer};

fn main() -> Result<()> {
    println!("Vision Transformer (ViT) Example");
    println!("================================");

    // Create a custom ViT configuration for small (CIFAR-like) images so the
    // forward pass stays lightweight.
    let config = ViTConfig {
        image_size: (32, 32), // CIFAR-10 image size
        patch_size: (8, 8),   // 8x8 patches -> 16 patches
        in_channels: 3,       // RGB images
        num_classes: 10,      // CIFAR-10 classes
        embed_dim: 32,        // Small embedding dimension
        num_layers: 2,        // Few transformer layers
        num_heads: 4,         // Few attention heads
        mlp_dim: 64,          // Small MLP dimension
        dropout_rate: 0.1,
        attention_dropout_rate: 0.1,
    };

    println!(
        "Creating custom ViT model: image {:?}, patch {:?}, {} channels, {} classes",
        config.image_size, config.patch_size, config.in_channels, config.num_classes
    );

    let model = VisionTransformer::<f32>::new(config)?;

    // Dummy input (batch_size=1, channels=3, height=32, width=32).
    let input = Array::from_shape_fn(IxDyn(&[1, 3, 32, 32]), |_| {
        scirs2_core::random::random::<f32>()
    });
    println!("Input shape: {:?}", input.shape());

    let output = model.forward(&input)?;
    println!("Output shape: {:?}", output.shape());
    println!("Output contains logits for {} classes", output.shape()[1]);

    // Also demonstrate the `vit_base` convenience constructor. ViT-Base is large
    // (12 transformer layers at embedding dimension 768), so we construct it and
    // report its configuration rather than running a full forward pass, keeping
    // the example fast. (The forward-pass demonstration above uses the small
    // custom model.)
    println!("\nCreating a ViT-Base model...");
    let base_model = VisionTransformer::<f32>::vit_base((224, 224), (16, 16), 3, 1000)?;
    let base_config = base_model.config();
    println!("ViT-Base model created successfully.");
    println!("  - embedding dimension: {}", base_config.embed_dim);
    println!("  - transformer layers: {}", base_config.num_layers);
    println!("  - attention heads: {}", base_config.num_heads);

    println!("\nVision Transformer example completed successfully!");
    Ok(())
}
