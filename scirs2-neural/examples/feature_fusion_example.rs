use ndarray::{Array, IxDyn};
use scirs2_core::error::Result;
use scirs2_neural::{
    models::architectures::{FeatureFusion, FeatureFusionConfig, FusionMethod},
    prelude::*,
};

fn main() -> Result<()> {
    println!("Feature Fusion Example");
    println!("---------------------");

    // Create random input tensors for two modalities
    // Modality 1: Image features (batch_size=2, features=512)
    let image_shape = [2, 512];
    let mut image_features = Array::<f32, _>::zeros(image_shape).into_dyn();

    // Modality 2: Text features (batch_size=2, features=256)
    let text_shape = [2, 256];
    let mut text_features = Array::<f32, _>::zeros(text_shape).into_dyn();

    // Fill with random values
    for elem in image_features.iter_mut() {
        *elem = rand::random::<f32>();
    }

    for elem in text_features.iter_mut() {
        *elem = rand::random::<f32>();
    }

    // 1. Concatenation Fusion
    println!("\nEarly Fusion (Concatenation):");
    let fusion_concat = FeatureFusion::create_early_fusion(
        512,  // image feature dimension
        256,  // text feature dimension
        128,  // hidden dimension
        10,   // number of classes
        true, // include classification head
    )?;

    let concat_output =
        fusion_concat.forward_multi(&[image_features.clone(), text_features.clone()])?;
    println!("Output shape: {:?}", concat_output.shape());

    // 2. Attention Fusion
    println!("\nAttention Fusion:");
    let fusion_attn = FeatureFusion::create_attention_fusion(
        512,  // image feature dimension
        256,  // text feature dimension
        128,  // hidden dimension
        10,   // number of classes
        true, // include classification head
    )?;

    let attn_output =
        fusion_attn.forward_multi(&[image_features.clone(), text_features.clone()])?;
    println!("Output shape: {:?}", attn_output.shape());

    // 3. FiLM Fusion
    println!("\nFiLM Fusion (Text conditions Image):");
    let fusion_film = FeatureFusion::create_film_fusion(
        512,  // image feature dimension
        256,  // text feature dimension
        128,  // hidden dimension
        10,   // number of classes
        true, // include classification head
    )?;

    let film_output =
        fusion_film.forward_multi(&[image_features.clone(), text_features.clone()])?;
    println!("Output shape: {:?}", film_output.shape());

    // 4. Create a custom fusion model
    println!("\nCustom Bilinear Fusion:");
    let custom_config = FeatureFusionConfig {
        input_dims: vec![512, 256],
        hidden_dim: 128,
        fusion_method: FusionMethod::Bilinear,
        dropout_rate: 0.2,
        num_classes: 10,
        include_head: true,
    };

    let fusion_bilinear = FeatureFusion::<f32>::new(custom_config)?;
    let bilinear_output =
        fusion_bilinear.forward_multi(&[image_features.clone(), text_features.clone()])?;
    println!("Output shape: {:?}", bilinear_output.shape());

    // 5. Multi-modality fusion (with 3 modalities)
    println!("\nMulti-modal Fusion (3 modalities):");

    // Create a third modality (e.g., audio features)
    let audio_shape = [2, 128];
    let mut audio_features = Array::<f32, _>::zeros(audio_shape).into_dyn();

    for elem in audio_features.iter_mut() {
        *elem = rand::random::<f32>();
    }

    let multi_config = FeatureFusionConfig {
        input_dims: vec![512, 256, 128],
        hidden_dim: 128,
        fusion_method: FusionMethod::Sum, // Using element-wise sum for 3+ modalities
        dropout_rate: 0.1,
        num_classes: 10,
        include_head: true,
    };

    let multi_fusion = FeatureFusion::<f32>::new(multi_config)?;
    let multi_output = multi_fusion.forward_multi(&[
        image_features.clone(),
        text_features.clone(),
        audio_features.clone(),
    ])?;

    println!("Output shape: {:?}", multi_output.shape());

    // Compare predictions from different fusion methods
    println!("\nPrediction comparison across fusion methods:");

    println!(
        "  Concatenation fusion - Top class: {}",
        get_top_class(&concat_output)
    );
    println!(
        "  Attention fusion - Top class: {}",
        get_top_class(&attn_output)
    );
    println!("  FiLM fusion - Top class: {}", get_top_class(&film_output));
    println!(
        "  Bilinear fusion - Top class: {}",
        get_top_class(&bilinear_output)
    );
    println!(
        "  Multi-modal fusion - Top class: {}",
        get_top_class(&multi_output)
    );

    Ok(())
}

// Helper function to get the top predicted class
fn get_top_class<F: num_traits::Float + std::fmt::Debug>(predictions: &Array<F, IxDyn>) -> usize {
    let mut max_val = F::neg_infinity();
    let mut max_idx = 0;

    for (i, &val) in predictions.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i % predictions.shape()[1];
        }
    }

    max_idx
}
