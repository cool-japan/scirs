use scirs2_core::error::Result;
use scirs2_neural::{
    config::{ConfigBuilder, ConfigFormat, ConfigSerializer, ModelConfig},
    models::architectures::{BertConfig, RNNCellType, ResNetConfig, Seq2SeqConfig, ViTConfig},
    prelude::*,
};

use std::path::Path;

fn main() -> Result<()> {
    println!("Model Configuration System Example");
    println!("---------------------------------");

    // 1. Create configurations using the builder
    println!("\nCreating model configurations with ConfigBuilder:");

    // Create a ResNet-50 configuration
    let resnet_config = ConfigBuilder::resnet(50, 1000, 3);
    println!("Created ResNet configuration");

    // Create a ViT-Base configuration
    let vit_config = ConfigBuilder::vit(224, 16, 1000);
    println!("Created Vision Transformer configuration");

    // Create a BERT-Base configuration
    let bert_config = ConfigBuilder::bert(30522, 768, 12);
    println!("Created BERT configuration");

    // Create a GPT configuration
    let gpt_config = ConfigBuilder::gpt(50257, 768, 12);
    println!("Created GPT configuration");

    // Create a Seq2Seq configuration
    let seq2seq_config = ConfigBuilder::seq2seq(10000, 8000, 512);
    println!("Created Seq2Seq configuration");

    // 2. Serialize configurations to JSON and YAML
    println!("\nSerializing configurations to JSON and YAML:");

    // Serialize ResNet configuration to JSON
    let resnet_json = ConfigSerializer::to_json(&resnet_config, true)?;
    println!("\nResNet JSON Configuration:");
    println!("{}", resnet_json);

    // Serialize ViT configuration to YAML
    let vit_yaml = ConfigSerializer::to_yaml(&vit_config)?;
    println!("\nViT YAML Configuration:");
    println!("{}", vit_yaml);

    // 3. Deserialize configurations from JSON and YAML
    println!("\nDeserializing configurations from JSON and YAML:");

    // Deserialize ResNet configuration from JSON
    let deserialized_resnet: ModelConfig = ConfigSerializer::from_json(&resnet_json)?;
    println!("Deserialized ResNet configuration successfully");

    // Deserialize ViT configuration from YAML
    let deserialized_vit: ModelConfig = ConfigSerializer::from_yaml(&vit_yaml)?;
    println!("Deserialized ViT configuration successfully");

    // 4. Save configurations to files
    println!("\nSaving configurations to files:");

    let config_dir = Path::new("./configs");
    std::fs::create_dir_all(config_dir).unwrap_or_default();

    // Save ResNet configuration to JSON file
    let resnet_path = config_dir.join("resnet50.json");
    resnet_config.to_file(&resnet_path, Some(ConfigFormat::JSON))?;
    println!("Saved ResNet configuration to {}", resnet_path.display());

    // Save ViT configuration to YAML file
    let vit_path = config_dir.join("vit_base.yaml");
    vit_config.to_file(&vit_path, Some(ConfigFormat::YAML))?;
    println!("Saved ViT configuration to {}", vit_path.display());

    // 5. Validate configurations
    println!("\nValidating configurations:");

    // Create an invalid ResNet configuration
    let invalid_resnet = ModelConfig::ResNet(ResNetConfig {
        num_layers: 42, // Invalid number of layers
        in_channels: 3,
        num_classes: 1000,
        zero_init_residual: false,
    });

    // Validate the configuration
    match invalid_resnet.validate() {
        Ok(_) => println!("ResNet configuration is valid"),
        Err(e) => println!("ResNet configuration validation failed: {}", e),
    }

    // Create a valid ViT configuration
    let valid_vit = ModelConfig::ViT(ViTConfig {
        image_size: 224,
        patch_size: 16,
        in_channels: 3,
        hidden_size: 768,
        num_layers: 12,
        num_heads: 12,
        mlp_dim: 3072,
        dropout_rate: 0.1,
        attention_dropout_rate: 0.0,
        classifier: "token".to_string(),
        num_classes: 1000,
        include_top: true,
    });

    // Validate the configuration
    match valid_vit.validate() {
        Ok(_) => println!("ViT configuration is valid"),
        Err(e) => println!("ViT configuration validation failed: {}", e),
    }

    // 6. Create models from configurations
    println!("\nCreating models from configurations:");

    // Create a ResNet model from configuration
    let resnet_model = resnet_config.create_model::<f32>()?;
    println!("Created ResNet model from configuration");

    // Create a BERT model from configuration
    let bert_model = bert_config.create_model::<f32>()?;
    println!("Created BERT model from configuration");

    // 7. Create hierarchical configurations
    println!("\nCreating hierarchical configurations:");

    // Create a Seq2Seq configuration with custom cell types
    let custom_seq2seq = ModelConfig::Seq2Seq(Seq2SeqConfig {
        input_vocab_size: 5000,
        output_vocab_size: 5000,
        embedding_dim: 256,
        hidden_dim: 512,
        num_layers: 2,
        encoder_cell_type: RNNCellType::GRU,
        decoder_cell_type: RNNCellType::LSTM,
        bidirectional_encoder: true,
        use_attention: true,
        dropout_rate: 0.1,
        max_seq_len: 100,
    });

    let custom_seq2seq_yaml = ConfigSerializer::to_yaml(&custom_seq2seq)?;
    println!("Hierarchical Seq2Seq YAML Configuration:");
    println!("{}", custom_seq2seq_yaml);

    println!("\nModel Configuration Example Completed Successfully!");

    Ok(())
}
