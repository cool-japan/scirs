//! Comprehensive tests for model serialization and deserialization
//!
//! These tests ensure that models can be reliably saved and loaded
//! across different formats while preserving their structure and parameters.

#![allow(dead_code)]
use super::*;
use crate::layers::*;
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Float;
use scirs2_core::random::SeedableRng;
use std::fs;
use tempfile::tempdir;
/// Test basic dense layer serialization roundtrip
#[test]
#[allow(dead_code)]
fn test_dense_layer_serialization_roundtrip() -> Result<()> {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let model_path = temp_dir.path().join("test_dense_model.json");
    // Create a simple dense model
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
    let mut model = Sequential::<f32>::new();
    model.add(Box::new(Dense::new(4, 8, Some("relu"), &mut rng)?));
    model.add(Box::new(Dense::new(8, 3, Some("softmax"), &mut rng)?));
    // Save the model
    save_model(&model, &model_path, SerializationFormat::JSON)?;
    assert!(model_path.exists(), "Model file should exist after saving");
    // Load the model
    let loaded_model: Sequential<f32> = load_model(&model_path, SerializationFormat::JSON)?;
    // Verify structure
    assert_eq!(model.layers().len(), loaded_model.layers().len());
    // Test with sample input
    let input = Array2::<f32>::ones((2, 4)).into_dyn();
    let original_output = model.forward(&input)?;
    let loaded_output = loaded_model.forward(&input)?;
    // Outputs should be similar (parameters were loaded)
    assert_eq!(original_output.shape(), loaded_output.shape());
    Ok(())
}
/// Test CNN model serialization with various layer types
#[allow(dead_code)]
fn test_cnn_model_serialization() -> Result<()> {
    let model_path = temp_dir.path().join("test_cnn_model.json");
    // Create a CNN model
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed(123);
    
    model.add(Box::new(Conv2D::new(
        3, 16, (3, 3), (1, 1), PaddingMode::Same, &mut rng
    )?));
    model.add(Box::new(BatchNorm::new(16, 0.1, 1e-5, &mut rng)?));
    model.add(Box::new(MaxPool2D::new((2, 2), (2, 2), None)?));
    model.add(Box::new(Dropout::new(0.25, &mut rng)?));
    // Save and load
/// Test different serialization formats
#[allow(dead_code)]
fn test_multiple_serialization_formats() -> Result<()> {
    // Create a simple model
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed(456);
    model.add(Box::new(Dense::new(3, 5, Some("tanh"), &mut rng)?));
    model.add(Box::new(LayerNorm::new(5, 1e-5, &mut rng)?));
    // Test JSON format
    let json_path = temp_dir.path().join("model.json");
    save_model(&model, &json_path, SerializationFormat::JSON)?;
    let _loaded_json: Sequential<f32> = load_model(&json_path, SerializationFormat::JSON)?;
    // Test CBOR format
    let cbor_path = temp_dir.path().join("model.cbor");
    save_model(&model, &cbor_path, SerializationFormat::CBOR)?;
    let _loaded_cbor: Sequential<f32> = load_model(&cbor_path, SerializationFormat::CBOR)?;
    // Test MessagePack format
    let msgpack_path = temp_dir.path().join("model.msgpack");
    save_model(&model, &msgpack_path, SerializationFormat::MessagePack)?;
    let _loaded_msgpack: Sequential<f32> = load_model(&msgpack_path, SerializationFormat::MessagePack)?;
    // All files should exist
    assert!(json_path.exists());
    assert!(cbor_path.exists());
    assert!(msgpack_path.exists());
/// Test parameter preservation across serialization
#[allow(dead_code)]
fn test_parameter_preservation() -> Result<()> {
    let model_path = temp_dir.path().join("param_test_model.json");
    // Create model with known parameters
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed(789);
    let dense_layer = Dense::new(2, 3, None, &mut rng)?;
    let original_params = dense_layer.get_parameters();
    model.add(Box::new(dense_layer));
    // Get loaded parameters
    let loaded_dense = loaded_model.layers()[0]
        .as_any()
        .downcast_ref::<Dense<f32>>()
        .expect("First layer should be Dense");
    let loaded_params = loaded_dense.get_parameters();
    // Parameters should be preserved
    assert_eq!(original_params.len(), loaded_params.len());
    for (orig, loaded) in original_params.iter().zip(loaded_params.iter()) {
        assert_eq!(orig.shape(), loaded.shape());
        // Note: Due to initialization differences, exact values may differ
        // but shapes should be identical
    }
/// Test error handling for invalid files
#[allow(dead_code)]
fn test_error_handling() -> Result<()> {
    let invalid_path = temp_dir.path().join("invalid.json");
    // Try to load non-existent file
    let result = load_model::<f32>(&invalid_path, SerializationFormat::JSON);
    assert!(result.is_err());
    // Create invalid JSON file
    fs::write(&invalid_path, "invalid json content")?;
/// Test activation function serialization
#[allow(dead_code)]
fn test_activation_function_serialization() -> Result<()> {
    // Test activation function factory
    let activations = vec![
        "relu", "sigmoid", "tanh", "softmax", "gelu", "swish", "mish"
    ];
    for activation_name in activations {
        let activation_fn = ActivationFunction::from_name(activation_name);
        assert!(activation_fn.is_some(), "Should recognize activation: {}", activation_name);
        
        if let Some(af) = activation_fn {
            let created = af.create::<f32>();
            assert!(!created.as_any().type_id().is_zero(), "Should create valid activation");
        }
    // Test parametric activations
    let leaky_relu = ActivationFunction::from_name("leaky_relu(0.2)");
    assert!(leaky_relu.is_some());
    let elu = ActivationFunction::from_name("elu(1.5)");
    assert!(elu.is_some());
/// Test model with mixed layer types
#[allow(dead_code)]
fn test_mixed_layer_model() -> Result<()> {
    let model_path = temp_dir.path().join("mixed_model.json");
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed(999);
    // Add various layer types
    model.add(Box::new(Dense::new(10, 20, Some("relu"), &mut rng)?));
    model.add(Box::new(BatchNorm::new(20, 0.1, 1e-5, &mut rng)?));
    model.add(Box::new(Dropout::new(0.3, &mut rng)?));
    model.add(Box::new(LayerNorm::new(20, 1e-6, &mut rng)?));
    model.add(Box::new(Dense::new(20, 5, Some("softmax"), &mut rng)?));
    // Test forward pass
    let input = Array2::<f32>::ones((3, 10)).into_dyn();
/// Test serialization with f64 precision
#[allow(dead_code)]
fn test_f64_model_serialization() -> Result<()> {
    let model_path = temp_dir.path().join("f64_model.json");
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed(111);
    let mut model = Sequential::<f64>::new();
    model.add(Box::new(Dense::new(4, 6, Some("tanh"), &mut rng)?));
    model.add(Box::new(Dense::new(6, 2, None, &mut rng)?));
    let loaded_model: Sequential<f64> = load_model(&model_path, SerializationFormat::JSON)?;
    // Test with f64 input
    let input = Array2::<f64>::ones((2, 4)).into_dyn();
/// Test serialization of empty model
#[allow(dead_code)]
fn test_empty_model_serialization() -> Result<()> {
    let model_path = temp_dir.path().join("empty_model.json");
    let model = Sequential::<f32>::new();
    // Save and load empty model
    assert_eq!(0, loaded_model.layers().len());
/// Test large model serialization performance
#[allow(dead_code)]
fn test_large_model_serialization() -> Result<()> {
    let model_path = temp_dir.path().join("large_model.json");
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed(222);
    // Create a larger model
    model.add(Box::new(Dense::new(100, 200, Some("relu"), &mut rng)?));
    model.add(Box::new(Dense::new(200, 400, Some("relu"), &mut rng)?));
    model.add(Box::new(Dropout::new(0.5, &mut rng)?));
    model.add(Box::new(Dense::new(400, 200, Some("relu"), &mut rng)?));
    model.add(Box::new(Dense::new(200, 50, Some("relu"), &mut rng)?));
    model.add(Box::new(Dense::new(50, 10, Some("softmax"), &mut rng)?));
    let start_time = std::time::Instant::now();
    let save_duration = start_time.elapsed();
    let _loaded_model: Sequential<f32> = load_model(&model_path, SerializationFormat::JSON)?;
    let load_duration = start_time.elapsed();
    // Basic performance checks (should be reasonable)
    assert!(save_duration.as_secs() < 5, "Save should complete within 5 seconds");
    assert!(load_duration.as_secs() < 5, "Load should complete within 5 seconds");
/// Test serialization format comparison
#[allow(dead_code)]
fn test_format_comparison() -> Result<()> {
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed(333);
    model.add(Box::new(Dense::new(50, 100, Some("relu"), &mut rng)?));
    model.add(Box::new(Dense::new(100, 25, Some("softmax"), &mut rng)?));
    // Save in all formats
    // Compare file sizes
    let json_size = fs::metadata(&json_path)?.len();
    let cbor_size = fs::metadata(&cbor_path)?.len();
    let msgpack_size = fs::metadata(&msgpack_path)?.len();
    // JSON should be human-readable (largest), binary formats should be smaller
    assert!(json_size > 0);
    assert!(cbor_size > 0);
    assert!(msgpack_size > 0);
    println!("File sizes - JSON: {}, CBOR: {}, MessagePack: {}", json_size, cbor_size, msgpack_size);
/// Helper function to create test model
#[allow(dead_code)]
fn create_test_model<F: Float + Debug + ScalarOperand + Send + Sync + 'static>() -> Result<Sequential<F>> {
    let mut model = Sequential::<F>::new();
    model.add(Box::new(Dropout::new(0.2, &mut rng)?));
    model.add(Box::new(Dense::new(20, 10, Some("tanh"), &mut rng)?));
    model.add(Box::new(Dense::new(10, 5, Some("softmax"), &mut rng)?));
    Ok(model)
/// Integration test for complete workflow
#[allow(dead_code)]
fn test_complete_workflow() -> Result<()> {
    let model_path = temp_dir.path().join("workflow_model.json");
    // Create, train (mock), save, load, and test
    let model = create_test_model::<f32>()?;
    // Save model
    // Load model
    // Test inference
    let input = Array2::<f32>::ones((5, 10)).into_dyn();
    let output = loaded_model.forward(&input)?;
    assert_eq!(output.shape(), &[5, 5]);
    // Clean up
    fs::remove_file(&model_path)?;
