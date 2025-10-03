//! NAS controller for building and managing architectures

use crate::error::Result;
use crate::layers::{Layer, ParamLayer};
use crate::models::sequential::Sequential;
use crate::nas::{
    architecture_encoding::ArchitectureEncoding,
    search_space::{Architecture, LayerType, SearchSpaceConfig},
};
use std::sync::Arc;
/// Configuration for the NAS controller
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Input shape for the models
    pub inputshape: Vec<usize>,
    /// Number of output classes
    pub num_classes: usize,
    /// Whether to add a final softmax layer
    pub add_softmax: bool,
    /// Global seed for reproducibility
    pub seed: Option<u64>,
    /// Device to use (cpu, cuda, etc.)
    pub device: String,
}
impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            inputshape: vec![32, 32, 3], // Default to CIFAR-10 like input
            num_classes: 10,
            add_softmax: true,
            seed: None,
            device: "cpu".to_string(),
        }
    }
/// NAS Controller for building models from architecture encodings
pub struct NASController {
    config: ControllerConfig,
    search_space: SearchSpaceConfig,
impl NASController {
    /// Create a new NAS controller
    pub fn new(_searchspace: SearchSpaceConfig) -> Result<Self> {
        Ok(Self {
            config: ControllerConfig::default(),
            search_space,
        })
    /// Create with custom configuration
    pub fn with_config(_searchspace: SearchSpaceConfig, config: ControllerConfig) -> Result<Self> {
            config,
    /// Build a model from an architecture encoding
    pub fn build_model(&self, encoding: &Arc<dyn ArchitectureEncoding>) -> Result<Sequential<f32>> {
        let architecture = encoding.to_architecture()?;
        self.build_from_architecture(&architecture)
    /// Build a model from an Architecture struct
    pub fn build_from_architecture(&self, architecture: &Architecture) -> Result<Sequential<f32>> {
        let mut model = Sequential::new();
        let mut currentshape = self.config.inputshape.clone();
        // Apply width and depth multipliers
        let effective_layers = self.apply_multipliers(
            &architecture.layers,
            architecture.width_multiplier,
            architecture.depth_multiplier,
        )?;
        // Build layers
        for (i, layer_type) in effective_layers.iter().enumerate() {
            match layer_type {
                LayerType::Dense(units) => {
                    let input_size = currentshape.iter().product();
                    let mut rng = rng();
                    model.add_layer(crate::layers::Dense::new(
                        input_size, *units, &mut rng,
                    )?);
                    currentshape = vec![*units];
                }
                LayerType::Dropout(rate) => {
                    model.add_layer(crate::layers::Dropout::new(*rate as f64, &mut rng)?);
                    // Dropout doesn't change shape
                LayerType::BatchNorm => {
                    let features = currentshape.last().copied().unwrap_or(1);
                    model.add_layer(crate::layers::BatchNorm::new(
                        features, 0.9, 1e-5, &mut rng,
                    // BatchNorm doesn't change shape
                LayerType::Activation(name) => {
                    let size = currentshape.iter().product();
                        size,
                        Some(name.as_str()),
                        &mut rng,
                    // Activation doesn't change shape
                LayerType::Flatten => {
                    // For now, simulate flatten with a dense layer
                    let input_size: usize = currentshape.iter().product();
                        input_size, input_size, None, &mut rng,
                    currentshape = vec![input_size];
                _ => {
                    // Skip unsupported layer types for now
                    continue;
            }
        // Add final classification layer if needed
        if self.config.add_softmax {
            let input_size = currentshape.iter().product();
            let mut rng = rng();
            model.add_layer(crate::layers::Dense::new(
                input_size,
                self.config.num_classes,
                Some("softmax"),
                &mut rng,
            )?);
        // Handle skip connections
        for (from, to) in &architecture.connections {
            // This would require a more sophisticated model builder
            // For now, we'll skip implementing skip connections in Sequential
            // In practice, would use a functional API
        Ok(model)
    /// Apply width and depth multipliers to layers
    fn apply_multipliers(
        &self,
        layers: &[LayerType],
        width_mult: f32,
        depth_mult: f32,
    ) -> Result<Vec<LayerType>> {
        let mut result = Vec::new();
        for layer in layers {
            // Apply depth multiplier (repeat layers)
            let repetitions = (depth_mult.max(0.1) as usize).max(1);
            for _ in 0..repetitions {
                // Apply width multiplier
                let modified_layer = match layer {
                    LayerType::Dense(units) => {
                        LayerType::Dense((*units as f32 * width_mult).round() as usize)
                    }
                    LayerType::Conv2D {
                        filters,
                        kernel_size,
                        stride,
                    } => LayerType::Conv2D {
                        filters: (*filters as f32 * width_mult).round() as usize,
                        kernel_size: *kernel_size,
                        stride: *stride,
                    },
                    LayerType::Conv1D {
                    } => LayerType::Conv1D {
                    LayerType::LSTM {
                        units,
                        return_sequences,
                    } => LayerType::LSTM {
                        units: (*units as f32 * width_mult).round() as usize,
                        return_sequences: *return_sequences,
                    LayerType::GRU {
                    } => LayerType::GRU {
                    LayerType::Attention { num_heads, key_dim } => LayerType::Attention {
                        num_heads: *num_heads,
                        key_dim: (*key_dim as f32 * width_mult).round() as usize,
                    // Other layers remain unchanged
                    other => other.clone(),
                };
                result.push(modified_layer);
        Ok(result)
    /// Create a layer from LayerType
    fn create_layer(
        layer_type: &LayerType,
        inputshape: &[usize],
    ) -> Result<Box<dyn Layer<f32>>> {
        use crate::layers::{BatchNorm, Dense, Dropout};
        match layer_type {
            LayerType::Dense(units) => {
                let input_size = inputshape.iter().product();
                let mut rng = rng();
                Ok(Box::new(Dense::new(input_size, *units, None, &mut rng)?))
            LayerType::Dropout(rate) => {
                Ok(Box::new(Dropout::new(*rate as f64, &mut rng)?))
            LayerType::BatchNorm => {
                let features = inputshape.last().copied().unwrap_or(1);
                Ok(Box::new(BatchNorm::new(features, 0.9, 1e-5, &mut rng)?))
            LayerType::Activation(name) => {
                // Create a simple dense layer with 1:1 mapping and activation
                let size = inputshape.iter().product();
                Ok(Box::new(Dense::new(
                    size,
                    Some(name.as_str()),
                    &mut rng,
                )?))
            LayerType::Flatten => {
                // Create a reshape layer that flattens
                struct FlattenLayer;
                impl crate::layers::Layer<f32> for FlattenLayer {
                    fn forward(
                        &self,
                        input: &scirs2_core::ndarray::ArrayD<f32>,
                    ) -> Result<scirs2_core::ndarray::ArrayD<f32>> {
                        let batch_size = input.shape()[0];
                        let flattened_size: usize = input.shape()[1..].iter().product();
                        Ok(input.clone().into_shape(vec![batch_size, flattened_size])?)
                    fn backward(
                        _input: &scirs2_core::ndarray::ArrayD<f32>,
                        grad_output: &scirs2_core::ndarray::ArrayD<f32>,
                        Ok(grad_output.clone())
                    fn update(&mut self, learningrate: f32) -> Result<()> {
                        Ok(())
                    fn as_any(&self) -> &dyn std::any::Any {
                        self
                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                    fn layer_type(&self) -> &str {
                        "Flatten"
                Ok(Box::new(FlattenLayer)), LayerType::Conv2D {
                filters,
                kernel_size,
                stride,
            } => {
                // Create a 2D convolutional layer
                // For now, create a simple dense layer as placeholder since Conv2D requires more complex implementation
                let output_size = filters
                    * ((inputshape[0] - kernel_size.0) / stride.0 + 1)
                    * ((inputshape[1] - kernel_size.1) / stride.1 + 1);
                    input_size,
                    output_size,
                    None,
            LayerType::Conv1D {
                // Create a 1D convolutional layer
                // For now, create a simple dense layer as placeholder
                let output_size = filters * ((inputshape[0] - kernel_size) / stride + 1);
            LayerType::MaxPool2D {
                pool_size: _,
                stride: _,
            | LayerType::AvgPool2D {
                // Pooling layers don't change the number of parameters, just spatial dimensions
                // For now, create an identity transformation
                Ok(Box::new(Dense::new(size, size, None, &mut rng)?))
            LayerType::GlobalMaxPool2D | LayerType::GlobalAvgPool2D => {
                // Global pooling reduces spatial dimensions to 1x1
                let channels = inputshape.last().copied().unwrap_or(1);
                Ok(Box::new(Dense::new(input_size, channels, None, &mut rng)?))
            LayerType::LayerNorm => {
                // Layer normalization
                let features = inputshape.iter().product();
                // Use BatchNorm as a placeholder since we don't have LayerNorm implemented
            LayerType::Residual => {
                // Residual connection - for now, just pass through
            LayerType::Attention {
                num_heads: _,
                key_dim,
                // Attention mechanism
                Ok(Box::new(Dense::new(input_size, *key_dim, None, &mut rng)?))
            LayerType::LSTM {
                units,
                return_sequences: _,
                // LSTM layer
                let input_size = inputshape.last().copied().unwrap_or(1);
                    *units,
                    Some("tanh"),
            LayerType::GRU {
                // GRU layer
            LayerType::Embedding {
                vocab_size,
                embedding_dim,
                // Embedding layer - map vocabulary indices to dense vectors
                    *vocab_size,
                    *embedding_dim,
            LayerType::Reshape(newshape) => {
                // Reshape layer
                struct ReshapeLayer {
                    targetshape: Vec<i32>,
                impl<
                        F: num_traits:: Float + std::fmt::Debug + scirs2_core::ndarray::ScalarOperand + Send + Sync,
                    > crate::layers::Layer<F> for ReshapeLayer
                {
                    fn forward(&self, input: &scirs2_core::ndarray::ArrayD<F>) -> Result<scirs2_core::ndarray::ArrayD<F>> {
                        let mut newshape = vec![batch_size];
                        for &dim in &self.targetshape {
                            if dim < 0 {
                                // Infer dimension
                                let known_size: usize = newshape[1..].iter().product::<usize>()
                                    * self
                                        .targetshape
                                        .iter()
                                        .filter(|&&x| x > 0)
                                        .map(|&x| x as usize)
                                        .product::<usize>();
                                let total_size = input.len() / batch_size;
                                let inferred_dim = total_size / known_size;
                                newshape.push(inferred_dim);
                            } else {
                                newshape.push(dim as usize);
                            }
                        }
                        Ok(input.clone().into_shape(newshape)?)
                        _input: &scirs2_core::ndarray::ArrayD<F>,
                        grad_output: &scirs2_core::ndarray::ArrayD<F>,
                    ) -> Result<scirs2_core::ndarray::ArrayD<F>> {
                    fn update(&mut self, learningrate: F) -> Result<()> {
                        "Reshape"
                Ok(Box::new(ReshapeLayer {
                    targetshape: newshape.clone(),
                }))
            LayerType::Conv3D {
                // Create a 3D convolutional layer
                    * ((inputshape[1] - kernel_size.1) / stride.1 + 1)
                    * ((inputshape[2] - kernel_size.2) / stride.2 + 1);
            LayerType::SeparableConv2D {
                kernel_size: _,
                depth_multiplier: _,
                // Separable convolution layer
                Ok(Box::new(Dense::new(input_size, *filters, None, &mut rng)?))
            LayerType::Conv2DTranspose {
                padding: _,
                // Transposed convolution (deconvolution) layer
            LayerType::MaxPool1D {
            | LayerType::AvgPool1D {
                // 1D pooling layers
            LayerType::MaxPool3D {
            | LayerType::AvgPool3D {
                // 3D pooling layers
            LayerType::GlobalMaxPool1D | LayerType::GlobalAvgPool1D => {
                // Global pooling for 1D
            LayerType::GlobalMaxPool3D | LayerType::GlobalAvgPool3D => {
                // Global pooling for 3D
            LayerType::UpSampling2D { size: _ } => {
                // Upsampling layer
                    input_size * 4,
            LayerType::ZeroPadding2D { padding: _ } => {
                // Zero padding layer
            LayerType::Cropping2D { cropping: _ } => {
                // Cropping layer
                Ok(Box::new(Dense::new(size, size / 2, None, &mut rng)?))
            LayerType::Concatenate { axis: _ } => {
                // Concatenation layer
                Ok(Box::new(Dense::new(size, size * 2, None, &mut rng)?))
            LayerType::Add => {
                // Element-wise addition layer
            LayerType::Multiply => {
                // Element-wise multiplication layer
    /// Compute output shape after a layer
    fn compute_outputshape(
    ) -> Result<Vec<usize>> {
            LayerType::Dense(units) => Ok(vec![*units]),
                if inputshape.len() < 3 {
                    return Err(crate::error::NeuralError::InvalidArgument(
                        "Conv2D requires 3D input (H, W, C)".to_string(),
                    ));
                let h = (inputshape[0] - kernel_size.0) / stride.0 + 1;
                let w = (inputshape[1] - kernel_size.1) / stride.1 + 1;
                Ok(vec![h, w, *filters])
            LayerType::MaxPool2D { pool_size, stride }
            | LayerType::AvgPool2D { pool_size, stride } => {
                        "Pooling requires 3D input (H, W, C)".to_string(),
                let h = (inputshape[0] - pool_size.0) / stride.0 + 1;
                let w = (inputshape[1] - pool_size.1) / stride.1 + 1;
                Ok(vec![h, w, inputshape[2]])
                        "Global pooling requires 3D input (H, W, C)".to_string(),
                Ok(vec![inputshape[2]])
                let total_size: usize = inputshape.iter().product();
                Ok(vec![total_size])
                let newshape_usize: Vec<usize> = newshape
                    .iter()
                    .map(|&x| {
                        if x < 0 {
                            // -1 means infer this dimension
                            let known_product: i32 = newshape.iter().filter(|&&y| y > 0).product();
                            let total: i32 = inputshape.iter().map(|&x| x as i32).product();
                            (total / known_product) as usize
                        } else {
                            x as usize
                    })
                    .collect();
                Ok(newshape_usize)
            // For layers that don't change shape
            LayerType::Dropout(_)
            | LayerType::BatchNorm
            | LayerType::LayerNorm
            | LayerType::Activation(_)
            | LayerType::Residual => Ok(inputshape.to_vec()),
            // Recurrent layers - output shape depends on return_sequences parameter
                return_sequences,
            | LayerType::GRU {
                if *return_sequences {
                    // Return sequences: (seq_len, units)
                    if inputshape.is_empty() {
                        Ok(vec![*units])
                    } else {
                        Ok(vec![inputshape[0], *units])
                } else {
                    // Return only last output: (units,)
                    Ok(vec![*units])
            // Attention layer output shape
                if inputshape.is_empty() {
                    Ok(vec![*key_dim])
                    let mut outputshape = inputshape.to_vec();
                    *outputshape.last_mut().unwrap() = *key_dim;
                    Ok(outputshape)
            // Embedding layer
                vocab_size: _,
                    Ok(vec![*embedding_dim])
                    outputshape.push(*embedding_dim);
            // 1D Convolution
                        "Conv1D requires at least 1D input".to_string(),
                let input_length = inputshape[0];
                let output_length = (input_length - kernel_size) / stride + 1;
                if inputshape.len() == 1 {
                    Ok(vec![output_length, *filters])
            // 1D Pooling layers
            LayerType::MaxPool1D { pool_size, stride }
            | LayerType::AvgPool1D { pool_size, stride } => {
                        "Pool1D requires at least 1D input".to_string(),
                let output_length = (input_length - pool_size) / stride + 1;
                let mut outputshape = inputshape.to_vec();
                outputshape[0] = output_length;
                Ok(outputshape)
            // 3D operations (if supported)
                if inputshape.len() < 4 {
                        "Conv3D requires 4D input (D, H, W, C)".to_string(),
                let d = (inputshape[0] - kernel_size.0) / stride.0 + 1;
                let h = (inputshape[1] - kernel_size.1) / stride.1 + 1;
                let w = (inputshape[2] - kernel_size.2) / stride.2 + 1;
                Ok(vec![d, h, w, *filters])
            LayerType::MaxPool3D { pool_size, stride }
            | LayerType::AvgPool3D { pool_size, stride } => {
                        "Pool3D requires 4D input (D, H, W, C)".to_string(),
                let d = (inputshape[0] - pool_size.0) / stride.0 + 1;
                let h = (inputshape[1] - pool_size.1) / stride.1 + 1;
                let w = (inputshape[2] - pool_size.2) / stride.2 + 1;
                Ok(vec![d, h, w, inputshape[3]])
            // Global pooling for 1D and 3D
                if inputshape.len() < 2 {
                        "Global Pool1D requires 2D input (length, channels)".to_string(),
                Ok(vec![inputshape[1]]) // Keep only channel dimension
                        "Global Pool3D requires 4D input (D, H, W, C)".to_string(),
                Ok(vec![inputshape[3]]) // Keep only channel dimension
            // Separable convolutions
                        "SeparableConv2D requires 3D input (H, W, C)".to_string(),
            // Upsampling layers
            LayerType::UpSampling2D { size } => {
                        "UpSampling2D requires 3D input (H, W, C)".to_string(),
                let h = inputshape[0] * size.0;
                let w = inputshape[1] * size.1;
            // Transpose/Deconvolution layers
                        "Conv2DTranspose requires 3D input (H, W, C)".to_string(),
                let h = (inputshape[0] - 1) * stride.0 + kernel_size.0;
                let w = (inputshape[1] - 1) * stride.1 + kernel_size.1;
            // Zero padding
            LayerType::ZeroPadding2D { padding } => {
                        "ZeroPadding2D requires 3D input (H, W, C)".to_string(),
                let h = inputshape[0] + 2 * padding.0;
                let w = inputshape[1] + 2 * padding.1;
            // Cropping layers
            LayerType::Cropping2D { cropping } => {
                        "Cropping2D requires 3D input (H, W, C)".to_string(),
                let h = inputshape[0].saturating_sub(2 * cropping.0);
                let w = inputshape[1].saturating_sub(2 * cropping.1);
            // Concatenation layer
            LayerType::Concatenate { axis } => {
                // For concatenation, we need multiple inputs
                // This is a simplified version assuming concatenation along the last axis
                    Ok(vec![1])
                    let concat_axis = if *axis < 0 {
                        outputshape.len() - 1
                        *axis as usize
                    };
                    if concat_axis < outputshape.len() {
                        // Assume doubling the size along concatenation axis for simplicity
                        outputshape[concat_axis] *= 2;
            // Add layer (element-wise addition)
            LayerType::Add => Ok(inputshape.to_vec()),
            // Multiply layer (element-wise multiplication)
            LayerType::Multiply => Ok(inputshape.to_vec()),
            // Other unimplemented layer types
            _ => {
                log::warn!(
                    "Shape computation not implemented for layer type: {:?}",
                    layer_type
                );
                Ok(inputshape.to_vec())
    /// Count parameters in a model
    pub fn count_parameters(&self, model: &Sequential<f32>) -> Result<usize> {
        let mut total_params = 0;
        
        for layer in model.layers() {
            // Try to cast to ParamLayer to count parameters
            if let Some(param_layer) = layer.as_any().downcast_ref::<dyn ParamLayer<f32>>() {
                for param in param_layer.get_parameters() {
                    total_params += param.len();
        Ok(total_params)
    /// Estimate FLOPs for a model
    pub fn estimate_flops(&self, model: &Sequential<f32>, inputshape: &[usize]) -> Result<usize> {
        let mut total_flops = 0;
        let mut currentshape = inputshape.to_vec();
            let layer_type_str = layer.layer_type();
            
            match layer_type_str {
                "Dense" => {
                    // Dense layer: input_size * output_size * 2 (multiply-add)
                    if let Some(param_layer) = layer.as_any().downcast_ref::<dyn ParamLayer<f32>>() {
                        let params = param_layer.get_parameters();
                        if !params.is_empty() {
                            let weightshape = params[0].shape();
                            if weightshape.len() >= 2 {
                                let flops = weightshape[0] * weightshape[1] * 2;
                                total_flops += flops;
                                currentshape = vec![weightshape[0]];
                "Conv2D" => {
                    // Conv2D: output_h * output_w * kernel_h * kernel_w * in_channels * out_channels * 2
                            if weightshape.len() >= 4 && currentshape.len() >= 3 {
                                let out_channels = weightshape[0];
                                let in_channels = weightshape[1];
                                let kernel_h = weightshape[2];
                                let kernel_w = weightshape[3];
                                
                                // Assume stride=1, padding=same for simplicity
                                let output_h = currentshape[0];
                                let output_w = currentshape[1];
                                let flops = output_h * output_w * kernel_h * kernel_w * in_channels * out_channels * 2;
                                currentshape = vec![output_h, output_w, out_channels];
                    // Other layers (activations, normalization, etc.) have minimal FLOPs
                    let elements = currentshape.iter().product::<usize>();
                    total_flops += elements; // Assume 1 FLOP per element
        Ok(total_flops)
    /// Validate an architecture
    pub fn validate_architecture(&self, architecture: &Architecture) -> Result<()> {
        // Check if architecture is valid
        if architecture.layers.is_empty() {
            return Err(crate::error::NeuralError::InvalidArgument(
                "Architecture must have at least one layer".to_string(),
            ));
        // Check skip connections
            if *from >= architecture.layers.len() || *to >= architecture.layers.len() {
                return Err(crate::error::NeuralError::InvalidArgument(format!(
                    "Invalid skip connection: {} -> {}",
                    from, to
                )));
            if from >= to {
                return Err(crate::error::NeuralError::InvalidArgument(
                    "Skip connections must be forward connections".to_string(),
                ));
        // Validate multipliers
        if architecture.width_multiplier <= 0.0 || architecture.depth_multiplier <= 0.0 {
                "Multipliers must be positive".to_string(),
        Ok(())
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search_space::Architecture;
    #[test]
    fn test_controller_creation() {
        let search_space = SearchSpaceConfig::default();
        let controller = NASController::new(search_space).unwrap();
        assert_eq!(controller.config.num_classes, 10);
    fn test_architecture_validation() {
        // Valid architecture
        let valid_arch = Architecture {
            layers: vec![
                LayerType::Dense(128),
                LayerType::Activation("relu".to_string()),
                LayerType::Dense(10),
            ],
            connections: vec![],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        assert!(controller.validate_architecture(&valid_arch).is_ok());
        // Empty architecture
        let empty_arch = Architecture {
            layers: vec![],
        assert!(controller.validate_architecture(&empty_arch).is_err());
        // Invalid skip connection
        let invalid_skip = Architecture {
            layers: vec![LayerType::Dense(128), LayerType::Dense(10)],
            connections: vec![(1, 0)], // Backward connection
        assert!(controller.validate_architecture(&invalid_skip).is_err());
    fn test_multiplier_application() {
        let controller = NASController::new(SearchSpaceConfig::default()).unwrap();
        let layers = vec![
            LayerType::Dense(100),
                filters: 32,
                kernel_size: (3, 3),
                stride: (1, 1),
            },
        ];
        let modified = controller.apply_multipliers(&layers, 2.0, 1.5).unwrap();
        // Width multiplier should double the units/filters
        match &modified[0] {
            LayerType::Dense(units) => assert_eq!(*units, 200, _ => unreachable!("Expected Dense layer"),
        // Depth multiplier should create repetitions
        assert!(modified.len() > layers.len());
