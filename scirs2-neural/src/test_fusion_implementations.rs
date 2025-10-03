//! Test file to verify fusion implementations

#[cfg(test)]
mod tests {
    use crate::models::architectures::fusion::*;
    use crate::layers::Layer;
    use scirs2_core::ndarray::Array;
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;
    #[test]
    fn test_feature_alignment_backward_update() -> crate::error::Result<()> {
        let mut rng = scirs2_core::random::rng();
        let mut alignment: FeatureAlignment<f32> = FeatureAlignment::new(10, 8, Some("test"))?;
        
        // Test forward pass
        let input = Array::ones((2, 10)).into_dyn();
        let output = alignment.forward(&input)?;
        assert_eq!(output.shape(), &[2, 8]);
        // Test backward pass
        let grad_output = Array::ones((2, 8)).into_dyn();
        let grad_input = alignment.backward(&input, &grad_output)?;
        assert_eq!(grad_input.shape(), input.shape());
        // Test update
        alignment.update(0.01)?;
        Ok(())
    }
    fn test_cross_modal_attention_backward_update() -> crate::error::Result<()> {
        let mut attention: CrossModalAttention<f32> = CrossModalAttention::new(8, 8, 8)?;
        // Test dedicated forward method
        let query = Array::ones((2, 4, 8)).into_dyn();
        let context = Array::ones((2, 6, 8)).into_dyn();
        let output = attention.forward(&query, &context)?;
        assert_eq!(output.shape(), &[2, 4, 8]);
        // Test Layer trait methods (simplified)
        let dummy_input = Array::ones((2, 4, 8)).into_dyn();
        let grad_output = Array::ones((2, 4, 8)).into_dyn();
        let grad_input = attention.backward(&dummy_input, &grad_output)?;
        assert_eq!(grad_input.shape(), grad_output.shape());
        attention.update(0.01)?;
    fn test_film_module_backward_update() -> crate::error::Result<()> {
        let mut film: FiLMModule<f32> = FiLMModule::new(8, 6)?;
        let features = Array::ones((2, 8)).into_dyn();
        let conditioning = Array::ones((2, 6)).into_dyn();
        let output = film.forward(&features, &conditioning)?;
        let dummy_input = Array::ones((2, 8)).into_dyn();
        let grad_input = film.backward(&dummy_input, &grad_output)?;
        film.update(0.01)?;
    fn test_bilinear_fusion_backward_update() -> crate::error::Result<()> {
        let mut bilinear: BilinearFusion<f32> = BilinearFusion::new(8, 6, 10, 4)?;
        let features_a = Array::ones((2, 8)).into_dyn();
        let features_b = Array::ones((2, 6)).into_dyn();
        let output = bilinear.forward(&features_a, &features_b)?;
        assert_eq!(output.shape(), &[2, 10]);
        let grad_output = Array::ones((2, 10)).into_dyn();
        let grad_input = bilinear.backward(&dummy_input, &grad_output)?;
        bilinear.update(0.01)?;
    fn test_feature_fusion_backward_update() -> crate::error::Result<()> {
        let config = FeatureFusionConfig {
            input_dims: vec![10, 8],
            hidden_dim: 6,
            fusion_method: FusionMethod::Concatenation,
            dropout_rate: 0.1,
            num_classes: 3,
            include_head: true,
        };
        let mut fusion: FeatureFusion<f32> = FeatureFusion::new(config)?;
        // Test forward_multi
        let inputs = vec![
            Array::ones((2, 10)).into_dyn(),
            Array::ones((2, 8)).into_dyn(),
        ];
        let output = fusion.forward_multi(&inputs)?;
        assert_eq!(output.shape(), &[2, 3]);
        let dummy_input = Array::ones((2, 10)).into_dyn();
        let grad_output = Array::ones((2, 3)).into_dyn();
        let grad_input = fusion.backward(&dummy_input, &grad_output)?;
        fusion.update(0.01)?;
    fn test_attention_fusion() -> crate::error::Result<()> {
            fusion_method: FusionMethod::Attention,
        let fusion: FeatureFusion<f32> = FeatureFusion::new(config)?;
        // Test forward_multi with attention fusion
    fn test_film_fusion() -> crate::error::Result<()> {
            fusion_method: FusionMethod::FiLM,
        // Test forward_multi with FiLM fusion
    fn test_bilinear_fusion_model() -> crate::error::Result<()> {
            fusion_method: FusionMethod::Bilinear,
        // Test forward_multi with bilinear fusion
}
