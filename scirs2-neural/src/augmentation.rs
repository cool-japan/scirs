//! Advanced data augmentation techniques for neural networks
//!
//! This module provides comprehensive data augmentation utilities including:
//! - Image augmentations (geometric, photometric, noise-based)
//! - Text augmentations (synonym replacement, random insertion/deletion)
//! - Audio augmentations (time-stretching, pitch shifting, noise injection)
//! - Mix-based augmentations (MixUp, CutMix, AugMix)

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD, Axis};
use num_traits::Float;
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;
use statrs::statistics::Statistics;
/// Image augmentation transforms
#[derive(Debug, Clone, PartialEq)]
pub enum ImageAugmentation {
    /// Random horizontal flip
    RandomHorizontalFlip {
        /// Probability of applying the flip (0.0 to 1.0)
        probability: f64,
    },
    /// Random vertical flip
    RandomVerticalFlip {
    /// Random rotation within angle range
    RandomRotation {
        /// Minimum rotation angle in degrees
        min_angle: f64,
        /// Maximum rotation angle in degrees
        max_angle: f64,
        /// How to fill empty areas after rotation
        fill_mode: FillMode,
    /// Random scaling
    RandomScale {
        /// Minimum scaling factor
        min_scale: f64,
        /// Maximum scaling factor
        max_scale: f64,
        /// Whether to preserve aspect ratio
        preserve_aspect_ratio: bool,
    /// Random crop and resize
    RandomCrop {
        /// Height of the crop
        crop_height: usize,
        /// Width of the crop
        crop_width: usize,
        /// Optional padding to add before cropping
        padding: Option<usize>,
    /// Color jittering
    ColorJitter {
        /// Brightness variation (None to disable)
        brightness: Option<f64>,
        /// Contrast variation (None to disable)
        contrast: Option<f64>,
        /// Saturation variation (None to disable)
        saturation: Option<f64>,
        /// Hue variation (None to disable)
        hue: Option<f64>,
    /// Gaussian noise injection
    GaussianNoise {
        /// Mean of the Gaussian noise
        mean: f64,
        /// Standard deviation of the Gaussian noise
        std: f64,
        /// Probability of applying noise (0.0 to 1.0)
    /// Random erasing (cutout)
    RandomErasing {
        /// Probability of applying erasing (0.0 to 1.0)
        /// Range of area ratios to erase (min, max)
        area_ratio_range: (f64, f64),
        /// Range of aspect ratios for erased area (min, max)
        aspect_ratio_range: (f64, f64),
        /// Value to fill erased area with
        fill_value: f64,
    /// Elastic deformation
    ElasticDeformation {
        /// Scaling factor for deformation strength
        alpha: f64,
        /// Standard deviation for Gaussian filter
        sigma: f64,
        /// Probability of applying deformation (0.0 to 1.0)
}
/// Fill modes for geometric transformations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillMode {
    /// Fill with constant value
    Constant(f64),
    /// Reflect across the edge
    Reflect,
    /// Wrap around
    Wrap,
    /// Nearest neighbor
    Nearest,
/// Text augmentation techniques
pub enum TextAugmentation {
    /// Random synonym replacement
    SynonymReplacement {
        /// Probability of replacing each word (0.0 to 1.0)
        /// Number of words to replace
        num_replacements: usize,
    /// Random word insertion
    RandomInsertion {
        /// Probability of inserting words (0.0 to 1.0)
        /// Number of words to insert
        num_insertions: usize,
    /// Random word deletion
    RandomDeletion {
        /// Probability of deleting each word (0.0 to 1.0)
    /// Random word swap
    RandomSwap {
        /// Probability of swapping words (0.0 to 1.0)
        /// Number of swaps to perform
        num_swaps: usize,
    /// Back translation
    BackTranslation {
        /// Intermediate language for back translation
        intermediate_language: String,
    /// Paraphrasing
    Paraphrasing {
        /// Type of paraphrasing model to use
        model_type: String,
/// Audio augmentation techniques
pub enum AudioAugmentation {
    /// Time stretching
    TimeStretch {
        /// Range of time stretch factors (min, max)
        stretch_factor_range: (f64, f64),
        /// Probability of applying time stretch (0.0 to 1.0)
    /// Pitch shifting
    PitchShift {
        /// Range of pitch shift in semitones (min, max)
        semitone_range: (f64, f64),
        /// Probability of applying pitch shift (0.0 to 1.0)
    /// Add background noise
    AddNoise {
        /// Factor for noise intensity
        noise_factor: f64,
        /// Probability of adding noise (0.0 to 1.0)
    /// Volume adjustment
    VolumeAdjust {
        /// Range of volume gain factors (min, max)
        gain_range: (f64, f64),
        /// Probability of applying volume adjustment (0.0 to 1.0)
    /// Frequency masking
    FrequencyMask {
        /// Number of frequency masks to apply
        num_masks: usize,
        /// Range of mask widths (min, max)
        mask_width_range: (usize, usize),
        /// Probability of applying frequency masking (0.0 to 1.0)
    /// Time masking
    TimeMask {
        /// Number of time masks to apply
        /// Range of mask lengths (min, max)
        mask_length_range: (usize, usize),
        /// Probability of applying time masking (0.0 to 1.0)
/// Mix-based augmentation strategies
pub enum MixAugmentation {
    /// MixUp augmentation
    MixUp {
        /// Beta distribution parameter for mixing ratio
    /// CutMix augmentation
    CutMix {
        /// Range of cut ratios (min, max)
        cut_ratio_range: (f64, f64),
    /// AugMix augmentation
    AugMix {
        /// Severity of augmentation operations
        severity: usize,
        /// Width of augmentation chain
        width: usize,
        /// Depth of augmentation chain
        depth: usize,
        /// Beta distribution parameter for mixing
    /// Manifold mixup
    ManifoldMix {
        /// Probability of mixing at each layer
        layer_mix_probability: f64,
/// Comprehensive data augmentation manager
pub struct AugmentationManager<
    F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    /// Image augmentation pipeline
    image_transforms: Vec<ImageAugmentation>,
    /// Text augmentation pipeline
    text_transforms: Vec<TextAugmentation>,
    /// Audio augmentation pipeline
    audio_transforms: Vec<AudioAugmentation>,
    /// Mix augmentation strategies
    mix_strategies: Vec<MixAugmentation>,
    /// Random number generator seed
    rng_seed: Option<u64>,
    /// Augmentation statistics
    stats: AugmentationStatistics<F>,
/// Statistics for tracking augmentation usage
#[derive(Debug, Clone)]
pub struct AugmentationStatistics<
    /// Number of samples processed
    pub samples_processed: usize,
    /// Average augmentation intensity
    pub avg_intensity: F,
    /// Transform usage counts
    pub transform_counts: HashMap<String, usize>,
    /// Performance metrics
    pub processing_time_ms: f64,
impl<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive>
    AugmentationManager<F>
{
    /// Create a new augmentation manager
    pub fn new(_rngseed: Option<u64>) -> Self {
        Self {
            image_transforms: Vec::new(),
            text_transforms: Vec::new(),
            audio_transforms: Vec::new(),
            mix_strategies: Vec::new(),
            rng_seed,
            stats: AugmentationStatistics {
                samples_processed: 0,
                avg_intensity: F::zero(),
                transform_counts: HashMap::new(),
                processing_time_ms: 0.0,
            },
        }
    }
    /// Add image augmentation transform
    pub fn add_image_transform(&mut self, transform: ImageAugmentation) {
        self.image_transforms.push(transform);
    /// Add text augmentation transform
    pub fn addtext_transform(&mut self, transform: TextAugmentation) {
        self.text_transforms.push(transform);
    /// Add audio augmentation transform
    pub fn add_audio_transform(&mut self, transform: AudioAugmentation) {
        self.audio_transforms.push(transform);
    /// Add mix augmentation strategy
    pub fn add_mix_strategy(&mut self, strategy: MixAugmentation) {
        self.mix_strategies.push(strategy);
    /// Apply image augmentations to a batch of images
    pub fn augment_images(&mut self, images: &ArrayD<F>) -> Result<ArrayD<F>> {
        let start_time = std::time::Instant::now();
        let mut augmented = images.clone();
        for transform in &self.image_transforms {
            augmented = self.apply_image_transform(&augmented, transform)?;
            // Update statistics
            let transform_name = format!("{transform:?}")
                .split(' ')
                .next()
                .unwrap_or("unknown")
                .to_string();
            *self
                .stats
                .transform_counts
                .entry(transform_name)
                .or_insert(0) += 1;
        self.stats.samples_processed += images.shape()[0];
        self.stats.processing_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(augmented)
    fn apply_image_transform(
        &self,
        images: &ArrayD<F>,
        transform: &ImageAugmentation,
    ) -> Result<ArrayD<F>> {
        match transform {
            ImageAugmentation::RandomHorizontalFlip { probability } => {
                self.random_horizontal_flip(images, *probability)
            }
            ImageAugmentation::RandomVerticalFlip { probability } => {
                self.random_vertical_flip(images, *probability)
            ImageAugmentation::RandomRotation {
                min_angle,
                max_angle,
                fill_mode,
            } => self.random_rotation(images, *min_angle, *max_angle, *fill_mode),
            ImageAugmentation::RandomScale {
                min_scale,
                max_scale,
                preserve_aspect_ratio,
            } => self.random_scale(images, *min_scale, *max_scale, *preserve_aspect_ratio),
            ImageAugmentation::RandomCrop {
                crop_height,
                crop_width,
                padding,
            } => self.random_crop(images, *crop_height, *crop_width, *padding),
            ImageAugmentation::ColorJitter {
                brightness,
                contrast,
                saturation,
                hue,
            } => self.color_jitter(images, *brightness, *contrast, *saturation, *hue),
            ImageAugmentation::GaussianNoise {
                mean,
                std,
                probability,
            } => self.gaussian_noise(images, *mean, *std, *probability),
            ImageAugmentation::RandomErasing {
                area_ratio_range,
                aspect_ratio_range,
                fill_value,
            } => self.random_erasing(
                images,
                *probability,
                *area_ratio_range,
                *aspect_ratio_range,
                *fill_value,
            ),
            ImageAugmentation::ElasticDeformation {
                alpha,
                sigma,
            } => self.elastic_deformation(images, *alpha, *sigma, *probability),
    fn random_horizontal_flip(&self, images: &ArrayD<F>, probability: f64) -> Result<ArrayD<F>> {
        let mut result = images.clone();
        let batch_size = images.shape()[0];
        for i in 0..batch_size {
            if rand::random::<f64>() < probability {
                // Flip horizontally by reversing the width dimension
                if images.ndim() >= 4 {
                    // Assuming NCHW format: (batch, channels, height, width)
                    let width_dim = images.ndim() - 1;
                    let mut sample = result.slice_mut(ndarray::s![i, .., .., ..]);
                    sample.invert_axis(Axis(width_dim - 1)); // width axis relative to sample
                }
        Ok(result)
    fn random_vertical_flip(&self, images: &ArrayD<F>, probability: f64) -> Result<ArrayD<F>> {
                // Flip vertically by reversing the height dimension
                    let height_dim = images.ndim() - 2;
                    sample.invert_axis(Axis(height_dim - 1)); // height axis relative to sample
    fn random_rotation(
        _fill_mode: FillMode,
        // Simplified rotation implementation
        // In practice, this would involve proper image rotation algorithms
        let result = images.clone();
        for _i in 0..batch_size {
            let _angle = rng().random_range(min_angle..=max_angle);
            // Apply rotation (simplified - just return original for now)
            // Real implementation would use affine transformations
    fn random_scale(
        _preserve_aspect_ratio: bool..// Simplified scaling implementation
        // In practice, this would involve proper image scaling algorithms
            let _scale = rng().random_range(min_scale..=max_scale);
            // Apply scaling (simplified - just return original for now)
            // Real implementation would use interpolation
    fn random_crop(
        _padding: Option<usize>..if images.ndim() < 4 {
            return Err(NeuralError::InvalidArchitecture(
                "Random crop requires 4D input (NCHW)".to_string(),
            ));
        let channels = images.shape()[1];
        let height = images.shape()[2];
        let width = images.shape()[3];
        if crop_height > height || crop_width > width {
                "Crop size cannot be larger than image size".to_string(),
        let mut result = Array::zeros((batch_size, channels, crop_height, crop_width));
            let start_h = rng().random_range(0..=(height - crop_height));
            let start_w = rng().random_range(0..=(width - crop_width));
            let crop = images.slice(ndarray::s![
                i....,
                start_h..start_h + crop_height,
                start_w..start_w + crop_width
            ]);
            result.slice_mut(ndarray::s![i, .., .., ..]).assign(&crop);
        Ok(result.into_dyn())
    fn color_jitter(
        _saturation: Option<f64>, _hue: Option<f64>,
        // Apply brightness adjustment
        if let Some(bright_factor) = brightness {
            let factor =
                F::from(1.0 + rng().random_range(-bright_factor..=bright_factor)).unwrap();
            result = result * factor;
        // Apply contrast adjustment
        if let Some(contrast_factor) = contrast {
                F::from(1.0 + rng().random_range(-contrast_factor..=contrast_factor))
                    .unwrap();
            let mean = result.mean().unwrap_or(F::zero());
            result = (result - mean) * factor + mean;
        // Clamp values to valid range [0..1] (assuming normalized images)
        result = result.mapv(|x| x.max(F::zero()).min(F::one()));
    fn gaussian_noise(
        if rand::random::<f64>() < probability {
            let noise = images.mapv(|_| {
                let noise_val = rng().random_range(-3.0 * std..=3.0 * std) + mean;
                F::from(noise_val).unwrap_or(F::zero())
            });
            result = result + noise;
    fn random_erasing(
                "Random erasing requires 4D input (NCHW)".to_string()..let fill_val = F::from(fill_value).unwrap_or(F::zero());
                let area_ratio = rng().random_range(area_ratio_range.0..=area_ratio_range.1);
                let aspect_ratio =
                    rng().random_range(aspect_ratio_range.0..=aspect_ratio_range.1);
                let target_area = (height * width) as f64 * area_ratio;
                let mask_height = ((target_area * aspect_ratio).sqrt() as usize).min(height);
                let mask_width = ((target_area / aspect_ratio).sqrt() as usize).min(width);
                if mask_height > 0 && mask_width > 0 {
                    let start_h = rng().random_range(0..=(height - mask_height));
                    let start_w = rng().random_range(0..=(width - mask_width));
                    result
                        .slice_mut(ndarray::s![
                            i....,
                            start_h..start_h + mask_height,
                            start_w..start_w + mask_width
                        ])
                        .fill(fill_val);
    fn elastic_deformation(
        _alpha: f64, sigma: f64,
        // Simplified elastic deformation implementation
        // In practice, this would involve complex displacement field generation
            // Apply simple noise as a placeholder for elastic deformation
            let noise_factor = F::from(0.01).unwrap();
                let noise_val = rng().random_range(-0.05..=0.05);
            result = result + noise * noise_factor;
    /// Apply MixUp augmentation to a batch
    pub fn apply_mixup(
        &mut self..labels: &ArrayD<F>,) -> Result<(ArrayD<F>, ArrayD<F>)> {
        if batch_size < 2 {
            return Ok((images.clone(), labels.clone()));
        let lambda = self.sample_beta_distribution(alpha)?;
        let lambda_f = F::from(lambda).unwrap_or(F::from(0.5).unwrap());
        // Create random permutation of indices
        let mut indices: Vec<usize> = (0..batch_size).collect();
            let j = rng().random_range(0..batch_size);
            indices.swap(i, j);
        let mut mixed_images = images.clone();
        let mut mixed_labels = labels.clone();
        for (i, &j) in indices.iter().enumerate().take(batch_size) {
            // Mix images: x_mixed = lambda * x_i + (1 - lambda) * x_j
            let x_i = images.index_axis(ndarray::Axis(0), i);
            let x_j = images.index_axis(ndarray::Axis(0), j);
            let mixed = &x_i * lambda_f + &x_j * (F::one() - lambda_f);
            mixed_images
                .index_axis_mut(ndarray::Axis(0), i)
                .assign(&mixed);
            // Mix labels: y_mixed = lambda * y_i + (1 - lambda) * y_j
            let y_i = labels.index_axis(ndarray::Axis(0), i);
            let y_j = labels.index_axis(ndarray::Axis(0), j);
            let mixed_label = &y_i * lambda_f + &y_j * (F::one() - lambda_f);
            mixed_labels
                .assign(&mixed_label);
        self.stats.samples_processed += batch_size;
        *self
            .stats
            .transform_counts
            .entry("MixUp".to_string())
            .or_insert(0) += 1;
        Ok((mixed_images, mixed_labels))
    /// Apply CutMix augmentation
    pub fn apply_cutmix(
                "CutMix requires 4D input (NCHW)".to_string(),
        let _lambda = self.sample_beta_distribution(alpha)?;
        let cut_ratio = rng().random_range(cut_ratio_range.0..=cut_ratio_range.1);
        let cut_height = ((height as f64 * cut_ratio).sqrt() as usize).min(height);
        let cut_width = ((width as f64 * cut_ratio).sqrt() as usize).min(width);
        // Create random permutation
            let j = indices[i];
            // Random cut position
            let start_h = rng().random_range(0..=(height - cut_height));
            let start_w = rng().random_range(0..=(width - cut_width));
            // Cut and paste
            let patch = images.slice(ndarray::s![
                j..start_h..start_h + cut_height,
                start_w..start_w + cut_width
                .slice_mut(ndarray::s![
                    i,
                    ..,
                    start_h..start_h + cut_height,
                    start_w..start_w + cut_width
                ])
                .assign(&patch);
            // Mix labels based on cut area ratio
            let actual_lambda = (cut_height * cut_width) as f64 / (height * width) as f64;
            let lambda_f = F::from(1.0 - actual_lambda).unwrap_or(F::from(0.5).unwrap());
            let y_i = labels.slice(ndarray::s![i, ..]);
            let y_j = labels.slice(ndarray::s![j, ..]);
                .slice_mut(ndarray::s![i, ..])
            .entry("CutMix".to_string())
    fn sample_beta_distribution(&self, alpha: f64) -> Result<f64> {
        // Simplified beta distribution sampling
        // In practice, you would use a proper beta distribution implementation
        if alpha <= 0.0 {
            return Ok(0.5);
        // Approximate beta distribution with uniform sampling for simplicity
        Ok(rand::random::<f64>())
    /// Get augmentation statistics
    pub fn get_statistics(&self) -> &AugmentationStatistics<F> {
        &self.stats
    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.stats = AugmentationStatistics {
            samples_processed: 0,
            avg_intensity: F::zero(),
            transform_counts: HashMap::new(),
            processing_time_ms: 0.0,
        };
    /// Create a standard image augmentation pipeline
    pub fn create_standard_image_pipeline() -> Vec<ImageAugmentation> {
        vec![
            ImageAugmentation::RandomHorizontalFlip { probability: 0.5 },
                brightness: Some(0.2),
                contrast: Some(0.2),
                saturation: Some(0.2),
                hue: Some(0.1),
                mean: 0.0,
                std: 0.01,
                probability: 0.3,
                probability: 0.25,
                area_ratio_range: (0.02, 0.33),
                aspect_ratio_range: (0.3, 3.3),
                fill_value: 0.0,
        ]
    /// Create a strong image augmentation pipeline
    pub fn create_strong_image_pipeline() -> Vec<ImageAugmentation> {
            ImageAugmentation::RandomVerticalFlip { probability: 0.2 },
                min_angle: -30.0,
                max_angle: 30.0,
                fill_mode: FillMode::Constant(0.0),
                min_scale: 0.8,
                max_scale: 1.2,
                preserve_aspect_ratio: true,
                brightness: Some(0.4),
                contrast: Some(0.4),
                saturation: Some(0.4),
                hue: Some(0.2),
                std: 0.02,
                probability: 0.5,
                area_ratio_range: (0.02, 0.4),
                alpha: 1.0,
                sigma: 0.1,
impl<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Default
    for AugmentationManager<F>
    fn default() -> Self {
        Self::new(None)
/// Augmentation pipeline builder for easy configuration
pub struct AugmentationPipelineBuilder<
    manager: AugmentationManager<F>,
    AugmentationPipelineBuilder<F>
    /// Create a new pipeline builder
    pub fn new() -> Self {
            manager: AugmentationManager::new(None),
    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.manager.rng_seed = Some(seed);
        self
    /// Add standard image augmentations
    pub fn with_standard_image_augmentations(mut self) -> Self {
        for transform in AugmentationManager::<F>::create_standard_image_pipeline() {
            self.manager.add_image_transform(transform);
    /// Add strong image augmentations
    pub fn with_strong_image_augmentations(mut self) -> Self {
        for transform in AugmentationManager::<F>::create_strong_image_pipeline() {
    /// Add MixUp augmentation
    pub fn with_mixup(mut self, alpha: f64) -> Self {
        self.manager
            .add_mix_strategy(MixAugmentation::MixUp { alpha });
    /// Add CutMix augmentation
    pub fn with_cutmix(mut self, alpha: f64, cut_ratiorange: (f64, f64)) -> Self {
        self.manager.add_mix_strategy(MixAugmentation::CutMix {
            alpha,
            cut_ratio_range,
        });
    /// Build the augmentation manager
    pub fn build(self) -> AugmentationManager<F> {
    for AugmentationPipelineBuilder<F>
        Self::new()
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array4};
    #[test]
    fn test_augmentation_manager_creation() {
        let manager = AugmentationManager::<f64>::new(Some(42));
        assert_eq!(manager.rng_seed, Some(42));
        assert_eq!(manager.image_transforms.len(), 0);
    fn test_random_horizontal_flip() {
        let mut manager = AugmentationManager::<f64>::new(Some(42));
        manager.add_image_transform(ImageAugmentation::RandomHorizontalFlip { probability: 1.0 });
        let input =
            Array4::<f64>::from_shape_fn((2, 3, 4, 4), |(____)| rand::random()).into_dyn();
        let result = manager.augment_images(&input).unwrap();
        assert_eq!(result.shape(), input.shape());
        assert!(manager.stats.samples_processed > 0);
    fn test_random_crop() {
        let manager = AugmentationManager::<f64>::new(None);
        let input = Array4::<f64>::ones((2, 3, 8, 8)).into_dyn();
        let result = manager.random_crop(&input, 4, 4, None).unwrap();
        assert_eq!(result.shape(), &[2, 3, 4, 4]);
    fn test_color_jitter() {
        let input = Array4::<f64>::from_elem((1, 3, 4, 4), 0.5).into_dyn();
        let result = manager
            .color_jitter(&input, Some(0.2), Some(0.2), None, None)
            .unwrap();
    fn test_gaussian_noise() {
        let input = Array4::<f64>::zeros((2, 3, 4, 4)).into_dyn();
        let result = manager.gaussian_noise(&input, 0.0, 0.1, 1.0).unwrap();
    fn test_random_erasing() {
            .random_erasing(&input, 1.0, (0.1, 0.3), (0.5, 2.0), 0.0)
    fn test_mixup() {
        let images = Array4::<f64>::ones((4, 3, 8, 8)).into_dyn();
        let labels = Array2::<f64>::from_elem((4, 10), 1.0).into_dyn();
        let (mixed_images, mixed_labels) = manager.apply_mixup(&images, &labels, 1.0).unwrap();
        assert_eq!(mixed_images.shape(), images.shape());
        assert_eq!(mixed_labels.shape(), labels.shape());
        assert!(manager.stats.transform_counts.contains_key("MixUp"));
    fn test_cutmix() {
        let (mixed_images, mixed_labels) = manager
            .apply_cutmix(&images, &labels, 1.0, (0.1, 0.5))
        assert!(manager.stats.transform_counts.contains_key("CutMix"));
    fn test_standard_pipeline() {
        let pipeline = AugmentationManager::<f64>::create_standard_image_pipeline();
        assert!(!pipeline.is_empty());
        assert!(pipeline.len() >= 3);
    fn test_strong_pipeline() {
        let pipeline = AugmentationManager::<f64>::create_strong_image_pipeline();
        assert!(
            pipeline.len() > AugmentationManager::<f64>::create_standard_image_pipeline().len()
        );
    fn test_pipeline_builder() {
        let manager = AugmentationPipelineBuilder::<f64>::new()
            .with_seed(42)
            .with_standard_image_augmentations()
            .with_mixup(1.0)
            .build();
        assert!(!manager.image_transforms.is_empty());
        assert!(!manager.mix_strategies.is_empty());
    fn test_augmentation_statistics() {
        let mut manager = AugmentationManager::<f64>::new(None);
        manager.add_image_transform(ImageAugmentation::RandomHorizontalFlip { probability: 0.5 });
        let input = Array4::<f64>::ones((2, 3, 4, 4)).into_dyn();
        let _ = manager.augment_images(&input).unwrap();
        let stats = manager.get_statistics();
        assert_eq!(stats.samples_processed, 2);
        assert!(stats.processing_time_ms >= 0.0);
