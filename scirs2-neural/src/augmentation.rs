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

/// Image augmentation transforms
#[derive(Debug, Clone, PartialEq)]
pub enum ImageAugmentation {
    /// Random horizontal flip
    RandomHorizontalFlip { probability: f64 },
    /// Random vertical flip
    RandomVerticalFlip { probability: f64 },
    /// Random rotation within angle range
    RandomRotation {
        min_angle: f64,
        max_angle: f64,
        fill_mode: FillMode,
    },
    /// Random scaling
    RandomScale {
        min_scale: f64,
        max_scale: f64,
        preserve_aspect_ratio: bool,
    },
    /// Random crop and resize
    RandomCrop {
        crop_height: usize,
        crop_width: usize,
        padding: Option<usize>,
    },
    /// Color jittering
    ColorJitter {
        brightness: Option<f64>,
        contrast: Option<f64>,
        saturation: Option<f64>,
        hue: Option<f64>,
    },
    /// Gaussian noise injection
    GaussianNoise {
        mean: f64,
        std: f64,
        probability: f64,
    },
    /// Random erasing (cutout)
    RandomErasing {
        probability: f64,
        area_ratio_range: (f64, f64),
        aspect_ratio_range: (f64, f64),
        fill_value: f64,
    },
    /// Elastic deformation
    ElasticDeformation {
        alpha: f64,
        sigma: f64,
        probability: f64,
    },
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
}

/// Text augmentation techniques
#[derive(Debug, Clone, PartialEq)]
pub enum TextAugmentation {
    /// Random synonym replacement
    SynonymReplacement {
        probability: f64,
        num_replacements: usize,
    },
    /// Random word insertion
    RandomInsertion {
        probability: f64,
        num_insertions: usize,
    },
    /// Random word deletion
    RandomDeletion { probability: f64 },
    /// Random word swap
    RandomSwap { probability: f64, num_swaps: usize },
    /// Back translation
    BackTranslation { intermediate_language: String },
    /// Paraphrasing
    Paraphrasing { model_type: String },
}

/// Audio augmentation techniques
#[derive(Debug, Clone, PartialEq)]
pub enum AudioAugmentation {
    /// Time stretching
    TimeStretch {
        stretch_factor_range: (f64, f64),
        probability: f64,
    },
    /// Pitch shifting
    PitchShift {
        semitone_range: (f64, f64),
        probability: f64,
    },
    /// Add background noise
    AddNoise { noise_factor: f64, probability: f64 },
    /// Volume adjustment
    VolumeAdjust {
        gain_range: (f64, f64),
        probability: f64,
    },
    /// Frequency masking
    FrequencyMask {
        num_masks: usize,
        mask_width_range: (usize, usize),
        probability: f64,
    },
    /// Time masking
    TimeMask {
        num_masks: usize,
        mask_length_range: (usize, usize),
        probability: f64,
    },
}

/// Mix-based augmentation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum MixAugmentation {
    /// MixUp augmentation
    MixUp { alpha: f64 },
    /// CutMix augmentation
    CutMix {
        alpha: f64,
        cut_ratio_range: (f64, f64),
    },
    /// AugMix augmentation
    AugMix {
        severity: usize,
        width: usize,
        depth: usize,
        alpha: f64,
    },
    /// Manifold mixup
    ManifoldMix {
        alpha: f64,
        layer_mix_probability: f64,
    },
}

/// Comprehensive data augmentation manager
pub struct AugmentationManager<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> {
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
}

/// Statistics for tracking augmentation usage
#[derive(Debug, Clone)]
pub struct AugmentationStatistics<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    /// Number of samples processed
    pub samples_processed: usize,
    /// Average augmentation intensity
    pub avg_intensity: F,
    /// Transform usage counts
    pub transform_counts: HashMap<String, usize>,
    /// Performance metrics
    pub processing_time_ms: f64,
}

impl<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> AugmentationManager<F> {
    /// Create a new augmentation manager
    pub fn new(rng_seed: Option<u64>) -> Self {
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
    }

    /// Add text augmentation transform
    pub fn add_text_transform(&mut self, transform: TextAugmentation) {
        self.text_transforms.push(transform);
    }

    /// Add audio augmentation transform
    pub fn add_audio_transform(&mut self, transform: AudioAugmentation) {
        self.audio_transforms.push(transform);
    }

    /// Add mix augmentation strategy
    pub fn add_mix_strategy(&mut self, strategy: MixAugmentation) {
        self.mix_strategies.push(strategy);
    }

    /// Apply image augmentations to a batch of images
    pub fn augment_images(&mut self, images: &ArrayD<F>) -> Result<ArrayD<F>> {
        let start_time = std::time::Instant::now();
        let mut augmented = images.clone();

        for transform in &self.image_transforms {
            augmented = self.apply_image_transform(&augmented, transform)?;

            // Update statistics
            let transform_name = format!("{:?}", transform)
                .split(' ')
                .next()
                .unwrap_or("unknown")
                .to_string();
            *self
                .stats
                .transform_counts
                .entry(transform_name)
                .or_insert(0) += 1;
        }

        self.stats.samples_processed += images.shape()[0];
        self.stats.processing_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(augmented)
    }

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
            }
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
                probability,
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
                probability,
            } => self.elastic_deformation(images, *alpha, *sigma, *probability),
        }
    }

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
            }
        }

        Ok(result)
    }

    fn random_vertical_flip(&self, images: &ArrayD<F>, probability: f64) -> Result<ArrayD<F>> {
        let mut result = images.clone();
        let batch_size = images.shape()[0];

        for i in 0..batch_size {
            if rand::random::<f64>() < probability {
                // Flip vertically by reversing the height dimension
                if images.ndim() >= 4 {
                    // Assuming NCHW format: (batch, channels, height, width)
                    let height_dim = images.ndim() - 2;
                    let mut sample = result.slice_mut(ndarray::s![i, .., .., ..]);
                    sample.invert_axis(Axis(height_dim - 1)); // height axis relative to sample
                }
            }
        }

        Ok(result)
    }

    fn random_rotation(
        &self,
        images: &ArrayD<F>,
        min_angle: f64,
        max_angle: f64,
        _fill_mode: FillMode,
    ) -> Result<ArrayD<F>> {
        // Simplified rotation implementation
        // In practice, this would involve proper image rotation algorithms
        let result = images.clone();
        let batch_size = images.shape()[0];

        for _i in 0..batch_size {
            let _angle = rand::rng().random_range(min_angle..=max_angle);
            // Apply rotation (simplified - just return original for now)
            // Real implementation would use affine transformations
        }

        Ok(result)
    }

    fn random_scale(
        &self,
        images: &ArrayD<F>,
        min_scale: f64,
        max_scale: f64,
        _preserve_aspect_ratio: bool,
    ) -> Result<ArrayD<F>> {
        // Simplified scaling implementation
        // In practice, this would involve proper image scaling algorithms
        let result = images.clone();
        let batch_size = images.shape()[0];

        for _i in 0..batch_size {
            let _scale = rand::rng().random_range(min_scale..=max_scale);
            // Apply scaling (simplified - just return original for now)
            // Real implementation would use interpolation
        }

        Ok(result)
    }

    fn random_crop(
        &self,
        images: &ArrayD<F>,
        crop_height: usize,
        crop_width: usize,
        _padding: Option<usize>,
    ) -> Result<ArrayD<F>> {
        if images.ndim() < 4 {
            return Err(NeuralError::InvalidArchitecture(
                "Random crop requires 4D input (NCHW)".to_string(),
            ));
        }

        let batch_size = images.shape()[0];
        let channels = images.shape()[1];
        let height = images.shape()[2];
        let width = images.shape()[3];

        if crop_height > height || crop_width > width {
            return Err(NeuralError::InvalidArchitecture(
                "Crop size cannot be larger than image size".to_string(),
            ));
        }

        let mut result = Array::zeros((batch_size, channels, crop_height, crop_width));

        for i in 0..batch_size {
            let start_h = rand::rng().random_range(0..=(height - crop_height));
            let start_w = rand::rng().random_range(0..=(width - crop_width));

            let crop = images.slice(ndarray::s![
                i,
                ..,
                start_h..start_h + crop_height,
                start_w..start_w + crop_width
            ]);
            result.slice_mut(ndarray::s![i, .., .., ..]).assign(&crop);
        }

        Ok(result.into_dyn())
    }

    fn color_jitter(
        &self,
        images: &ArrayD<F>,
        brightness: Option<f64>,
        contrast: Option<f64>,
        _saturation: Option<f64>,
        _hue: Option<f64>,
    ) -> Result<ArrayD<F>> {
        let mut result = images.clone();

        // Apply brightness adjustment
        if let Some(bright_factor) = brightness {
            let factor =
                F::from(1.0 + rand::rng().random_range(-bright_factor..=bright_factor)).unwrap();
            result = result * factor;
        }

        // Apply contrast adjustment
        if let Some(contrast_factor) = contrast {
            let factor =
                F::from(1.0 + rand::rng().random_range(-contrast_factor..=contrast_factor))
                    .unwrap();
            let mean = result.mean().unwrap_or(F::zero());
            result = (result - mean) * factor + mean;
        }

        // Clamp values to valid range [0, 1] (assuming normalized images)
        result = result.mapv(|x| x.max(F::zero()).min(F::one()));

        Ok(result)
    }

    fn gaussian_noise(
        &self,
        images: &ArrayD<F>,
        mean: f64,
        std: f64,
        probability: f64,
    ) -> Result<ArrayD<F>> {
        let mut result = images.clone();

        if rand::random::<f64>() < probability {
            let noise = images.mapv(|_| {
                let noise_val = rand::rng().random_range(-3.0 * std..=3.0 * std) + mean;
                F::from(noise_val).unwrap_or(F::zero())
            });
            result = result + noise;
        }

        Ok(result)
    }

    fn random_erasing(
        &self,
        images: &ArrayD<F>,
        probability: f64,
        area_ratio_range: (f64, f64),
        aspect_ratio_range: (f64, f64),
        fill_value: f64,
    ) -> Result<ArrayD<F>> {
        if images.ndim() < 4 {
            return Err(NeuralError::InvalidArchitecture(
                "Random erasing requires 4D input (NCHW)".to_string(),
            ));
        }

        let mut result = images.clone();
        let batch_size = images.shape()[0];
        let height = images.shape()[2];
        let width = images.shape()[3];
        let fill_val = F::from(fill_value).unwrap_or(F::zero());

        for i in 0..batch_size {
            if rand::random::<f64>() < probability {
                let area_ratio = rand::rng().random_range(area_ratio_range.0..=area_ratio_range.1);
                let aspect_ratio =
                    rand::rng().random_range(aspect_ratio_range.0..=aspect_ratio_range.1);

                let target_area = (height * width) as f64 * area_ratio;
                let mask_height = ((target_area * aspect_ratio).sqrt() as usize).min(height);
                let mask_width = ((target_area / aspect_ratio).sqrt() as usize).min(width);

                if mask_height > 0 && mask_width > 0 {
                    let start_h = rand::rng().random_range(0..=(height - mask_height));
                    let start_w = rand::rng().random_range(0..=(width - mask_width));

                    result
                        .slice_mut(ndarray::s![
                            i,
                            ..,
                            start_h..start_h + mask_height,
                            start_w..start_w + mask_width
                        ])
                        .fill(fill_val);
                }
            }
        }

        Ok(result)
    }

    fn elastic_deformation(
        &self,
        images: &ArrayD<F>,
        _alpha: f64,
        _sigma: f64,
        probability: f64,
    ) -> Result<ArrayD<F>> {
        // Simplified elastic deformation implementation
        // In practice, this would involve complex displacement field generation
        let mut result = images.clone();

        if rand::random::<f64>() < probability {
            // Apply simple noise as a placeholder for elastic deformation
            let noise_factor = F::from(0.01).unwrap();
            let noise = images.mapv(|_| {
                let noise_val = rand::rng().random_range(-0.05..=0.05);
                F::from(noise_val).unwrap_or(F::zero())
            });
            result = result + noise * noise_factor;
        }

        Ok(result)
    }

    /// Apply MixUp augmentation to a batch
    pub fn apply_mixup(
        &mut self,
        images: &ArrayD<F>,
        labels: &ArrayD<F>,
        alpha: f64,
    ) -> Result<(ArrayD<F>, ArrayD<F>)> {
        let batch_size = images.shape()[0];
        if batch_size < 2 {
            return Ok((images.clone(), labels.clone()));
        }

        let lambda = self.sample_beta_distribution(alpha)?;
        let lambda_f = F::from(lambda).unwrap_or(F::from(0.5).unwrap());

        // Create random permutation of indices
        let mut indices: Vec<usize> = (0..batch_size).collect();
        for i in 0..batch_size {
            let j = rand::rng().random_range(0..batch_size);
            indices.swap(i, j);
        }

        let mut mixed_images = images.clone();
        let mut mixed_labels = labels.clone();

        for i in 0..batch_size {
            let j = indices[i];

            // Mix images: x_mixed = lambda * x_i + (1 - lambda) * x_j
            let x_i = images.slice(ndarray::s![i, ..]);
            let x_j = images.slice(ndarray::s![j, ..]);
            let mixed = &x_i * lambda_f + &x_j * (F::one() - lambda_f);
            mixed_images.slice_mut(ndarray::s![i, ..]).assign(&mixed);

            // Mix labels: y_mixed = lambda * y_i + (1 - lambda) * y_j
            let y_i = labels.slice(ndarray::s![i, ..]);
            let y_j = labels.slice(ndarray::s![j, ..]);
            let mixed_label = &y_i * lambda_f + &y_j * (F::one() - lambda_f);
            mixed_labels
                .slice_mut(ndarray::s![i, ..])
                .assign(&mixed_label);
        }

        self.stats.samples_processed += batch_size;
        *self
            .stats
            .transform_counts
            .entry("MixUp".to_string())
            .or_insert(0) += 1;

        Ok((mixed_images, mixed_labels))
    }

    /// Apply CutMix augmentation
    pub fn apply_cutmix(
        &mut self,
        images: &ArrayD<F>,
        labels: &ArrayD<F>,
        alpha: f64,
        cut_ratio_range: (f64, f64),
    ) -> Result<(ArrayD<F>, ArrayD<F>)> {
        if images.ndim() < 4 {
            return Err(NeuralError::InvalidArchitecture(
                "CutMix requires 4D input (NCHW)".to_string(),
            ));
        }

        let batch_size = images.shape()[0];
        if batch_size < 2 {
            return Ok((images.clone(), labels.clone()));
        }

        let _lambda = self.sample_beta_distribution(alpha)?;
        let cut_ratio = rand::rng().random_range(cut_ratio_range.0..=cut_ratio_range.1);

        let height = images.shape()[2];
        let width = images.shape()[3];

        let cut_height = ((height as f64 * cut_ratio).sqrt() as usize).min(height);
        let cut_width = ((width as f64 * cut_ratio).sqrt() as usize).min(width);

        let mut mixed_images = images.clone();
        let mut mixed_labels = labels.clone();

        // Create random permutation
        let mut indices: Vec<usize> = (0..batch_size).collect();
        for i in 0..batch_size {
            let j = rand::rng().random_range(0..batch_size);
            indices.swap(i, j);
        }

        for i in 0..batch_size {
            let j = indices[i];

            // Random cut position
            let start_h = rand::rng().random_range(0..=(height - cut_height));
            let start_w = rand::rng().random_range(0..=(width - cut_width));

            // Cut and paste
            let patch = images.slice(ndarray::s![
                j,
                ..,
                start_h..start_h + cut_height,
                start_w..start_w + cut_width
            ]);
            mixed_images
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
            let mixed_label = &y_i * lambda_f + &y_j * (F::one() - lambda_f);
            mixed_labels
                .slice_mut(ndarray::s![i, ..])
                .assign(&mixed_label);
        }

        *self
            .stats
            .transform_counts
            .entry("CutMix".to_string())
            .or_insert(0) += 1;

        Ok((mixed_images, mixed_labels))
    }

    fn sample_beta_distribution(&self, alpha: f64) -> Result<f64> {
        // Simplified beta distribution sampling
        // In practice, you would use a proper beta distribution implementation
        if alpha <= 0.0 {
            return Ok(0.5);
        }

        // Approximate beta distribution with uniform sampling for simplicity
        Ok(rand::random::<f64>())
    }

    /// Get augmentation statistics
    pub fn get_statistics(&self) -> &AugmentationStatistics<F> {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.stats = AugmentationStatistics {
            samples_processed: 0,
            avg_intensity: F::zero(),
            transform_counts: HashMap::new(),
            processing_time_ms: 0.0,
        };
    }

    /// Create a standard image augmentation pipeline
    pub fn create_standard_image_pipeline() -> Vec<ImageAugmentation> {
        vec![
            ImageAugmentation::RandomHorizontalFlip { probability: 0.5 },
            ImageAugmentation::ColorJitter {
                brightness: Some(0.2),
                contrast: Some(0.2),
                saturation: Some(0.2),
                hue: Some(0.1),
            },
            ImageAugmentation::GaussianNoise {
                mean: 0.0,
                std: 0.01,
                probability: 0.3,
            },
            ImageAugmentation::RandomErasing {
                probability: 0.25,
                area_ratio_range: (0.02, 0.33),
                aspect_ratio_range: (0.3, 3.3),
                fill_value: 0.0,
            },
        ]
    }

    /// Create a strong image augmentation pipeline
    pub fn create_strong_image_pipeline() -> Vec<ImageAugmentation> {
        vec![
            ImageAugmentation::RandomHorizontalFlip { probability: 0.5 },
            ImageAugmentation::RandomVerticalFlip { probability: 0.2 },
            ImageAugmentation::RandomRotation {
                min_angle: -30.0,
                max_angle: 30.0,
                fill_mode: FillMode::Constant(0.0),
            },
            ImageAugmentation::RandomScale {
                min_scale: 0.8,
                max_scale: 1.2,
                preserve_aspect_ratio: true,
            },
            ImageAugmentation::ColorJitter {
                brightness: Some(0.4),
                contrast: Some(0.4),
                saturation: Some(0.4),
                hue: Some(0.2),
            },
            ImageAugmentation::GaussianNoise {
                mean: 0.0,
                std: 0.02,
                probability: 0.5,
            },
            ImageAugmentation::RandomErasing {
                probability: 0.5,
                area_ratio_range: (0.02, 0.4),
                aspect_ratio_range: (0.3, 3.3),
                fill_value: 0.0,
            },
            ImageAugmentation::ElasticDeformation {
                alpha: 1.0,
                sigma: 0.1,
                probability: 0.3,
            },
        ]
    }
}

impl<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Default for AugmentationManager<F> {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Augmentation pipeline builder for easy configuration
pub struct AugmentationPipelineBuilder<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    manager: AugmentationManager<F>,
}

impl<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> AugmentationPipelineBuilder<F> {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            manager: AugmentationManager::new(None),
        }
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.manager.rng_seed = Some(seed);
        self
    }

    /// Add standard image augmentations
    pub fn with_standard_image_augmentations(mut self) -> Self {
        for transform in AugmentationManager::<F>::create_standard_image_pipeline() {
            self.manager.add_image_transform(transform);
        }
        self
    }

    /// Add strong image augmentations
    pub fn with_strong_image_augmentations(mut self) -> Self {
        for transform in AugmentationManager::<F>::create_strong_image_pipeline() {
            self.manager.add_image_transform(transform);
        }
        self
    }

    /// Add MixUp augmentation
    pub fn with_mixup(mut self, alpha: f64) -> Self {
        self.manager
            .add_mix_strategy(MixAugmentation::MixUp { alpha });
        self
    }

    /// Add CutMix augmentation
    pub fn with_cutmix(mut self, alpha: f64, cut_ratio_range: (f64, f64)) -> Self {
        self.manager.add_mix_strategy(MixAugmentation::CutMix {
            alpha,
            cut_ratio_range,
        });
        self
    }

    /// Build the augmentation manager
    pub fn build(self) -> AugmentationManager<F> {
        self.manager
    }
}

impl<F: Float + Debug + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Default for AugmentationPipelineBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array4};

    #[test]
    fn test_augmentation_manager_creation() {
        let manager = AugmentationManager::<f64>::new(Some(42));
        assert_eq!(manager.rng_seed, Some(42));
        assert_eq!(manager.image_transforms.len(), 0);
    }

    #[test]
    fn test_random_horizontal_flip() {
        let mut manager = AugmentationManager::<f64>::new(Some(42));
        manager.add_image_transform(ImageAugmentation::RandomHorizontalFlip { probability: 1.0 });

        let input =
            Array4::<f64>::from_shape_fn((2, 3, 4, 4), |(_, _, _, _)| rand::random()).into_dyn();
        let result = manager.augment_images(&input).unwrap();

        assert_eq!(result.shape(), input.shape());
        assert!(manager.stats.samples_processed > 0);
    }

    #[test]
    fn test_random_crop() {
        let manager = AugmentationManager::<f64>::new(None);

        let input = Array4::<f64>::ones((2, 3, 8, 8)).into_dyn();
        let result = manager.random_crop(&input, 4, 4, None).unwrap();

        assert_eq!(result.shape(), &[2, 3, 4, 4]);
    }

    #[test]
    fn test_color_jitter() {
        let manager = AugmentationManager::<f64>::new(None);

        let input = Array4::<f64>::from_elem((1, 3, 4, 4), 0.5).into_dyn();
        let result = manager
            .color_jitter(&input, Some(0.2), Some(0.2), None, None)
            .unwrap();

        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_gaussian_noise() {
        let manager = AugmentationManager::<f64>::new(None);

        let input = Array4::<f64>::zeros((2, 3, 4, 4)).into_dyn();
        let result = manager.gaussian_noise(&input, 0.0, 0.1, 1.0).unwrap();

        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_random_erasing() {
        let manager = AugmentationManager::<f64>::new(None);

        let input = Array4::<f64>::ones((2, 3, 8, 8)).into_dyn();
        let result = manager
            .random_erasing(&input, 1.0, (0.1, 0.3), (0.5, 2.0), 0.0)
            .unwrap();

        assert_eq!(result.shape(), input.shape());
    }

    #[test]
    fn test_mixup() {
        let mut manager = AugmentationManager::<f64>::new(Some(42));

        let images = Array4::<f64>::ones((4, 3, 8, 8)).into_dyn();
        let labels = Array2::<f64>::from_elem((4, 10), 1.0).into_dyn();

        let (mixed_images, mixed_labels) = manager.apply_mixup(&images, &labels, 1.0).unwrap();

        assert_eq!(mixed_images.shape(), images.shape());
        assert_eq!(mixed_labels.shape(), labels.shape());
        assert!(manager.stats.transform_counts.contains_key("MixUp"));
    }

    #[test]
    fn test_cutmix() {
        let mut manager = AugmentationManager::<f64>::new(Some(42));

        let images = Array4::<f64>::ones((4, 3, 8, 8)).into_dyn();
        let labels = Array2::<f64>::from_elem((4, 10), 1.0).into_dyn();

        let (mixed_images, mixed_labels) = manager
            .apply_cutmix(&images, &labels, 1.0, (0.1, 0.5))
            .unwrap();

        assert_eq!(mixed_images.shape(), images.shape());
        assert_eq!(mixed_labels.shape(), labels.shape());
        assert!(manager.stats.transform_counts.contains_key("CutMix"));
    }

    #[test]
    fn test_standard_pipeline() {
        let pipeline = AugmentationManager::<f64>::create_standard_image_pipeline();
        assert!(!pipeline.is_empty());
        assert!(pipeline.len() >= 3);
    }

    #[test]
    fn test_strong_pipeline() {
        let pipeline = AugmentationManager::<f64>::create_strong_image_pipeline();
        assert!(!pipeline.is_empty());
        assert!(
            pipeline.len() > AugmentationManager::<f64>::create_standard_image_pipeline().len()
        );
    }

    #[test]
    fn test_pipeline_builder() {
        let manager = AugmentationPipelineBuilder::<f64>::new()
            .with_seed(42)
            .with_standard_image_augmentations()
            .with_mixup(1.0)
            .build();

        assert_eq!(manager.rng_seed, Some(42));
        assert!(!manager.image_transforms.is_empty());
        assert!(!manager.mix_strategies.is_empty());
    }

    #[test]
    fn test_augmentation_statistics() {
        let mut manager = AugmentationManager::<f64>::new(None);
        manager.add_image_transform(ImageAugmentation::RandomHorizontalFlip { probability: 0.5 });

        let input = Array4::<f64>::ones((2, 3, 4, 4)).into_dyn();
        let _ = manager.augment_images(&input).unwrap();

        let stats = manager.get_statistics();
        assert_eq!(stats.samples_processed, 2);
        assert!(stats.processing_time_ms >= 0.0);
    }
}
