//! ImageNet-100-class synthetic image classification dataset generator.
//!
//! Generates a synthetic dataset mimicking ImageNet's structure:
//! - 100 classes with configurable samples per class
//! - Images stored as `Array4<f32>` in NCHW format, normalised to `[0, 1]`
//! - Each class has a distinct mean colour/texture; images = mean + Normal(0, 0.1) noise
//! - Class names `class_000` through `class_099`

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{s, Array1, Array3, Array4, ArrayView3};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

/// Number of ImageNet-100 classes (fixed constant).
pub const IMAGENET100_N_CLASSES: usize = 100;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the ImageNet-100 synthetic dataset generator.
#[derive(Debug, Clone)]
pub struct ImageNet100Config {
    /// Number of samples per class (default: 10).
    pub n_samples_per_class: usize,
    /// Spatial dimension of each square image in pixels (default: 64).
    ///
    /// Images are stored as `[C, H, W]` with `C = 3`.  Use a small value
    /// (e.g. 32 or 64) in tests to avoid excessive memory use.
    pub image_size: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for ImageNet100Config {
    fn default() -> Self {
        Self {
            n_samples_per_class: 10,
            image_size: 64,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ImageNet100Dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Synthetic ImageNet-100-class image classification dataset.
///
/// Images have shape `[N, 3, H, W]` where `N = n_samples_per_class × 100`.
/// All pixel values are clamped to `[0, 1]`.
#[derive(Debug, Clone)]
pub struct ImageNet100Dataset {
    /// `[N, C, H, W]` float array, values in `[0, 1]`.
    images: Array4<f32>,
    /// Class index for each sample (0..99).
    labels: Array1<u32>,
    /// Human-readable class names (`class_000`..`class_099`).
    class_names: Vec<String>,
    config: ImageNet100Config,
}

impl ImageNet100Dataset {
    /// Generate a synthetic ImageNet-100 dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or distribution
    /// construction fails.
    pub fn generate(config: ImageNet100Config) -> Result<Self> {
        if config.n_samples_per_class == 0 {
            return Err(DatasetsError::InvalidFormat(
                "ImageNet100Config: n_samples_per_class must be > 0".to_string(),
            ));
        }
        if config.image_size == 0 {
            return Err(DatasetsError::InvalidFormat(
                "ImageNet100Config: image_size must be > 0".to_string(),
            ));
        }

        let n_total = config.n_samples_per_class * IMAGENET100_N_CLASSES;
        let h = config.image_size;
        let w = config.image_size;

        let mut rng = StdRng::seed_from_u64(config.seed);
        let noise_dist = Normal::new(0.0_f32, 0.1_f32).map_err(|e| {
            DatasetsError::ComputationError(format!("Normal dist construction failed: {e}"))
        })?;

        // Per-class mean colours (RGB), distinct for each class
        // Use evenly-spaced hues to give visual separation
        let class_means: Vec<[f32; 3]> = (0..IMAGENET100_N_CLASSES)
            .map(|c| {
                // Map class index to a unique colour in [0.1, 0.9]^3
                let hue = c as f32 / IMAGENET100_N_CLASSES as f32; // 0..1
                                                                   // Convert hue to approximate RGB (simplified)
                let r = (hue * 6.0).sin().abs() * 0.8 + 0.1;
                let g = ((hue + 0.333) * 6.0).sin().abs() * 0.8 + 0.1;
                let b = ((hue + 0.667) * 6.0).sin().abs() * 0.8 + 0.1;
                [r, g, b]
            })
            .collect();

        let mut images = Array4::zeros((n_total, 3, h, w));
        let mut labels = Array1::zeros(n_total);
        let class_names: Vec<String> = (0..IMAGENET100_N_CLASSES)
            .map(|c| format!("class_{c:03}"))
            .collect();

        for (class_id, mean_rgb) in class_means.iter().enumerate() {
            for sample_in_class in 0..config.n_samples_per_class {
                let sample_idx = class_id * config.n_samples_per_class + sample_in_class;
                labels[sample_idx] = class_id as u32;
                for (c, &mean) in mean_rgb.iter().enumerate() {
                    for row in 0..h {
                        for col in 0..w {
                            let noise: f32 = noise_dist.sample(&mut rng);
                            let pixel = (mean + noise).clamp(0.0, 1.0);
                            images[[sample_idx, c, row, col]] = pixel;
                        }
                    }
                }
            }
        }

        Ok(Self {
            images,
            labels,
            class_names,
            config,
        })
    }

    /// All images as a 4-D array of shape `[N, 3, H, W]`, values in `[0, 1]`.
    pub fn images(&self) -> &Array4<f32> {
        &self.images
    }

    /// Class label for each sample (values in `0..99`).
    pub fn labels(&self) -> &Array1<u32> {
        &self.labels
    }

    /// Human-readable class names (`class_000`..`class_099`).
    pub fn class_names(&self) -> &[String] {
        &self.class_names
    }

    /// Number of classes (always 100).
    pub fn n_classes(&self) -> usize {
        IMAGENET100_N_CLASSES
    }

    /// Total number of samples (`n_samples_per_class × 100`).
    pub fn n_samples(&self) -> usize {
        self.config.n_samples_per_class * IMAGENET100_N_CLASSES
    }

    /// Return the image and label at position `idx`.
    ///
    /// The returned `ArrayView3<'_, f32>` has shape `[3, H, W]` and is a view
    /// into the backing array, borrowing from `&self`.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.n_samples()`.
    pub fn get_sample(&self, idx: usize) -> (ArrayView3<'_, f32>, u32) {
        let view: ArrayView3<'_, f32> = self.images.slice(s![idx, .., .., ..]);
        (view, self.labels[idx])
    }

    /// Return a standalone `Array3<f32>` copy for sample `idx`.
    ///
    /// Shape `[3, H, W]`, values in `[0, 1]`.
    ///
    /// Returns an error if `idx >= self.n_samples()`.
    pub fn get_sample_owned(&self, idx: usize) -> Result<(Array3<f32>, u32)> {
        if idx >= self.n_samples() {
            return Err(DatasetsError::InvalidFormat(format!(
                "ImageNet100Dataset: index {idx} out of bounds (n_samples = {})",
                self.n_samples()
            )));
        }
        let view = self.images.slice(s![idx, .., .., ..]);
        Ok((view.to_owned(), self.labels[idx]))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Smaller config for tests to keep memory usage low.
    fn small_config() -> ImageNet100Config {
        ImageNet100Config {
            n_samples_per_class: 2,
            image_size: 8,
            seed: 42,
        }
    }

    #[test]
    fn test_imagenet100_shape() {
        let cfg = small_config();
        let ds = ImageNet100Dataset::generate(cfg.clone()).expect("generate failed");
        let n = cfg.n_samples_per_class * IMAGENET100_N_CLASSES;
        assert_eq!(ds.n_samples(), n);
        assert_eq!(ds.n_classes(), IMAGENET100_N_CLASSES);
        let imgs = ds.images();
        assert_eq!(imgs.shape(), &[n, 3, cfg.image_size, cfg.image_size]);
        assert_eq!(ds.labels().len(), n);
    }

    #[test]
    fn test_imagenet100_deterministic() {
        let cfg = small_config();
        let ds1 = ImageNet100Dataset::generate(cfg.clone()).expect("generate failed");
        let ds2 = ImageNet100Dataset::generate(cfg).expect("generate failed");
        assert_eq!(ds1.images(), ds2.images());
        assert_eq!(ds1.labels(), ds2.labels());
    }

    #[test]
    fn test_imagenet100_pixel_range() {
        let cfg = small_config();
        let ds = ImageNet100Dataset::generate(cfg).expect("generate failed");
        let imgs = ds.images();
        let imgs_ref = imgs.view();
        let slice = imgs_ref.as_slice().expect("contiguous");
        for &v in slice {
            assert!((0.0..=1.0).contains(&v), "pixel value {v} out of [0,1]");
            assert!(!v.is_nan(), "NaN pixel found");
        }
    }

    #[test]
    fn test_imagenet100_labels_in_range() {
        let cfg = small_config();
        let ds = ImageNet100Dataset::generate(cfg).expect("generate failed");
        for &label in ds.labels().iter() {
            assert!(
                (label as usize) < IMAGENET100_N_CLASSES,
                "label {label} out of range"
            );
        }
    }

    #[test]
    fn test_imagenet100_class_names() {
        let cfg = small_config();
        let ds = ImageNet100Dataset::generate(cfg).expect("generate failed");
        let names = ds.class_names();
        assert_eq!(names.len(), IMAGENET100_N_CLASSES);
        assert_eq!(names[0], "class_000");
        assert_eq!(names[99], "class_099");
    }

    #[test]
    fn test_imagenet100_get_sample() {
        let cfg = small_config();
        let ds = ImageNet100Dataset::generate(cfg.clone()).expect("generate failed");
        let (view, label) = ds.get_sample(0);
        assert_eq!(view.shape(), &[3, cfg.image_size, cfg.image_size]);
        assert_eq!(label, 0u32); // first sample is class 0
    }

    #[test]
    fn test_imagenet100_get_sample_owned() {
        let cfg = small_config();
        let ds = ImageNet100Dataset::generate(cfg.clone()).expect("generate failed");
        let (arr, label) = ds.get_sample_owned(0).expect("get_sample_owned failed");
        assert_eq!(arr.shape(), &[3, cfg.image_size, cfg.image_size]);
        assert_eq!(label, 0u32);
    }

    #[test]
    fn test_imagenet100_error_zero_samples_per_class() {
        let cfg = ImageNet100Config {
            n_samples_per_class: 0,
            ..ImageNet100Config::default()
        };
        assert!(ImageNet100Dataset::generate(cfg).is_err());
    }

    #[test]
    fn test_imagenet100_get_sample_owned_out_of_bounds() {
        let cfg = small_config();
        let ds = ImageNet100Dataset::generate(cfg.clone()).expect("generate failed");
        assert!(ds.get_sample_owned(ds.n_samples()).is_err());
    }

    #[test]
    fn test_imagenet100_labels_balanced() {
        let cfg = ImageNet100Config {
            n_samples_per_class: 3,
            image_size: 4,
            seed: 1,
        };
        let ds = ImageNet100Dataset::generate(cfg).expect("generate failed");
        let mut counts = vec![0u32; IMAGENET100_N_CLASSES];
        for &lbl in ds.labels().iter() {
            counts[lbl as usize] += 1;
        }
        for (cls, &cnt) in counts.iter().enumerate() {
            assert_eq!(cnt, 3, "class {cls} should have exactly 3 samples");
        }
    }
}
