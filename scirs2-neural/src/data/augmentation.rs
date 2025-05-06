//! Data augmentation for training neural networks

use crate::error::Result;
use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::fmt::Debug;

/// Trait for data augmentation
pub trait Augmentation<F: Float + Debug + ScalarOperand> {
    /// Apply augmentation to the input
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>>;

    /// Get a description of the augmentation
    fn description(&self) -> String;
}

/// Gaussian noise augmentation
#[derive(Debug, Clone)]
pub struct GaussianNoise<F: Float + Debug + ScalarOperand> {
    /// Standard deviation of the noise
    std: F,
}

impl<F: Float + Debug + ScalarOperand> GaussianNoise<F> {
    /// Create a new Gaussian noise augmentation
    pub fn new(std: F) -> Self {
        Self { std }
    }
}

impl<F: Float + Debug + ScalarOperand> Augmentation<F> for GaussianNoise<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = rand::rng();
        let mut result = input.clone();

        for item in result.iter_mut() {
            let noise = F::from(rng.sample::<f64, _>(rand_distr::Normal::new(
                0.0,
                self.std.to_f64().unwrap_or(0.1),
            )))
            .unwrap_or(F::zero());
            *item = *item + noise;
        }

        Ok(result)
    }

    fn description(&self) -> String {
        format!(
            "GaussianNoise (std: {:.3})",
            self.std.to_f64().unwrap_or(0.0)
        )
    }
}

/// Random erasing augmentation
#[derive(Debug, Clone)]
pub struct RandomErasing<F: Float + Debug + ScalarOperand> {
    /// Probability of applying the augmentation
    probability: f64,
    /// Value to use for erasing
    value: F,
}

impl<F: Float + Debug + ScalarOperand> RandomErasing<F> {
    /// Create a new random erasing augmentation
    pub fn new(probability: f64, value: F) -> Self {
        Self { probability, value }
    }
}

impl<F: Float + Debug + ScalarOperand> Augmentation<F> for RandomErasing<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = rand::rng();
        let mut result = input.clone();

        // Only apply augmentation based on probability
        if rng.gen::<f64>() > self.probability {
            return Ok(result);
        }

        // Only apply to 2D or higher arrays (like images)
        if result.ndim() < 3 {
            return Ok(result);
        }

        // Pick a random region to erase
        let h = result.shape()[1];
        let w = result.shape()[2];

        // Determine erase size (10% to 30% of the image)
        let erase_h = ((h as f64) * rng.gen_range(0.1..0.3)) as usize;
        let erase_w = ((w as f64) * rng.gen_range(0.1..0.3)) as usize;

        // Determine erase position
        let i = rng.gen_range(0..(h - erase_h).max(1));
        let j = rng.gen_range(0..(w - erase_w).max(1));

        // Erase the region
        for img in 0..result.shape()[0] {
            for c in 0..result.shape()[3].min(3) {
                // Assuming channel-last format
                for ii in i..(i + erase_h).min(h) {
                    for jj in j..(j + erase_w).min(w) {
                        result[[img, ii, jj, c]] = self.value;
                    }
                }
            }
        }

        Ok(result)
    }

    fn description(&self) -> String {
        format!(
            "RandomErasing (prob: {:.2}, value: {:.2})",
            self.probability,
            self.value.to_f64().unwrap_or(0.0)
        )
    }
}

/// Random horizontal flip augmentation
#[derive(Debug, Clone)]
pub struct RandomHorizontalFlip<F: Float + Debug + ScalarOperand> {
    /// Probability of applying the flip
    probability: f64,
    /// Phantom data for generic type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand> RandomHorizontalFlip<F> {
    /// Create a new random horizontal flip augmentation
    pub fn new(probability: f64) -> Self {
        Self {
            probability,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Debug + ScalarOperand> Augmentation<F> for RandomHorizontalFlip<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = rand::rng();
        let mut result = input.clone();

        // Only apply augmentation based on probability
        if rng.gen::<f64>() > self.probability {
            return Ok(result);
        }

        // Only apply to 2D or higher arrays (like images)
        if result.ndim() < 3 {
            return Ok(result);
        }

        // Flip horizontally (assuming CHW format)
        let w = result.shape()[2];

        for img in 0..result.shape()[0] {
            for c in 0..result.shape()[1] {
                for h in 0..result.shape()[1] {
                    for j in 0..(w / 2) {
                        let temp = result[[img, c, h, j]];
                        result[[img, c, h, j]] = result[[img, c, h, w - 1 - j]];
                        result[[img, c, h, w - 1 - j]] = temp;
                    }
                }
            }
        }

        Ok(result)
    }

    fn description(&self) -> String {
        format!("RandomHorizontalFlip (prob: {:.2})", self.probability)
    }
}

/// Compose multiple augmentations into a single augmentation
#[derive(Debug, Clone)]
pub struct ComposeAugmentation<F: Float + Debug + ScalarOperand> {
    /// List of augmentations to apply in sequence
    augmentations: Vec<Box<dyn Augmentation<F>>>,
}

impl<F: Float + Debug + ScalarOperand> ComposeAugmentation<F> {
    /// Create a new composition of augmentations
    pub fn new(augmentations: Vec<Box<dyn Augmentation<F>>>) -> Self {
        Self { augmentations }
    }
}

impl<F: Float + Debug + ScalarOperand> Augmentation<F> for ComposeAugmentation<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut data = input.clone();

        for augmentation in &self.augmentations {
            data = augmentation.apply(&data)?;
        }

        Ok(data)
    }

    fn description(&self) -> String {
        let descriptions: Vec<String> =
            self.augmentations.iter().map(|a| a.description()).collect();

        format!("Compose({})", descriptions.join(", "))
    }
}
