//! Data augmentation for training neural networks

use crate::error::Result;
use scirs2_core::ndarray::{Array, Axis, IxDyn, ScalarOperand, Slice};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, RngExt, SeedableRng};
use std::fmt::Debug;

/// Trait for data augmentation
pub trait Augmentation<F: Float + NumAssign + Debug + ScalarOperand> {
    /// Apply augmentation to the input
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>>;

    /// Get a description of the augmentation
    fn description(&self) -> String;

    /// Clone this augmentation into a boxed trait object.
    ///
    /// This enables `ComposeAugmentation` (which stores `Box<dyn Augmentation>`)
    /// to be cloned without losing its contents.
    fn clone_box(&self) -> Box<dyn Augmentation<F>>;
}

/// Gaussian noise augmentation
#[derive(Debug, Clone)]
pub struct GaussianNoise<F: Float + NumAssign + Debug + ScalarOperand> {
    /// Standard deviation of the noise
    std: F,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> GaussianNoise<F> {
    /// Create a new Gaussian noise augmentation
    pub fn new(std: F) -> Self {
        Self { std }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand + 'static> Augmentation<F> for GaussianNoise<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = SmallRng::from_rng(&mut thread_rng());
        let mut result = input.clone();

        // Create a normal distribution once for the whole tensor
        let normal = scirs2_core::random::Normal::new(0.0, self.std.to_f64().unwrap_or(0.1))
            .expect("Failed to create normal distribution");

        for item in result.iter_mut() {
            // Sample from the distribution
            let noise = F::from(rng.sample(normal)).unwrap_or(F::zero());
            *item += noise;
        }

        Ok(result)
    }

    fn description(&self) -> String {
        format!(
            "GaussianNoise (std: {:.3})",
            self.std.to_f64().unwrap_or(0.0)
        )
    }

    fn clone_box(&self) -> Box<dyn Augmentation<F>> {
        Box::new(self.clone())
    }
}

/// Random erasing augmentation
#[derive(Debug, Clone)]
pub struct RandomErasing<F: Float + NumAssign + Debug + ScalarOperand> {
    /// Probability of applying the augmentation
    probability: f64,
    /// Value to use for erasing
    value: F,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> RandomErasing<F> {
    /// Create a new random erasing augmentation
    pub fn new(probability: f64, value: F) -> Self {
        Self { probability, value }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand + 'static> Augmentation<F> for RandomErasing<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = SmallRng::from_rng(&mut thread_rng());
        let mut result = input.clone();

        // Only apply augmentation based on probability
        if rng.random::<f64>() > self.probability {
            return Ok(result);
        }

        // Only apply to 3D or higher arrays (like images with channels).
        // The last two axes are treated as the spatial (height, width) plane.
        let ndim = result.ndim();
        if ndim < 3 {
            return Ok(result);
        }

        let shape = result.shape().to_vec();
        let height = shape[ndim - 2];
        let width = shape[ndim - 1];
        if height == 0 || width == 0 {
            return Ok(result);
        }

        // Sample an erasing rectangle covering between ~2% and ~40% of the area,
        // following the spirit of Zhong et al. (2017) "Random Erasing".
        let area = (height * width) as f64;
        let target_area = area * (0.02 + rng.random::<f64>() * 0.38);
        let aspect = 0.3 + rng.random::<f64>() * (3.0 - 0.3);
        let mut rect_h = ((target_area * aspect).sqrt().round() as usize).max(1);
        let mut rect_w = ((target_area / aspect).sqrt().round() as usize).max(1);
        rect_h = rect_h.min(height);
        rect_w = rect_w.min(width);

        let top =
            ((rng.random::<f64>() * (height - rect_h + 1) as f64) as usize).min(height - rect_h);
        let left =
            ((rng.random::<f64>() * (width - rect_w + 1) as f64) as usize).min(width - rect_w);

        // Erase the sampled rectangle on the spatial plane for every leading
        // (batch/channel) index. `slice_each_axis_mut` keeps this layout-agnostic.
        let erase_value = self.value;
        result
            .slice_each_axis_mut(|ax| {
                let axis = ax.axis.index();
                if axis == ndim - 2 {
                    Slice::new(top as isize, Some((top + rect_h) as isize), 1)
                } else if axis == ndim - 1 {
                    Slice::new(left as isize, Some((left + rect_w) as isize), 1)
                } else {
                    Slice::new(0, None, 1)
                }
            })
            .fill(erase_value);

        Ok(result)
    }

    fn description(&self) -> String {
        format!(
            "RandomErasing (prob: {:.2}, value: {:.2})",
            self.probability,
            self.value.to_f64().unwrap_or(0.0)
        )
    }

    fn clone_box(&self) -> Box<dyn Augmentation<F>> {
        Box::new(self.clone())
    }
}

/// Random horizontal flip augmentation
#[derive(Debug, Clone)]
pub struct RandomHorizontalFlip<F: Float + NumAssign + Debug + ScalarOperand> {
    /// Probability of applying the flip
    probability: f64,
    /// Phantom data for generic type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> RandomHorizontalFlip<F> {
    /// Create a new random horizontal flip augmentation
    pub fn new(probability: f64) -> Self {
        Self {
            probability,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand + 'static> Augmentation<F>
    for RandomHorizontalFlip<F>
{
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = SmallRng::from_rng(&mut thread_rng());
        let mut result = input.clone();

        // Only apply based on probability
        if rng.random::<f64>() > self.probability {
            return Ok(result);
        }

        // A horizontal flip reverses the width axis. We treat the last axis as
        // width (valid for CHW, HWC and plain HW layouts). For scalars there is
        // nothing to flip.
        let ndim = result.ndim();
        if ndim == 0 {
            return Ok(result);
        }
        result.invert_axis(Axis(ndim - 1));

        Ok(result)
    }

    fn description(&self) -> String {
        format!("RandomHorizontalFlip (prob: {:.2})", self.probability)
    }

    fn clone_box(&self) -> Box<dyn Augmentation<F>> {
        Box::new(self.clone())
    }
}

/// Debug wrapper for a trait object
struct DebugAugmentationWrapper<'a, F: Float + NumAssign + Debug + ScalarOperand> {
    /// Reference to the augmentation
    inner: &'a dyn Augmentation<F>,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Debug for DebugAugmentationWrapper<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Augmentation({})", self.inner.description())
    }
}

/// Compose multiple augmentations into a single augmentation
pub struct ComposeAugmentation<F: Float + NumAssign + Debug + ScalarOperand> {
    /// List of augmentations to apply in sequence
    augmentations: Vec<Box<dyn Augmentation<F>>>,
}

impl<F: Float + NumAssign + Debug + ScalarOperand + 'static> Clone for ComposeAugmentation<F> {
    fn clone(&self) -> Self {
        // Clone each contained augmentation through its `clone_box` method.
        // This preserves the full pipeline instead of dropping it.
        Self {
            augmentations: self
                .augmentations
                .iter()
                .map(|augmentation| augmentation.clone_box())
                .collect(),
        }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Debug for ComposeAugmentation<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_list = f.debug_list();
        for augmentation in &self.augmentations {
            debug_list.entry(&DebugAugmentationWrapper {
                inner: augmentation.as_ref(),
            });
        }
        debug_list.finish()
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand> ComposeAugmentation<F> {
    /// Create a new composition of augmentations
    pub fn new(augmentations: Vec<Box<dyn Augmentation<F>>>) -> Self {
        Self { augmentations }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand + 'static> Augmentation<F>
    for ComposeAugmentation<F>
{
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

    fn clone_box(&self) -> Box<dyn Augmentation<F>> {
        Box::new(self.clone())
    }
}
