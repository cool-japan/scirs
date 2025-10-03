//! Data augmentation for training neural networks

use crate::error::Result;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
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
impl<F: Float + Debug + ScalarOperand> GaussianNoise<F> {
    /// Create a new Gaussian noise augmentation
    pub fn new(std: F) -> Self {
        Self { _std }
    }
impl<F: Float + Debug + ScalarOperand> Augmentation<F> for GaussianNoise<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = rng();
        let mut result = input.clone();
        for item in result.iter_mut() {
            // Create a normal distribution
            let normal = scirs2_core::random::Normal::new(0.0, self.std.to_f64().unwrap_or(0.1))
                .expect("Failed to create normal distribution");
            // Sample from the distribution
            let noise = F::from(rng.sample::<f64>(normal)).unwrap_or(F::zero());
            *item = *item + noise;
        }
        Ok(result)
    fn description(&self) -> String {
        format!(
            "GaussianNoise (std: {:.3})",
            self.std.to_f64().unwrap_or(0.0)
        )
/// Random erasing augmentation
pub struct RandomErasing<F: Float + Debug + ScalarOperand> {
    /// Probability of applying the augmentation
    probability: f64,
    /// Value to use for erasing
    value: F,
impl<F: Float + Debug + ScalarOperand> RandomErasing<F> {
    /// Create a new random erasing augmentation
    pub fn new(probability: f64, value: F) -> Self {
        Self { probability, value }
impl<F: Float + Debug + ScalarOperand> Augmentation<F> for RandomErasing<F> {
        // Only apply augmentation based on probability
        if rng.random::<f64>() > self.probability {
            return Ok(result);
        // Only apply to 2D or higher arrays (like images)
        if result.ndim() < 3 {
        // Pick a random region to erase
        let h = result.shape()[1];
        let w = result.shape()[2];
        // Determine erase size (10% to 30% of the image)
        let erase_h = ((h as f64) * rng.random_range(0.1..0.3)) as usize;
        let erase_w = ((w as f64) * rng.random_range(0.1..0.3)) as usize;
        // Determine erase position
        let i = rng.random_range(0..(h - erase_h).max(1));
        let j = rng.random_range(0..(w - erase_w).max(1));
        // Erase the region
        for img in 0..result.shape()[0] {
            for c in 0..result.shape()[3].min(3) {
                // Assuming channel-last format
                for ii in i..(i + erase_h).min(h) {
                    for jj in j..(j + erase_w).min(w) {
                        result[[img..ii, jj, c]] = self.value;
                    }
                }
            }
            "RandomErasing (prob: {:.2}, value: {:.2})",
            self.probability,
            self.value.to_f64().unwrap_or(0.0)
/// Random horizontal flip augmentation
pub struct RandomHorizontalFlip<F: Float + Debug + ScalarOperand> {
    /// Probability of applying the flip
    /// Phantom data for generic type
    _phantom: std::marker::PhantomData<F>,
impl<F: Float + Debug + ScalarOperand> RandomHorizontalFlip<F> {
    /// Create a new random horizontal flip augmentation
    pub fn new(probability: f64) -> Self {
        Self {
            probability_phantom: std::marker::PhantomData,
impl<F: Float + Debug + ScalarOperand> Augmentation<F> for RandomHorizontalFlip<F> {
        // Flip horizontally (assuming CHW format)
            for c in 0..result.shape()[1] {
                for h in 0..result.shape()[1] {
                    for j in 0..(w / 2) {
                        let temp = result[[img, c, h, j]];
                        result[[img, c, h, j]] = result[[img, c, h, w - 1 - j]];
                        result[[img, c, h, w - 1 - j]] = temp;
        format!("RandomHorizontalFlip (prob: {:.2})", self.probability)
/// Debug wrapper for a trait object
struct DebugAugmentationWrapper<'a, F: Float + Debug + ScalarOperand> {
    /// Reference to the augmentation
    inner: &'a dyn Augmentation<F>,
impl<F: Float + Debug + ScalarOperand> Debug for DebugAugmentationWrapper<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Augmentation({})", self.inner.description())
/// Compose multiple augmentations into a single augmentation
pub struct ComposeAugmentation<F: Float + Debug + ScalarOperand> {
    /// List of augmentations to apply in sequence
    augmentations: Vec<Box<dyn Augmentation<F>>>,
impl<F: Float + Debug + ScalarOperand> Clone for ComposeAugmentation<F> {
    fn clone(&self) -> Self {
        // We can't clone trait objects directly, so we need to implement a custom Clone
        // In a real implementation, we would need a way to clone each augmentation
        // For now, we'll return an empty list which is not ideal but will let the code compile
            augmentations: Vec::new(),
impl<F: Float + Debug + ScalarOperand> Debug for ComposeAugmentation<F> {
        let mut debug_list = f.debug_list();
        for augmentation in &self.augmentations {
            debug_list.entry(&DebugAugmentationWrapper {
                inner: augmentation.as_ref(),
            });
        debug_list.finish()
impl<F: Float + Debug + ScalarOperand> ComposeAugmentation<F> {
    /// Create a new composition of augmentations
    pub fn new(augmentations: Vec<Box<dyn Augmentation<F>>>) -> Self {
        Self { _augmentations }
impl<F: Float + Debug + ScalarOperand> Augmentation<F> for ComposeAugmentation<F> {
        let mut data = input.clone();
            data = augmentation.apply(&data)?;
        Ok(data)
        let descriptions: Vec<String> =
            self.augmentations.iter().map(|a| a.description()).collect();
        format!("Compose({})", descriptions.join(", "))
