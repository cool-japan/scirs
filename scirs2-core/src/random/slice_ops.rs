//! Enhanced slice random operations for SCIRS2 ecosystem
//!
//! This module provides robust, flexible slice random operations that serve as
//! alternatives to native rand:: slice operations with enhanced functionality
//! for scientific computing.

use crate::random::core::{thread_rng, Random};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::Uniform;

/// Re-export SliceRandom trait from rand for enhanced compatibility
pub use rand::seq::SliceRandom as SliceRandomExt;

/// Enhanced slice random operations for scientific computing
pub trait ScientificSliceRandom<T> {
    /// Shuffle the slice in place with enhanced performance
    fn scientific_shuffle<R: Rng>(&mut self, rng: &mut Random<R>);

    /// Choose a random element with guaranteed uniform distribution
    fn scientific_choose<R: Rng>(&self, rng: &mut Random<R>) -> Option<&T>;

    /// Choose multiple elements without replacement using optimal algorithms
    fn scientific_choose_multiple<R: Rng>(&self, rng: &mut Random<R>, amount: usize) -> Vec<&T>;

    /// Sample with replacement
    fn scientific_sample_with_replacement<R: Rng>(
        &self,
        rng: &mut Random<R>,
        amount: usize,
    ) -> Vec<&T>;

    /// Weighted sampling (requires weights)
    fn scientific_weighted_sample<R: Rng, W>(
        &self,
        rng: &mut Random<R>,
        weights: &[W],
        amount: usize,
    ) -> Result<Vec<&T>, String>
    where
        W: Into<f64> + Copy;

    /// Reservoir sampling for streaming data
    fn scientific_reservoir_sample<R: Rng>(&self, rng: &mut Random<R>, k: usize) -> Vec<&T>;
}

impl<T> ScientificSliceRandom<T> for [T] {
    fn scientific_shuffle<R: Rng>(&mut self, rng: &mut Random<R>) {
        // Use Fisher-Yates shuffle for guaranteed uniform distribution
        for i in (1..self.len()).rev() {
            let j = rng.sample(Uniform::new(0, i + 1).unwrap());
            self.swap(i, j);
        }
    }

    fn scientific_choose<R: Rng>(&self, rng: &mut Random<R>) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            let index = rng.gen_range(0..self.len());
            Some(&self[index])
        }
    }

    fn scientific_choose_multiple<R: Rng>(&self, rng: &mut Random<R>, amount: usize) -> Vec<&T> {
        if amount >= self.len() {
            return self.iter().collect();
        }

        // Use Floyd's algorithm for optimal sampling without replacement
        let mut selected = std::collections::HashSet::new();
        let n = self.len();
        let k = amount;

        for i in (n - k)..n {
            let mut j = rng.gen_range(0..=i);
            if selected.contains(&j) {
                j = i;
            }
            selected.insert(j);
        }

        selected.into_iter().map(|i| &self[i]).collect()
    }

    fn scientific_sample_with_replacement<R: Rng>(
        &self,
        rng: &mut Random<R>,
        amount: usize,
    ) -> Vec<&T> {
        (0..amount)
            .map(|_| &self[rng.gen_range(0..self.len())])
            .collect()
    }

    fn scientific_weighted_sample<R: Rng, W>(
        &self,
        rng: &mut Random<R>,
        weights: &[W],
        amount: usize,
    ) -> Result<Vec<&T>, String>
    where
        W: Into<f64> + Copy,
    {
        if self.len() != weights.len() {
            return Err("Items and weights must have the same length".to_string());
        }

        if self.is_empty() {
            return Ok(Vec::new());
        }

        // Convert weights to f64 and compute cumulative distribution
        let weights_f64: Vec<f64> = weights.iter().map(|&w| w.into()).collect();
        let total_weight: f64 = weights_f64.iter().sum();

        if total_weight <= 0.0 {
            return Err("Total weight must be positive".to_string());
        }

        let mut cumulative = Vec::with_capacity(weights_f64.len());
        let mut cum_sum = 0.0;
        for &weight in &weights_f64 {
            cum_sum += weight / total_weight;
            cumulative.push(cum_sum);
        }

        let mut result = Vec::with_capacity(amount);
        for _ in 0..amount {
            let u = rng.gen_range(0.0..1.0);
            match cumulative.binary_search_by(|&x| x.partial_cmp(&u).unwrap()) {
                Ok(idx) => result.push(&self[idx]),
                Err(idx) => result.push(&self[idx.min(self.len() - 1)]),
            }
        }

        Ok(result)
    }

    fn scientific_reservoir_sample<R: Rng>(&self, rng: &mut Random<R>, k: usize) -> Vec<&T> {
        if k >= self.len() {
            return self.iter().collect();
        }

        let mut reservoir: Vec<&T> = Vec::with_capacity(k);

        // Fill reservoir with first k elements
        for item in self.iter().take(k) {
            reservoir.push(item);
        }

        // Replace elements with gradually decreasing probability
        for (i, item) in self.iter().enumerate().skip(k) {
            let j = rng.sample(Uniform::new(0, i + 1).unwrap());
            if j < k {
                reservoir[j] = item;
            }
        }

        reservoir
    }
}

impl<T> ScientificSliceRandom<T> for Vec<T> {
    fn scientific_shuffle<R: Rng>(&mut self, rng: &mut Random<R>) {
        self.as_mut_slice().scientific_shuffle(rng);
    }

    fn scientific_choose<R: Rng>(&self, rng: &mut Random<R>) -> Option<&T> {
        self.as_slice().scientific_choose(rng)
    }

    fn scientific_choose_multiple<R: Rng>(&self, rng: &mut Random<R>, amount: usize) -> Vec<&T> {
        self.as_slice().scientific_choose_multiple(rng, amount)
    }

    fn scientific_sample_with_replacement<R: Rng>(
        &self,
        rng: &mut Random<R>,
        amount: usize,
    ) -> Vec<&T> {
        self.as_slice()
            .scientific_sample_with_replacement(rng, amount)
    }

    fn scientific_weighted_sample<R: Rng, W>(
        &self,
        rng: &mut Random<R>,
        weights: &[W],
        amount: usize,
    ) -> Result<Vec<&T>, String>
    where
        W: Into<f64> + Copy,
    {
        self.as_slice()
            .scientific_weighted_sample(rng, weights, amount)
    }

    fn scientific_reservoir_sample<R: Rng>(&self, rng: &mut Random<R>, k: usize) -> Vec<&T> {
        self.as_slice().scientific_reservoir_sample(rng, k)
    }
}

/// Convenience functions for slice random operations
pub mod convenience {
    use super::*;

    /// Shuffle a slice in place using the default thread-local RNG
    pub fn shuffle<T>(slice: &mut [T]) {
        use rand::seq::SliceRandom as _;
        let mut rng = thread_rng();
        slice.shuffle(&mut rng.rng);
    }

    /// Sample n elements from a slice without replacement
    pub fn sample<T>(slice: &[T], n: usize) -> Vec<T>
    where
        T: Clone,
    {
        // Simplified implementation to avoid trait import conflicts
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..slice.len()).collect();
        indices.shuffle(&mut rng.rng);
        indices
            .into_iter()
            .take(n)
            .map(|i| slice[i].clone())
            .collect()
    }

    /// Choose a random element from a slice
    pub fn choose<T>(slice: &[T]) -> Option<&T> {
        // Simplified implementation to avoid trait import conflicts
        if slice.is_empty() {
            None
        } else {
            let mut rng = thread_rng();
            let index = rng.gen_range(0..slice.len());
            Some(&slice[index])
        }
    }

    /// Scientific shuffle with guaranteed uniform distribution
    pub fn scientific_shuffle<T>(slice: &mut [T]) {
        let mut rng = thread_rng();
        slice.scientific_shuffle(&mut rng);
    }

    /// Scientific sampling without replacement
    pub fn scientific_sample<T>(slice: &[T], n: usize) -> Vec<&T> {
        let mut rng = thread_rng();
        slice.scientific_choose_multiple(&mut rng, n)
    }

    /// Scientific weighted sampling
    pub fn scientific_weighted_sample<'a, T, W>(
        slice: &'a [T],
        weights: &[W],
        n: usize,
    ) -> Result<Vec<&'a T>, String>
    where
        W: Into<f64> + Copy,
    {
        let mut rng = thread_rng();
        slice.scientific_weighted_sample(&mut rng, weights, n)
    }

    /// Reservoir sampling for streaming data
    pub fn reservoir_sample<T>(slice: &[T], k: usize) -> Vec<&T> {
        let mut rng = thread_rng();
        slice.scientific_reservoir_sample(&mut rng, k)
    }
}

/// Advanced sampling algorithms for scientific computing
pub mod algorithms {
    use super::*;
    use std::collections::HashMap;

    /// Stratified sampling for experimental design
    pub fn stratified_sample<'a, T, K>(
        data: &'a [(T, K)],
        strata_sizes: &HashMap<K, usize>,
        rng: &mut Random<impl Rng>,
    ) -> Vec<&'a T>
    where
        K: Eq + std::hash::Hash + Clone,
    {
        let mut result = Vec::new();
        let mut strata: HashMap<K, Vec<&T>> = HashMap::new();

        // Group data by strata
        for (item, key) in data {
            strata.entry(key.clone()).or_default().push(item);
        }

        // Sample from each stratum
        for (key, desired_size) in strata_sizes {
            if let Some(stratum_data) = strata.get(key) {
                let sample = stratum_data.scientific_choose_multiple(rng, *desired_size);
                result.extend(sample);
            }
        }

        result
    }

    /// Systematic sampling with random start
    pub fn systematic_sample<'a, T>(
        data: &'a [T],
        n: usize,
        rng: &mut Random<impl Rng>,
    ) -> Vec<&'a T> {
        if n == 0 || data.is_empty() {
            return Vec::new();
        }

        if n >= data.len() {
            return data.iter().collect();
        }

        let interval = data.len() as f64 / n as f64;
        let start = rng.gen_range(0.0..interval);

        (0..n)
            .map(|i| {
                let index = (start + i as f64 * interval) as usize;
                &data[index.min(data.len() - 1)]
            })
            .collect()
    }

    /// Cluster sampling
    pub fn cluster_sample<'a, T, C>(
        clusters: &'a [(C, Vec<T>)],
        n_clusters: usize,
        rng: &mut Random<impl Rng>,
    ) -> Vec<&'a T>
    where
        C: Clone,
    {
        if clusters.is_empty() || n_clusters == 0 {
            return Vec::new();
        }

        let cluster_refs: Vec<&(C, Vec<T>)> = clusters.iter().collect();
        let selected_clusters = cluster_refs.scientific_choose_multiple(rng, n_clusters);

        let mut result = Vec::new();
        for (_, cluster_data) in selected_clusters {
            result.extend(cluster_data.iter());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::core::seeded_rng;

    #[test]
    fn test_scientific_shuffle() {
        let mut data = vec![1, 2, 3, 4, 5];
        let mut rng = seeded_rng(42);

        data.scientific_shuffle(&mut rng);
        assert_eq!(data.len(), 5);
        assert!(data.contains(&1));
        assert!(data.contains(&5));
    }

    #[test]
    fn test_scientific_choose() {
        let data = [1, 2, 3, 4, 5];
        let mut rng = seeded_rng(123);

        let choice = data.scientific_choose(&mut rng);
        assert!(choice.is_some());
        assert!(data.contains(choice.unwrap()));
    }

    #[test]
    fn test_scientific_choose_multiple() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut rng = seeded_rng(456);

        let choices = data.scientific_choose_multiple(&mut rng, 3);
        assert_eq!(choices.len(), 3);

        // Check uniqueness
        let mut unique_values = std::collections::HashSet::new();
        for &choice in &choices {
            unique_values.insert(*choice);
        }
        assert_eq!(unique_values.len(), 3);
    }

    #[test]
    fn test_weighted_sampling() {
        let items = ["A", "B", "C"];
        let weights = [0.1, 0.3, 0.6];
        let mut rng = seeded_rng(789);

        let samples = items
            .scientific_weighted_sample(&mut rng, &weights, 100)
            .unwrap();
        assert_eq!(samples.len(), 100);

        // Check that all samples are valid
        for &sample in &samples {
            assert!(items.contains(sample));
        }
    }

    #[test]
    fn test_reservoir_sampling() {
        let data: Vec<i32> = (0..1000).collect();
        let mut rng = seeded_rng(101112);

        let sample = data.scientific_reservoir_sample(&mut rng, 10);
        assert_eq!(sample.len(), 10);

        // Check uniqueness
        let mut unique_values = std::collections::HashSet::new();
        for &value in &sample {
            unique_values.insert(*value);
        }
        assert_eq!(unique_values.len(), 10);
    }

    #[test]
    fn test_stratified_sampling() {
        let data = vec![(1, "A"), (2, "A"), (3, "B"), (4, "B"), (5, "C"), (6, "C")];
        let mut strata_sizes = std::collections::HashMap::new();
        strata_sizes.insert("A", 1);
        strata_sizes.insert("B", 1);
        strata_sizes.insert("C", 1);

        let mut rng = seeded_rng(131415);
        let sample = algorithms::stratified_sample(&data, &strata_sizes, &mut rng);

        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn test_systematic_sampling() {
        let data: Vec<i32> = (0..100).collect();
        let mut rng = seeded_rng(161718);

        let sample = algorithms::systematic_sample(&data, 10, &mut rng);
        assert_eq!(sample.len(), 10);

        // Check that samples are roughly evenly spaced
        let indices: Vec<usize> = sample.iter().map(|&&x| x as usize).collect();
        for i in 1..indices.len() {
            let gap = indices[i] - indices[i - 1];
            assert!((8..=12).contains(&gap)); // Approximately interval of 10
        }
    }

    #[test]
    fn test_convenience_functions() {
        let mut data = vec![1, 2, 3, 4, 5];
        convenience::shuffle(&mut data);
        assert_eq!(data.len(), 5);

        let original = vec![1, 2, 3, 4, 5];
        let choice = convenience::choose(&original);
        assert!(choice.is_some());

        let sample = convenience::sample(&original, 3);
        assert_eq!(sample.len(), 3);
    }
}
