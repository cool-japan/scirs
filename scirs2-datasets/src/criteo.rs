//! Criteo Display Advertising synthetic dataset generator.
//!
//! Generates synthetic data mimicking the Criteo click-through rate (CTR)
//! prediction dataset:
//! - 13 integer features (log-normalised counts)
//! - 26 categorical features (hashed to a uniform hash space)
//! - Binary click label with configurable base CTR
//! - Slight positive correlation between label and some integer features

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// Config & Record
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Criteo synthetic dataset generator.
#[derive(Debug, Clone)]
pub struct CriteoConfig {
    /// Number of samples (default: 10_000).
    pub n_samples: usize,
    /// Hash-space size for categorical features (default: 1_000).
    pub n_categories: usize,
    /// Base click-through rate, in (0, 1) (default: 0.04).
    pub ctr: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for CriteoConfig {
    fn default() -> Self {
        Self {
            n_samples: 10_000,
            n_categories: 1_000,
            ctr: 0.04,
            seed: 42,
        }
    }
}

/// A single record in the synthetic Criteo dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct CriteoRecord {
    /// Click label: 0 (no click) or 1 (click).
    pub label: u8,
    /// 13 integer count features (Poisson-distributed, possibly 0 or negative
    /// after log-normalisation).
    pub integer_features: [i32; 13],
    /// 26 hashed categorical feature values, uniform over `[0, n_categories)`.
    pub categorical_features: [u32; 26],
}

// ─────────────────────────────────────────────────────────────────────────────
// CriteoDataset
// ─────────────────────────────────────────────────────────────────────────────

/// Synthetic Criteo-style CTR prediction dataset.
///
/// Integer features follow `Poisson(λ=5)`.  Categorical features are uniform
/// over `[0, n_categories)`.  The label is Bernoulli with probability
/// `p = ctr + delta` where `delta` is a small positive bias correlated with
/// the sum of the first 4 integer features.
#[derive(Debug, Clone)]
pub struct CriteoDataset {
    records: Vec<CriteoRecord>,
    config: CriteoConfig,
}

impl CriteoDataset {
    /// Generate a synthetic Criteo dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or distribution
    /// construction fails.
    pub fn generate(config: CriteoConfig) -> Result<Self> {
        if config.n_samples == 0 {
            return Err(DatasetsError::InvalidFormat(
                "CriteoConfig: n_samples must be > 0".to_string(),
            ));
        }
        if config.n_categories == 0 {
            return Err(DatasetsError::InvalidFormat(
                "CriteoConfig: n_categories must be > 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&config.ctr) {
            return Err(DatasetsError::InvalidFormat(
                "CriteoConfig: ctr must be in [0, 1)".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(config.seed);
        let pois = Poisson::new(5.0_f64).map_err(|e| {
            DatasetsError::ComputationError(format!("Poisson dist construction failed: {e}"))
        })?;
        let cat_dist = Uniform::new(0u64, config.n_categories as u64).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform dist construction failed: {e}"))
        })?;

        let mut records = Vec::with_capacity(config.n_samples);

        for _ in 0..config.n_samples {
            // Sample 13 integer features
            let mut int_feats = [0i32; 13];
            for feat in int_feats.iter_mut() {
                let raw: f64 = pois.sample(&mut rng);
                // Store as raw count (integer) — callers can apply log1p normalisation
                *feat = raw as i32;
            }

            // Sample 26 categorical features
            let mut cat_feats = [0u32; 26];
            for feat in cat_feats.iter_mut() {
                *feat = cat_dist.sample(&mut rng) as u32;
            }

            // Label: Bernoulli with slight positive correlation to int_feats[0..4]
            // The higher the sum of first 4 features, the slightly higher the click prob.
            let feature_sum: f64 = int_feats[..4].iter().map(|&v| v as f64).sum();
            // Scale factor: +0.5% per unit of feature_sum, capped at +2%
            let delta = (feature_sum * 0.005_f64).clamp(0.0, 0.02_f64);
            let p = ((config.ctr as f64) + delta).min(1.0);
            let bernoulli = Bernoulli::new(p).map_err(|e| {
                DatasetsError::ComputationError(format!("Bernoulli dist construction failed: {e}"))
            })?;
            let label = if bernoulli.sample(&mut rng) { 1u8 } else { 0u8 };

            records.push(CriteoRecord {
                label,
                integer_features: int_feats,
                categorical_features: cat_feats,
            });
        }

        Ok(Self { records, config })
    }

    /// All records in the dataset.
    pub fn records(&self) -> &[CriteoRecord] {
        &self.records
    }

    /// Convert to feature matrix `X` of shape `[n_samples, 39]` (13 int + 26 cat
    /// features, all cast to `f32`) and label vector `y` of shape `[n_samples]`.
    ///
    /// Column layout: columns 0–12 are integer features, columns 13–38 are
    /// categorical features.
    pub fn to_feature_matrix(&self) -> (Array2<f32>, Array1<u8>) {
        let n = self.records.len();
        let mut x = Array2::zeros((n, 39));
        let mut y = Array1::zeros(n);
        for (i, rec) in self.records.iter().enumerate() {
            for (j, &v) in rec.integer_features.iter().enumerate() {
                x[[i, j]] = v as f32;
            }
            for (j, &v) in rec.categorical_features.iter().enumerate() {
                x[[i, 13 + j]] = v as f32;
            }
            y[i] = rec.label;
        }
        (x, y)
    }

    /// Observed click-through rate in the dataset.
    pub fn ctr_rate(&self) -> f32 {
        if self.records.is_empty() {
            return 0.0;
        }
        let clicks: f32 = self.records.iter().map(|r| r.label as f32).sum();
        clicks / self.records.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_criteo_shape() {
        let cfg = CriteoConfig {
            n_samples: 500,
            n_categories: 100,
            ctr: 0.05,
            seed: 1,
        };
        let ds = CriteoDataset::generate(cfg.clone()).expect("generate failed");
        assert_eq!(ds.records().len(), cfg.n_samples);
        let (x, y) = ds.to_feature_matrix();
        assert_eq!(x.nrows(), cfg.n_samples);
        assert_eq!(x.ncols(), 39);
        assert_eq!(y.len(), cfg.n_samples);
    }

    #[test]
    fn test_criteo_deterministic() {
        let cfg = CriteoConfig {
            n_samples: 200,
            n_categories: 50,
            ctr: 0.04,
            seed: 77,
        };
        let ds1 = CriteoDataset::generate(cfg.clone()).expect("generate failed");
        let ds2 = CriteoDataset::generate(cfg).expect("generate failed");
        assert_eq!(ds1.records(), ds2.records());
    }

    #[test]
    fn test_criteo_valid_ranges() {
        let cfg = CriteoConfig {
            n_samples: 300,
            n_categories: 200,
            ctr: 0.04,
            seed: 5,
        };
        let ds = CriteoDataset::generate(cfg.clone()).expect("generate failed");
        for rec in ds.records() {
            assert!(rec.label == 0 || rec.label == 1, "label must be 0 or 1");
            for &cf in &rec.categorical_features {
                assert!(
                    (cf as usize) < cfg.n_categories,
                    "categorical feature out of hash range"
                );
            }
            for &iv in &rec.integer_features {
                assert!(iv >= 0, "integer feature must be non-negative (Poisson)");
            }
        }
    }

    #[test]
    fn test_criteo_ctr_reasonable() {
        // With 10k samples and ctr=0.04 we expect roughly 2–8 % clicks.
        let cfg = CriteoConfig {
            n_samples: 10_000,
            n_categories: 1_000,
            ctr: 0.04,
            seed: 42,
        };
        let ds = CriteoDataset::generate(cfg).expect("generate failed");
        let rate = ds.ctr_rate();
        assert!(rate > 0.01, "CTR too low: {rate}");
        assert!(rate < 0.15, "CTR too high: {rate}");
    }

    #[test]
    fn test_criteo_feature_matrix_no_nan() {
        let cfg = CriteoConfig {
            n_samples: 100,
            n_categories: 50,
            ctr: 0.05,
            seed: 8,
        };
        let ds = CriteoDataset::generate(cfg).expect("generate failed");
        let (x, _y) = ds.to_feature_matrix();
        let x_ref = x.view();
        let slice = x_ref.as_slice().expect("contiguous");
        assert!(slice.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn test_criteo_error_zero_samples() {
        let cfg = CriteoConfig {
            n_samples: 0,
            ..CriteoConfig::default()
        };
        assert!(CriteoDataset::generate(cfg).is_err());
    }

    #[test]
    fn test_criteo_error_invalid_ctr() {
        let cfg = CriteoConfig {
            ctr: 1.5,
            ..CriteoConfig::default()
        };
        assert!(CriteoDataset::generate(cfg).is_err());
    }
}
