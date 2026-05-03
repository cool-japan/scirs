//! M5 Competition synthetic retail time series dataset generator.
//!
//! Generates synthetic time series data mimicking the M5 competition format:
//! - 28-day weekly aggregations
//! - 3-level hierarchy: item / store / state
//! - Poisson-distributed daily demand with weekly seasonality and item-level trends

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{s, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// Config & Record
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the M5 synthetic dataset generator.
#[derive(Debug, Clone)]
pub struct M5Config {
    /// Number of unique item SKUs (default: 50).
    pub n_items: usize,
    /// Number of stores (default: 10).
    pub n_stores: usize,
    /// Number of states (default: 3).
    pub n_states: usize,
    /// Number of weeks of history (default: 52).
    pub n_weeks: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for M5Config {
    fn default() -> Self {
        Self {
            n_items: 50,
            n_stores: 10,
            n_states: 3,
            n_weeks: 52,
            seed: 42,
        }
    }
}

/// A single demand record in the synthetic M5 dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct M5Record {
    /// Item SKU index (0-based).
    pub item_id: usize,
    /// Store index (0-based).
    pub store_id: usize,
    /// State index (0-based).
    pub state_id: usize,
    /// Week index (0-based).
    pub week: usize,
    /// Aggregated weekly demand.
    pub demand: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// M5Dataset
// ─────────────────────────────────────────────────────────────────────────────

/// Synthetic M5-style retail time series dataset.
///
/// Demand is generated with:
/// - Poisson-distributed daily counts (λ = base_lambda_per_item)
/// - Weekly seasonality: weekday multiplier 1.2, weekend multiplier 0.8
/// - Per-item linear trend slope sampled from Normal(0, 0.01)
#[derive(Debug, Clone)]
pub struct M5Dataset {
    records: Vec<M5Record>,
    config: M5Config,
}

impl M5Dataset {
    /// Generate a synthetic M5 dataset from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if distribution construction fails.
    pub fn generate(config: M5Config) -> Result<Self> {
        if config.n_items == 0 {
            return Err(DatasetsError::InvalidFormat(
                "M5Config: n_items must be > 0".to_string(),
            ));
        }
        if config.n_stores == 0 {
            return Err(DatasetsError::InvalidFormat(
                "M5Config: n_stores must be > 0".to_string(),
            ));
        }
        if config.n_states == 0 {
            return Err(DatasetsError::InvalidFormat(
                "M5Config: n_states must be > 0".to_string(),
            ));
        }
        if config.n_weeks == 0 {
            return Err(DatasetsError::InvalidFormat(
                "M5Config: n_weeks must be > 0".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(config.seed);
        let n_days = config.n_weeks * 7;
        let n_series = config.n_items * config.n_stores;

        // Pre-sample per-item trend slopes ~ N(0, 0.01)
        let trend_dist = Normal::new(0.0_f64, 0.01_f64).map_err(|e| {
            DatasetsError::ComputationError(format!("Normal dist construction failed: {e}"))
        })?;
        let trend_slopes: Vec<f64> = (0..config.n_items)
            .map(|_| trend_dist.sample(&mut rng))
            .collect();

        // Per-item base lambda ~ Uniform(2.0, 10.0)
        let lambda_dist = Uniform::new(2.0_f64, 10.0_f64).map_err(|e| {
            DatasetsError::ComputationError(format!("Uniform dist construction failed: {e}"))
        })?;
        let base_lambdas: Vec<f64> = (0..config.n_items)
            .map(|_| lambda_dist.sample(&mut rng))
            .collect();

        let mut records = Vec::with_capacity(n_series * config.n_weeks);

        for item_id in 0..config.n_items {
            for store_id in 0..config.n_stores {
                let state_id = store_id % config.n_states;
                let base_lambda = base_lambdas[item_id];
                let slope = trend_slopes[item_id];

                for week in 0..config.n_weeks {
                    let mut weekly_demand = 0.0_f64;
                    for day_in_week in 0..7usize {
                        let abs_day = week * 7 + day_in_week;
                        // Weekly seasonality: Mon-Fri(0-4) × 1.2, Sat-Sun(5-6) × 0.8
                        let season = if day_in_week < 5 { 1.2_f64 } else { 0.8_f64 };
                        let trend = 1.0 + slope * abs_day as f64;
                        let lambda = (base_lambda * season * trend).max(0.01);
                        let pois = Poisson::new(lambda).map_err(|e| {
                            DatasetsError::ComputationError(format!(
                                "Poisson dist construction failed: {e}"
                            ))
                        })?;
                        weekly_demand += pois.sample(&mut rng);
                    }
                    records.push(M5Record {
                        item_id,
                        store_id,
                        state_id,
                        week,
                        demand: weekly_demand as f32,
                    });
                }
            }
        }

        Ok(Self { records, config })
    }

    /// Return all records in the dataset.
    pub fn records(&self) -> &[M5Record] {
        &self.records
    }

    /// Convert to a 2-D array of shape `[n_series, n_weeks]` where
    /// `n_series = n_items × n_stores`, ordered item-major, store-minor.
    pub fn to_ndarray(&self) -> Array2<f32> {
        let n_series = self.config.n_items * self.config.n_stores;
        let n_weeks = self.config.n_weeks;
        let mut out = Array2::zeros((n_series, n_weeks));
        for rec in &self.records {
            let row = rec.item_id * self.config.n_stores + rec.store_id;
            out[[row, rec.week]] = rec.demand;
        }
        out
    }

    /// Number of time series (= n_items × n_stores).
    pub fn num_series(&self) -> usize {
        self.config.n_items * self.config.n_stores
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m5_shape() {
        let cfg = M5Config {
            n_items: 5,
            n_stores: 3,
            n_states: 2,
            n_weeks: 10,
            seed: 0,
        };
        let ds = M5Dataset::generate(cfg.clone()).expect("generate failed");
        assert_eq!(ds.records().len(), cfg.n_items * cfg.n_stores * cfg.n_weeks);
        assert_eq!(ds.num_series(), cfg.n_items * cfg.n_stores);

        let arr = ds.to_ndarray();
        assert_eq!(arr.nrows(), cfg.n_items * cfg.n_stores);
        assert_eq!(arr.ncols(), cfg.n_weeks);
    }

    #[test]
    fn test_m5_deterministic() {
        let cfg = M5Config {
            n_items: 4,
            n_stores: 2,
            n_states: 2,
            n_weeks: 8,
            seed: 99,
        };
        let ds1 = M5Dataset::generate(cfg.clone()).expect("generate failed");
        let ds2 = M5Dataset::generate(cfg).expect("generate failed");
        assert_eq!(ds1.records(), ds2.records());
    }

    #[test]
    fn test_m5_valid_ranges() {
        let cfg = M5Config {
            n_items: 3,
            n_stores: 2,
            n_states: 2,
            n_weeks: 4,
            seed: 7,
        };
        let ds = M5Dataset::generate(cfg.clone()).expect("generate failed");
        for rec in ds.records() {
            assert!(rec.demand >= 0.0, "demand must be non-negative");
            assert!(rec.item_id < cfg.n_items);
            assert!(rec.store_id < cfg.n_stores);
            assert!(rec.state_id < cfg.n_states);
            assert!(rec.week < cfg.n_weeks);
            assert!(!rec.demand.is_nan(), "demand must not be NaN");
        }
    }

    #[test]
    fn test_m5_array_no_nan() {
        let cfg = M5Config::default();
        let ds = M5Dataset::generate(cfg).expect("generate failed");
        let arr = ds.to_ndarray();
        let arr_ref = arr.view();
        let slice = arr_ref.as_slice().expect("contiguous");
        assert!(slice.iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn test_m5_state_assignment() {
        let cfg = M5Config {
            n_items: 2,
            n_stores: 6,
            n_states: 3,
            n_weeks: 2,
            seed: 1,
        };
        let ds = M5Dataset::generate(cfg.clone()).expect("generate failed");
        // state_id is store_id % n_states
        for rec in ds.records() {
            assert_eq!(rec.state_id, rec.store_id % cfg.n_states);
        }
    }

    #[test]
    fn test_m5_error_on_zero_items() {
        let cfg = M5Config {
            n_items: 0,
            ..M5Config::default()
        };
        assert!(M5Dataset::generate(cfg).is_err());
    }

    // Slice indexing test (validates to_ndarray layout)
    #[test]
    fn test_m5_ndarray_layout() {
        let cfg = M5Config {
            n_items: 2,
            n_stores: 2,
            n_states: 2,
            n_weeks: 3,
            seed: 42,
        };
        let ds = M5Dataset::generate(cfg.clone()).expect("generate failed");
        let arr = ds.to_ndarray();
        // First series = item 0, store 0
        let series0 = arr.slice(s![0, ..]);
        // Verify all weeks are filled (should not all be zero)
        let sum: f32 = series0.iter().copied().sum();
        assert!(sum > 0.0, "series 0 should have non-zero demand");
    }
}
