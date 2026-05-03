//! Landmark (inducing-point) selection strategies for Nyström approximation.
//!
//! Three strategies are provided:
//!
//! * **UniformRandom** — sample m indices uniformly at random (O(m)).
//! * **KMeansCenters** — run Lloyd's k-means (k = m) for `n_iter` iterations;
//!   centroids become the landmarks (O(n · m · n_iter · d)).
//! * **LeverageScore** — estimate Drineas-Mahoney leverage scores via a
//!   random sketch of K_{n,m_sketch}; sample proportional to ℓ_i (O(n · m_sketch)).
//!
//! All strategies accept a `seed: u64` for reproducibility.

use crate::error::InterpolateError;
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

// ─── LandmarkStrategy ────────────────────────────────────────────────────────

/// Strategy for selecting landmark (inducing) points used in the Nyström
/// approximation.
#[derive(Debug, Clone)]
pub enum LandmarkStrategy {
    /// Pick `m` training indices uniformly at random.
    UniformRandom,
    /// Run Lloyd's k-means for `n_iter` iterations; use the k = m centroids
    /// as landmarks.
    KMeansCenters {
        /// Number of Lloyd's iterations.  20 works well in practice.
        n_iter: usize,
    },
    /// Approximate leverage-score sampling (Drineas & Mahoney).
    ///
    /// Build a random sketch of K_{n, m_sketch} using Gaussian weights to
    /// estimate each row's leverage score, then sample m points proportional
    /// to those scores.
    LeverageScore {
        /// Width of the random sketch matrix (columns).
        n_random_features: usize,
    },
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Select `m` landmark rows from `x` (shape n × d) using the given strategy.
///
/// Returns a 2-D array of shape `m × d`.  If `m >= n`, all rows of `x` are
/// returned.
pub fn select_landmarks(
    x: &Array2<f64>,
    m: usize,
    strategy: &LandmarkStrategy,
    seed: u64,
    bandwidth: f64,
) -> Result<Array2<f64>, InterpolateError> {
    let n = x.nrows();
    let d = x.ncols();
    let m = m.min(n);

    if m == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "Number of landmarks m must be > 0".to_string(),
        });
    }
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "Training set is empty".to_string(),
        ));
    }
    if m == n {
        return Ok(x.to_owned());
    }

    let indices = match strategy {
        LandmarkStrategy::UniformRandom => uniform_random_indices(n, m, seed),
        LandmarkStrategy::KMeansCenters { n_iter } => kmeans_indices(x, m, *n_iter, seed)?,
        LandmarkStrategy::LeverageScore { n_random_features } => {
            leverage_score_indices(x, m, *n_random_features, seed, bandwidth)?
        }
    };

    // Gather selected rows
    let mut landmarks = Array2::zeros((indices.len(), d));
    for (new_row, &src_row) in indices.iter().enumerate() {
        for col in 0..d {
            landmarks[[new_row, col]] = x[[src_row, col]];
        }
    }
    Ok(landmarks)
}

// ─── Uniform random ───────────────────────────────────────────────────────────

fn uniform_random_indices(n: usize, m: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    // Partial Fisher-Yates shuffle — O(n) but produces m unique indices
    for i in 0..m {
        let j = rng.random_range(i..n);
        indices.swap(i, j);
    }
    indices[..m].to_vec()
}

// ─── K-means (Lloyd's algorithm) ─────────────────────────────────────────────

fn kmeans_indices(
    x: &Array2<f64>,
    m: usize,
    n_iter: usize,
    seed: u64,
) -> Result<Vec<usize>, InterpolateError> {
    let n = x.nrows();
    let d = x.ncols();
    let n_iter = n_iter.max(1);

    // Initialise centroids via k-means++ seeding
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids: Array2<f64> = Array2::zeros((m, d));

    // First centroid chosen uniformly at random
    let first = rng.random_range(0..n);
    for col in 0..d {
        centroids[[0, col]] = x[[first, col]];
    }

    // k-means++ remaining centroids: proportional to D²
    let mut min_dists: Vec<f64> = (0..n).map(|i| sq_dist_row(x, i, &centroids, 0)).collect();

    for c in 1..m {
        let total: f64 = min_dists.iter().sum();
        if total < 1e-300 {
            // All remaining points coincide with existing centroids — duplicate
            let fallback = rng.random_range(0..n);
            for col in 0..d {
                centroids[[c, col]] = x[[fallback, col]];
            }
        } else {
            let mut thresh = rng.random::<f64>() * total;
            let mut chosen = n - 1;
            for (i, &d_val) in min_dists.iter().enumerate() {
                thresh -= d_val;
                if thresh <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            for col in 0..d {
                centroids[[c, col]] = x[[chosen, col]];
            }
        }

        // Update min_dists
        for i in 0..n {
            let dist_to_new = sq_dist_row(x, i, &centroids, c);
            if dist_to_new < min_dists[i] {
                min_dists[i] = dist_to_new;
            }
        }
    }

    // Lloyd iterations
    let mut labels = vec![0usize; n];
    for _ in 0..n_iter {
        // Assignment step
        let mut changed = false;
        for i in 0..n {
            let best = (0..m)
                .min_by(|&a, &b| {
                    sq_dist_row(x, i, &centroids, a)
                        .partial_cmp(&sq_dist_row(x, i, &centroids, b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update step
        let mut sums: Array2<f64> = Array2::zeros((m, d));
        let mut counts = vec![0usize; m];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for col in 0..d {
                sums[[c, col]] += x[[i, col]];
            }
        }
        for c in 0..m {
            if counts[c] > 0 {
                for col in 0..d {
                    centroids[[c, col]] = sums[[c, col]] / counts[c] as f64;
                }
            }
        }
    }

    // For each centroid find the nearest training point (centroids may be
    // off-grid; we want actual data points as landmarks).
    let indices: Vec<usize> = (0..m)
        .map(|c| {
            (0..n)
                .min_by(|&i, &j| {
                    sq_dist_row(x, i, &centroids, c)
                        .partial_cmp(&sq_dist_row(x, j, &centroids, c))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0)
        })
        .collect();

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    let unique: Vec<usize> = indices
        .into_iter()
        .filter(|idx| seen.insert(*idx))
        .collect();

    Ok(unique)
}

// ─── Leverage-score sampling (Drineas-Mahoney) ────────────────────────────────

fn leverage_score_indices(
    x: &Array2<f64>,
    m: usize,
    n_random_features: usize,
    seed: u64,
    bandwidth: f64,
) -> Result<Vec<usize>, InterpolateError> {
    let n = x.nrows();
    let d = x.ncols();
    let s = n_random_features.min(n).max(1);

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(7919));

    // Build an n × s sketch matrix Z where Z[i,j] = Gaussian kernel between
    // x_i and a random "anchor" point drawn uniformly from x.  This gives a
    // fast O(n · s · d) approximation to the leverage scores.
    let anchor_indices: Vec<usize> = {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..s {
            let j = rng.random_range(i..n);
            idx.swap(i, j);
        }
        idx[..s].to_vec()
    };

    // Compute Z  (n × s) : Gaussian kernel to anchors
    let mut sketch = vec![0.0f64; n * s];
    for i in 0..n {
        for (j, &aj) in anchor_indices.iter().enumerate() {
            let sq: f64 = (0..d).map(|k| (x[[i, k]] - x[[aj, k]]).powi(2)).sum();
            sketch[i * s + j] = (-sq / (2.0 * bandwidth * bandwidth)).exp();
        }
    }

    // Leverage score ℓ_i ≈ ||Z[i,:]||²  (proportional to diagonal of Z Z^T Z (Z^T Z)^{-1})
    // As an approximation we just use the row 2-norm squared.
    let mut scores: Vec<f64> = (0..n)
        .map(|i| (0..s).map(|j| sketch[i * s + j].powi(2)).sum::<f64>())
        .collect();

    // Normalise
    let total: f64 = scores.iter().sum();
    if total < 1e-300 {
        // Fall back to uniform sampling
        return Ok(uniform_random_indices(n, m, seed));
    }
    for s_val in scores.iter_mut() {
        *s_val /= total;
    }

    // Sample m distinct indices proportional to scores
    let mut selected: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut attempts = 0usize;
    while selected.len() < m && attempts < n * 10 {
        attempts += 1;
        let mut thresh = rng.random::<f64>();
        let mut cum = 0.0;
        for (i, &score) in scores.iter().enumerate() {
            cum += score;
            if thresh <= cum {
                selected.insert(i);
                break;
            }
        }
        // Ensure we always make progress even if rng stalls at tail
        if attempts % (n + 1) == 0 && selected.len() < m {
            let fallback = rng.random_range(0..n);
            selected.insert(fallback);
        }
    }

    // Pad with uniform random if necessary
    if selected.len() < m {
        let extra = uniform_random_indices(n, m, seed.wrapping_add(1));
        for idx in extra {
            if selected.len() >= m {
                break;
            }
            selected.insert(idx);
        }
    }

    let mut indices: Vec<usize> = selected.into_iter().collect();
    indices.sort_unstable();
    indices.truncate(m);
    Ok(indices)
}

// ─── Geometry helpers ─────────────────────────────────────────────────────────

/// Squared Euclidean distance between `x[row, :]` and `centroids[c, :]`.
fn sq_dist_row(x: &Array2<f64>, row: usize, centroids: &Array2<f64>, c: usize) -> f64 {
    let d = x.ncols();
    (0..d)
        .map(|k| (x[[row, k]] - centroids[[c, k]]).powi(2))
        .sum()
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_grid(n: usize) -> Array2<f64> {
        let mut x = Array2::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = i as f64 / n as f64;
            x[[i, 1]] = (i as f64 / n as f64).powi(2);
        }
        x
    }

    #[test]
    fn uniform_produces_m_landmarks() {
        let x = make_grid(50);
        let lm = select_landmarks(&x, 10, &LandmarkStrategy::UniformRandom, 0, 1.0)
            .expect("uniform landmark selection");
        assert_eq!(lm.nrows(), 10);
        assert_eq!(lm.ncols(), 2);
    }

    #[test]
    fn kmeans_produces_up_to_m_landmarks() {
        let x = make_grid(40);
        let lm = select_landmarks(
            &x,
            8,
            &LandmarkStrategy::KMeansCenters { n_iter: 20 },
            1,
            1.0,
        )
        .expect("kmeans landmark selection");
        assert!(lm.nrows() > 0 && lm.nrows() <= 8);
    }

    #[test]
    fn leverage_score_produces_landmarks() {
        let x = make_grid(60);
        let lm = select_landmarks(
            &x,
            12,
            &LandmarkStrategy::LeverageScore {
                n_random_features: 20,
            },
            2,
            1.0,
        )
        .expect("leverage-score landmark selection");
        assert!(lm.nrows() > 0 && lm.nrows() <= 12);
    }

    #[test]
    fn uniform_when_m_equals_n() {
        let x = make_grid(5);
        let lm =
            select_landmarks(&x, 5, &LandmarkStrategy::UniformRandom, 0, 1.0).expect("m == n case");
        assert_eq!(lm.nrows(), 5);
    }
}
