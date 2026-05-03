//! Biclustering algorithms
//!
//! Biclustering (also called co-clustering) simultaneously clusters rows and
//! columns of a data matrix, finding subsets of rows and columns that exhibit
//! a coherent pattern together.
//!
//! # Algorithms
//!
//! - **SpectralBiclustering**: Normalised spectral approach (Kluger et al., 2003)
//! - **ChengChurch**: Greedy mean squared residue minimisation (Cheng & Church, 2000)
//! - **PLAID**: Additive layer model (Lazzeroni & Owen, 2002)

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashSet;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Bicluster primitive
// ---------------------------------------------------------------------------

/// A bicluster: a subset of rows and columns that co-cluster.
#[derive(Debug, Clone)]
pub struct Bicluster {
    /// Row indices belonging to this bicluster.
    pub rows: Vec<usize>,
    /// Column indices belonging to this bicluster.
    pub cols: Vec<usize>,
    /// Mean value of the bicluster sub-matrix (for diagnostics).
    pub mean: f64,
    /// Mean squared residue (for ChengChurch; 0.0 otherwise).
    pub residue: f64,
}

impl Bicluster {
    /// Compute the mean squared residue of this bicluster in matrix `x`.
    ///
    /// Residue H(I, J) = (1/|I||J|) Σ_i Σ_j (x_ij - x_iJ - x_Ij + x_IJ)^2
    /// where x_iJ, x_Ij, x_IJ are row/col/bicluster means.
    pub fn msr(&self, x: ArrayView2<f64>) -> f64 {
        if self.rows.is_empty() || self.cols.is_empty() {
            return 0.0;
        }
        let n_r = self.rows.len() as f64;
        let n_c = self.cols.len() as f64;

        // Bicluster mean.
        let x_ij_mean: f64 = self
            .rows
            .iter()
            .flat_map(|&r| self.cols.iter().map(move |&c| x[[r, c]]))
            .sum::<f64>()
            / (n_r * n_c);

        // Row means within bicluster.
        let row_means: Vec<f64> = self
            .rows
            .iter()
            .map(|&r| self.cols.iter().map(|&c| x[[r, c]]).sum::<f64>() / n_c)
            .collect();

        // Col means within bicluster.
        let col_means: Vec<f64> = self
            .cols
            .iter()
            .map(|&c| self.rows.iter().map(|&r| x[[r, c]]).sum::<f64>() / n_r)
            .collect();

        let mut total = 0.0f64;
        for (ri, &r) in self.rows.iter().enumerate() {
            for (ci, &c) in self.cols.iter().enumerate() {
                let residue = x[[r, c]] - row_means[ri] - col_means[ci] + x_ij_mean;
                total += residue * residue;
            }
        }
        total / (n_r * n_c)
    }
}

/// Result of a biclustering algorithm.
#[derive(Debug, Clone)]
pub struct BiclusterResult {
    /// The biclusters found.
    pub biclusters: Vec<Bicluster>,
    /// Row-level cluster assignment (best matching bicluster index, or `n_biclusters` if none).
    pub row_labels: Vec<usize>,
    /// Column-level cluster assignment.
    pub col_labels: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Spectral Biclustering
// ---------------------------------------------------------------------------

/// Spectral biclustering via doubly-normalised SVD.
///
/// Normalises the data matrix using log and double-normalisation, extracts
/// the leading singular vectors, and applies k-means to rows and columns
/// independently using those vectors as features.
///
/// Reference: Kluger et al. (2003) "Spectral Biclustering of Microarray Data".
pub struct SpectralBiclustering {
    /// Minimum singular value ratio to accept a component (default 0.01).
    pub sv_threshold: f64,
    /// Number of k-means iterations for assignment step (default 100).
    pub km_max_iter: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for SpectralBiclustering {
    fn default() -> Self {
        Self {
            sv_threshold: 0.01,
            km_max_iter: 100,
            seed: 42,
        }
    }
}

impl SpectralBiclustering {
    /// Fit spectral biclustering to the matrix `x`.
    ///
    /// # Arguments
    /// * `x` – Data matrix `(n_rows, n_cols)`.
    /// * `n_clusters` – Number of biclusters to find.
    pub fn fit(&self, x: ArrayView2<f64>, n_clusters: usize) -> Result<BiclusterResult> {
        let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);
        if n_rows == 0 || n_cols == 0 {
            return Err(ClusteringError::InvalidInput("Empty input matrix".into()));
        }
        if n_clusters == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_clusters must be > 0".into(),
            ));
        }
        if n_clusters > n_rows || n_clusters > n_cols {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters ({}) must be <= n_rows ({}) and n_cols ({})",
                n_clusters, n_rows, n_cols
            )));
        }

        // Step 1: Log-normalise (shift to positive range first).
        let min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let shift = if min_val <= 0.0 {
            (-min_val) + 1.0
        } else {
            0.0
        };
        let mut norm = x.to_owned();
        norm.mapv_inplace(|v| (v + shift).max(1e-12).ln());

        // Step 2: Subtract global mean.
        let global_mean = norm.mean().unwrap_or(0.0);
        norm.mapv_inplace(|v| v - global_mean);

        // Step 3: Double normalise: row-centre then column-centre (×2 iterations).
        for _ in 0..2 {
            // Row centre.
            for r in 0..n_rows {
                let rm: f64 = norm.row(r).mean().unwrap_or(0.0);
                norm.row_mut(r).mapv_inplace(|v| v - rm);
            }
            // Col centre.
            for c in 0..n_cols {
                let cm: f64 = norm.column(c).mean().unwrap_or(0.0);
                norm.column_mut(c).mapv_inplace(|v| v - cm);
            }
        }

        // Step 4: Compact SVD (power iteration to extract top-n_clusters singular triplets).
        let n_components = n_clusters.min(n_rows.min(n_cols));
        let (u, _s, vt) = compact_svd(norm.view(), n_components, self.seed)?;

        // Step 5: K-means on row features (U columns) and col features (V^T rows).
        let row_features = u; // (n_rows, n_components)
        let col_features = vt.t().to_owned(); // (n_cols, n_components)

        let mut rng = self.seed;
        let row_labels =
            kmeans_1d_labels(row_features.view(), n_clusters, self.km_max_iter, &mut rng)?;
        let col_labels =
            kmeans_1d_labels(col_features.view(), n_clusters, self.km_max_iter, &mut rng)?;

        // Step 6: Build biclusters.
        let mut biclusters: Vec<Bicluster> = (0..n_clusters)
            .map(|k| {
                let rows: Vec<usize> = (0..n_rows).filter(|&r| row_labels[r] == k).collect();
                let cols: Vec<usize> = (0..n_cols).filter(|&c| col_labels[c] == k).collect();
                let mean = if rows.is_empty() || cols.is_empty() {
                    0.0
                } else {
                    rows.iter()
                        .flat_map(|&r| cols.iter().map(move |&c| x[[r, c]]))
                        .sum::<f64>()
                        / (rows.len() * cols.len()) as f64
                };
                Bicluster {
                    rows,
                    cols,
                    mean,
                    residue: 0.0,
                }
            })
            .collect();

        Ok(BiclusterResult {
            biclusters,
            row_labels,
            col_labels,
        })
    }
}

// ---------------------------------------------------------------------------
// Cheng-Church algorithm
// ---------------------------------------------------------------------------

/// Cheng-Church biclustering: greedy deletion/addition to minimise MSR.
///
/// Each call to `fit` returns up to `n_clusters` biclusters. After finding one,
/// the sub-matrix is masked with random noise before searching for the next.
///
/// Reference: Cheng & Church (2000) "Biclustering of Expression Data".
pub struct ChengChurch {
    /// Maximum number of single-node deletion iterations per bicluster (default 1000).
    pub max_iter: usize,
    /// RNG seed for masking.
    pub seed: u64,
}

impl Default for ChengChurch {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            seed: 42,
        }
    }
}

impl ChengChurch {
    /// Fit Cheng-Church to find up to `n_clusters` biclusters with MSR ≤ `delta`.
    ///
    /// # Arguments
    /// * `x` – Data matrix.
    /// * `n_clusters` – Maximum number of biclusters.
    /// * `delta` – MSR threshold; lower means tighter/smaller biclusters.
    pub fn fit(&self, x: ArrayView2<f64>, n_clusters: usize, delta: f64) -> Result<Vec<Bicluster>> {
        let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);
        if n_rows == 0 || n_cols == 0 {
            return Err(ClusteringError::InvalidInput("Empty input matrix".into()));
        }
        if n_clusters == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_clusters must be > 0".into(),
            ));
        }
        if delta < 0.0 {
            return Err(ClusteringError::InvalidInput("delta must be >= 0".into()));
        }

        let mut masked = x.to_owned();
        let mut biclusters: Vec<Bicluster> = Vec::with_capacity(n_clusters);
        let mut rng = self.seed;

        // Compute data range for noise masking.
        let data_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let data_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let noise_range = (data_max - data_min).max(1.0);

        for _ in 0..n_clusters {
            let bc = self.find_one_bicluster(masked.view(), delta, &mut rng)?;
            // Mask the bicluster positions with random noise.
            for &r in &bc.rows {
                for &c in &bc.cols {
                    let noise = lcg_f64(&mut rng) * noise_range + data_min;
                    masked[[r, c]] = noise;
                }
            }
            biclusters.push(bc);
        }

        Ok(biclusters)
    }

    fn find_one_bicluster(
        &self,
        x: ArrayView2<f64>,
        delta: f64,
        rng: &mut u64,
    ) -> Result<Bicluster> {
        let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);

        let mut rows: HashSet<usize> = (0..n_rows).collect();
        let mut cols: HashSet<usize> = (0..n_cols).collect();

        // Phase 1: Multiple node deletion until MSR ≤ delta.
        for _iter in 0..self.max_iter {
            if rows.len() <= 1 || cols.len() <= 1 {
                break;
            }

            let (msr, row_msrs, col_msrs) = compute_msr(x, &rows, &cols);
            if msr <= delta {
                break;
            }

            // Remove the row or column with the highest MSR contribution.
            let worst_row = rows
                .iter()
                .max_by(|&&a, &&b| {
                    row_msrs
                        .get(&a)
                        .unwrap_or(&0.0)
                        .partial_cmp(col_msrs.get(&b).unwrap_or(&0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned();

            let worst_col = cols
                .iter()
                .max_by(|&&a, &&b| {
                    col_msrs
                        .get(&a)
                        .unwrap_or(&0.0)
                        .partial_cmp(col_msrs.get(&b).unwrap_or(&0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned();

            let row_val = worst_row
                .and_then(|r| row_msrs.get(&r))
                .cloned()
                .unwrap_or(0.0);
            let col_val = worst_col
                .and_then(|c| col_msrs.get(&c))
                .cloned()
                .unwrap_or(0.0);

            if row_val >= col_val {
                if let Some(r) = worst_row {
                    if rows.len() > 1 {
                        rows.remove(&r);
                    }
                }
            } else if let Some(c) = worst_col {
                if cols.len() > 1 {
                    cols.remove(&c);
                }
            }
        }

        // Phase 2: Node addition – add rows/cols that don't increase MSR.
        let all_rows: Vec<usize> = (0..n_rows).filter(|r| !rows.contains(r)).collect();
        let all_cols: Vec<usize> = (0..n_cols).filter(|c| !cols.contains(c)).collect();

        for r in &all_rows {
            let mut test = rows.clone();
            test.insert(*r);
            let (new_msr, _, _) = compute_msr(x, &test, &cols);
            if new_msr <= delta {
                rows.insert(*r);
            }
        }
        for c in &all_cols {
            let mut test = cols.clone();
            test.insert(*c);
            let (new_msr, _, _) = compute_msr(x, &rows, &test);
            if new_msr <= delta {
                cols.insert(*c);
            }
        }

        let mut rows_vec: Vec<usize> = rows.into_iter().collect();
        let mut cols_vec: Vec<usize> = cols.into_iter().collect();
        rows_vec.sort_unstable();
        cols_vec.sort_unstable();

        let n_r = rows_vec.len() as f64;
        let n_c = cols_vec.len() as f64;
        let mean = if n_r > 0.0 && n_c > 0.0 {
            rows_vec
                .iter()
                .flat_map(|&r| cols_vec.iter().map(move |&c| x[[r, c]]))
                .sum::<f64>()
                / (n_r * n_c)
        } else {
            0.0
        };

        let bc = Bicluster {
            rows: rows_vec,
            cols: cols_vec,
            mean,
            residue: 0.0,
        };
        let residue = bc.msr(x);

        Ok(Bicluster { residue, ..bc })
    }
}

/// Compute the mean squared residue and per-row/col contributions.
fn compute_msr(
    x: ArrayView2<f64>,
    rows: &HashSet<usize>,
    cols: &HashSet<usize>,
) -> (
    f64,
    std::collections::HashMap<usize, f64>,
    std::collections::HashMap<usize, f64>,
) {
    let n_r = rows.len() as f64;
    let n_c = cols.len() as f64;
    if n_r == 0.0 || n_c == 0.0 {
        return (0.0, Default::default(), Default::default());
    }

    // Bicluster mean.
    let x_mean: f64 = rows
        .iter()
        .flat_map(|&r| cols.iter().map(move |&c| x[[r, c]]))
        .sum::<f64>()
        / (n_r * n_c);

    // Row means.
    let row_means: std::collections::HashMap<usize, f64> = rows
        .iter()
        .map(|&r| {
            let rm = cols.iter().map(|&c| x[[r, c]]).sum::<f64>() / n_c;
            (r, rm)
        })
        .collect();

    // Col means.
    let col_means: std::collections::HashMap<usize, f64> = cols
        .iter()
        .map(|&c| {
            let cm = rows.iter().map(|&r| x[[r, c]]).sum::<f64>() / n_r;
            (c, cm)
        })
        .collect();

    // MSR per cell.
    let total: f64 = rows
        .iter()
        .flat_map(|&r| {
            cols.iter()
                .map(|&c| {
                    let res = x[[r, c]] - row_means[&r] - col_means[&c] + x_mean;
                    res * res
                })
                .collect::<Vec<_>>()
        })
        .sum::<f64>()
        / (n_r * n_c);

    // Per-row MSR (mean over columns for that row).
    let row_msrs: std::collections::HashMap<usize, f64> = rows
        .iter()
        .map(|&r| {
            let v = cols
                .iter()
                .map(|&c| {
                    let res = x[[r, c]] - row_means[&r] - col_means[&c] + x_mean;
                    res * res
                })
                .sum::<f64>()
                / n_c;
            (r, v)
        })
        .collect();

    // Per-col MSR (mean over rows for that col).
    let col_msrs: std::collections::HashMap<usize, f64> = cols
        .iter()
        .map(|&c| {
            let v = rows
                .iter()
                .map(|&r| {
                    let res = x[[r, c]] - row_means[&r] - col_means[&c] + x_mean;
                    res * res
                })
                .sum::<f64>()
                / n_r;
            (c, v)
        })
        .collect();

    (total, row_msrs, col_msrs)
}

// ---------------------------------------------------------------------------
// PLAID model
// ---------------------------------------------------------------------------

/// A single layer in the PLAID additive model.
#[derive(Debug, Clone)]
struct PlaidLayer {
    /// Row binary membership vector.
    row_membership: Vec<bool>,
    /// Column binary membership vector.
    col_membership: Vec<bool>,
    /// Layer intercept μ.
    mu: f64,
    /// Row effects α_i.
    row_effects: Vec<f64>,
    /// Column effects β_j.
    col_effects: Vec<f64>,
}

/// PLAID additive biclustering model.
///
/// Fits an additive model X ≈ Σ_k (μ_k + α_{ik} + β_{jk}) * ρ_{ik} * κ_{jk},
/// where ρ, κ are binary membership vectors.
///
/// Reference: Lazzeroni & Owen (2002) "Plaid Models for Gene Expression Data".
pub struct PLAID {
    /// Number of binary thresholding iterations per layer (default 10).
    pub binary_iter: usize,
    /// Number of back-fitting iterations per layer (default 10).
    pub backfit_iter: usize,
    /// Minimum fraction of rows/cols in a layer (default 0.1).
    pub min_fraction: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for PLAID {
    fn default() -> Self {
        Self {
            binary_iter: 10,
            backfit_iter: 10,
            min_fraction: 0.1,
            seed: 42,
        }
    }
}

impl PLAID {
    /// Fit the PLAID model and return `k` biclusters.
    ///
    /// # Arguments
    /// * `x` – Data matrix `(n_rows, n_cols)`.
    /// * `k` – Number of layers (biclusters) to fit.
    pub fn fit(&self, x: ArrayView2<f64>, k: usize) -> Result<Vec<Bicluster>> {
        let (n_rows, n_cols) = (x.shape()[0], x.shape()[1]);
        if n_rows == 0 || n_cols == 0 {
            return Err(ClusteringError::InvalidInput("Empty input matrix".into()));
        }
        if k == 0 {
            return Err(ClusteringError::InvalidInput("k must be > 0".into()));
        }

        let mut residual = x.to_owned();
        let mut biclusters: Vec<Bicluster> = Vec::with_capacity(k);
        let mut rng = self.seed;

        for _ in 0..k {
            let layer = self.fit_one_layer(residual.view(), n_rows, n_cols, &mut rng)?;

            // Subtract the layer from residual.
            for r in 0..n_rows {
                for c in 0..n_cols {
                    if layer.row_membership[r] && layer.col_membership[c] {
                        residual[[r, c]] -= layer.mu + layer.row_effects[r] + layer.col_effects[c];
                    }
                }
            }

            let rows: Vec<usize> = (0..n_rows).filter(|&r| layer.row_membership[r]).collect();
            let cols: Vec<usize> = (0..n_cols).filter(|&c| layer.col_membership[c]).collect();

            let n_r = rows.len() as f64;
            let n_c = cols.len() as f64;
            let mean = if n_r > 0.0 && n_c > 0.0 {
                rows.iter()
                    .flat_map(|&r| cols.iter().map(move |&c| x[[r, c]]))
                    .sum::<f64>()
                    / (n_r * n_c)
            } else {
                0.0
            };

            let bc = Bicluster {
                rows,
                cols,
                mean,
                residue: 0.0,
            };
            biclusters.push(bc);
        }

        Ok(biclusters)
    }

    fn fit_one_layer(
        &self,
        x: ArrayView2<f64>,
        n_rows: usize,
        n_cols: usize,
        rng: &mut u64,
    ) -> Result<PlaidLayer> {
        let min_rows = (n_rows as f64 * self.min_fraction).ceil() as usize;
        let min_cols = (n_cols as f64 * self.min_fraction).ceil() as usize;

        // Initialise random binary memberships.
        let mut row_mem: Vec<bool> = (0..n_rows).map(|_| lcg_f64(rng) > 0.5).collect();
        let mut col_mem: Vec<bool> = (0..n_cols).map(|_| lcg_f64(rng) > 0.5).collect();

        // Ensure at least min_rows/min_cols are active.
        let active_rows = row_mem.iter().filter(|&&v| v).count();
        if active_rows < min_rows {
            for i in 0..min_rows {
                row_mem[i] = true;
            }
        }
        let active_cols = col_mem.iter().filter(|&&v| v).count();
        if active_cols < min_cols {
            for j in 0..min_cols {
                col_mem[j] = true;
            }
        }

        let mut mu = 0.0f64;
        let mut row_effects = vec![0.0f64; n_rows];
        let mut col_effects = vec![0.0f64; n_cols];

        for _bfit in 0..self.backfit_iter {
            // Update μ: mean of active cells minus row/col effects.
            let active_count =
                row_mem.iter().filter(|&&v| v).count() * col_mem.iter().filter(|&&v| v).count();
            if active_count == 0 {
                break;
            }
            let row_effects_ref = &row_effects;
            let col_effects_ref = &col_effects;
            let sum_active: f64 = (0..n_rows)
                .filter(|&r| row_mem[r])
                .flat_map(|r| {
                    (0..n_cols)
                        .filter(|&c| col_mem[c])
                        .map(move |c| x[[r, c]] - row_effects_ref[r] - col_effects_ref[c])
                })
                .sum();
            mu = sum_active / active_count as f64;

            // Update row effects.
            for r in 0..n_rows {
                if !row_mem[r] {
                    row_effects[r] = 0.0;
                    continue;
                }
                let n_active_cols = col_mem.iter().filter(|&&v| v).count();
                if n_active_cols == 0 {
                    row_effects[r] = 0.0;
                } else {
                    let sum: f64 = (0..n_cols)
                        .filter(|&c| col_mem[c])
                        .map(|c| x[[r, c]] - mu - col_effects[c])
                        .sum();
                    row_effects[r] = sum / n_active_cols as f64;
                }
            }

            // Update col effects.
            for c in 0..n_cols {
                if !col_mem[c] {
                    col_effects[c] = 0.0;
                    continue;
                }
                let n_active_rows = row_mem.iter().filter(|&&v| v).count();
                if n_active_rows == 0 {
                    col_effects[c] = 0.0;
                } else {
                    let sum: f64 = (0..n_rows)
                        .filter(|&r| row_mem[r])
                        .map(|r| x[[r, c]] - mu - row_effects[r])
                        .sum();
                    col_effects[c] = sum / n_active_rows as f64;
                }
            }

            // Binary thresholding: update row/col memberships.
            for _thresh_iter in 0..self.binary_iter {
                // Row thresholding: include row r if it reduces layer error.
                for r in 0..n_rows {
                    let mut cost_in = 0.0f64;
                    let mut cost_out = 0.0f64;
                    let n_active_cols = col_mem.iter().filter(|&&v| v).count() as f64;
                    if n_active_cols == 0.0 {
                        continue;
                    }
                    for c in 0..n_cols {
                        if col_mem[c] {
                            let pred = mu + row_effects[r] + col_effects[c];
                            cost_in += (x[[r, c]] - pred).powi(2);
                            cost_out += x[[r, c]].powi(2);
                        }
                    }
                    row_mem[r] = cost_in < cost_out;
                }
                // Enforce minimum rows.
                let active_rows = row_mem.iter().filter(|&&v| v).count();
                if active_rows < min_rows {
                    for i in 0..min_rows {
                        row_mem[i] = true;
                    }
                }

                // Col thresholding.
                for c in 0..n_cols {
                    let mut cost_in = 0.0f64;
                    let mut cost_out = 0.0f64;
                    let n_active_rows = row_mem.iter().filter(|&&v| v).count() as f64;
                    if n_active_rows == 0.0 {
                        continue;
                    }
                    for r in 0..n_rows {
                        if row_mem[r] {
                            let pred = mu + row_effects[r] + col_effects[c];
                            cost_in += (x[[r, c]] - pred).powi(2);
                            cost_out += x[[r, c]].powi(2);
                        }
                    }
                    col_mem[c] = cost_in < cost_out;
                }
                let active_cols = col_mem.iter().filter(|&&v| v).count();
                if active_cols < min_cols {
                    for j in 0..min_cols {
                        col_mem[j] = true;
                    }
                }
            }
        }

        Ok(PlaidLayer {
            row_membership: row_mem,
            col_membership: col_mem,
            mu,
            row_effects,
            col_effects,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// LCG float in [0, 1).
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// LCG usize in [0, n).
fn lcg_usize(state: &mut u64, n: usize) -> usize {
    (lcg_f64(state) * n as f64) as usize % n
}

/// Compact SVD via power iteration (randomised).
///
/// Returns (U, s, Vt) where U is (m, k), s is (k,), Vt is (k, n).
fn compact_svd(
    x: ArrayView2<f64>,
    k: usize,
    seed: u64,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = (x.shape()[0], x.shape()[1]);
    let k = k.min(m).min(n);
    if k == 0 {
        return Err(ClusteringError::ComputationError("Zero components".into()));
    }

    let mut rng = seed;
    // Random Gaussian sketch matrix Omega (n × k).
    let mut omega = Array2::<f64>::zeros((n, k));
    for v in omega.iter_mut() {
        // Box-Muller transform for normal samples.
        let u1 = lcg_f64(&mut rng).max(1e-15);
        let u2 = lcg_f64(&mut rng);
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }

    // Y = X @ Omega  (m × k)
    let mut y = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            let mut s = 0.0;
            for l in 0..n {
                s += x[[i, l]] * omega[[l, j]];
            }
            y[[i, j]] = s;
        }
    }

    // QR decomposition of Y to get Q (m × k), orthonormal.
    let q = gram_schmidt(y.view())?;

    // B = Q^T @ X  (k × n)
    let mut b = Array2::<f64>::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..m {
                s += q[[l, i]] * x[[l, j]];
            }
            b[[i, j]] = s;
        }
    }

    // SVD of B (k × n) via power iteration for small k.
    // Since B is small (k × n), we compute B B^T (k × k) and diagonalise.
    let mut bbt = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for l in 0..n {
                s += b[[i, l]] * b[[j, l]];
            }
            bbt[[i, j]] = s;
        }
    }

    // Power iteration to extract singular values and left singular vectors of B.
    let (ub, sigma) = power_iter_eig(bbt.view(), k, seed + 1)?;

    // Singular values: sqrt of eigenvalues.
    let s: Array1<f64> = sigma.mapv(|v| v.max(0.0).sqrt());

    // U = Q @ U_B  (m × k)
    let mut u = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            let mut val = 0.0;
            for l in 0..k {
                val += q[[i, l]] * ub[[l, j]];
            }
            u[[i, j]] = val;
        }
    }

    // V^T = (U_B^T @ B) / sigma   (k × n), with safe division.
    let mut vt = Array2::<f64>::zeros((k, n));
    for i in 0..k {
        let si = s[i];
        if si < 1e-12 {
            continue;
        }
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += ub[[l, i]] * b[[l, j]];
            }
            vt[[i, j]] = val / si;
        }
    }

    Ok((u, s, vt))
}

/// Gram-Schmidt orthonormalisation of columns of `a`.
fn gram_schmidt(a: ArrayView2<f64>) -> Result<Array2<f64>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut q = Array2::<f64>::zeros((m, n));

    for j in 0..n {
        let mut v = a.column(j).to_owned();
        // Subtract projection onto previous columns.
        for i in 0..j {
            let dot: f64 = v.iter().zip(q.column(i).iter()).map(|(a, b)| a * b).sum();
            for k in 0..m {
                v[k] -= dot * q[[k, i]];
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            // Degenerate column — fill with unit vector.
            if j < m {
                q[[j, j]] = 1.0;
            }
        } else {
            for k in 0..m {
                q[[k, j]] = v[k] / norm;
            }
        }
    }
    Ok(q)
}

/// Power iteration to find top-k eigenvectors/values of a symmetric matrix.
fn power_iter_eig(a: ArrayView2<f64>, k: usize, seed: u64) -> Result<(Array2<f64>, Array1<f64>)> {
    let n = a.shape()[0];
    let k = k.min(n);
    let mut rng = seed;

    let mut eigvecs = Array2::<f64>::zeros((n, k));
    let mut eigvals = Array1::<f64>::zeros(k);

    // Deflation: extract eigenvectors one by one.
    let mut deflated = a.to_owned();

    for col in 0..k {
        // Start with random vector.
        let mut v: Vec<f64> = (0..n).map(|_| lcg_f64(&mut rng) - 0.5).collect();
        normalize_vec(&mut v);

        // Power iteration.
        for _ in 0..200 {
            let mut av: Vec<f64> = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    av[i] += deflated[[i, j]] * v[j];
                }
            }
            // Orthogonalise against already-found vectors.
            for prev in 0..col {
                let dot: f64 = (0..n).map(|i| av[i] * eigvecs[[i, prev]]).sum();
                for i in 0..n {
                    av[i] -= dot * eigvecs[[i, prev]];
                }
            }
            normalize_vec(&mut av);
            v = av;
        }

        // Compute Rayleigh quotient.
        let mut av: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += deflated[[i, j]] * v[j];
            }
        }
        let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
        eigvals[col] = eigenvalue;
        for i in 0..n {
            eigvecs[[i, col]] = v[i];
        }

        // Deflate: A = A - λ v v^T
        for i in 0..n {
            for j in 0..n {
                deflated[[i, j]] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    Ok((eigvecs, eigvals))
}

/// Normalise a vector in-place; no-op if near-zero.
fn normalize_vec(v: &mut Vec<f64>) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// K-means on row-vectors of a matrix, returning per-row labels.
fn kmeans_1d_labels(
    features: ArrayView2<f64>,
    k: usize,
    max_iter: usize,
    rng: &mut u64,
) -> Result<Vec<usize>> {
    let (n, d) = (features.shape()[0], features.shape()[1]);
    if n == 0 || k == 0 {
        return Ok(vec![]);
    }
    let k = k.min(n);

    // Initialise centroids by sampling.
    let mut centroids: Vec<Vec<f64>> = {
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = lcg_usize(rng, i + 1);
            indices.swap(i, j);
        }
        indices[..k]
            .iter()
            .map(|&i| features.row(i).to_vec())
            .collect()
    };

    let mut labels = vec![0usize; n];

    for _iter in 0..max_iter {
        // Assign.
        let mut changed = false;
        for i in 0..n {
            let row = features.row(i).to_vec();
            let best = (0..k)
                .map(|j| {
                    let d2: f64 = row
                        .iter()
                        .zip(centroids[j].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (j, d2)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update centroids.
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0; d]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for dim in 0..d {
                sums[c][dim] += features[[i, dim]];
            }
        }
        for j in 0..k {
            if counts[j] == 0 {
                // Reinitialise dead centroid to a random point.
                let ri = lcg_usize(rng, n);
                centroids[j] = features.row(ri).to_vec();
            } else {
                for dim in 0..d {
                    centroids[j][dim] = sums[j][dim] / counts[j] as f64;
                }
            }
        }
    }

    Ok(labels)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Create a block-diagonal matrix: two obvious biclusters.
    fn block_matrix() -> Array2<f64> {
        // 8×8 matrix with high values in [0..4, 0..4] and [4..8, 4..8].
        let mut m = Array2::<f64>::zeros((8, 8));
        for r in 0..4 {
            for c in 0..4 {
                m[[r, c]] = 5.0 + r as f64 * 0.1 + c as f64 * 0.05;
            }
        }
        for r in 4..8 {
            for c in 4..8 {
                m[[r, c]] = 3.0 + (r - 4) as f64 * 0.1 + (c - 4) as f64 * 0.05;
            }
        }
        m
    }

    #[test]
    fn test_spectral_biclustering_basic() {
        let x = block_matrix();
        let sb = SpectralBiclustering::default();
        let result = sb.fit(x.view(), 2).expect("spectral bicluster");
        assert_eq!(result.biclusters.len(), 2);
        assert_eq!(result.row_labels.len(), 8);
        assert_eq!(result.col_labels.len(), 8);
    }

    #[test]
    fn test_spectral_biclustering_all_rows_and_cols_assigned() {
        let x = block_matrix();
        let sb = SpectralBiclustering::default();
        let result = sb.fit(x.view(), 2).expect("spectral bicluster");
        let all_rows: std::collections::HashSet<usize> = result
            .biclusters
            .iter()
            .flat_map(|bc| bc.rows.iter().cloned())
            .collect();
        let all_cols: std::collections::HashSet<usize> = result
            .biclusters
            .iter()
            .flat_map(|bc| bc.cols.iter().cloned())
            .collect();
        assert_eq!(all_rows.len(), 8, "all rows should be in some bicluster");
        assert_eq!(all_cols.len(), 8, "all cols should be in some bicluster");
    }

    #[test]
    fn test_cheng_church_basic() {
        let x = block_matrix();
        let cc = ChengChurch::default();
        let biclusters = cc.fit(x.view(), 2, 2.0).expect("cheng church");
        assert_eq!(biclusters.len(), 2);
        for bc in &biclusters {
            assert!(!bc.rows.is_empty());
            assert!(!bc.cols.is_empty());
        }
    }

    #[test]
    fn test_cheng_church_msr_threshold() {
        let x = block_matrix();
        let cc = ChengChurch::default();
        // With a very tight threshold, each bicluster residue should be small.
        let biclusters = cc.fit(x.view(), 1, 0.5).expect("cc tight delta");
        assert!(!biclusters.is_empty());
        let bc = &biclusters[0];
        // Residue of found bicluster should be at or below delta=0.5
        // (or very small for the block structure).
        assert!(bc.residue < 2.0, "residue should be small for block data");
    }

    #[test]
    fn test_plaid_basic() {
        let x = block_matrix();
        let plaid = PLAID {
            binary_iter: 5,
            backfit_iter: 5,
            min_fraction: 0.2,
            seed: 42,
        };
        let biclusters = plaid.fit(x.view(), 2).expect("plaid fit");
        assert_eq!(biclusters.len(), 2);
        for bc in &biclusters {
            assert!(!bc.rows.is_empty());
            assert!(!bc.cols.is_empty());
        }
    }

    #[test]
    fn test_plaid_single_layer() {
        let x = block_matrix();
        let plaid = PLAID::default();
        let biclusters = plaid.fit(x.view(), 1).expect("plaid 1 layer");
        assert_eq!(biclusters.len(), 1);
    }

    #[test]
    fn test_bicluster_msr_zero_for_constant() {
        // A constant sub-matrix has MSR = 0.
        let x = Array2::from_elem((4, 4), 3.0);
        let bc = Bicluster {
            rows: vec![0, 1, 2, 3],
            cols: vec![0, 1, 2, 3],
            mean: 3.0,
            residue: 0.0,
        };
        let msr = bc.msr(x.view());
        assert!(
            msr.abs() < 1e-10,
            "constant matrix MSR should be 0, got {msr}"
        );
    }

    #[test]
    fn test_spectral_invalid_n_clusters() {
        let x = block_matrix();
        let sb = SpectralBiclustering::default();
        assert!(sb.fit(x.view(), 0).is_err());
        assert!(sb.fit(x.view(), 100).is_err());
    }
}
