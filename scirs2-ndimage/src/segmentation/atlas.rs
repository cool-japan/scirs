//! Atlas-based segmentation algorithms.
//!
//! Provides label fusion methods for multi-atlas segmentation:
//!
//! - [`AtlasSegmentation`]: orchestrates multi-atlas label fusion pipelines
//! - [`MajorityVoting`]: simple voxel-wise majority voting over atlas labels
//! - [`STAPLE`]: Simultaneous Truth and Performance Level Estimation
//! - [`JointLabelFusion`]: locally weighted voting using patch similarity (Wang et al. 2013)
//!
//! # References
//!
//! - Artaechevarria et al. (2009), "Combination Strategies in Multi-Atlas Image Segmentation",
//!   IEEE TMI 28(8):1266-1277.
//! - Warfield et al. (2004), "Simultaneous Truth and Performance Level Estimation (STAPLE)",
//!   IEEE TMI 23(7):903-921.
//! - Wang et al. (2013), "Multi-Atlas Segmentation with Joint Label Fusion", IEEE TPAMI 35(3).

use std::collections::HashMap;

use scirs2_core::ndarray::Array3;

use crate::error::{NdimageError, NdimageResult};

// ─── Shared helpers ───────────────────────────────────────────────────────────

/// Validate that all atlas label volumes have the same shape.
fn check_shapes(labels: &[Array3<u32>]) -> NdimageResult<(usize, usize, usize)> {
    if labels.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Atlas segmentation: must provide at least one atlas".to_string(),
        ));
    }
    let s = labels[0].shape();
    let shape = (s[0], s[1], s[2]);
    for (i, lab) in labels.iter().enumerate().skip(1) {
        if lab.shape() != labels[0].shape() {
            return Err(NdimageError::DimensionError(format!(
                "Atlas segmentation: atlas {} has shape {:?}, expected {:?}",
                i,
                lab.shape(),
                labels[0].shape()
            )));
        }
    }
    Ok(shape)
}

// ─── MajorityVoting ───────────────────────────────────────────────────────────

/// Voxel-wise majority voting over multiple atlas label maps.
///
/// At each voxel the label that appears most frequently among the atlases is
/// selected.  Ties are broken by choosing the smallest label index.
pub struct MajorityVoting;

impl MajorityVoting {
    /// Fuse `labels` by majority voting.
    ///
    /// Each element of `labels` is a 3-D array of label values with shape
    /// `(nz, ny, nx)`.  Returns a single label volume of the same shape.
    pub fn fuse(labels: &[Array3<u32>]) -> NdimageResult<Array3<u32>> {
        let (nz, ny, nx) = check_shapes(labels)?;
        let n_atlases = labels.len();
        let mut result = Array3::<u32>::zeros((nz, ny, nx));

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let mut counts: HashMap<u32, usize> = HashMap::new();
                    for a in 0..n_atlases {
                        let lv = labels[a][[iz, iy, ix]];
                        *counts.entry(lv).or_insert(0) += 1;
                    }
                    // Select label with highest count (ties: smallest label wins)
                    let winner = counts
                        .iter()
                        .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                        .map(|(&lv, _)| lv)
                        .unwrap_or(0);
                    result[[iz, iy, ix]] = winner;
                }
            }
        }
        Ok(result)
    }

    /// Return per-voxel confidence (fraction of atlases agreeing with the
    /// majority label).
    pub fn confidence(labels: &[Array3<u32>]) -> NdimageResult<Array3<f64>> {
        let fused = Self::fuse(labels)?;
        let (nz, ny, nx) = check_shapes(labels)?;
        let n_atlases = labels.len() as f64;
        let mut conf = Array3::<f64>::zeros((nz, ny, nx));
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let winner = fused[[iz, iy, ix]];
                    let agree = labels.iter().filter(|l| l[[iz, iy, ix]] == winner).count();
                    conf[[iz, iy, ix]] = agree as f64 / n_atlases;
                }
            }
        }
        Ok(conf)
    }
}

// ─── STAPLE ───────────────────────────────────────────────────────────────────

/// STAPLE (Simultaneous Truth and Performance Level Estimation) algorithm.
///
/// Estimates the "ground truth" probabilistic segmentation and each rater's
/// sensitivity/specificity via expectation-maximisation.
///
/// This implementation supports binary (foreground/background) segmentation.
/// Multi-label inputs are binarised by treating label > 0 as foreground.
#[derive(Debug, Clone)]
pub struct StapleConfig {
    /// Maximum number of EM iterations (default 20).
    pub max_iterations: usize,
    /// Convergence threshold on the max absolute parameter change (default 1e-5).
    pub convergence_threshold: f64,
    /// Initial sensitivity estimate for each rater (default 0.99).
    pub init_sensitivity: f64,
    /// Initial specificity estimate for each rater (default 0.99).
    pub init_specificity: f64,
}

impl Default for StapleConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            convergence_threshold: 1e-5,
            init_sensitivity: 0.99,
            init_specificity: 0.99,
        }
    }
}

/// Per-rater performance parameters estimated by STAPLE.
#[derive(Debug, Clone)]
pub struct RaterPerformance {
    /// Sensitivity (true positive rate) for this rater.
    pub sensitivity: f64,
    /// Specificity (true negative rate) for this rater.
    pub specificity: f64,
}

/// Result of a STAPLE estimation run.
#[derive(Debug, Clone)]
pub struct StapleResult {
    /// Probabilistic foreground segmentation `W[z,y,x] ∈ [0, 1]`.
    pub probability: Array3<f64>,
    /// Hard binary label map (W > 0.5 → 1, else 0).
    pub label: Array3<u32>,
    /// Estimated performance parameters per rater.
    pub performance: Vec<RaterPerformance>,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Whether the EM converged.
    pub converged: bool,
}

/// STAPLE algorithm for binary segmentation quality estimation.
pub struct STAPLE {
    config: StapleConfig,
}

impl STAPLE {
    /// Create a new STAPLE estimator with default configuration.
    pub fn new() -> Self {
        Self {
            config: StapleConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: StapleConfig) -> Self {
        Self { config }
    }

    /// Run the STAPLE EM algorithm on a set of binary atlas segmentations.
    ///
    /// `labels` should contain binary or multi-label volumes; values > 0 are
    /// treated as foreground.  All volumes must have the same shape.
    pub fn estimate(&self, labels: &[Array3<u32>]) -> NdimageResult<StapleResult> {
        let (nz, ny, nx) = check_shapes(labels)?;
        let n = nz * ny * nx;
        let r = labels.len();

        // Flatten observations: d[rater][voxel] ∈ {0, 1}
        let d: Vec<Vec<u8>> = labels
            .iter()
            .map(|l| l.iter().map(|&v| if v > 0 { 1u8 } else { 0u8 }).collect())
            .collect();

        // Initial performance parameters
        let mut p: Vec<f64> = vec![self.config.init_sensitivity; r]; // sensitivity
        let mut q: Vec<f64> = vec![self.config.init_specificity; r]; // specificity

        // Prior probability of foreground (uniform 0.5)
        let prior_fg = 0.5_f64;

        // Initial W: fraction of raters labelling each voxel as foreground
        let mut w: Vec<f64> = (0..n)
            .map(|i| d.iter().map(|rater| rater[i] as f64).sum::<f64>() / r as f64)
            .collect();

        let mut converged = false;
        let mut n_iter = 0;

        for _iter in 0..self.config.max_iterations {
            n_iter += 1;

            // M-step: update performance parameters
            let sum_w: f64 = w.iter().sum();
            let sum_w0: f64 = w.iter().map(|&wi| 1.0 - wi).sum();

            let mut new_p = vec![0.0_f64; r];
            let mut new_q = vec![0.0_f64; r];

            for j in 0..r {
                // Sensitivity: TP / (TP + FN) — foreground agreement
                let tp: f64 = (0..n).map(|i| d[j][i] as f64 * w[i]).sum();
                new_p[j] = (tp + 1e-10) / (sum_w + 1e-10);

                // Specificity: TN / (TN + FP)
                let tn: f64 = (0..n).map(|i| (1.0 - d[j][i] as f64) * (1.0 - w[i])).sum();
                new_q[j] = (tn + 1e-10) / (sum_w0 + 1e-10);

                // Clamp to valid probability range
                new_p[j] = new_p[j].clamp(1e-6, 1.0 - 1e-6);
                new_q[j] = new_q[j].clamp(1e-6, 1.0 - 1e-6);
            }

            // E-step: update W
            let mut max_change = 0.0_f64;
            let mut new_w = vec![0.0_f64; n];
            for i in 0..n {
                // Log likelihood of observing d[*][i] given W=1 and W=0
                let mut ll1 = prior_fg.ln();
                let mut ll0 = (1.0 - prior_fg).ln();
                for j in 0..r {
                    if d[j][i] == 1 {
                        ll1 += new_p[j].ln();
                        ll0 += (1.0 - new_q[j]).ln();
                    } else {
                        ll1 += (1.0 - new_p[j]).ln();
                        ll0 += new_q[j].ln();
                    }
                }
                let max_ll = ll1.max(ll0);
                let p1 = (ll1 - max_ll).exp();
                let p0 = (ll0 - max_ll).exp();
                new_w[i] = p1 / (p1 + p0 + 1e-10);
                max_change = max_change.max((new_w[i] - w[i]).abs());
            }

            // Check convergence
            let param_change = (0..r)
                .map(|j| (new_p[j] - p[j]).abs().max((new_q[j] - q[j]).abs()))
                .fold(0.0_f64, f64::max);

            p = new_p;
            q = new_q;
            w = new_w;

            if param_change < self.config.convergence_threshold
                && max_change < self.config.convergence_threshold
            {
                converged = true;
                break;
            }
        }

        // Build output arrays
        let mut probability = Array3::<f64>::zeros((nz, ny, nx));
        let mut label = Array3::<u32>::zeros((nz, ny, nx));
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let idx = iz * ny * nx + iy * nx + ix;
                    let wi = w[idx];
                    probability[[iz, iy, ix]] = wi;
                    label[[iz, iy, ix]] = if wi > 0.5 { 1 } else { 0 };
                }
            }
        }

        let performance: Vec<RaterPerformance> = (0..r)
            .map(|j| RaterPerformance {
                sensitivity: p[j],
                specificity: q[j],
            })
            .collect();

        Ok(StapleResult {
            probability,
            label,
            performance,
            iterations: n_iter,
            converged,
        })
    }
}

// ─── JointLabelFusion ─────────────────────────────────────────────────────────

/// Configuration for Joint Label Fusion (JLF).
#[derive(Debug, Clone)]
pub struct JlfConfig {
    /// Half-width of the patch (neighbourhood) used for similarity weighting.
    /// Patch is `(2*patch_radius+1)^dim` voxels.
    pub patch_radius: usize,
    /// Alpha parameter controlling the steepness of the similarity weight decay
    /// (default 0.1).  Larger → more uniform weighting.
    pub alpha: f64,
    /// Beta parameter for patch normalisation (default 2.0).
    pub beta: f64,
}

impl Default for JlfConfig {
    fn default() -> Self {
        Self {
            patch_radius: 2,
            alpha: 0.1,
            beta: 2.0,
        }
    }
}

/// Result of joint label fusion.
#[derive(Debug, Clone)]
pub struct JlfResult {
    /// Final fused label volume.
    pub label: Array3<u32>,
    /// Per-voxel weight normalisation factor.
    pub weight_sum: Array3<f64>,
}

/// Joint Label Fusion (Wang et al., 2013) for 3D label volumes.
///
/// Each voxel in the target image is labelled by a weighted majority vote over
/// the atlas labels.  The weight for each atlas at each location is derived from
/// the normalised cross-correlation between patches in the target image and
/// the corresponding atlas image.
pub struct JointLabelFusion {
    config: JlfConfig,
}

impl JointLabelFusion {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: JlfConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: JlfConfig) -> Self {
        Self { config }
    }

    /// Perform joint label fusion.
    ///
    /// # Arguments
    ///
    /// * `target` – intensity volume of the target subject (`f64`, shape `[nz,ny,nx]`).
    /// * `atlas_images` – intensity volumes of the registered atlas subjects.
    /// * `atlas_labels` – corresponding label volumes.
    ///
    /// All volumes must have the same shape.
    pub fn fuse(
        &self,
        target: &Array3<f64>,
        atlas_images: &[Array3<f64>],
        atlas_labels: &[Array3<u32>],
    ) -> NdimageResult<JlfResult> {
        if atlas_images.len() != atlas_labels.len() {
            return Err(NdimageError::InvalidInput(
                "JointLabelFusion: atlas_images and atlas_labels must have equal length"
                    .to_string(),
            ));
        }
        let n_atlases = atlas_images.len();
        if n_atlases == 0 {
            return Err(NdimageError::InvalidInput(
                "JointLabelFusion: must provide at least one atlas".to_string(),
            ));
        }

        let ts = target.shape();
        for (i, ai) in atlas_images.iter().enumerate() {
            if ai.shape() != ts {
                return Err(NdimageError::DimensionError(format!(
                    "JointLabelFusion: atlas_images[{}] shape {:?} ≠ target shape {:?}",
                    i,
                    ai.shape(),
                    ts
                )));
            }
        }
        for (i, al) in atlas_labels.iter().enumerate() {
            if al.shape() != ts {
                return Err(NdimageError::DimensionError(format!(
                    "JointLabelFusion: atlas_labels[{}] shape {:?} ≠ target shape {:?}",
                    i,
                    al.shape(),
                    ts
                )));
            }
        }

        let (nz, ny, nx) = (ts[0], ts[1], ts[2]);
        let pr = self.config.patch_radius as isize;

        // Accumulate weighted label votes
        // For efficiency, accumulate per-label probability maps
        let mut label_votes: HashMap<u32, Array3<f64>> = HashMap::new();
        let mut weight_sum = Array3::<f64>::zeros((nz, ny, nx));

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    // Extract target patch
                    let t_patch =
                        extract_patch_3d(target, iz as isize, iy as isize, ix as isize, pr);

                    // Compute weights for each atlas
                    let mut weights = Vec::with_capacity(n_atlases);
                    for a in 0..n_atlases {
                        let a_patch = extract_patch_3d(
                            &atlas_images[a],
                            iz as isize,
                            iy as isize,
                            ix as isize,
                            pr,
                        );
                        let w = self.patch_weight(&t_patch, &a_patch);
                        weights.push(w);
                    }

                    // Normalise weights
                    let w_sum: f64 = weights.iter().sum();
                    let w_norm: Vec<f64> = if w_sum > 1e-12 {
                        weights.iter().map(|&w| w / w_sum).collect()
                    } else {
                        vec![1.0 / n_atlases as f64; n_atlases]
                    };

                    // Accumulate votes
                    let total_w: f64 = w_norm.iter().sum();
                    weight_sum[[iz, iy, ix]] = total_w;
                    for (a, &wn) in w_norm.iter().enumerate() {
                        let lv = atlas_labels[a][[iz, iy, ix]];
                        label_votes
                            .entry(lv)
                            .or_insert_with(|| Array3::<f64>::zeros((nz, ny, nx)))[[iz, iy, ix]] +=
                            wn;
                    }
                }
            }
        }

        // Winner-take-all label selection
        let mut label_result = Array3::<u32>::zeros((nz, ny, nx));
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let winner_label = label_votes
                        .iter()
                        .max_by(|a, b| {
                            a.1[[iz, iy, ix]]
                                .partial_cmp(&b.1[[iz, iy, ix]])
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| b.0.cmp(a.0))
                        })
                        .map(|(&lv, _)| lv)
                        .unwrap_or(0);
                    label_result[[iz, iy, ix]] = winner_label;
                }
            }
        }

        Ok(JlfResult {
            label: label_result,
            weight_sum,
        })
    }

    /// Compute the similarity weight between a target patch and an atlas patch.
    ///
    /// Uses a normalised sum of squared differences (NSSD) decay function.
    fn patch_weight(&self, target_patch: &[f64], atlas_patch: &[f64]) -> f64 {
        if target_patch.is_empty() {
            return 1.0;
        }
        let n = target_patch.len().min(atlas_patch.len()) as f64;
        let ssd: f64 = target_patch
            .iter()
            .zip(atlas_patch.iter())
            .map(|(t, a)| (t - a).powi(2))
            .sum();
        let normalised_ssd = ssd / (n * self.config.beta + 1e-10);
        (-normalised_ssd / (self.config.alpha + 1e-10)).exp()
    }
}

/// Extract a cubic patch of half-width `pr` centred at `(iz, iy, ix)`.
///
/// Out-of-bounds positions are clamped (edge replication).
fn extract_patch_3d(vol: &Array3<f64>, iz: isize, iy: isize, ix: isize, pr: isize) -> Vec<f64> {
    let shape = vol.shape();
    let (nz, ny, nx) = (shape[0] as isize, shape[1] as isize, shape[2] as isize);
    let mut patch = Vec::with_capacity(((2 * pr + 1) as usize).pow(3));
    for dz in -pr..=pr {
        for dy in -pr..=pr {
            for dx in -pr..=pr {
                let z = (iz + dz).clamp(0, nz - 1) as usize;
                let y = (iy + dy).clamp(0, ny - 1) as usize;
                let x = (ix + dx).clamp(0, nx - 1) as usize;
                patch.push(vol[[z, y, x]]);
            }
        }
    }
    patch
}

// ─── AtlasSegmentation ───────────────────────────────────────────────────────

/// Fusion method to use in the multi-atlas pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMethod {
    /// Simple majority voting.
    MajorityVoting,
    /// STAPLE probabilistic estimation.
    Staple,
    /// Joint label fusion with patch similarity weighting.
    JointLabelFusion,
}

/// Configuration for the multi-atlas segmentation pipeline.
#[derive(Debug, Clone)]
pub struct AtlasConfig {
    /// Label fusion method.
    pub fusion_method: FusionMethod,
    /// STAPLE configuration (used when `fusion_method == Staple`).
    pub staple_config: StapleConfig,
    /// JLF configuration (used when `fusion_method == JointLabelFusion`).
    pub jlf_config: JlfConfig,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            fusion_method: FusionMethod::MajorityVoting,
            staple_config: StapleConfig::default(),
            jlf_config: JlfConfig::default(),
        }
    }
}

/// Result of multi-atlas segmentation.
#[derive(Debug, Clone)]
pub struct AtlasSegmentationResult {
    /// Final label volume.
    pub label: Array3<u32>,
    /// Number of atlases used.
    pub n_atlases: usize,
    /// Fusion method employed.
    pub fusion_method: FusionMethod,
    /// Optional STAPLE result (populated when `fusion_method == Staple`).
    pub staple_result: Option<StapleResult>,
}

/// Multi-atlas label fusion segmentation pipeline.
///
/// Accepts pre-registered atlas label volumes and fuses them using the
/// configured method.  If JLF is used, corresponding atlas intensity images
/// and the target image must also be provided.
pub struct AtlasSegmentation {
    config: AtlasConfig,
}

impl AtlasSegmentation {
    /// Create with default configuration (majority voting).
    pub fn new() -> Self {
        Self {
            config: AtlasConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: AtlasConfig) -> Self {
        Self { config }
    }

    /// Fuse atlas labels.
    ///
    /// `atlas_labels` are the registered atlas segmentations.
    /// `target_image` / `atlas_images` are required only when
    /// `fusion_method == JointLabelFusion`.
    pub fn segment(
        &self,
        atlas_labels: &[Array3<u32>],
        target_image: Option<&Array3<f64>>,
        atlas_images: Option<&[Array3<f64>]>,
    ) -> NdimageResult<AtlasSegmentationResult> {
        let n_atlases = atlas_labels.len();
        match self.config.fusion_method {
            FusionMethod::MajorityVoting => {
                let label = MajorityVoting::fuse(atlas_labels)?;
                Ok(AtlasSegmentationResult {
                    label,
                    n_atlases,
                    fusion_method: FusionMethod::MajorityVoting,
                    staple_result: None,
                })
            }
            FusionMethod::Staple => {
                let staple = STAPLE::with_config(self.config.staple_config.clone());
                let sr = staple.estimate(atlas_labels)?;
                let label = sr.label.clone();
                Ok(AtlasSegmentationResult {
                    label,
                    n_atlases,
                    fusion_method: FusionMethod::Staple,
                    staple_result: Some(sr),
                })
            }
            FusionMethod::JointLabelFusion => {
                let target = target_image.ok_or_else(|| {
                    NdimageError::InvalidInput(
                        "AtlasSegmentation: JLF requires target_image".to_string(),
                    )
                })?;
                let imgs = atlas_images.ok_or_else(|| {
                    NdimageError::InvalidInput(
                        "AtlasSegmentation: JLF requires atlas_images".to_string(),
                    )
                })?;
                let jlf = JointLabelFusion::with_config(self.config.jlf_config.clone());
                let jr = jlf.fuse(target, imgs, atlas_labels)?;
                Ok(AtlasSegmentationResult {
                    label: jr.label,
                    n_atlases,
                    fusion_method: FusionMethod::JointLabelFusion,
                    staple_result: None,
                })
            }
        }
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    /// Create a simple test label volume: foreground sphere in the centre.
    fn sphere_labels(nz: usize, ny: usize, nx: usize, label: u32) -> Array3<u32> {
        let mut a = Array3::<u32>::zeros((nz, ny, nx));
        let cz = nz as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        let cx = nx as f64 / 2.0;
        let r2 = ((nz.min(ny).min(nx)) as f64 / 3.0).powi(2);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let d2 = (iz as f64 - cz).powi(2)
                        + (iy as f64 - cy).powi(2)
                        + (ix as f64 - cx).powi(2);
                    if d2 < r2 {
                        a[[iz, iy, ix]] = label;
                    }
                }
            }
        }
        a
    }

    #[test]
    fn test_majority_voting_identical_atlases() {
        let a = sphere_labels(8, 8, 8, 1);
        let labels = vec![a.clone(), a.clone(), a.clone()];
        let fused = MajorityVoting::fuse(&labels)
            .expect("MajorityVoting::fuse should succeed with identical atlases");
        // All atlases agree → output must equal input
        for iz in 0..8 {
            for iy in 0..8 {
                for ix in 0..8 {
                    assert_eq!(fused[[iz, iy, ix]], a[[iz, iy, ix]]);
                }
            }
        }
    }

    #[test]
    fn test_majority_voting_confidence_perfect() {
        let a = sphere_labels(6, 6, 6, 1);
        let labels = vec![a.clone(), a.clone()];
        let conf = MajorityVoting::confidence(&labels)
            .expect("MajorityVoting::confidence should succeed with identical atlases");
        for v in conf.iter() {
            assert!((*v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_majority_voting_tie_breaks() {
        // Two atlases disagree at every voxel: label 1 vs label 2
        let a = sphere_labels(4, 4, 4, 1);
        let b = sphere_labels(4, 4, 4, 2);
        let labels = vec![a, b];
        let fused = MajorityVoting::fuse(&labels)
            .expect("MajorityVoting::fuse should succeed with two atlases");
        // Tie: smallest label (1) should win when counts are equal
        for v in fused.iter() {
            assert!(*v == 1 || *v == 0, "unexpected label {}", v);
        }
    }

    #[test]
    fn test_staple_smoke() {
        let a = sphere_labels(4, 4, 4, 1);
        let b = sphere_labels(4, 4, 4, 1);
        let labels = vec![a, b];
        let staple = STAPLE::new();
        let result = staple
            .estimate(&labels)
            .expect("STAPLE::estimate should succeed with valid atlases");
        assert_eq!(result.performance.len(), 2);
        // Sensitivities should be high for identical atlases
        for perf in &result.performance {
            assert!(
                perf.sensitivity > 0.5,
                "Expected high sensitivity, got {}",
                perf.sensitivity
            );
        }
    }

    #[test]
    fn test_staple_single_atlas() {
        let a = sphere_labels(4, 4, 4, 1);
        let labels = vec![a];
        let result = STAPLE::new()
            .estimate(&labels)
            .expect("STAPLE::estimate should succeed with single atlas");
        assert_eq!(result.performance.len(), 1);
    }

    #[test]
    fn test_jlf_smoke() {
        let n = 6;
        let target = Array3::<f64>::from_elem((n, n, n), 100.0);
        let atlas_img = Array3::<f64>::from_elem((n, n, n), 100.0);
        let atlas_label = sphere_labels(n, n, n, 1);
        let jlf = JointLabelFusion::new();
        let result = jlf
            .fuse(&target, &[atlas_img], std::slice::from_ref(&atlas_label))
            .expect("JLF::fuse should succeed with single identical atlas");
        // Single identical atlas → output == input labels
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    assert_eq!(result.label[[iz, iy, ix]], atlas_label[[iz, iy, ix]]);
                }
            }
        }
    }

    #[test]
    fn test_atlas_segmentation_majority_voting() {
        let a = sphere_labels(6, 6, 6, 1);
        let labels = vec![a.clone(), a.clone()];
        let seg = AtlasSegmentation::new();
        let result = seg
            .segment(&labels, None, None)
            .expect("AtlasSegmentation::segment should succeed with valid atlases");
        assert_eq!(result.fusion_method, FusionMethod::MajorityVoting);
        assert_eq!(result.n_atlases, 2);
    }

    #[test]
    fn test_atlas_segmentation_staple() {
        let a = sphere_labels(4, 4, 4, 1);
        let labels = vec![a.clone(), a.clone()];
        let config = AtlasConfig {
            fusion_method: FusionMethod::Staple,
            ..Default::default()
        };
        let seg = AtlasSegmentation::with_config(config);
        let result = seg
            .segment(&labels, None, None)
            .expect("AtlasSegmentation STAPLE should succeed with valid atlases");
        assert_eq!(result.fusion_method, FusionMethod::Staple);
        assert!(result.staple_result.is_some());
    }
}
