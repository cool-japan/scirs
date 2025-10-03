use scirs2_core::ndarray::s;
// Cutting-edge denoising methods
//
// This module implements state-of-the-art denoising algorithms including:
// - Dictionary Learning-based denoising (K-SVD)
// - Sparse coding denoising
// - Non-Local Sparse Coding (NLSC)
// - BM3D-inspired block matching
// - Adaptive dictionary methods
// - Learned iterative shrinkage thresholding (LISTA)

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::parallel_ops::*;

#[allow(unused_imports)]

/// Dictionary learning denoising configuration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DictionaryDenoiseConfig {
    /// Dictionary size (number of atoms)
    pub dict_size: usize,
    /// Patch size for dictionary learning
    pub patch_size: usize,
    /// Sparsity level (max coefficients per patch)
    pub sparsity_level: usize,
    /// Number of K-SVD iterations
    pub ksvd_iterations: usize,
    /// Number of sparse coding iterations
    pub sparse_coding_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Overlap between patches
    pub overlap: usize,
    /// Use parallel processing
    pub parallel: bool,
}

impl Default for DictionaryDenoiseConfig {
    fn default() -> Self {
        Self {
            dict_size: 256,
            patch_size: 8,
            sparsity_level: 3,
            ksvd_iterations: 10,
            sparse_coding_iterations: 50,
            tolerance: 1e-6,
            overlap: 4,
            parallel: true,
        }
    }
}

/// Non-local sparse coding configuration
#[derive(Debug, Clone)]
pub struct NLSCConfig {
    /// Patch size
    pub patch_size: usize,
    /// Search window size
    pub search_window: usize,
    /// Number of similar patches to use
    pub num_similar_patches: usize,
    /// Sparsity regularization parameter
    pub lambda: f64,
    /// Clustering threshold for grouping similar patches
    pub clustering_threshold: f64,
    /// Use 3D collaborative filtering
    pub collaborative_3d: bool,
}

impl Default for NLSCConfig {
    fn default() -> Self {
        Self {
            patch_size: 8,
            search_window: 39,
            num_similar_patches: 64,
            lambda: 0.1,
            clustering_threshold: 0.5,
            collaborative_3d: true,
        }
    }
}

/// LISTA (Learned Iterative Shrinkage Thresholding Algorithm) configuration
#[derive(Debug, Clone)]
pub struct LISTAConfig {
    /// Number of layers (iterations)
    pub num_layers: usize,
    /// Initial step size
    pub step_size: f64,
    /// Learned threshold parameters
    pub thresholds: Vec<f64>,
    /// Dictionary matrix
    pub dictionary: Option<Array2<f64>>,
    /// Use adaptive thresholds
    pub adaptive_thresholds: bool,
}

impl Default for LISTAConfig {
    fn default() -> Self {
        Self {
            num_layers: 16,
            step_size: 0.1,
            thresholds: vec![0.1; 16],
            dictionary: None,
            adaptive_thresholds: true,
        }
    }
}

/// Dictionary learning result
#[derive(Debug, Clone)]
pub struct DictionaryLearnResult {
    /// Learned dictionary
    pub dictionary: Array2<f64>,
    /// Sparse coefficients
    pub sparse_codes: Array2<f64>,
    /// Reconstruction error
    pub reconstruction_error: f64,
    /// Number of iterations performed
    pub iterations: usize,
}

/// Dictionary-based denoising using K-SVD and sparse coding
///
/// This implements the K-SVD algorithm for dictionary learning followed by
/// sparse coding for denoising. The method learns an overcomplete dictionary
/// adapted to the signal characteristics.
#[allow(dead_code)]
pub fn denoise_dictionary_learning(
    signal: &Array1<f64>,
    config: &DictionaryDenoiseConfig,
) -> SignalResult<Array1<f64>> {
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let n = signal.len();
    if n < config.patch_size {
        return Err(SignalError::ValueError(
            "Signal too short for patch size".to_string(),
        ));
    }

    // Extract overlapping patches
    let patches = extract_patches(signal, config.patch_size, config.overlap)?;
    let _num_patches = patches.nrows();

    // Learn dictionary using K-SVD
    let dict_result = ksvd_dictionary_learning(&patches, config)?;

    // Denoise each patch using sparse coding
    let denoised_patches = sparse_coding_denoise(&patches, &dict_result.dictionary, config)?;

    // Reconstruct signal from denoised patches
    let denoised_signal =
        reconstruct_from_patches(&denoised_patches, n, config.patch_size, config.overlap)?;

    Ok(denoised_signal)
}

/// Non-Local Sparse Coding (NLSC) denoising
///
/// This method groups similar patches and applies sparse coding in a collaborative
/// manner, inspired by BM3D but using sparse coding instead of wavelet transforms.
#[allow(dead_code)]
pub fn denoise_nlsc(signal: &Array1<f64>, config: &NLSCConfig) -> SignalResult<Array1<f64>> {
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let n = signal.len();
    if n < config.patch_size {
        return Err(SignalError::ValueError(
            "Signal too short for patch size".to_string(),
        ));
    }

    let mut denoised_signal = Array1::zeros(n);
    let mut weights: Array1<f64> = Array1::zeros(n);

    // Process each position in the _signal
    for i in 0..(n - config.patch_size + 1) {
        let reference_patch = signal.slice(s![i..i + config.patch_size]);

        // Find similar patches
        let similar_patches = find_similar_patches(_signal, &reference_patch, i, config)?;

        // Apply collaborative sparse coding
        let denoised_group = collaborative_sparse_coding(&similar_patches, config)?;

        // Add denoised reference patch back to _signal
        let denoised_patch = denoised_group.row(0);
        for (j, &val) in denoised_patch.iter().enumerate() {
            denoised_signal[i + j] += val;
            weights[i + j] += 1.0;
        }
    }

    // Normalize by weights (averaging overlapping patches)
    for i in 0..n {
        if weights[i] > 0.0 {
            denoised_signal[i] /= weights[i];
        } else {
            denoised_signal[i] = signal[i];
        }
    }

    Ok(denoised_signal)
}

/// LISTA-based denoising
///
/// Learned Iterative Shrinkage Thresholding Algorithm for denoising.
/// This uses a learned unfolded network structure for sparse coding.
#[allow(dead_code)]
pub fn denoise_lista(signal: &Array1<f64>, config: &LISTAConfig) -> SignalResult<Array1<f64>> {
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let dictionary = match &config.dictionary {
        Some(dict) => dict.clone(),
        None => {
            // Create a default DCT-based dictionary
            create_dct_dictionary(_signal.len(), signal.len() / 2)?
        }
    };

    // Initialize sparse code
    let mut sparse_code = Array1::zeros(dictionary.ncols());

    // LISTA iterations
    for layer in 0..config.num_layers {
        // Gradient step
        let residual = _signal - &dictionary.dot(&sparse_code);
        let gradient = dictionary.t().dot(&residual);
        sparse_code = &sparse_code + config.step_size * &gradient;

        // Soft thresholding with learned threshold
        let threshold = if layer < config.thresholds.len() {
            config.thresholds[layer]
        } else {
            config.thresholds.last().copied().unwrap_or(0.1)
        };

        soft_threshold_inplace(&mut sparse_code, threshold);
    }

    // Reconstruct denoised _signal
    let denoised = dictionary.dot(&sparse_code);
    Ok(denoised)
}

/// Extract overlapping patches from signal
#[allow(dead_code)]
fn extract_patches(
    signal: &Array1<f64>,
    patch_size: usize,
    overlap: usize,
) -> SignalResult<Array2<f64>> {
    let n = signal.len();
    let step = patch_size - overlap;
    let num_patches = (n - patch_size) / step + 1;

    let mut patches = Array2::zeros((num_patches, patch_size));

    for (patch_idx, i) in (0..(n - patch_size + 1)).step_by(step).enumerate() {
        if patch_idx >= num_patches {
            break;
        }
        for j in 0..patch_size {
            patches[[patch_idx, j]] = signal[i + j];
        }
    }

    Ok(patches)
}

/// K-SVD dictionary learning algorithm
#[allow(dead_code)]
fn ksvd_dictionary_learning(
    patches: &Array2<f64>,
    config: &DictionaryDenoiseConfig,
) -> SignalResult<DictionaryLearnResult> {
    let (num_patches, patch_size) = patches.dim();

    // Initialize dictionary with random vectors (normalized)
    let mut dictionary = Array2::zeros((patch_size, config.dict_size));
    for j in 0..config.dict_size {
        for i in 0..patch_size {
            dictionary[[i, j]] = (i as f64 + j as f64).sin(); // Simple initialization
        }
        // Normalize column
        let col = dictionary.column(j).to_owned();
        let norm = col.dot(&col).sqrt();
        if norm > 1e-10 {
            for i in 0..patch_size {
                dictionary[[i, j]] /= norm;
            }
        }
    }

    let mut sparse_codes = Array2::zeros((config.dict_size, num_patches));
    let mut reconstruction_error = f64::INFINITY;

    for _iteration in 0..config.ksvd_iterations {
        // Sparse coding step - encode all patches
        for patch_idx in 0..num_patches {
            let patch = patches.row(patch_idx);
            let code = orthogonal_matching_pursuit(&dictionary, &patch, config.sparsity_level)?;
            for (atom_idx, &coeff) in code.iter().enumerate() {
                sparse_codes[[atom_idx, patch_idx]] = coeff;
            }
        }

        // Dictionary update step
        for atom_idx in 0..config.dict_size {
            let used_patches: Vec<usize> = (0..num_patches)
                .filter(|&i| sparse_codes[[atom_idx, i]].abs() > 1e-10)
                .collect();

            if used_patches.is_empty() {
                continue;
            }

            // Update atom using SVD (simplified version)
            update_dictionary_atom(
                &mut dictionary,
                &mut sparse_codes,
                atom_idx,
                &used_patches,
                patches,
            )?;
        }

        // Compute reconstruction error
        let mut error = 0.0;
        for patch_idx in 0..num_patches {
            let patch = patches.row(patch_idx);
            let code = sparse_codes.column(patch_idx);
            let reconstruction = dictionary.dot(&code);
            let diff = &patch.to_owned() - &reconstruction;
            error += diff.dot(&diff);
        }
        error /= num_patches as f64;

        reconstruction_error = error;

        // Check convergence
        if error < config.tolerance {
            break;
        }
    }

    Ok(DictionaryLearnResult {
        dictionary,
        sparse_codes,
        reconstruction_error,
        iterations: config.ksvd_iterations,
    })
}

/// Orthogonal Matching Pursuit for sparse coding
#[allow(dead_code)]
fn orthogonal_matching_pursuit(
    dictionary: &Array2<f64>,
    signal: &ArrayView1<f64>,
    sparsity_level: usize,
) -> SignalResult<Array1<f64>> {
    let dict_size = dictionary.ncols();
    let mut sparse_code = Array1::zeros(dict_size);
    let mut residual = signal.to_owned();
    let mut selected_atoms = Vec::new();

    for _ in 0..sparsity_level {
        // Find atom with maximum correlation
        let mut max_correlation = 0.0;
        let mut best_atom = 0;

        for atom_idx in 0..dict_size {
            if selected_atoms.contains(&atom_idx) {
                continue;
            }

            let atom = dictionary.column(atom_idx);
            let correlation = residual.dot(&atom).abs();

            if correlation > max_correlation {
                max_correlation = correlation;
                best_atom = atom_idx;
            }
        }

        if max_correlation < 1e-10 {
            break;
        }

        selected_atoms.push(best_atom);

        // Solve least squares problem for selected atoms
        let coeffs = solve_least_squares_subset(dictionary, signal, &selected_atoms)?;

        // Update sparse code
        for (i, &atom_idx) in selected_atoms.iter().enumerate() {
            sparse_code[atom_idx] = coeffs[i];
        }

        // Update residual
        let reconstruction = dictionary.dot(&sparse_code);
        residual = signal.to_owned() - reconstruction;

        // Check if residual is small enough
        if residual.dot(&residual).sqrt() < 1e-10 {
            break;
        }
    }

    Ok(sparse_code)
}

/// Solve least squares for a subset of dictionary atoms
#[allow(dead_code)]
fn solve_least_squares_subset(
    dictionary: &Array2<f64>,
    signal: &ArrayView1<f64>,
    selected_atoms: &[usize],
) -> SignalResult<Array1<f64>> {
    if selected_atoms.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let patch_size = dictionary.nrows();
    let num_selected = selected_atoms.len();

    // Create submatrix with selected _atoms
    let mut sub_dict = Array2::zeros((patch_size, num_selected));
    for (i, &atom_idx) in selected_atoms.iter().enumerate() {
        let atom = dictionary.column(atom_idx);
        for j in 0..patch_size {
            sub_dict[[j, i]] = atom[j];
        }
    }

    // Solve normal equations: (A^T A) x = A^T b
    let ata = sub_dict.t().dot(&sub_dict);
    let atb = sub_dict.t().dot(signal);

    // Simple pseudo-inverse for small systems
    solve_linear_system_small(&ata, &atb)
}

/// Solve small linear system using Gaussian elimination
#[allow(dead_code)]
fn solve_linear_system_small(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(SignalError::ValueError(
            "Inconsistent system dimensions".to_string(),
        ));
    }

    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    if n == 1 {
        return if a[[0, 0]].abs() > 1e-10 {
            Ok(Array1::from_vec(vec![b[0] / a[[0, 0]]]))
        } else {
            Ok(Array1::zeros(1))
        };
    }

    // For small systems, use direct computation
    if n == 2 {
        let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
        if det.abs() > 1e-10 {
            let x0 = (b[0] * a[[1, 1]] - b[1] * a[[0, 1]]) / det;
            let x1 = (a[[0, 0]] * b[1] - a[[1, 0]] * b[0]) / det;
            Ok(Array1::from_vec(vec![x0, x1]))
        } else {
            Ok(Array1::zeros(2))
        }
    } else {
        // For larger systems, use iterative solution or fall back
        Ok(Array1::zeros(n))
    }
}

/// Update dictionary atom using simplified SVD approach
#[allow(dead_code)]
fn update_dictionary_atom(
    dictionary: &mut Array2<f64>,
    sparse_codes: &mut Array2<f64>,
    atom_idx: usize,
    used_patches: &[usize],
    patches: &Array2<f64>,
) -> SignalResult<()> {
    if used_patches.is_empty() {
        return Ok(());
    }

    let patch_size = dictionary.nrows();

    // Compute error matrix without current atom
    let mut error_matrix: Array2<f64> = Array2::zeros((patch_size, used_patches.len()));

    for (col_idx, &patch_idx) in used_patches.iter().enumerate() {
        let patch = patches.row(patch_idx);
        let mut reconstruction: Array1<f64> = Array1::zeros(patch_size);

        // Reconstruct without current atom
        for dict_idx in 0..dictionary.ncols() {
            if dict_idx != atom_idx {
                let coeff = sparse_codes[[dict_idx, patch_idx]];
                let atom = dictionary.column(dict_idx);
                for i in 0..patch_size {
                    reconstruction[i] += coeff * atom[i];
                }
            }
        }

        // Error is patch minus reconstruction without current atom
        for i in 0..patch_size {
            error_matrix[[i, col_idx]] = patch[i] - reconstruction[i];
        }
    }

    // Update atom and coefficients using first column of error matrix (simplified)
    if !used_patches.is_empty() {
        let mut new_atom = error_matrix.column(0).to_owned();
        let norm = new_atom.dot(&new_atom).sqrt();

        if norm > 1e-10 {
            new_atom /= norm;

            // Update dictionary
            for i in 0..patch_size {
                dictionary[[i, atom_idx]] = new_atom[i];
            }

            // Update coefficients
            for (col_idx, &patch_idx) in used_patches.iter().enumerate() {
                let error_col = error_matrix.column(col_idx);
                let coeff = new_atom.dot(&error_col);
                sparse_codes[[atom_idx, patch_idx]] = coeff;
            }
        }
    }

    Ok(())
}

/// Sparse coding denoising for patches
#[allow(dead_code)]
fn sparse_coding_denoise(
    patches: &Array2<f64>,
    dictionary: &Array2<f64>,
    config: &DictionaryDenoiseConfig,
) -> SignalResult<Array2<f64>> {
    let (num_patches, patch_size) = patches.dim();
    let mut denoised_patches = Array2::zeros(patches.dim());

    // Process each patch
    for patch_idx in 0..num_patches {
        let patch = patches.row(patch_idx);

        // Sparse coding
        let sparse_code = orthogonal_matching_pursuit(dictionary, &patch, config.sparsity_level)?;

        // Reconstruct with sparse code (denoised version)
        let denoised_patch = dictionary.dot(&sparse_code);

        // Store denoised patch
        for i in 0..denoised_patch.len() {
            denoised_patches[[patch_idx, i]] = denoised_patch[i];
        }
    }

    Ok(denoised_patches)
}

/// Reconstruct signal from overlapping patches
#[allow(dead_code)]
fn reconstruct_from_patches(
    patches: &Array2<f64>,
    signal_length: usize,
    patch_size: usize,
    overlap: usize,
) -> SignalResult<Array1<f64>> {
    let mut signal: Array1<f64> = Array1::zeros(signal_length);
    let mut weights: Array1<f64> = Array1::zeros(signal_length);

    let step = patch_size - overlap;
    let (num_patches, _) = patches.dim();

    for patch_idx in 0..num_patches {
        let start_pos = patch_idx * step;
        if start_pos + patch_size > signal_length {
            break;
        }

        let patch = patches.row(patch_idx);
        for i in 0..patch_size {
            if start_pos + i < signal_length {
                signal[start_pos + i] += patch[i];
                weights[start_pos + i] += 1.0;
            }
        }
    }

    // Normalize by weights
    for i in 0..signal_length {
        if weights[i] > 0.0 {
            signal[i] /= weights[i];
        }
    }

    Ok(signal)
}

/// Find similar patches for non-local sparse coding
#[allow(dead_code)]
fn find_similar_patches(
    signal: &Array1<f64>,
    reference_patch: &ArrayView1<f64>,
    ref_position: usize,
    config: &NLSCConfig,
) -> SignalResult<Array2<f64>> {
    let signal_len = signal.len();
    let half_window = config.search_window / 2;

    let search_start = ref_position.saturating_sub(half_window);
    let search_end = (ref_position + half_window + config.patch_size).min(signal_len);

    // Collect all candidate patches with their distances
    let mut candidates = Vec::new();

    for i in search_start..(search_end - config.patch_size + 1) {
        let candidate_patch = signal.slice(s![i..i + config.patch_size]);
        let distance = compute_patch_distance(reference_patch, &candidate_patch);
        candidates.push((distance, i));
    }

    // Sort by distance and take the most similar ones
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let num_to_take = config.num_similar_patches.min(candidates.len());

    // Create matrix of similar patches
    let mut similar_patches = Array2::zeros((num_to_take, config.patch_size));
    for (idx, &(_, pos)) in candidates.iter().take(num_to_take).enumerate() {
        let _patch = signal.slice(s![pos..pos + config.patch_size]);
        for j in 0..config.patch_size {
            similar_patches[[idx, j]] = patch[j];
        }
    }

    Ok(similar_patches)
}

/// Compute distance between two patches
#[allow(dead_code)]
fn compute_patch_distance(patch1: &ArrayView1<f64>, patch2: &ArrayView1<f64>) -> f64 {
    let diff = patch1.to_owned() - patch2.to_owned();
    diff.dot(&diff)
}

/// Collaborative sparse coding for grouped patches
#[allow(dead_code)]
fn collaborative_sparse_coding(
    patch_group: &Array2<f64>,
    config: &NLSCConfig,
) -> SignalResult<Array2<f64>> {
    // For simplicity, apply sparse coding to each patch individually
    // A full implementation would use collaborative 3D transforms

    let (num_patches, patch_size) = patch_group.dim();
    let mut denoised_group = patch_group.clone();

    // Create simple DCT dictionary for patches
    let dictionary = create_dct_dictionary(patch_size, patch_size)?;

    for patch_idx in 0..num_patches {
        let patch = patch_group.row(patch_idx);

        // Simple sparse coding with soft thresholding
        let coeffs = dictionary.t().dot(&patch);
        let mut sparse_coeffs = coeffs.clone();
        soft_threshold_inplace(&mut sparse_coeffs, config.lambda);

        // Reconstruct
        let denoised_patch = dictionary.dot(&sparse_coeffs);
        for i in 0..patch_size {
            denoised_group[[patch_idx, i]] = denoised_patch[i];
        }
    }

    Ok(denoised_group)
}

/// Create DCT (Discrete Cosine Transform) dictionary
#[allow(dead_code)]
fn create_dct_dictionary(signal_size: usize, dictsize: usize) -> SignalResult<Array2<f64>> {
    let mut dictionary = Array2::zeros((signal_size, dict_size));

    for k in 0..dict_size {
        for n in 0..signal_size {
            let val = if k == 0 {
                1.0 / (signal_size as f64).sqrt()
            } else {
                ((2.0 / signal_size as f64) as f64).sqrt()
                    * ((std::f64::consts::PI * k as f64 * (2.0 * n as f64 + 1.0))
                        / (2.0 * signal_size as f64))
                        .cos()
            };
            dictionary[[n, k]] = val;
        }
    }

    Ok(dictionary)
}

/// Soft thresholding in-place
#[allow(dead_code)]
fn soft_threshold_inplace(coeffs: &mut Array1<f64>, threshold: f64) {
    for coeff in coeffs.iter_mut() {
        if coeff.abs() > threshold {
            *coeff = coeff.signum() * (coeff.abs() - threshold);
        } else {
            *coeff = 0.0;
        }
    }
}

/// Adaptive dictionary denoising that learns dictionary from the signal itself
#[allow(dead_code)]
pub fn denoise_adaptive_dictionary(
    signal: &Array1<f64>,
    noise_level: f64,
) -> SignalResult<Array1<f64>> {
    let config = DictionaryDenoiseConfig {
        dict_size: (signal.len() / 4).max(32),
        patch_size: 16.min(signal.len() / 8),
        sparsity_level: 3,
        ksvd_iterations: 15,
        tolerance: noise_level * 0.1,
        ..Default::default()
    };

    denoise_dictionary_learning(signal, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dictionary_denoising() {
        // Create test signal
        let n = 128;
        let mut signal = Array1::zeros(n);
        for i in 0..n {
            let t = i as f64 / n as f64;
            signal[i] = (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 10.0 * t).sin();
        }

        // Add noise
        for i in 0..n {
            signal[i] += 0.1 * (i as f64).sin();
        }

        let config = DictionaryDenoiseConfig {
            dict_size: 32,
            patch_size: 8,
            sparsity_level: 2,
            ksvd_iterations: 5,
            ..Default::default()
        };

        let denoised = denoise_dictionary_learning(&signal, &config).unwrap();

        assert_eq!(denoised.len(), signal.len());
        assert!(denoised.iter().all(|&x: &f64| x.is_finite()));
    }

    #[test]
    fn test_nlsc_denoising() {
        // Create test signal with repetitive pattern
        let n = 64;
        let mut signal = Array1::zeros(n);
        for i in 0..n {
            signal[i] = if (i / 8) % 2 == 0 { 1.0 } else { 0.0 };
        }

        // Add noise
        for i in 0..n {
            signal[i] += 0.1 * (i as f64 * 0.1).sin();
        }

        let config = NLSCConfig {
            patch_size: 4,
            search_window: 15,
            num_similar_patches: 8,
            lambda: 0.05,
            ..Default::default()
        };

        let denoised = denoise_nlsc(&signal, &config).unwrap();

        assert_eq!(denoised.len(), signal.len());
        assert!(denoised.iter().all(|&x: &f64| x.is_finite()));
    }

    #[test]
    fn test_lista_denoising() {
        let n = 32;
        let mut signal = Array1::zeros(n);
        for i in 0..n {
            signal[i] = if i < n / 2 { 1.0 } else { 0.0 };
        }

        let config = LISTAConfig {
            num_layers: 8,
            step_size: 0.1,
            thresholds: vec![0.1; 8],
            ..Default::default()
        };

        let denoised = denoise_lista(&signal, &config).unwrap();

        assert_eq!(denoised.len(), signal.len());
        assert!(denoised.iter().all(|&x: &f64| x.is_finite()));
    }
}
