// 2D Wavelet Packet Transform (WPT2D)
//
// This module provides implementations of the 2D Wavelet Packet Transform (WPT2D),
// which is a generalization of the 2D wavelet transform that offers richer signal
// analysis. Unlike standard wavelet transforms that decompose only the approximation
// coefficients, wavelet packets also decompose the detail coefficients, resulting
// in a full quad-tree of subbands.
//
// The 2D WPT is useful for applications such as:
// * Texture analysis and classification
// * Feature extraction for pattern recognition
// * Image compression with adaptive basis selection
// * Image denoising with selective reconstruction
// * Edge detection with customized subband selection
//
// # Performance Optimizations
//
// This implementation builds on the validated separable 2D DWT primitives
// (`dwt2d_decompose` / `dwt2d_reconstruct`), which means the analysis and
// synthesis stages form a true inverse pair. Reconstruction is performed by
// recombining the four sibling sub-band packets of every parent node, level by
// level, starting from the leaf nodes and ascending to the root.
//
// # Examples
//
// Basic usage:
//
// ```
// use scirs2_core::ndarray::Array2;
// use scirs2_signal::dwt::Wavelet;
// use scirs2_signal::wpt2d::wpt2d_full;
//
// // Create a test image
// let mut image = Array2::zeros((64, 64));
// for i in 0..64 {
//     for j in 0..64 {
//         image[[i, j]] = (i * j) as f64 / 64.0;
//     }
// }
//
// // Perform 2D wavelet packet decomposition up to level 2
// let decomp = wpt2d_full(&image, Wavelet::Haar, 2, None).expect("Operation failed");
//
// // Access the packet at level 2, position (1, 2)
// // This corresponds to the pattern LH-HL
// let packet = decomp.get_packet(2, 1, 2).expect("Operation failed");
//
// // Reconstruct the original image
// let reconstructed = decomp.reconstruct().expect("Operation failed");
// ```

use crate::dwt::Wavelet;
use crate::dwt2d_advanced::{dwt2d_decompose, dwt2d_reconstruct, Dwt2DCoeffs, EdgeMode2D};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, NumCast};
use std::collections::HashMap;
use std::fmt::Debug;

/// Maps the textual extension-mode names accepted by the public WPT2D API onto
/// the [`EdgeMode2D`] variants used by the underlying separable 2D DWT.
///
/// Unknown names fall back to `Symmetric`, matching the default behaviour of the
/// rest of the wavelet API.
fn edge_mode_from_str(mode: Option<&str>) -> EdgeMode2D {
    match mode.unwrap_or("symmetric") {
        "symmetric" => EdgeMode2D::Symmetric,
        "reflect" => EdgeMode2D::Reflect,
        "periodic" | "wrap" | "circular" => EdgeMode2D::Periodic,
        "zero" | "constant" => EdgeMode2D::Zero,
        "replicate" | "edge" | "nearest" => EdgeMode2D::Replicate,
        "antisymmetric" | "asymmetric" => EdgeMode2D::AntiSymmetric,
        _ => EdgeMode2D::Symmetric,
    }
}

/// Represents a 2D wavelet packet node with its position in the tree and coefficient array.
#[derive(Clone)]
pub struct WaveletPacket2D {
    /// The level in the decomposition tree (0 is the root)
    pub level: usize,
    /// The row index within the level (0-indexed)
    pub row: usize,
    /// The column index within the level (0-indexed)
    pub col: usize,
    /// The 2D array of coefficients for this packet
    pub coeffs: Array2<f64>,
    /// The path to this node in the decomposition tree
    /// e.g., "LH-HL" means "low-high" at first level, then "high-low" at second level
    pub path: String,
}

impl WaveletPacket2D {
    /// Creates a new wavelet packet node.
    pub fn new(level: usize, row: usize, col: usize, coeffs: Array2<f64>, path: String) -> Self {
        WaveletPacket2D {
            level,
            row,
            col,
            coeffs,
            path,
        }
    }

    /// Returns the dimensions of the coefficients.
    pub fn shape(&self) -> (usize, usize) {
        self.coeffs.dim()
    }

    /// Returns the energy of this packet (sum of squared coefficients).
    pub fn energy(&self) -> f64 {
        self.coeffs.iter().map(|&x| x * x).sum()
    }
}

/// Represents a full 2D wavelet packet decomposition tree.
pub struct WaveletPacketTree2D {
    /// The wavelet used for the decomposition
    pub wavelet: Wavelet,
    /// The maximum decomposition level
    pub max_level: usize,
    /// The signal extension mode used during decomposition
    edge_mode: EdgeMode2D,
    /// The collection of wavelet packets organized by (level, row, col)
    packets: HashMap<(usize, usize, usize), WaveletPacket2D>,
    /// The shape of the original signal
    originalshape: (usize, usize),
}

impl WaveletPacketTree2D {
    /// Creates a new wavelet packet tree.
    pub fn new(
        wavelet: Wavelet,
        max_level: usize,
        root_coeffs: Array2<f64>,
        mode: Option<&str>,
    ) -> Self {
        let mut packets = HashMap::new();
        let shape = root_coeffs.dim();

        // Create the root node (level 0)
        let root = WaveletPacket2D::new(0, 0, 0, root_coeffs, String::new());
        packets.insert((0, 0, 0), root);

        WaveletPacketTree2D {
            wavelet,
            max_level,
            edge_mode: edge_mode_from_str(mode),
            packets,
            originalshape: shape,
        }
    }

    /// Retrieves a wavelet packet at the specified position.
    pub fn get_packet(&self, level: usize, row: usize, col: usize) -> Option<&WaveletPacket2D> {
        self.packets.get(&(level, row, col))
    }

    /// Retrieves a mutable reference to a wavelet packet at the specified position.
    pub fn get_packet_mut(
        &mut self,
        level: usize,
        row: usize,
        col: usize,
    ) -> Option<&mut WaveletPacket2D> {
        self.packets.get_mut(&(level, row, col))
    }

    /// Adds a wavelet packet to the tree.
    pub fn add_packet(&mut self, packet: WaveletPacket2D) {
        let key = (packet.level, packet.row, packet.col);
        self.packets.insert(key, packet);
    }

    /// Returns all packets at a specific level.
    pub fn get_level_packets(&self, level: usize) -> Vec<&WaveletPacket2D> {
        self.packets
            .iter()
            .filter_map(
                |((l, _, _), packet)| {
                    if *l == level {
                        Some(packet)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Gets the number of packets in the tree.
    pub fn len(&self) -> usize {
        self.packets.len()
    }

    /// Checks if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }

    /// Returns the shape of the original signal.
    pub fn originalshape(&self) -> (usize, usize) {
        self.originalshape
    }

    /// Reconstructs the original signal from the full decomposition.
    ///
    /// This performs the real inverse 2D wavelet packet transform. Starting from
    /// the leaf nodes, every group of four sibling sub-bands (`LL`, `LH`, `HL`,
    /// `HH`) of a parent node is recombined via the single-level inverse 2D DWT
    /// (`dwt2d_reconstruct`). The process is repeated up the quad-tree until the
    /// root (level 0) is reconstructed.
    ///
    /// # Errors
    ///
    /// Returns an error if a required sibling packet is missing from the tree, in
    /// which case a full reconstruction is impossible (e.g. for a tree produced by
    /// [`reconstruct_selective`](Self::reconstruct_selective) with an incomplete
    /// sibling set). The error names the offending node so the caller can supply
    /// the missing coefficients instead of receiving a silently fabricated result.
    pub fn reconstruct(&self) -> SignalResult<Array2<f64>> {
        // If no decomposition was done, the root *is* the signal.
        if self.max_level == 0 {
            return Ok(self
                .get_packet(0, 0, 0)
                .ok_or_else(|| SignalError::ValueError("Root packet missing".to_string()))?
                .coeffs
                .clone());
        }

        // Bottom-up reconstruction: collapse one level at a time.
        //
        // `current` holds the coefficients available at `level`. We start from the
        // deepest level for which packets exist and fold pairs of levels until we
        // reach level 0.
        let deepest = self.deepest_populated_level();

        // Working map from (row, col) -> coefficients at the level currently being
        // collapsed. Seed it from the deepest populated level.
        let mut current: HashMap<(usize, usize), Array2<f64>> = HashMap::new();
        for packet in self.get_level_packets(deepest) {
            current.insert((packet.row, packet.col), packet.coeffs.clone());
        }

        for level in (1..=deepest).rev() {
            let parent_level = level - 1;
            // Determine the set of distinct parents at `parent_level` from the
            // children present in `current`.
            let mut parents: Vec<(usize, usize)> = current
                .keys()
                .map(|&(row, col)| (row / 2, col / 2))
                .collect();
            parents.sort_unstable();
            parents.dedup();

            let mut next: HashMap<(usize, usize), Array2<f64>> = HashMap::new();

            for &(prow, pcol) in &parents {
                let ll = self.child_coeffs(&current, parent_level, prow, pcol, 0, 0, "LL")?;
                let lh = self.child_coeffs(&current, parent_level, prow, pcol, 0, 1, "LH")?;
                let hl = self.child_coeffs(&current, parent_level, prow, pcol, 1, 0, "HL")?;
                let hh = self.child_coeffs(&current, parent_level, prow, pcol, 1, 1, "HH")?;

                // The parent dimensions are twice the child sub-band dimensions.
                let (sub_rows, sub_cols) = ll.dim();
                let parent_shape = (sub_rows * 2, sub_cols * 2);

                let coeffs = Dwt2DCoeffs {
                    ll,
                    lh,
                    hl,
                    hh,
                    wavelet: self.wavelet,
                    edge_mode: self.edge_mode,
                    original_shape: parent_shape,
                };

                let parent_coeffs = dwt2d_reconstruct(&coeffs)?;
                next.insert((prow, pcol), parent_coeffs);
            }

            current = next;
        }

        // After collapsing all levels, `current` must contain exactly the root.
        let root = current.remove(&(0, 0)).ok_or_else(|| {
            SignalError::ValueError(
                "Reconstruction failed to produce a root node from the decomposition tree"
                    .to_string(),
            )
        })?;

        // The separable inverse DWT rounds dimensions up to even sizes. Crop back
        // to the original shape if the root was odd-sized.
        let (orig_rows, orig_cols) = self.originalshape;
        if root.dim() == (orig_rows, orig_cols) {
            Ok(root)
        } else {
            let (rrows, rcols) = root.dim();
            let rows = orig_rows.min(rrows);
            let cols = orig_cols.min(rcols);
            let mut cropped = Array2::zeros((orig_rows, orig_cols));
            for i in 0..rows {
                for j in 0..cols {
                    cropped[[i, j]] = root[[i, j]];
                }
            }
            Ok(cropped)
        }
    }

    /// Returns the coefficients of a specific child sub-band of `(parent_level, prow, pcol)`.
    ///
    /// `row_off`/`col_off` select the quadrant (0/1 in each dimension), and `label`
    /// is used purely to build a descriptive error message when the child is absent.
    fn child_coeffs(
        &self,
        current: &HashMap<(usize, usize), Array2<f64>>,
        parent_level: usize,
        prow: usize,
        pcol: usize,
        row_off: usize,
        col_off: usize,
        label: &str,
    ) -> SignalResult<Array2<f64>> {
        let child_row = prow * 2 + row_off;
        let child_col = pcol * 2 + col_off;
        current
            .get(&(child_row, child_col))
            .cloned()
            .ok_or_else(|| {
                SignalError::ValueError(format!(
                "Cannot reconstruct: missing {} child at level {}, position ({}, {}) for parent \
                 at level {}, position ({}, {})",
                label,
                parent_level + 1,
                child_row,
                child_col,
                parent_level,
                prow,
                pcol
            ))
            })
    }

    /// Returns the deepest level that contains at least one packet.
    fn deepest_populated_level(&self) -> usize {
        self.packets
            .keys()
            .map(|&(level, _, _)| level)
            .max()
            .unwrap_or(0)
    }

    /// Reconstructs the signal using only the specified packets.
    ///
    /// The supplied packets must form a complete quad-tree partition of the signal
    /// (i.e. every parent on a path to a selected leaf must have all four children
    /// available). Otherwise [`reconstruct`](Self::reconstruct) returns an honest
    /// error naming the missing sub-band rather than fabricating coefficients.
    pub fn reconstruct_selective(
        &self,
        selected_packets: &[(usize, usize, usize)],
    ) -> SignalResult<Array2<f64>> {
        // Create a new tree seeded with a zero root of the correct shape, copying
        // the exact edge mode so reconstruction matches the original transform.
        let mut packets = HashMap::new();
        packets.insert(
            (0, 0, 0),
            WaveletPacket2D::new(0, 0, 0, Array2::zeros(self.originalshape), String::new()),
        );
        let mut selective_tree = WaveletPacketTree2D {
            wavelet: self.wavelet,
            max_level: self.max_level,
            edge_mode: self.edge_mode,
            packets,
            originalshape: self.originalshape,
        };

        // Add the selected packets to the new tree.
        for &(level, row, col) in selected_packets {
            if let Some(packet) = self.get_packet(level, row, col) {
                selective_tree.add_packet(packet.clone());
            }
        }

        // Reconstruct from the selective tree.
        selective_tree.reconstruct()
    }
}

/// Performs a 2D wavelet packet transform with full decomposition.
///
/// This function decomposes all subbands at each level, creating a complete
/// quad-tree of wavelet packets.
///
/// # Arguments
///
/// * `data` - The input 2D array (image)
/// * `wavelet` - The wavelet to use for the transform
/// * `max_level` - The maximum decomposition level
/// * `mode` - The signal extension mode (default: "symmetric")
///
/// # Returns
///
/// * A `WaveletPacketTree2D` containing the full decomposition
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::wpt2d::wpt2d_full;
///
/// // Create a test image
/// let mut image = Array2::zeros((16, 16));
/// for i in 0..16 {
///     for j in 0..16 {
///         image[[i, j]] = (i * j) as f64 / 16.0;
///     }
/// }
///
/// // Perform full wavelet packet decomposition up to level 2
/// let decomp = wpt2d_full(&image, Wavelet::Haar, 2, None).expect("Operation failed");
///
/// // Check that we have the expected number of packets:
/// // 1 at level 0, 4 at level 1, 16 at level 2
/// assert_eq!(decomp.len(), 1 + 4 + 16);
/// ```
pub fn wpt2d_full<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    max_level: usize,
    mode: Option<&str>,
) -> SignalResult<WaveletPacketTree2D>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Convert input to f64.
    let root_coeffs = convert_to_f64(data)?;

    if max_level == 0 {
        return Ok(WaveletPacketTree2D::new(wavelet, 0, root_coeffs, mode));
    }

    // Check if the data dimensions are sufficient for the requested level.
    let min_size = 2_usize.pow(max_level as u32);
    let (rows, cols) = data.dim();

    if rows < min_size || cols < min_size {
        return Err(SignalError::ValueError(format!(
            "Input dimensions ({}, {}) are too small for {} decomposition levels. Need at least \
             ({}, {})",
            rows, cols, max_level, min_size, min_size
        )));
    }

    // Initialize the wavelet packet tree.
    let mut tree = WaveletPacketTree2D::new(wavelet, max_level, root_coeffs, mode);

    // Perform the decomposition.
    decompose_node(&mut tree, 0, 0, 0, max_level, mode)?;

    Ok(tree)
}

/// Recursively decomposes a node in the wavelet packet tree.
fn decompose_node(
    tree: &mut WaveletPacketTree2D,
    level: usize,
    row: usize,
    col: usize,
    max_level: usize,
    mode: Option<&str>,
) -> SignalResult<()> {
    // If we've reached the maximum level, stop recursion.
    if level >= max_level {
        return Ok(());
    }

    // Get the current node's coefficients.
    let parent = tree
        .get_packet(level, row, col)
        .ok_or_else(|| {
            SignalError::ValueError(format!(
                "Missing wavelet packet at level {}, position ({}, {})",
                level, row, col
            ))
        })?
        .clone();

    // A node can only be decomposed if both dimensions are at least 2.
    let (prows, pcols) = parent.coeffs.dim();
    if prows < 2 || pcols < 2 {
        return Err(SignalError::ValueError(format!(
            "Cannot decompose packet at level {}, position ({}, {}): dimensions ({}, {}) are too \
             small for a further wavelet packet level",
            level, row, col, prows, pcols
        )));
    }

    // Decompose the coefficients into four subbands using the validated 2D DWT.
    let decomposition = dwt2d_decompose(&parent.coeffs, tree.wavelet, tree.edge_mode)?;

    // Calculate child positions in the next level.
    let child_level = level + 1;
    let child_row_base = row * 2;
    let child_col_base = col * 2;

    // Create child nodes with appropriate paths.
    let sep = if parent.path.is_empty() { "" } else { "-" };

    let child_ll = WaveletPacket2D::new(
        child_level,
        child_row_base,
        child_col_base,
        decomposition.ll,
        format!("{}{}{}", parent.path, sep, "LL"),
    );
    let child_lh = WaveletPacket2D::new(
        child_level,
        child_row_base,
        child_col_base + 1,
        decomposition.lh,
        format!("{}{}{}", parent.path, sep, "LH"),
    );
    let child_hl = WaveletPacket2D::new(
        child_level,
        child_row_base + 1,
        child_col_base,
        decomposition.hl,
        format!("{}{}{}", parent.path, sep, "HL"),
    );
    let child_hh = WaveletPacket2D::new(
        child_level,
        child_row_base + 1,
        child_col_base + 1,
        decomposition.hh,
        format!("{}{}{}", parent.path, sep, "HH"),
    );

    // Add children to the tree.
    tree.add_packet(child_ll);
    tree.add_packet(child_lh);
    tree.add_packet(child_hl);
    tree.add_packet(child_hh);

    // Recursively decompose each child sequentially.
    decompose_node(
        tree,
        child_level,
        child_row_base,
        child_col_base,
        max_level,
        mode,
    )?;
    decompose_node(
        tree,
        child_level,
        child_row_base,
        child_col_base + 1,
        max_level,
        mode,
    )?;
    decompose_node(
        tree,
        child_level,
        child_row_base + 1,
        child_col_base,
        max_level,
        mode,
    )?;
    decompose_node(
        tree,
        child_level,
        child_row_base + 1,
        child_col_base + 1,
        max_level,
        mode,
    )?;

    Ok(())
}

/// Performs a selective 2D wavelet packet transform, expanding only nodes
/// that meet certain criteria.
///
/// This function creates a wavelet packet tree where only nodes that satisfy
/// the provided criterion function are further decomposed.
///
/// # Arguments
///
/// * `data` - The input 2D array (image)
/// * `wavelet` - The wavelet to use for the transform
/// * `max_level` - The maximum decomposition level
/// * `criterion` - A function that decides whether to further decompose a node
/// * `mode` - The signal extension mode (default: "symmetric")
///
/// # Returns
///
/// * A `WaveletPacketTree2D` containing the selective decomposition
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::wpt2d::{wpt2d_selective, WaveletPacket2D};
///
/// // Create a test image
/// let mut image = Array2::zeros((32, 32));
/// for i in 0..32 {
///     for j in 0..32 {
///         image[[i, j]] = (i * j) as f64 / 32.0;
///     }
/// }
///
/// // Define a criterion that only decomposes packets with high energy
/// let energy_criterion = |packet: &WaveletPacket2D| -> bool {
///     // Only decompose nodes with energy above a threshold
///     packet.energy() > 1000.0
/// };
///
/// // Perform selective wavelet packet decomposition
/// let decomp = wpt2d_selective(&image, Wavelet::Haar, 3, energy_criterion, None).expect("Operation failed");
///
/// // The resulting tree will have fewer nodes than the full decomposition
/// assert!(decomp.len() < 1 + 4 + 16 + 64); // Max possible for level 3
/// ```
pub fn wpt2d_selective<T, F>(
    data: &Array2<T>,
    wavelet: Wavelet,
    max_level: usize,
    criterion: F,
    mode: Option<&str>,
) -> SignalResult<WaveletPacketTree2D>
where
    T: Float + NumCast + Debug,
    F: Fn(&WaveletPacket2D) -> bool + Copy,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    let root_coeffs = convert_to_f64(data)?;

    if max_level == 0 {
        return Ok(WaveletPacketTree2D::new(wavelet, 0, root_coeffs, mode));
    }

    // Initialize the wavelet packet tree.
    let mut tree = WaveletPacketTree2D::new(wavelet, max_level, root_coeffs, mode);

    // Perform the selective decomposition.
    decompose_node_selective(&mut tree, 0, 0, 0, max_level, criterion, mode)?;

    Ok(tree)
}

/// Recursively decomposes a node in the wavelet packet tree if it meets the criterion.
fn decompose_node_selective<F>(
    tree: &mut WaveletPacketTree2D,
    level: usize,
    row: usize,
    col: usize,
    max_level: usize,
    criterion: F,
    mode: Option<&str>,
) -> SignalResult<()>
where
    F: Fn(&WaveletPacket2D) -> bool + Copy,
{
    // If we've reached the maximum level, stop recursion.
    if level >= max_level {
        return Ok(());
    }

    // Get the current node's coefficients.
    let parent = tree
        .get_packet(level, row, col)
        .ok_or_else(|| {
            SignalError::ValueError(format!(
                "Missing wavelet packet at level {}, position ({}, {})",
                level, row, col
            ))
        })?
        .clone();

    // Check if this node should be decomposed.
    if !criterion(&parent) {
        return Ok(());
    }

    // A node can only be decomposed if both dimensions are large enough for one
    // more level of the separable transform.
    let (prows, pcols) = parent.coeffs.dim();
    if prows < 2 || pcols < 2 {
        return Ok(());
    }

    // Decompose the coefficients into four subbands.
    let decomposition = dwt2d_decompose(&parent.coeffs, tree.wavelet, tree.edge_mode)?;

    // Calculate child positions in the next level.
    let child_level = level + 1;
    let child_row_base = row * 2;
    let child_col_base = col * 2;
    let sep = if parent.path.is_empty() { "" } else { "-" };

    let child_ll = WaveletPacket2D::new(
        child_level,
        child_row_base,
        child_col_base,
        decomposition.ll,
        format!("{}{}{}", parent.path, sep, "LL"),
    );
    let child_lh = WaveletPacket2D::new(
        child_level,
        child_row_base,
        child_col_base + 1,
        decomposition.lh,
        format!("{}{}{}", parent.path, sep, "LH"),
    );
    let child_hl = WaveletPacket2D::new(
        child_level,
        child_row_base + 1,
        child_col_base,
        decomposition.hl,
        format!("{}{}{}", parent.path, sep, "HL"),
    );
    let child_hh = WaveletPacket2D::new(
        child_level,
        child_row_base + 1,
        child_col_base + 1,
        decomposition.hh,
        format!("{}{}{}", parent.path, sep, "HH"),
    );

    // Add children to the tree.
    tree.add_packet(child_ll);
    tree.add_packet(child_lh);
    tree.add_packet(child_hl);
    tree.add_packet(child_hh);

    // Recursively decompose each child.
    decompose_node_selective(
        tree,
        child_level,
        child_row_base,
        child_col_base,
        max_level,
        criterion,
        mode,
    )?;
    decompose_node_selective(
        tree,
        child_level,
        child_row_base,
        child_col_base + 1,
        max_level,
        criterion,
        mode,
    )?;
    decompose_node_selective(
        tree,
        child_level,
        child_row_base + 1,
        child_col_base,
        max_level,
        criterion,
        mode,
    )?;
    decompose_node_selective(
        tree,
        child_level,
        child_row_base + 1,
        child_col_base + 1,
        max_level,
        criterion,
        mode,
    )?;

    Ok(())
}

/// Converts a generic numeric 2D array into an `Array2<f64>`, returning an honest
/// error if any value cannot be represented as `f64`.
fn convert_to_f64<T>(data: &Array2<T>) -> SignalResult<Array2<f64>>
where
    T: Float + NumCast + Debug,
{
    let (rows, cols) = data.dim();
    let mut out = Array2::zeros((rows, cols));
    for ((i, j), &val) in data.indexed_iter() {
        out[[i, j]] = NumCast::from(val).ok_or_else(|| {
            SignalError::ValueError(format!("Could not convert {:?} to f64", val))
        })?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a test image
    fn create_test_image(size: usize) -> Array2<f64> {
        let mut image = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                image[[i, j]] = (i * j) as f64;
            }
        }
        image
    }

    #[test]
    fn test_wpt2d_full_decomposition() {
        // Create a test image (16x16 for 2 levels of decomposition)
        let image = create_test_image(16);

        // Perform 2-level wavelet packet decomposition
        let decomp = wpt2d_full(&image, Wavelet::Haar, 2, None).expect("Operation failed");

        // Check that we have the expected number of packets
        // Level 0: 1 node, Level 1: 4 nodes, Level 2: 16 nodes
        assert_eq!(decomp.len(), 1 + 4 + 16);

        // Root
        assert!(decomp.get_packet(0, 0, 0).is_some());

        // Level 1 (4 nodes)
        for row in 0..2 {
            for col in 0..2 {
                assert!(decomp.get_packet(1, row, col).is_some());
            }
        }

        // Level 2 (16 nodes)
        for row in 0..4 {
            for col in 0..4 {
                assert!(decomp.get_packet(2, row, col).is_some());
            }
        }
    }

    #[test]
    fn test_wpt2d_selective_decomposition() {
        // Create a test image (32x32 for 3 levels of decomposition)
        let image = create_test_image(32);

        // Define a criterion that only decomposes the LL subband
        let ll_only_criterion = |packet: &WaveletPacket2D| -> bool {
            packet.path.is_empty() || packet.path.ends_with("LL")
        };

        // Perform selective wavelet packet decomposition
        let decomp = wpt2d_selective(&image, Wavelet::Haar, 3, ll_only_criterion, None)
            .expect("Operation failed");

        // Level 0: 1, Level 1: 4 (LL,LH,HL,HH), Level 2: 4 (LL-*), Level 3: 4 (LL-LL-*)
        // Total: 13 nodes
        assert_eq!(decomp.len(), 13);

        // Check that the LL path nodes exist at all levels
        assert!(decomp.get_packet(0, 0, 0).is_some()); // Root
        assert!(decomp.get_packet(1, 0, 0).is_some()); // LL
        assert!(decomp.get_packet(2, 0, 0).is_some()); // LL-LL
        assert!(decomp.get_packet(3, 0, 0).is_some()); // LL-LL-LL

        // Check that non-LL nodes at level 1 exist (root is always decomposed)
        assert!(decomp.get_packet(1, 0, 1).is_some()); // LH
        assert!(decomp.get_packet(1, 1, 0).is_some()); // HL
        assert!(decomp.get_packet(1, 1, 1).is_some()); // HH

        // Check that level 2 non-LL-LL nodes do not exist
        assert!(decomp.get_packet(2, 2, 0).is_none()); // HL-LL should not exist
    }

    #[test]
    fn test_packet_paths() {
        // Create a test image (16x16 for 2 levels of decomposition)
        let image = create_test_image(16);

        // Perform 2-level wavelet packet decomposition
        let decomp = wpt2d_full(&image, Wavelet::Haar, 2, None).expect("Operation failed");

        // Check root path (empty string)
        assert_eq!(
            decomp.get_packet(0, 0, 0).expect("Operation failed").path,
            ""
        );

        // Check level 1 paths
        assert_eq!(
            decomp.get_packet(1, 0, 0).expect("Operation failed").path,
            "LL"
        );
        assert_eq!(
            decomp.get_packet(1, 0, 1).expect("Operation failed").path,
            "LH"
        );
        assert_eq!(
            decomp.get_packet(1, 1, 0).expect("Operation failed").path,
            "HL"
        );
        assert_eq!(
            decomp.get_packet(1, 1, 1).expect("Operation failed").path,
            "HH"
        );

        // Check a few level 2 paths
        assert_eq!(
            decomp.get_packet(2, 0, 0).expect("Operation failed").path,
            "LL-LL"
        );
        assert_eq!(
            decomp.get_packet(2, 3, 3).expect("Operation failed").path,
            "HH-HH"
        );
    }

    #[test]
    fn test_wpt2d_perfect_reconstruction_haar() {
        // For an orthogonal wavelet (Haar) with periodic extension the full WPT
        // reconstruction must recover the original image to within numerical
        // precision.
        let image = create_test_image(16);

        let decomp =
            wpt2d_full(&image, Wavelet::Haar, 2, Some("periodic")).expect("decomposition failed");
        let reconstructed = decomp.reconstruct().expect("reconstruction failed");

        assert_eq!(reconstructed.dim(), image.dim());

        let max_err = image
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-8,
            "Haar WPT reconstruction error too large: {}",
            max_err
        );
    }

    #[test]
    fn test_wpt2d_reconstruction_level0() {
        // With max_level = 0 the reconstruction is the identity.
        let image = create_test_image(8);
        let decomp = wpt2d_full(&image, Wavelet::Haar, 0, None).expect("decomposition failed");
        let reconstructed = decomp.reconstruct().expect("reconstruction failed");
        assert_eq!(reconstructed, image);
    }

    #[test]
    fn test_reconstruct_missing_sibling_errors() {
        // A tree with an incomplete sibling set must produce an honest error
        // rather than silently fabricating a result.
        let image = create_test_image(16);
        let decomp = wpt2d_full(&image, Wavelet::Haar, 1, None).expect("decomposition failed");

        // Select only the LL child at level 1 (missing LH/HL/HH).
        let err = decomp.reconstruct_selective(&[(1, 0, 0)]);
        assert!(err.is_err(), "expected an error for incomplete sibling set");
    }
}
