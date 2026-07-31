//! Automatic padding strategies for optimal FFT performance
//!
//! This module provides functionality to automatically pad input data
//! to optimal sizes for FFT computation, improving performance by
//! ensuring the FFT size has small prime factors.

use crate::{next_fast_len, FFTError, FFTResult};
use scirs2_core::ndarray::{s, Array1, ArrayBase, ArrayD, Data, Dimension, IxDyn, Slice};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::Zero;

/// Padding mode for FFT operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    /// No padding
    None,
    /// Zero padding
    Zero,
    /// Constant value padding
    Constant(f64),
    /// Edge value replication
    Edge,
    /// Reflect padding (mirror at edge)
    Reflect,
    /// Symmetric padding (mirror with edge duplication)
    Symmetric,
    /// Wrap around (circular)
    Wrap,
    /// Linear ramp to zero
    LinearRamp,
}

/// Auto-padding configuration
#[derive(Debug, Clone)]
pub struct AutoPadConfig {
    /// Padding mode
    pub mode: PaddingMode,
    /// Minimum padding length (default: 0)
    pub min_pad: usize,
    /// Maximum padding length (default: input length)
    pub max_pad: Option<usize>,
    /// Whether to pad to power of 2
    pub power_of_2: bool,
    /// Whether to center the data in padded array
    pub center: bool,
}

impl Default for AutoPadConfig {
    fn default() -> Self {
        Self {
            mode: PaddingMode::Zero,
            min_pad: 0,
            max_pad: None,
            power_of_2: false,
            center: false,
        }
    }
}

impl AutoPadConfig {
    /// Create a new auto-padding configuration
    pub fn new(mode: PaddingMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Set minimum padding
    pub fn with_min_pad(mut self, minpad: usize) -> Self {
        self.min_pad = minpad;
        self
    }

    /// Set maximum padding
    pub fn with_max_pad(mut self, maxpad: usize) -> Self {
        self.max_pad = Some(maxpad);
        self
    }

    /// Require power of 2 size
    pub fn with_power_of_2(mut self) -> Self {
        self.power_of_2 = true;
        self
    }

    /// Center the data in padded array
    pub fn with_center(mut self) -> Self {
        self.center = true;
        self
    }
}

/// Automatically pad a 1D array for optimal FFT performance
#[allow(dead_code)]
pub fn auto_pad_1d<T>(x: &Array1<T>, config: &AutoPadConfig) -> FFTResult<Array1<T>>
where
    T: Clone + Zero,
{
    let n = x.len();

    // Determine target size
    let target_size = if config.power_of_2 {
        // Next power of 2
        let min_size = n + config.min_pad;
        let mut size = 1;
        while size < min_size {
            size *= 2;
        }
        size
    } else {
        // Next fast length
        next_fast_len(n + config.min_pad, false)
    };

    // Apply maximum padding constraint
    let padded_size = if let Some(max_pad) = config.max_pad {
        target_size.min(n + max_pad)
    } else {
        target_size
    };

    // No padding needed
    if padded_size == n {
        return Ok(x.clone());
    }

    // Create padded array
    let mut padded = Array1::zeros(padded_size);

    // Determine where to place the original data
    let start_idx = if config.center {
        (padded_size - n) / 2
    } else {
        0
    };

    // Copy original data
    padded.slice_mut(s![start_idx..start_idx + n]).assign(x);

    // Apply padding based on mode
    match config.mode {
        PaddingMode::None | PaddingMode::Zero => {
            // Already zero-initialized
        }
        PaddingMode::Constant(_value) => {
            let const_val = T::zero(); // Need to convert f64 to T properly
            if start_idx > 0 {
                padded.slice_mut(s![..start_idx]).fill(const_val.clone());
            }
            if start_idx + n < padded_size {
                padded.slice_mut(s![start_idx + n..]).fill(const_val);
            }
        }
        PaddingMode::Edge => {
            // Replicate edge values
            if start_idx > 0 {
                let left_val = x[0].clone();
                padded.slice_mut(s![..start_idx]).fill(left_val);
            }
            if start_idx + n < padded_size {
                let right_val = x[n - 1].clone();
                padded.slice_mut(s![start_idx + n..]).fill(right_val);
            }
        }
        PaddingMode::Reflect => {
            // Mirror at edges
            for i in 0..start_idx {
                let offset = start_idx - i - 1;
                let cycle = 2 * (n - 1);
                let src_idx = offset % cycle;
                let src_idx = if src_idx >= n {
                    cycle - src_idx
                } else {
                    src_idx
                };
                padded[i] = x[src_idx].clone();
            }
            for i in (start_idx + n)..padded_size {
                let offset = i - (start_idx + n);
                let cycle = 2 * (n - 1);
                let src_idx = n - 1 - (offset % cycle);
                padded[i] = x[src_idx].clone();
            }
        }
        PaddingMode::Symmetric => {
            // Mirror with edge duplication
            for i in 0..start_idx {
                let offset = start_idx - i;
                let cycle = 2 * n;
                let src_idx = (offset - 1) % cycle;
                let src_idx = if src_idx >= n {
                    cycle - 1 - src_idx
                } else {
                    src_idx
                };
                padded[i] = x[src_idx].clone();
            }
            for i in (start_idx + n)..padded_size {
                let offset = i - (start_idx + n);
                let cycle = 2 * n;
                let src_idx = n - 1 - (offset % cycle);
                padded[i] = x[src_idx].clone();
            }
        }
        PaddingMode::Wrap => {
            // Circular padding
            for i in 0..start_idx {
                let src_idx = (n - (start_idx - i) % n) % n;
                padded[i] = x[src_idx].clone();
            }
            for i in (start_idx + n)..padded_size {
                let src_idx = (i - start_idx) % n;
                padded[i] = x[src_idx].clone();
            }
        }
        PaddingMode::LinearRamp => {
            // Linear fade to zero
            if start_idx > 0 {
                for i in 0..start_idx {
                    // This would need proper numeric operations for type T
                    padded[i] = T::zero();
                }
            }
            if start_idx + n < padded_size {
                for i in (start_idx + n)..padded_size {
                    // This would need proper numeric operations for type T
                    padded[i] = T::zero();
                }
            }
        }
    }

    Ok(padded)
}

/// Automatically pad a complex array for optimal FFT performance
#[allow(dead_code)]
pub fn auto_pad_complex(
    x: &Array1<Complex<f64>>,
    config: &AutoPadConfig,
) -> FFTResult<Array1<Complex<f64>>> {
    let n = x.len();

    // Determine target size
    let target_size = if config.power_of_2 {
        let min_size = n + config.min_pad;
        let mut size = 1;
        while size < min_size {
            size *= 2;
        }
        size
    } else {
        next_fast_len(n + config.min_pad, false)
    };

    // Apply maximum padding constraint
    let padded_size = if let Some(max_pad) = config.max_pad {
        target_size.min(n + max_pad)
    } else {
        target_size
    };

    if padded_size == n {
        return Ok(x.clone());
    }

    let mut padded = Array1::zeros(padded_size);
    let start_idx = if config.center {
        (padded_size - n) / 2
    } else {
        0
    };

    padded.slice_mut(s![start_idx..start_idx + n]).assign(x);

    // Apply padding
    match config.mode {
        PaddingMode::None | PaddingMode::Zero => {}
        PaddingMode::Constant(value) => {
            let const_val = Complex::new(value, 0.0);
            if start_idx > 0 {
                padded.slice_mut(s![..start_idx]).fill(const_val);
            }
            if start_idx + n < padded_size {
                padded.slice_mut(s![start_idx + n..]).fill(const_val);
            }
        }
        PaddingMode::Edge => {
            if start_idx > 0 {
                let left_val = x[0];
                padded.slice_mut(s![..start_idx]).fill(left_val);
            }
            if start_idx + n < padded_size {
                let right_val = x[n - 1];
                padded.slice_mut(s![start_idx + n..]).fill(right_val);
            }
        }
        PaddingMode::LinearRamp => {
            // Linear fade from edges to zero
            if start_idx > 0 {
                let edge_val = x[0];
                for i in 0..start_idx {
                    let t = i as f64 / start_idx as f64;
                    padded[start_idx - 1 - i] = edge_val * t;
                }
            }
            if start_idx + n < padded_size {
                let edge_val = x[n - 1];
                let pad_len = padded_size - (start_idx + n);
                for i in 0..pad_len {
                    let t = 1.0 - (i as f64 / pad_len as f64);
                    padded[start_idx + n + i] = edge_val * t;
                }
            }
        }
        _ => {
            // For other modes, use simpler implementations or delegate to auto_pad_1d
            return auto_pad_1d(x, config);
        }
    }

    Ok(padded)
}

/// Remove padding from a 1D array after FFT
#[allow(dead_code)]
pub fn remove_padding_1d<T>(
    padded: &Array1<T>,
    original_size: usize,
    config: &AutoPadConfig,
) -> Array1<T>
where
    T: Clone,
{
    let padded_size = padded.len();

    if padded_size == original_size {
        return padded.clone();
    }

    let start_idx = if config.center {
        (padded_size - original_size) / 2
    } else {
        0
    };

    padded
        .slice(s![start_idx..start_idx + original_size])
        .to_owned()
}

/// Automatic padding for N-dimensional arrays of any dimensionality.
///
/// Every axis listed in `axes` (all axes, by default) is padded up to the
/// size [`AutoPadConfig`] computes for it, and the newly-added border
/// region is filled according to `config.mode`. All [`PaddingMode`]
/// variants are supported, for arbitrary `ndim` (not just 1D/2D).
///
/// Border fills are applied one axis at a time, in ascending axis order,
/// each pass building on the previous axis's already-filled data. This
/// mirrors NumPy's own `numpy.pad`, which composes independent per-axis 1D
/// fills the same way for `edge`/`reflect`/`symmetric`/`wrap` -- so corner
/// values (where two or more padded axes meet) match NumPy for those modes
/// too, not just the interior border.
#[allow(dead_code)]
pub fn auto_pad_nd<S, D>(
    x: &ArrayBase<S, D>,
    config: &AutoPadConfig,
    axes: Option<&[usize]>,
) -> FFTResult<ArrayD<Complex<f64>>>
where
    S: Data<Elem = Complex<f64>>,
    D: Dimension,
{
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    let default_axes: Vec<usize> = (0..ndim).collect();
    let axes: Vec<usize> = axes.map(<[usize]>::to_vec).unwrap_or(default_axes);

    for &axis in &axes {
        if axis >= ndim {
            return Err(FFTError::ValueError(format!(
                "Axis {axis} is out of bounds for array of dimension {ndim}"
            )));
        }
    }

    let mut paddedshape = shape.clone();
    // Offset of the original data along each axis within the padded array.
    let mut start_idx = vec![0usize; ndim];

    // Calculate padded sizes (and placement offsets) for the specified axes.
    for &axis in &axes {
        let n = shape[axis];
        let target_size = if config.power_of_2 {
            let min_size = n + config.min_pad;
            let mut size = 1;
            while size < min_size {
                size *= 2;
            }
            size
        } else {
            next_fast_len(n + config.min_pad, false)
        };

        let padded_axis_len = if let Some(max_pad) = config.max_pad {
            target_size.min(n + max_pad)
        } else {
            target_size
        };
        paddedshape[axis] = padded_axis_len;
        start_idx[axis] = if config.center {
            (padded_axis_len - n) / 2
        } else {
            0
        };
    }

    // Nothing to do: every requested axis is already at its target size.
    if paddedshape == shape {
        return Ok(x.to_owned().into_dyn());
    }

    let mut padded = ArrayD::<Complex<f64>>::zeros(paddedshape.clone());

    // Place the original data into its (possibly centered) sub-block. This
    // works for any `ndim` via `slice_each_axis_mut`, unlike the previous
    // hand-written 1D/2D-only implementation.
    {
        let mut core = padded.slice_each_axis_mut(|desc| {
            let axis = desc.axis.index();
            let s = start_idx[axis];
            Slice::from(s..s + shape[axis])
        });
        core.assign(&x.view().into_dyn());
    }

    // Fill the newly-added border region for every padded axis, in
    // ascending axis order (see doc comment above for why the order
    // matters for modes that depend on already-padded neighboring data).
    if !matches!(config.mode, PaddingMode::None | PaddingMode::Zero) {
        let mut ordered_axes = axes.clone();
        ordered_axes.sort_unstable();
        ordered_axes.dedup();
        for axis in ordered_axes {
            let n = shape[axis];
            let total = paddedshape[axis];
            if n == 0 || total == n {
                continue;
            }
            fill_axis_border(&mut padded, axis, start_idx[axis], n, total, config.mode);
        }
    }

    Ok(padded)
}

/// Enumerate every fixed combination of indices for all axes other than
/// `axis`, as full `shape.len()`-length index vectors (the slot at `axis`
/// is left at `0`; callers overwrite it while walking the fiber along that
/// axis). This is what lets [`fill_axis_border`] sweep *every* row/column/
/// etc. along an axis instead of just the one at the origin.
fn other_axis_index_combinations(shape: &[usize], axis: usize) -> Vec<Vec<usize>> {
    let ndim = shape.len();
    let mut combos: Vec<Vec<usize>> = vec![vec![0; ndim]];
    for (dim, &size) in shape.iter().enumerate() {
        if dim == axis {
            continue;
        }
        let mut expanded = Vec::with_capacity(combos.len() * size.max(1));
        for combo in &combos {
            for v in 0..size {
                let mut next = combo.clone();
                next[dim] = v;
                expanded.push(next);
            }
        }
        combos = expanded;
    }
    combos
}

/// Source index within `[0, n)` for NumPy's `reflect` padding mode (mirrors
/// without repeating the edge sample; period `2*(n-1)`), given a logical
/// position `p` relative to the start of the original data (`p` in
/// `[0, n)` is the data itself; `p < 0` or `p >= n` is padding).
fn reflect_source_index(p: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let m = p.rem_euclid(period);
    if m < n as isize {
        m as usize
    } else {
        (period - m) as usize
    }
}

/// Source index within `[0, n)` for NumPy's `symmetric` padding mode
/// (mirrors *including* the edge sample; period `2*n`).
fn symmetric_source_index(p: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let period = 2 * n as isize;
    let m = p.rem_euclid(period);
    if m < n as isize {
        m as usize
    } else {
        (period - 1 - m) as usize
    }
}

/// Source index within `[0, n)` for NumPy's `wrap` (circular) padding mode.
fn wrap_source_index(p: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    p.rem_euclid(n as isize) as usize
}

/// Fill the border region of `padded` along `axis` (positions before
/// `start` and from `start + n` to `total`) from the `n` already-placed
/// elements at `[start, start + n)`, per `mode`. Every fiber along `axis`
/// (every combination of the *other* axes' indices) is swept
/// independently, so this handles arrays of any dimensionality.
fn fill_axis_border(
    padded: &mut ArrayD<Complex<f64>>,
    axis: usize,
    start: usize,
    n: usize,
    total: usize,
    mode: PaddingMode,
) {
    let shape = padded.shape().to_vec();
    let end_value = Complex::new(0.0, 0.0);

    for mut indices in other_axis_index_combinations(&shape, axis) {
        // Snapshot the current core fiber. On the second (and later) axis
        // processed by `auto_pad_nd`, this may itself already contain
        // border values filled in by an earlier axis's pass, which is
        // exactly what makes corner regions come out right.
        let mut core = Vec::with_capacity(n);
        for i in 0..n {
            indices[axis] = start + i;
            core.push(padded[IxDyn(&indices)]);
        }

        match mode {
            PaddingMode::None | PaddingMode::Zero => {}
            PaddingMode::Constant(value) => {
                let fill = Complex::new(value, 0.0);
                for i in 0..start {
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = fill;
                }
                for i in (start + n)..total {
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = fill;
                }
            }
            PaddingMode::Edge => {
                let left = core[0];
                let right = core[n - 1];
                for i in 0..start {
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = left;
                }
                for i in (start + n)..total {
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = right;
                }
            }
            PaddingMode::Reflect => {
                for i in 0..start {
                    let src = reflect_source_index(i as isize - start as isize, n);
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = core[src];
                }
                for i in (start + n)..total {
                    let src = reflect_source_index(i as isize - start as isize, n);
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = core[src];
                }
            }
            PaddingMode::Symmetric => {
                for i in 0..start {
                    let src = symmetric_source_index(i as isize - start as isize, n);
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = core[src];
                }
                for i in (start + n)..total {
                    let src = symmetric_source_index(i as isize - start as isize, n);
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = core[src];
                }
            }
            PaddingMode::Wrap => {
                for i in 0..start {
                    let src = wrap_source_index(i as isize - start as isize, n);
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = core[src];
                }
                for i in (start + n)..total {
                    let src = wrap_source_index(i as isize - start as isize, n);
                    indices[axis] = i;
                    padded[IxDyn(&indices)] = core[src];
                }
            }
            PaddingMode::LinearRamp => {
                // Linearly ramp from the edge sample down to `end_value`
                // (0) at the outermost padded position, independently on
                // each side (matching NumPy's `linear_ramp` with its
                // default `end_values=0`).
                let left_edge = core[0];
                if start > 0 {
                    let w = start as f64;
                    for i in 0..start {
                        let k = (start - 1 - i) as f64;
                        let val = left_edge - (left_edge - end_value) * ((k + 1.0) / w);
                        indices[axis] = i;
                        padded[IxDyn(&indices)] = val;
                    }
                }
                let right_edge = core[n - 1];
                let right_len = total - (start + n);
                if right_len > 0 {
                    let w = right_len as f64;
                    for i in (start + n)..total {
                        let k = (i - (start + n)) as f64;
                        let val = right_edge - (right_edge - end_value) * ((k + 1.0) / w);
                        indices[axis] = i;
                        padded[IxDyn(&indices)] = val;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_auto_pad_zero() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let config = AutoPadConfig::new(PaddingMode::Zero);

        let padded =
            auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).expect("Operation failed");

        // Should pad to next fast length
        assert!(padded.len() >= x.len());

        // Original values should be preserved
        for i in 0..x.len() {
            assert_abs_diff_eq!(padded[i].re, x[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_auto_pad_power_of_2() {
        let x = Array1::from_vec(vec![1.0; 5]);
        let config = AutoPadConfig::new(PaddingMode::Zero).with_power_of_2();

        let padded =
            auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).expect("Operation failed");

        // Should pad to 8 (next power of 2)
        assert_eq!(padded.len(), 8);
    }

    #[test]
    fn test_remove_padding() {
        let padded = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
        let config = AutoPadConfig::new(PaddingMode::Zero);

        let unpadded = remove_padding_1d(&padded, 4, &config);
        assert_eq!(unpadded.len(), 4);
        assert_eq!(
            unpadded.as_slice().expect("Operation failed"),
            &[0.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_auto_pad_center() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let config = AutoPadConfig::new(PaddingMode::Zero)
            .with_center()
            .with_min_pad(3);

        let padded =
            auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).expect("Operation failed");

        // Should center the data
        assert!(padded.len() >= 6);
        let start = (padded.len() - 3) / 2;
        assert_abs_diff_eq!(padded[start].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(padded[start + 1].re, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(padded[start + 2].re, 3.0, epsilon = 1e-10);
    }

    fn complex_vec(vals: &[f64]) -> Array1<Complex<f64>> {
        Array1::from_vec(vals.iter().map(|&v| Complex::new(v, 0.0)).collect())
    }

    fn assert_real_close(actual: &ArrayD<Complex<f64>>, expected: &[f64], eps: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a.re, e, epsilon = eps);
            assert_abs_diff_eq!(a.im, 0.0, epsilon = eps);
        }
    }

    // Reference values throughout this section were computed with
    // `numpy.pad` for the exact input and pad widths described in each
    // test (non-constant data throughout, so a fabricated/constant stub
    // could not pass).

    #[test]
    fn test_auto_pad_nd_1d_noncentered_all_modes() {
        // x has length 5; `power_of_2` rounds up to 8, so all 3 padding
        // elements land on the right (non-centered). Reference: `np.pad(x,
        // (0, 3), mode=...)`.
        let x = complex_vec(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let cases: [(PaddingMode, [f64; 8]); 6] = [
            (
                PaddingMode::Constant(9.0),
                [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 9.0, 9.0],
            ),
            (PaddingMode::Edge, [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]),
            (
                PaddingMode::Reflect,
                [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0],
            ),
            (
                PaddingMode::Symmetric,
                [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0],
            ),
            (PaddingMode::Wrap, [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0]),
            (
                PaddingMode::LinearRamp,
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    3.333_333_333_333_333_5,
                    1.666_666_666_666_666_7,
                    0.0,
                ],
            ),
        ];
        for (mode, expected) in cases {
            let config = AutoPadConfig::new(mode).with_power_of_2();
            let result = auto_pad_nd(&x, &config, None).expect("auto_pad_nd failed");
            assert_eq!(result.shape(), &[8]);
            assert_real_close(&result, &expected, 1e-9);
        }
    }

    #[test]
    fn test_auto_pad_nd_1d_centered_all_modes() {
        // x has length 3; `power_of_2` with `min_pad(5)` forces a minimum
        // size of 8, split as 2 left / 3 right when centered. Reference:
        // `np.pad(x, (2, 3), mode=...)`.
        let x = complex_vec(&[1.0, 2.0, 3.0]);
        let cases: [(PaddingMode, [f64; 8]); 6] = [
            (PaddingMode::Edge, [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]),
            (
                PaddingMode::Reflect,
                [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0],
            ),
            (
                PaddingMode::Symmetric,
                [2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            ),
            (PaddingMode::Wrap, [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            (
                PaddingMode::LinearRamp,
                [0.0, 0.5, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0],
            ),
            (
                PaddingMode::Constant(-4.0),
                [-4.0, -4.0, 1.0, 2.0, 3.0, -4.0, -4.0, -4.0],
            ),
        ];
        for (mode, expected) in cases {
            let config = AutoPadConfig::new(mode)
                .with_power_of_2()
                .with_min_pad(5)
                .with_center();
            let result = auto_pad_nd(&x, &config, None).expect("auto_pad_nd failed");
            assert_eq!(result.shape(), &[8]);
            assert_real_close(&result, &expected, 1e-9);
        }
    }

    #[test]
    fn test_auto_pad_nd_invalid_axis_is_an_error() {
        let x = complex_vec(&[1.0, 2.0, 3.0]);
        let config = AutoPadConfig::new(PaddingMode::Zero);
        let err = auto_pad_nd(&x, &config, Some(&[5])).unwrap_err();
        assert!(matches!(err, FFTError::ValueError(_)));
    }

    /// Build a real 3x5 `Complex<f64>` array from a row-major flat `Vec`.
    fn complex_2d(
        rows: usize,
        cols: usize,
        vals: &[f64],
    ) -> scirs2_core::ndarray::Array2<Complex<f64>> {
        scirs2_core::ndarray::Array2::from_shape_vec(
            (rows, cols),
            vals.iter().map(|&v| Complex::new(v, 0.0)).collect(),
        )
        .expect("valid shape")
    }

    #[test]
    fn test_auto_pad_nd_2d_corners_match_numpy() {
        // shape (3,5); `power_of_2` rounds axis0 3->4 (pad 1) and axis1
        // 5->8 (pad 3), non-centered. This exercises the sequential
        // per-axis composition (the corner region depends on axis-0's
        // padding already being present when axis-1 is padded), which a
        // naive "pad each axis independently from the original data only"
        // implementation would get wrong. Reference: `np.pad(x, ((0,1),
        // (0,3)), mode=...)`.
        #[rustfmt::skip]
        let input = [
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let x = complex_2d(3, 5, &input);

        let cases: [(PaddingMode, [f64; 32]); 6] = [
            (
                PaddingMode::Edge,
                [
                    1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0,
                    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.0, 15.0, 15.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 15.0, 15.0, 15.0,
                ],
            ),
            (
                PaddingMode::Reflect,
                [
                    1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0, 9.0, 8.0,
                    7.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    9.0, 8.0, 7.0,
                ],
            ),
            (
                PaddingMode::Symmetric,
                [
                    1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 9.0,
                    8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.0, 14.0, 13.0, 11.0, 12.0, 13.0, 14.0,
                    15.0, 15.0, 14.0, 13.0,
                ],
            ),
            (
                PaddingMode::Wrap,
                [
                    1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 10.0, 6.0, 7.0,
                    8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                    1.0, 2.0, 3.0,
                ],
            ),
            (
                PaddingMode::Constant(7.0),
                [
                    1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 7.0, 6.0, 7.0, 8.0, 9.0, 10.0, 7.0, 7.0,
                    7.0, 11.0, 12.0, 13.0, 14.0, 15.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
                    7.0, 7.0,
                ],
            ),
            (
                PaddingMode::LinearRamp,
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    3.333_333_333_333_333_5,
                    1.666_666_666_666_666_7,
                    0.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    6.666_666_666_666_667,
                    3.333_333_333_333_333_5,
                    0.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    10.0,
                    5.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
        ];

        for (mode, expected) in cases {
            let config = AutoPadConfig::new(mode).with_power_of_2();
            let result = auto_pad_nd(&x, &config, None).expect("auto_pad_nd failed");
            assert_eq!(result.shape(), &[4, 8]);
            assert_real_close(&result, &expected, 1e-9);
        }
    }

    #[test]
    fn test_auto_pad_nd_3d_reflect_matches_numpy() {
        // shape (2,3,4); `power_of_2` + per-axis `min_pad(1)` (applied via
        // 3 separate calls is not supported by `AutoPadConfig`, so instead
        // this uses axis sizes chosen so the *same* config -- power_of_2
        // with min_pad=1, centered -- produces pad (1,1) on axis 0, (0,1)
        // on axis 1, and (2,2) on axis 2). Confirms `auto_pad_nd` no
        // longer rejects 3D input, and that corner blending across all 3
        // axes matches NumPy. Reference: `np.pad(x, ((1,1),(0,1),(2,2)),
        // mode='reflect')`.
        let input: Vec<f64> = (0..24).map(|v| v as f64).collect();
        let x = ArrayD::from_shape_vec(vec![2, 3, 4], input)
            .expect("valid shape")
            .mapv(|v| Complex::new(v, 0.0));

        let config = AutoPadConfig::new(PaddingMode::Reflect)
            .with_power_of_2()
            .with_min_pad(1)
            .with_center();
        let result = auto_pad_nd(&x, &config, None).expect("auto_pad_nd failed");
        assert_eq!(result.shape(), &[4, 4, 8]);

        #[rustfmt::skip]
        let expected = [
            14.0, 13.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 18.0, 17.0,
            22.0, 21.0, 20.0, 21.0, 22.0, 23.0, 22.0, 21.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 18.0, 17.0, 2.0,
            1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0, 10.0, 9.0, 8.0, 9.0,
            10.0, 11.0, 10.0, 9.0, 6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0, 14.0, 13.0, 12.0, 13.0, 14.0, 15.0,
            14.0, 13.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 18.0, 17.0, 22.0, 21.0, 20.0, 21.0, 22.0, 23.0,
            22.0, 21.0, 18.0, 17.0, 16.0, 17.0, 18.0, 19.0, 18.0, 17.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0,
            6.0, 5.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 9.0, 6.0, 5.0, 4.0,
            5.0, 6.0, 7.0, 6.0, 5.0,
        ];
        assert_real_close(&result, &expected, 1e-9);
    }

    #[test]
    fn test_auto_pad_nd_no_padding_needed_returns_input() {
        // Every axis is already at its `power_of_2` target size, so
        // `auto_pad_nd` must return the data unchanged (not an all-zero
        // array of the same shape, which a careless implementation might
        // produce by always allocating a fresh `ArrayD::zeros` and
        // forgetting the early-return).
        let x = complex_vec(&[1.0, 2.0, 3.0, 4.0]);
        let config = AutoPadConfig::new(PaddingMode::Reflect).with_power_of_2();
        let result = auto_pad_nd(&x, &config, None).expect("auto_pad_nd failed");
        assert_eq!(result.shape(), &[4]);
        assert_real_close(&result, &[1.0, 2.0, 3.0, 4.0], 1e-9);
    }
}
