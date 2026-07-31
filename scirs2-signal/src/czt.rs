// Chirp Z-Transform (CZT)
//
// This module provides functions for computing the Chirp Z-Transform (CZT),
// which is a generalization of the Discrete Fourier Transform (DFT) that
// allows evaluation of the Z-transform on arbitrary contours in the complex plane.
//
// The CZT is particularly useful for analyzing frequency components with
// non-uniform spacing or for "zooming in" on specific frequency ranges.
//
// This module also provides targeted frequency-domain analysis tools built on
// (or complementary to) the CZT:
//
// - **Zoom FFT** (`zoom_fft`): high-resolution DFT in a specific frequency band
//   `[f1, f2]`, implemented directly in terms of [`czt`].
// - **Goertzel algorithm** (`goertzel`): O(N) per-frequency DFT coefficient
//   computation, more efficient than a full FFT when only a few frequencies
//   are of interest.
// - **Sliding DFT** (`SlidingDft`): recursive, O(1)-per-sample streaming DFT
//   update for real-time / streaming applications.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::fmt::Debug;

/// Calculate the points at which the chirp z-transform is computed
///
/// # Arguments
///
/// * `m` - Number of points to evaluate
/// * `w` - Step size between points on the contour (default: unit circle w = exp(-j*2π/m))
/// * `a` - Starting point on the contour (default: a = 1)
///
/// # Returns
///
/// * A vector of points in the complex plane
///
/// # Examples
///
/// ```
/// use scirs2_signal::czt::czt_points;
///
/// // Generate 10 points on the unit circle
/// let points = czt_points(10, None, None).expect("Operation failed");
/// assert_eq!(points.len(), 10);
/// ```
pub fn czt_points(
    m: usize,
    w: Option<Complex64>,
    a: Option<Complex64>,
) -> SignalResult<Vec<Complex64>> {
    // Default values
    let a_val = a.unwrap_or(Complex64::new(1.0, 0.0));
    let w_val = w.unwrap_or_else(|| {
        // Default to unit circle: w = exp(-j*2π/m)
        let arg = -2.0 * PI / m as f64;
        Complex64::new(arg.cos(), arg.sin())
    });

    // Create the points
    let mut points = Vec::with_capacity(m);
    let mut current = a_val;

    for _ in 0..m {
        points.push(current);
        current *= w_val;
    }

    Ok(points)
}

/// Compute the Chirp Z-Transform
///
/// Evaluates `X(k) = sum_{n=0}^{N-1} x[n] * A^{-n} * W^{nk}` for `k = 0..M-1`.
///
/// When called with default parameters (`m = N`, `w = exp(-j*2π/N)`, `a = 1`),
/// the result is identical to the standard DFT (FFT output).
///
/// # Arguments
///
/// * `x` - Input signal
/// * `m` - Number of output points (default: same as input length)
/// * `w` - Step size between points on the contour (default: unit circle w = exp(-j*2π/m))
/// * `a` - Starting point on the contour (default: a = 1)
/// * `axis` - Axis along which to compute the transform (only -1 or 0 supported)
///
/// # Returns
///
/// * The Chirp Z-Transform of the input signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::czt::czt;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Compute the CZT (equivalent to the DFT in this case)
/// let result = czt(&signal, None, None, None, None).expect("Operation failed");
/// assert_eq!(result.len(), 4);
/// ```
///
/// Zoom in on a specific frequency range:
///
/// ```
/// use scirs2_signal::czt::czt;
/// use scirs2_core::numeric::Complex64;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // w = exp(-j*π/8) -> 1/8 of a full circle per step
/// let arg = -std::f64::consts::PI / 8.0;
/// let w = Complex64::new(arg.cos(), arg.sin());
/// let result = czt(&signal, Some(16), Some(w), None, None).expect("Operation failed");
/// assert_eq!(result.len(), 16);
/// ```
pub fn czt<T>(
    x: &[T],
    m: Option<usize>,
    w: Option<Complex64>,
    a: Option<Complex64>,
    axis: Option<isize>,
) -> SignalResult<Vec<Complex64>>
where
    T: Float + NumCast + Debug,
{
    // Check input
    if x.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Default values
    let n = x.len();
    let m_val = m.unwrap_or(n);
    let a_val = a.unwrap_or(Complex64::new(1.0, 0.0));
    let w_val = w.unwrap_or_else(|| {
        // Default to unit circle: w = exp(-j*2π/m)
        let arg = -2.0 * PI / m_val as f64;
        Complex64::new(arg.cos(), arg.sin())
    });

    // Convert input to complex
    let x_complex: Vec<Complex64> = x
        .iter()
        .map(|&val| {
            let val_f64 = NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })?;
            Ok(Complex64::new(val_f64, 0.0))
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Ignore axis parameter for now (only 1D transform is implemented)
    if let Some(ax) = axis {
        if ax != -1 && ax != 0 {
            return Err(SignalError::ValueError(
                "Only axis=-1 or axis=0 is supported".to_string(),
            ));
        }
    }

    // Compute the CZT using Bluestein's algorithm
    czt_bluestein(&x_complex, m_val, w_val, a_val)
}

/// Compute the Chirp Z-Transform using Bluestein's algorithm.
///
/// Evaluates X(k) = sum_{n=0}^{N-1} x[n] * a^{-n} * w^{nk}  for k=0..M-1
///
/// Using the Bluestein identity `nk = n²/2 - (k-n)²/2 + k²/2`:
///
/// ```text
/// X(k) = w^{k²/2} * sum_{n=0}^{N-1} [x[n] * a^{-n} * w^{n²/2}] * w^{-(k-n)²/2}
/// ```
///
/// The inner sum is a convolution that can be computed with FFTs.
fn czt_bluestein(
    x: &[Complex64],
    m: usize,
    w: Complex64,
    a: Complex64,
) -> SignalResult<Vec<Complex64>> {
    let n = x.len();

    // Length for the FFT-based convolution: next power of 2 >= (n + m - 1)
    let conv_len = next_power_of_two(n + m - 1);

    // Extract |w| and angle(w) to handle complex w with |w| != 1 correctly.
    let w_angle = w.im.atan2(w.re);
    let w_mag = (w.re * w.re + w.im * w.im).sqrt();

    // chirp_w(n_val) = w^{n_val^2 / 2}
    //   magnitude:  w_mag^{n_val^2 / 2}
    //   phase:      exp(j * w_angle * n_val^2 / 2)
    let chirp_w = |n_val: i64| -> Complex64 {
        let sq = n_val * n_val;
        let mag = w_mag.powf(sq as f64 / 2.0);
        let phase = w_angle * sq as f64 / 2.0;
        Complex64::new(mag * phase.cos(), mag * phase.sin())
    };

    // Build yn[n] = x[n] * a^{-n} * w^{n²/2},  n = 0..N-1, zero-padded to conv_len
    let mut yn: Vec<Complex64> = Vec::with_capacity(conv_len);
    let a_inv = Complex64::new(1.0, 0.0) / a;
    let mut a_pow = Complex64::new(1.0, 0.0); // tracks a^{-n}
    for ni in 0..n {
        let chirp_n = chirp_w(ni as i64);
        yn.push(x[ni] * a_pow * chirp_n);
        a_pow *= a_inv;
    }
    while yn.len() < conv_len {
        yn.push(Complex64::new(0.0, 0.0));
    }

    // Build hn: the filter kernel h[k] = w^{-k²/2}
    //   h[k]              for k = 0..M-1  (positive lags)
    //   h[conv_len - k]   for k = 1..N-1  (negative lags, wrapped)
    let mut hn: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); conv_len];
    for ki in 0..m {
        let c = chirp_w(ki as i64);
        hn[ki] = Complex64::new(c.re, -c.im); // conjugate = w^{-k^2/2}
    }
    for ni in 1..n {
        let c = chirp_w(ni as i64);
        hn[conv_len - ni] = Complex64::new(c.re, -c.im);
    }

    // FFT-based convolution
    let yn_fft = fft_complex(&yn)?;
    let hn_fft = fft_complex(&hn)?;

    let mut product: Vec<Complex64> = yn_fft
        .iter()
        .zip(hn_fft.iter())
        .map(|(&y, &h)| y * h)
        .collect();

    ifft_in_place(&mut product)?;

    // Extract M points, multiply by post-chirp w^{k²/2}
    let result: Vec<Complex64> = (0..m).map(|ki| product[ki] * chirp_w(ki as i64)).collect();

    Ok(result)
}

/// Find the next power of 2 greater than or equal to n.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Compute FFT of a complex sequence using in-place Cooley-Tukey.
fn fft_complex(x: &[Complex64]) -> SignalResult<Vec<Complex64>> {
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let mut buf = x.to_vec();
    fft_inplace(&mut buf, false)?;
    Ok(buf)
}

/// In-place Cooley-Tukey radix-2 DIT FFT/IFFT. Length must be a power of 2.
fn fft_inplace(buf: &mut Vec<Complex64>, inverse: bool) -> SignalResult<()> {
    let n = buf.len();
    if n <= 1 {
        return Ok(());
    }
    if n & (n - 1) != 0 {
        return Err(SignalError::ValueError(format!(
            "FFT length must be a power of 2, got {}",
            n
        )));
    }

    // Bit-reversal permutation
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            buf.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let mut len = 2_usize;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex64::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..half {
                let u = buf[i + j];
                let v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for c in buf.iter_mut() {
            *c = Complex64::new(c.re * scale, c.im * scale);
        }
    }

    Ok(())
}

/// Reverse the bits of `x` using `bits` significant bits.
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place IFFT.
fn ifft_in_place(buf: &mut Vec<Complex64>) -> SignalResult<()> {
    fft_inplace(buf, true)
}

// ---------------------------------------------------------------------------
// Zoom FFT
// ---------------------------------------------------------------------------

/// Compute high-resolution DFT in a specific frequency band using the Chirp Z-Transform.
///
/// The Zoom FFT computes `m` equally-spaced DFT samples in the frequency band
/// `[f1, f2]`, providing higher frequency resolution within that band compared
/// to a standard FFT of the same length.
///
/// Internally uses [`czt`] (Bluestein's algorithm) to evaluate the Z-transform
/// along an arc from `exp(j*2π*f1/fs)` to `exp(j*2π*f2/fs)`.
///
/// # Arguments
///
/// * `x` - Input signal (time domain)
/// * `f1` - Lower frequency bound (Hz), must be >= 0
/// * `f2` - Upper frequency bound (Hz), must be > f1 and <= fs/2
/// * `m` - Number of output frequency points (>= 1)
/// * `fs` - Sampling frequency (Hz)
///
/// # Returns
///
/// Complex spectrum of length `m` covering frequencies `[f1, f2]`.
///
/// # Examples
///
/// ```
/// use scirs2_signal::czt::{zoom_fft, zoom_fft_freqs};
///
/// let fs = 2000.0_f64;
/// let n = 1024;
/// // 500 Hz tone (Nyquist is fs/2 = 1000 Hz, so 400-600 Hz is a valid zoom band)
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * std::f64::consts::PI * 500.0 * i as f64 / fs).sin())
///     .collect();
///
/// // Zoom into 400-600 Hz with 256 points
/// let spectrum = zoom_fft(&signal, 400.0, 600.0, 256, fs).expect("zoom_fft failed");
/// assert_eq!(spectrum.len(), 256);
///
/// let freqs = zoom_fft_freqs(400.0, 600.0, 256);
/// assert_eq!(freqs.len(), 256);
/// // 500 Hz should be near the center
/// ```
pub fn zoom_fft(x: &[f64], f1: f64, f2: f64, m: usize, fs: f64) -> SignalResult<Vec<Complex64>> {
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".into()));
    }
    if m == 0 {
        return Err(SignalError::ValueError(
            "Number of output points m must be >= 1".into(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency fs must be positive".into(),
        ));
    }
    if f1 < 0.0 || f2 <= f1 {
        return Err(SignalError::ValueError(
            "Frequency bounds must satisfy 0 <= f1 < f2".into(),
        ));
    }
    if f2 > fs / 2.0 {
        return Err(SignalError::ValueError(
            "f2 must not exceed the Nyquist frequency (fs/2)".into(),
        ));
    }

    // Starting point on the unit circle: a = exp(j*2π*f1/fs)
    let theta_start = 2.0 * PI * f1 / fs;
    let a = Complex64::new(theta_start.cos(), theta_start.sin());

    // Step between consecutive frequency samples:
    // w = exp(-j * 2π * (f2-f1) / (fs * (m-1)))  for m > 1, else no step
    let delta_f = if m > 1 {
        (f2 - f1) / (m - 1) as f64
    } else {
        0.0
    };
    let theta_step = -2.0 * PI * delta_f / fs;
    let w = Complex64::new(theta_step.cos(), theta_step.sin());

    // Compute via the shared Bluestein CZT implementation.
    czt(x, Some(m), Some(w), Some(a), None)
}

/// Compute the frequency axis for `zoom_fft` output.
///
/// Returns `m` linearly spaced frequencies between `f1` and `f2` (inclusive).
///
/// # Arguments
///
/// * `f1` - Lower frequency bound (Hz)
/// * `f2` - Upper frequency bound (Hz)
/// * `m` - Number of frequency points
///
/// # Returns
///
/// Vector of length `m` with frequency values in Hz.
pub fn zoom_fft_freqs(f1: f64, f2: f64, m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![f1];
    }
    (0..m)
        .map(|i| f1 + i as f64 * (f2 - f1) / (m - 1) as f64)
        .collect()
}

// ---------------------------------------------------------------------------
// Goertzel Algorithm
// ---------------------------------------------------------------------------

/// Compute DFT coefficients at specific frequencies using the Goertzel algorithm.
///
/// The Goertzel algorithm computes the DFT at arbitrary frequencies with O(N)
/// complexity per frequency. It is more efficient than FFT when only a small
/// number of specific frequencies are of interest.
///
/// The algorithm uses a second-order IIR filter approach equivalent to:
/// ```text
/// X(f) = sum_{n=0}^{N-1} x[n] * exp(-j * 2π * f * n / fs)
/// ```
///
/// # Arguments
///
/// * `x` - Input signal
/// * `freqs` - Frequencies at which to evaluate the DFT (Hz)
/// * `fs` - Sampling frequency (Hz)
///
/// # Returns
///
/// Complex DFT values at each of the requested frequencies.
///
/// # Examples
///
/// ```
/// use scirs2_signal::czt::goertzel;
///
/// let fs = 8000.0_f64;
/// let n = 256;
/// let freq = 1000.0_f64;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
///     .collect();
///
/// let result = goertzel(&signal, &[freq], fs).expect("goertzel failed");
/// // The magnitude at 1 kHz should be large
/// assert!(result[0].norm() > 10.0);
/// ```
pub fn goertzel(x: &[f64], freqs: &[f64], fs: f64) -> SignalResult<Vec<Complex64>> {
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".into()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency fs must be positive".into(),
        ));
    }

    let n = x.len();
    let mut results = Vec::with_capacity(freqs.len());

    for &freq in freqs {
        if freq < 0.0 || freq > fs / 2.0 {
            return Err(SignalError::ValueError(format!(
                "Frequency {} is outside valid range [0, {}]",
                freq,
                fs / 2.0
            )));
        }

        // Normalized frequency: k = f * N / fs (real-valued for arbitrary f)
        let k = freq * n as f64 / fs;
        let omega = 2.0 * PI * k / n as f64;
        let coeff = 2.0 * omega.cos();

        // Goertzel IIR filter
        let mut s_prev2 = 0.0_f64;
        let mut s_prev1 = 0.0_f64;
        for &sample in x {
            let s = sample + coeff * s_prev1 - s_prev2;
            s_prev2 = s_prev1;
            s_prev1 = s;
        }

        // Final complex output: X = s_prev1 - s_prev2 * exp(-j*omega)
        let re = s_prev1 - s_prev2 * omega.cos();
        let im = s_prev2 * omega.sin();
        results.push(Complex64::new(re, im));
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Sliding DFT
// ---------------------------------------------------------------------------

/// Sliding DFT for efficient streaming frequency analysis.
///
/// Maintains a sliding window DFT that updates in O(K) per sample (where K is
/// the number of tracked frequencies), compared to O(N log N) for recomputing
/// the full FFT every sample. The sliding DFT is exact for frequencies that are
/// exact DFT bin frequencies (i.e., `f = k * fs / N` for integer k).
///
/// For arbitrary frequencies the algorithm uses the frequency-domain update rule:
/// ```text
/// X_new[k] = (X_old[k] - x_out + x_in) * W[k]
/// ```
/// where `W[k] = exp(j * 2π * f_k / fs)` and `x_out` is the oldest sample.
///
/// # Examples
///
/// ```
/// use scirs2_signal::czt::SlidingDft;
///
/// let fs = 1000.0_f64;
/// let freqs = vec![50.0, 100.0, 200.0];
/// let window_size = 128;
/// let mut sdft = SlidingDft::new(freqs, fs, window_size);
///
/// // Push samples one at a time
/// for i in 0..256_usize {
///     let sample = (2.0 * std::f64::consts::PI * 100.0 * i as f64 / fs).sin();
///     let spectrum = sdft.push(sample);
///     assert_eq!(spectrum.len(), 3); // one value per tracked frequency
/// }
/// ```
pub struct SlidingDft {
    /// Tracked frequencies (Hz)
    freqs: Vec<f64>,
    /// Sampling frequency
    fs: f64,
    /// Number of tracked frequencies
    n_freqs: usize,
    /// Window size (N)
    window_size: usize,
    /// Current DFT state (one complex value per tracked frequency)
    state: Vec<Complex64>,
    /// Circular buffer of input samples
    buffer: VecDeque<f64>,
    /// Rotation factors W[k] = exp(j * 2π * f_k / fs)
    rotation: Vec<Complex64>,
}

impl SlidingDft {
    /// Create a new SlidingDft tracker.
    ///
    /// # Arguments
    ///
    /// * `freqs` - Frequencies to track (Hz). Must be in [0, fs/2].
    /// * `fs` - Sampling frequency (Hz)
    /// * `window_size` - Analysis window length (number of samples)
    pub fn new(freqs: Vec<f64>, fs: f64, window_size: usize) -> Self {
        let n_freqs = freqs.len();

        // Precompute rotation factors W[k] = exp(j * 2π * f_k / fs)
        let rotation: Vec<Complex64> = freqs
            .iter()
            .map(|&f| {
                let theta = 2.0 * PI * f / fs;
                Complex64::new(theta.cos(), theta.sin())
            })
            .collect();

        let state = vec![Complex64::new(0.0, 0.0); n_freqs];
        let buffer = VecDeque::with_capacity(window_size + 1);

        Self {
            freqs,
            fs,
            n_freqs,
            window_size,
            state,
            buffer,
            rotation,
        }
    }

    /// Push a new sample and return the updated DFT at all tracked frequencies.
    ///
    /// Uses the recursive update: `X_new[k] = (X_old[k] - x_out + x_in) * W[k]`
    ///
    /// # Returns
    ///
    /// Vector of complex DFT values at each tracked frequency. The values are
    /// normalized by `1/window_size` so they are comparable to a DFT output.
    pub fn push(&mut self, sample: f64) -> Vec<Complex64> {
        // Get the oldest sample that is about to leave the window
        let x_out = if self.buffer.len() >= self.window_size {
            self.buffer.pop_front().unwrap_or(0.0)
        } else {
            0.0
        };

        // Add new sample to buffer
        self.buffer.push_back(sample);

        // Update DFT state for each tracked frequency
        for k in 0..self.n_freqs {
            // Sliding DFT update rule
            self.state[k] = (self.state[k] - Complex64::new(x_out, 0.0)
                + Complex64::new(sample, 0.0))
                * self.rotation[k];
        }

        // Return normalized copy of state
        let scale = 1.0 / self.window_size as f64;
        self.state
            .iter()
            .map(|&c| Complex64::new(c.re * scale, c.im * scale))
            .collect()
    }

    /// Return the tracked frequencies.
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Return the sampling frequency.
    pub fn fs(&self) -> f64 {
        self.fs
    }

    /// Return the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Return the current number of samples buffered.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Reset internal state (clear buffer and DFT state).
    pub fn reset(&mut self) {
        self.buffer.clear();
        for s in self.state.iter_mut() {
            *s = Complex64::new(0.0, 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_tone(freq: f64, n: usize, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_czt_points() {
        // Generate 4 points on the unit circle
        let points = czt_points(4, None, None).expect("Operation failed");

        // Check length
        assert_eq!(points.len(), 4);

        // First point should be 1+0j
        assert_relative_eq!(points[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(points[0].im, 0.0, epsilon = 1e-10);

        // Check that points are evenly spaced on unit circle (W = exp(-j*2π/4))
        points.iter().enumerate().take(4).for_each(|(i, point)| {
            let angle = -2.0 * PI * i as f64 / 4.0;
            let expected = Complex64::new(angle.cos(), angle.sin());

            assert_relative_eq!(point.re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(point.im, expected.im, epsilon = 1e-10);
        });
    }

    /// Helper: compute an N-point DFT directly (O(N²)) for reference comparison.
    fn dft_direct(x: &[f64]) -> Vec<Complex64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut sum = Complex64::new(0.0, 0.0);
                for (ni, &xn) in x.iter().enumerate() {
                    let angle = -2.0 * PI * k as f64 * ni as f64 / n as f64;
                    sum += Complex64::new(xn, 0.0) * Complex64::new(angle.cos(), angle.sin());
                }
                sum
            })
            .collect()
    }

    #[test]
    fn test_czt_dft_equivalence() {
        // CZT with default parameters (w = exp(-j*2π/N), a = 1, m = N)
        // must be identical to the standard DFT within floating-point tolerance.
        let signals: &[&[f64]] = &[
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 0.0, -1.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ];

        for signal in signals {
            let czt_result = czt(*signal, None, None, None, None).expect("CZT failed");
            let dft_result = dft_direct(signal);

            assert_eq!(czt_result.len(), dft_result.len());

            for (k, (c, d)) in czt_result.iter().zip(dft_result.iter()).enumerate() {
                let err_re = (c.re - d.re).abs();
                let err_im = (c.im - d.im).abs();
                assert!(
                    err_re < 1e-9,
                    "signal index {}, bin {}: CZT re={} vs DFT re={} (err={})",
                    signal.len(),
                    k,
                    c.re,
                    d.re,
                    err_re
                );
                assert!(
                    err_im < 1e-9,
                    "signal index {}, bin {}: CZT im={} vs DFT im={} (err={})",
                    signal.len(),
                    k,
                    c.im,
                    d.im,
                    err_im
                );
            }
        }
    }

    #[test]
    fn test_czt_dft_properties() {
        // Verify DFT properties when CZT uses default parameters.
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
        let czt_result = czt(&signal, None, None, None, None).expect("CZT failed");

        // 1. Length
        assert_eq!(czt_result.len(), 4);

        // 2. DC component should be purely real and equal to sum of signal
        let dc_sum: f64 = signal.iter().sum();
        assert_relative_eq!(czt_result[0].re, dc_sum, epsilon = 1e-9);
        assert_relative_eq!(czt_result[0].im, 0.0, epsilon = 1e-9);

        // 3. Nyquist bin (index N/2) should be real for real input
        assert_relative_eq!(czt_result[2].im, 0.0, epsilon = 1e-9);

        // 4. Conjugate symmetry: X[N-k] = conj(X[k])
        assert_relative_eq!(czt_result[1].re, czt_result[3].re, epsilon = 1e-9);
        assert_relative_eq!(czt_result[1].im, -czt_result[3].im, epsilon = 1e-9);

        // 5. Linearity: CZT(2x) = 2 * CZT(x)
        let signal2: Vec<f64> = signal.iter().map(|&v| 2.0 * v).collect();
        let czt_result2 = czt(&signal2, None, None, None, None).expect("CZT failed");
        for k in 0..4 {
            assert_relative_eq!(czt_result2[k].re, 2.0 * czt_result[k].re, epsilon = 1e-9);
            assert_relative_eq!(czt_result2[k].im, 2.0 * czt_result[k].im, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_czt_zoom() {
        // Test CZT for "zooming in" on a specific frequency range
        let signal = vec![1.0, 0.0, 1.0, 0.0]; // Simple 2Hz signal (when sampled at 8Hz)

        // Compute 8-point CZT that zooms in on the first quarter of the spectrum
        // This means w = exp(-j*π/16)
        let arg = -PI / 16.0;
        let w = Complex64::new(arg.cos(), arg.sin());

        let czt_result = czt(&signal, Some(8), Some(w), None, None).expect("Operation failed");

        // Check length
        assert_eq!(czt_result.len(), 8);

        // Find max magnitude bin
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for (i, val) in czt_result.iter().enumerate() {
            let mag = val.norm();
            if mag > max_val {
                max_val = mag;
                max_idx = i;
            }
        }

        // Check that we have significant energy somewhere in the array
        assert!(max_val > 1.0);

        println!(
            "Max energy found at bin {} with magnitude {}",
            max_idx, max_val
        );
    }

    #[test]
    fn test_czt_non_power_of_two_length() {
        // CZT can handle non-power-of-2 input lengths unlike standard FFT
        let signal: Vec<f64> = (0..7).map(|i| i as f64).collect();
        let czt_result = czt(&signal, None, None, None, None).expect("CZT failed");
        let dft_result = dft_direct(&signal);

        assert_eq!(czt_result.len(), 7);

        for (k, (c, d)) in czt_result.iter().zip(dft_result.iter()).enumerate() {
            let err = (c - d).norm();
            assert!(
                err < 1e-8,
                "bin {}: CZT=({},{}) vs DFT=({},{}) err={}",
                k,
                c.re,
                c.im,
                d.re,
                d.im,
                err
            );
        }
    }

    #[test]
    fn test_czt_m_greater_than_n() {
        // CZT can produce more output points than input length
        let signal = vec![1.0_f64, 0.0, 0.0, 0.0];
        // With a = 1 and w = exp(-j*2π/8), 8 output points on the unit circle
        let w_arg = -2.0 * PI / 8.0;
        let w = Complex64::new(w_arg.cos(), w_arg.sin());
        let czt_result = czt(&signal, Some(8), Some(w), None, None).expect("CZT failed");

        // For x = [1, 0, 0, 0], all DFT bins have value 1.0+0j
        assert_eq!(czt_result.len(), 8);
        for (k, c) in czt_result.iter().enumerate() {
            let err = (c.norm() - 1.0).abs();
            assert!(
                err < 1e-9,
                "bin {} magnitude should be 1.0, got {}",
                k,
                c.norm()
            );
        }
    }

    // --- zoom_fft tests ---

    #[test]
    fn test_zoom_fft_output_length() {
        let fs = 1000.0;
        let signal = make_tone(200.0, 512, fs);
        let m = 64;
        let result = zoom_fft(&signal, 100.0, 300.0, m, fs).expect("zoom_fft failed");
        assert_eq!(result.len(), m);
    }

    #[test]
    fn test_zoom_fft_freqs_length() {
        let freqs = zoom_fft_freqs(100.0, 300.0, 64);
        assert_eq!(freqs.len(), 64);
        assert_relative_eq!(freqs[0], 100.0, epsilon = 1e-10);
        assert_relative_eq!(freqs[63], 300.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zoom_fft_peak_at_tone_frequency() {
        let fs = 1000.0;
        let freq = 250.0;
        let n = 512;
        let signal = make_tone(freq, n, fs);
        let m = 128;
        // Zoom into [200, 300] Hz
        let spectrum = zoom_fft(&signal, 200.0, 300.0, m, fs).expect("zoom_fft failed");
        let freqs = zoom_fft_freqs(200.0, 300.0, m);

        // Find peak
        let (peak_idx, _) = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.norm()
                    .partial_cmp(&b.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("empty spectrum");

        // The peak frequency should be close to 250 Hz
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - freq).abs() < 2.0,
            "Expected peak near {} Hz, got {} Hz",
            freq,
            peak_freq
        );
    }

    #[test]
    fn test_zoom_fft_empty_signal_error() {
        assert!(zoom_fft(&[], 100.0, 300.0, 64, 1000.0).is_err());
    }

    #[test]
    fn test_zoom_fft_invalid_freqs_error() {
        let signal = vec![1.0; 64];
        assert!(zoom_fft(&signal, 300.0, 100.0, 64, 1000.0).is_err()); // f1 > f2
        assert!(zoom_fft(&signal, 100.0, 600.0, 64, 1000.0).is_err()); // f2 > fs/2
    }

    #[test]
    fn test_zoom_fft_freqs_single_point() {
        let freqs = zoom_fft_freqs(300.0, 500.0, 1);
        assert_eq!(freqs.len(), 1);
        assert_relative_eq!(freqs[0], 300.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zoom_fft_freqs_empty() {
        let freqs = zoom_fft_freqs(100.0, 200.0, 0);
        assert!(freqs.is_empty());
    }

    // --- Goertzel tests ---

    #[test]
    fn test_goertzel_matches_fft_magnitude() {
        let fs = 8000.0;
        let n = 256;
        let freq = 1000.0_f64;
        let signal = make_tone(freq, n, fs);

        // Goertzel at exact DFT bin frequency
        let bin_freq = (freq * n as f64 / fs).round() * fs / n as f64;
        let goertzel_result = goertzel(&signal, &[bin_freq], fs).expect("goertzel failed");

        // Compute DFT reference using the module's own FFT
        let complex_signal: Vec<Complex64> =
            signal.iter().map(|&s| Complex64::new(s, 0.0)).collect();
        let mut buf = complex_signal;
        // pad to power of 2 = 256
        fft_inplace(&mut buf, false).expect("fft failed");

        // Find the bin corresponding to bin_freq
        let bin_idx = (bin_freq * n as f64 / fs).round() as usize;
        let fft_mag = buf[bin_idx].norm();
        let goertzel_mag = goertzel_result[0].norm();

        // Magnitudes should match within 0.1%
        assert_relative_eq!(goertzel_mag, fft_mag, epsilon = fft_mag * 0.001 + 0.01);
    }

    #[test]
    fn test_goertzel_output_length() {
        let signal = make_tone(1000.0, 256, 8000.0);
        let freqs = [500.0, 1000.0, 2000.0, 3000.0];
        let result = goertzel(&signal, &freqs, 8000.0).expect("goertzel failed");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_goertzel_empty_signal_error() {
        assert!(goertzel(&[], &[1000.0], 8000.0).is_err());
    }

    #[test]
    fn test_goertzel_out_of_range_freq_error() {
        let signal = vec![0.0; 64];
        assert!(goertzel(&signal, &[5000.0], 8000.0).is_err()); // f > fs/2
    }

    #[test]
    fn test_goertzel_dc_component() {
        // DC signal: all ones
        let signal = vec![1.0_f64; 64];
        let result = goertzel(&signal, &[0.0], 1000.0).expect("goertzel failed");
        // DC DFT value should be N (sum of all samples)
        assert_relative_eq!(result[0].re, 64.0, epsilon = 1e-8);
        assert_relative_eq!(result[0].im, 0.0, epsilon = 1e-8);
    }

    // --- SlidingDft tests ---

    #[test]
    fn test_sliding_dft_output_length() {
        let mut sdft = SlidingDft::new(vec![100.0, 200.0, 300.0], 1000.0, 64);
        let spectrum = sdft.push(1.0);
        assert_eq!(spectrum.len(), 3);
    }

    #[test]
    fn test_sliding_dft_window_fills() {
        let fs = 1000.0;
        let window = 32;
        let freq = 100.0;
        let mut sdft = SlidingDft::new(vec![freq], fs, window);

        // Push a full window of a 100 Hz tone
        for i in 0..(window * 2) {
            let s = (2.0 * PI * freq * i as f64 / fs).sin();
            let spectrum = sdft.push(s);
            assert_eq!(spectrum.len(), 1);
        }
        // After a full window, the DFT at 100 Hz should have non-zero magnitude
        let final_spectrum = sdft.push(0.0);
        let mag = final_spectrum[0].norm();
        // Should detect significant energy at 100 Hz
        assert!(
            mag > 0.0,
            "SlidingDft should have non-zero output after window fills"
        );
    }

    #[test]
    fn test_sliding_dft_reset() {
        let mut sdft = SlidingDft::new(vec![100.0], 1000.0, 32);
        for i in 0..32_usize {
            sdft.push(i as f64 * 0.1);
        }
        sdft.reset();
        assert_eq!(sdft.buffered(), 0);
        let spectrum = sdft.push(0.0);
        assert_relative_eq!(spectrum[0].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sliding_dft_accessors() {
        let freqs = vec![50.0, 100.0];
        let fs = 500.0;
        let window = 64;
        let sdft = SlidingDft::new(freqs.clone(), fs, window);
        assert_eq!(sdft.freqs(), freqs.as_slice());
        assert_relative_eq!(sdft.fs(), fs, epsilon = 1e-10);
        assert_eq!(sdft.window_size(), window);
        assert_eq!(sdft.buffered(), 0);
    }
}
