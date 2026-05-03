//! Async streaming API for WASM environments.
//!
//! This module provides async functions that consume JavaScript
//! [`ReadableStream`] objects and process them incrementally using
//! [`wasm_bindgen_futures::JsFuture`].
//!
//! ## Design
//!
//! Two layers are provided:
//!
//! 1. **Pure-Rust core logic** (`accumulate_samples`, `compute_spectrum`)
//!    — testable on native targets without a JS runtime.
//!
//! 2. **`#[wasm_bindgen]` async adapters** — thin wrappers that drive the
//!    JS `ReadableStream` protocol and call the core logic.
//!
//! ## Transform types for `async_transform`
//!
//! | `transform_type` | Operation |
//! |-----------------|-----------|
//! | `0` | FFT (power spectrum) |
//! | `1` | DCT-II |
//! | `2` | Normalize to `[-1, 1]` |
//!
//! ## Threading note
//!
//! These functions run on the WASM main thread / event loop.  For
//! CPU-intensive transforms you can offload to a `WorkerMessage` using the
//! `worker` module; this module focuses on the stream-reading half.

use std::f32::consts::PI;

use js_sys::{Float32Array, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{ReadableStream, ReadableStreamDefaultReader};

// ============================================================================
// Pure-Rust core logic (testable on all targets)
// ============================================================================

/// Radix-2 Cooley–Tukey in-place DIT FFT.
///
/// `buf` must have a length that is a power of two.
fn fft_inplace(buf: &mut [num_complex::Complex<f32>]) {
    let n = buf.len();
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // Butterfly stages.
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / len as f32;
        let w_n = num_complex::Complex::new(ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w = num_complex::Complex::new(1.0_f32, 0.0_f32);
            for k in 0..len / 2 {
                let u = buf[i + k];
                let v = buf[i + k + len / 2] * w;
                buf[i + k] = u + v;
                buf[i + k + len / 2] = u - v;
                w *= w_n;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Compute the power spectrum of `samples` using an FFT of size `fft_size`.
///
/// * `samples` — time-domain f32 samples (at least `fft_size` elements).
/// * `fft_size` — must be a power of two; excess samples are ignored.
///
/// Returns a `Vec<f32>` of length `fft_size / 2 + 1` (non-negative frequencies).
///
/// # Errors
///
/// Returns `Err` if `fft_size` is zero, not a power of two, or if `samples`
/// is shorter than `fft_size`.
pub fn compute_spectrum(samples: &[f32], fft_size: usize) -> Result<Vec<f32>, String> {
    if fft_size == 0 || !fft_size.is_power_of_two() {
        return Err(format!(
            "fft_size must be a non-zero power of two, got {fft_size}"
        ));
    }
    if samples.len() < fft_size {
        return Err(format!(
            "need at least {fft_size} samples, got {}",
            samples.len()
        ));
    }

    let mut buf: Vec<num_complex::Complex<f32>> = samples[..fft_size]
        .iter()
        .copied()
        .map(|x| num_complex::Complex::new(x, 0.0_f32))
        .collect();
    fft_inplace(&mut buf);

    let n_bins = fft_size / 2 + 1;
    let spectrum: Vec<f32> = buf[..n_bins].iter().map(|c| c.norm()).collect();
    Ok(spectrum)
}

/// DCT-II of `samples` (same-length output).
///
/// Uses the definition: `X[k] = 2 * Σ_{n=0}^{N-1} x[n] * cos(π*(n+0.5)*k/N)`.
pub fn compute_dct(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|k| {
            2.0 * samples
                .iter()
                .enumerate()
                .map(|(m, &x)| x * (PI * (m as f32 + 0.5) * k as f32 / n as f32).cos())
                .sum::<f32>()
        })
        .collect()
}

/// Normalize `samples` to the range `[-1, 1]`.
///
/// If all values are equal the slice is returned unchanged (already uniform).
pub fn normalize_samples(samples: &[f32]) -> Vec<f32> {
    let max_abs = samples
        .iter()
        .copied()
        .fold(0.0_f32, |acc, x| acc.max(x.abs()));
    if max_abs == 0.0 {
        return samples.to_vec();
    }
    samples.iter().map(|&x| x / max_abs).collect()
}

/// Accumulate chunks of `f32` samples from a sequence of JS `Float32Array`
/// chunks into a single `Vec<f32>`.
///
/// This is the pure-Rust version used by both the async WASM adapter and
/// unit tests.  `chunks` is an iterator of `Vec<f32>`.
pub fn accumulate_samples<I>(chunks: I, min_len: usize) -> Vec<f32>
where
    I: IntoIterator<Item = Vec<f32>>,
{
    let mut buf = Vec::with_capacity(min_len);
    for chunk in chunks {
        buf.extend_from_slice(&chunk);
        if buf.len() >= min_len {
            break;
        }
    }
    buf
}

// ============================================================================
// WASM async adapters
// ============================================================================

/// Async streaming FFT processor — consumes a JS `ReadableStream` of
/// `Float32Array` chunks.
///
/// Pulls chunks from the stream until `fft_size` samples have been collected,
/// then returns the power spectrum as a `Float32Array` of length
/// `fft_size / 2 + 1`.
///
/// ## Errors
///
/// Returns a `JsValue` error when:
/// - acquiring the reader fails,
/// - any `read()` promise rejects,
/// - `fft_size` is zero or not a power of two,
/// - the stream ends before `fft_size` samples are available.
///
/// ## COOP / COEP note
///
/// This function does not require `SharedArrayBuffer`; it works in any secure
/// context (HTTPS or `localhost`).
#[wasm_bindgen]
pub async fn streaming_fft_from_readable(
    stream: ReadableStream,
    fft_size: usize,
) -> Result<Float32Array, JsValue> {
    if fft_size == 0 || !fft_size.is_power_of_two() {
        return Err(JsValue::from_str(&format!(
            "fft_size must be a non-zero power of two, got {fft_size}"
        )));
    }

    let reader = ReadableStreamDefaultReader::new(&stream)
        .map_err(|e| JsValue::from_str(&format!("Failed to acquire reader: {e:?}")))?;

    let mut samples: Vec<f32> = Vec::with_capacity(fft_size);

    loop {
        if samples.len() >= fft_size {
            break;
        }

        let read_promise = reader.read();
        let result = JsFuture::from(read_promise)
            .await
            .map_err(|e| JsValue::from_str(&format!("Stream read error: {e:?}")))?;

        // result is a plain JS object {value: Float32Array|undefined, done: bool}
        let done = Reflect::get(&result, &JsValue::from_str("done"))
            .map(|v| v.as_bool().unwrap_or(false))
            .unwrap_or(false);

        if done {
            break;
        }

        let chunk_val =
            Reflect::get(&result, &JsValue::from_str("value")).unwrap_or(JsValue::UNDEFINED);
        if chunk_val.is_undefined() || chunk_val.is_null() {
            continue;
        }

        let chunk = Float32Array::from(chunk_val);
        let n = chunk.length() as usize;
        let remaining = fft_size - samples.len();
        let take = n.min(remaining);

        let mut tmp = vec![0.0_f32; take];
        chunk.slice(0, take as u32).copy_to(&mut tmp);
        samples.extend_from_slice(&tmp);
    }

    // Release the lock on the stream.
    reader.release_lock();

    if samples.len() < fft_size {
        return Err(JsValue::from_str(&format!(
            "Stream ended with only {} samples; need {fft_size}",
            samples.len()
        )));
    }

    let spectrum = compute_spectrum(&samples, fft_size)
        .map_err(|e| JsValue::from_str(&format!("FFT error: {e}")))?;

    let out = Float32Array::new_with_length(spectrum.len() as u32);
    out.copy_from(&spectrum);
    Ok(out)
}

/// Async batch transform — processes a flat `f32` slice asynchronously.
///
/// `transform_type`:
/// - `0` — FFT power spectrum (output length `len/2 + 1` where `len` is
///   the largest power-of-two ≤ `data.len()`).
/// - `1` — DCT-II (output same length as input).
/// - `2` — Normalize to `[-1, 1]` (output same length as input).
///
/// Any other value returns an error.
#[wasm_bindgen]
pub async fn async_transform(data: &[f32], transform_type: u32) -> Result<Float32Array, JsValue> {
    // Yielding to the event loop is not directly possible without a
    // JS-exposed scheduler, but wrapping in `async` allows callers to
    // `await` and gives the runtime a natural suspension point between
    // other promises.
    let result: Vec<f32> = match transform_type {
        0 => {
            // FFT: use the largest power-of-two that fits in data.
            let fft_size = data.len().next_power_of_two();
            let fft_size = if fft_size > data.len() {
                fft_size >> 1
            } else {
                fft_size
            };
            if fft_size == 0 {
                return Err(JsValue::from_str("async_transform(FFT): data is empty"));
            }
            compute_spectrum(data, fft_size)
                .map_err(|e| JsValue::from_str(&format!("FFT error: {e}")))?
        }
        1 => compute_dct(data),
        2 => normalize_samples(data),
        other => {
            return Err(JsValue::from_str(&format!(
                "Unknown transform_type {other}; expected 0=FFT, 1=DCT, 2=normalize"
            )));
        }
    };

    let out = Float32Array::new_with_length(result.len() as u32);
    out.copy_from(&result);
    Ok(out)
}

// ============================================================================
// Tests (native target only — no JS runtime needed)
// ============================================================================

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    /// Helper: assert close to within an absolute tolerance.
    fn assert_near(a: f32, b: f32, tol: f32, label: &str) {
        assert!(
            (a - b).abs() <= tol,
            "{label}: |{a} - {b}| = {} > {tol}",
            (a - b).abs()
        );
    }

    // ------------------------------------------------------------------
    // fft_inplace
    // ------------------------------------------------------------------

    #[test]
    fn fft_dc_signal() {
        // All-ones input → DC bin = N, all other bins = 0.
        let n = 8;
        let mut buf: Vec<num_complex::Complex<f32>> = (0..n)
            .map(|_| num_complex::Complex::new(1.0_f32, 0.0))
            .collect();
        fft_inplace(&mut buf);
        assert_near(buf[0].norm(), n as f32, 1e-4, "DC bin");
        for (k, entry) in buf.iter().enumerate().skip(1) {
            assert_near(entry.norm(), 0.0, 1e-4, &format!("bin {k}"));
        }
    }

    #[test]
    fn fft_single_tone() {
        // cos(2π k0 n / N) → peak at bin k0 and N-k0.
        let n = 16usize;
        let k0 = 3usize;
        let mut buf: Vec<num_complex::Complex<f32>> = (0..n)
            .map(|i| {
                let angle = 2.0 * PI * k0 as f32 * i as f32 / n as f32;
                num_complex::Complex::new(angle.cos(), 0.0_f32)
            })
            .collect();
        fft_inplace(&mut buf);
        let peak_idx = buf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.norm()
                    .partial_cmp(&b.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!(
            peak_idx == k0 || peak_idx == n - k0,
            "Expected peak at {k0} or {}, got {peak_idx}",
            n - k0
        );
    }

    // ------------------------------------------------------------------
    // compute_spectrum
    // ------------------------------------------------------------------

    #[test]
    fn spectrum_length() {
        let samples: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let spec = compute_spectrum(&samples, 32).expect("spectrum ok");
        assert_eq!(spec.len(), 32 / 2 + 1);
    }

    #[test]
    fn spectrum_rejects_non_power_of_two() {
        let samples = vec![1.0_f32; 100];
        assert!(compute_spectrum(&samples, 100).is_err());
    }

    #[test]
    fn spectrum_rejects_empty_fft_size() {
        let samples = vec![1.0_f32; 64];
        assert!(compute_spectrum(&samples, 0).is_err());
    }

    #[test]
    fn spectrum_rejects_insufficient_samples() {
        let samples = vec![1.0_f32; 16];
        assert!(compute_spectrum(&samples, 32).is_err());
    }

    // ------------------------------------------------------------------
    // compute_dct
    // ------------------------------------------------------------------

    #[test]
    fn dct_length() {
        let samples: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let dct = compute_dct(&samples);
        assert_eq!(dct.len(), samples.len());
    }

    #[test]
    fn dct_empty() {
        assert!(compute_dct(&[]).is_empty());
    }

    #[test]
    fn dct_constant_signal() {
        // For x[n] = c: X[0] = 2Nc, X[k>0] = 0.
        let n = 8;
        let c = 3.0_f32;
        let samples = vec![c; n];
        let dct = compute_dct(&samples);
        assert_near(dct[0], 2.0 * n as f32 * c, 1e-3, "DC term");
        for (k, &val) in dct.iter().enumerate().skip(1) {
            assert_near(val, 0.0, 1e-3, &format!("AC term {k}"));
        }
    }

    // ------------------------------------------------------------------
    // normalize_samples
    // ------------------------------------------------------------------

    #[test]
    fn normalize_basic() {
        let s = vec![-4.0_f32, 0.0, 2.0, 4.0];
        let n = normalize_samples(&s);
        assert_near(n[0], -1.0, 1e-6, "min");
        assert_near(n[3], 1.0, 1e-6, "max");
    }

    #[test]
    fn normalize_zeros() {
        let s = vec![0.0_f32; 8];
        let n = normalize_samples(&s);
        assert_eq!(n, s);
    }

    // ------------------------------------------------------------------
    // accumulate_samples
    // ------------------------------------------------------------------

    #[test]
    fn accumulate_stops_at_min_len() {
        let chunks = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![4.0_f32, 5.0, 6.0],
            vec![7.0_f32, 8.0, 9.0],
        ];
        let acc = accumulate_samples(chunks, 5);
        // Should stop as soon as len >= 5.
        assert_eq!(acc.len(), 6, "stops at chunk boundary");
        assert_eq!(&acc[..3], &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn accumulate_returns_all_if_below_min() {
        let chunks = vec![vec![1.0_f32, 2.0], vec![3.0_f32]];
        let acc = accumulate_samples(chunks, 100);
        assert_eq!(acc, vec![1.0_f32, 2.0, 3.0]);
    }

    // ------------------------------------------------------------------
    // wasm_bindgen_test placeholders (wasm32 only)
    // ------------------------------------------------------------------

    // The functions `streaming_fft_from_readable` and `async_transform` are
    // exercised via `wasm-pack test --headless` in CI.  Their pure-Rust
    // compute cores are covered above.
}
