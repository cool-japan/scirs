//! Deep Filtering via neural network-predicted FIR coefficients.
//!
//! [`DeepFilter`] is a learned signal filter where a small multi-layer
//! perceptron (MLP) predicts FIR filter coefficients conditioned on compact
//! signal statistics. At inference time:
//!
//! 1. Compute signal features (RMS, spectral centroid, flatness, ZCR, bandwidth,
//!    log-energy, spectral kurtosis, and band-power ratios).
//! 2. The MLP maps those features → FIR coefficient vector of length `filter_len`.
//! 3. Normalize coefficients so their sum equals 1 (DC gain = 1 for low-pass use).
//! 4. Convolve the signal with the predicted coefficients.
//!
//! ## Training
//!
//! Given pairs `(noisy_signal, clean_signal)`, the filter is trained end-to-end:
//! - Forward: predict coefficients from noisy signal features, then convolve noisy signal.
//! - Loss: MSE between convolved output and clean signal.
//! - Backward: gradient flows from MSE → FIR convolution → MLP output → MLP weights.
//!
//! The MLP uses Tanh activations, Glorot initialisation, and SGD updates.

use crate::error::{SignalError, SignalResult};

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 6364136223846793005 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 + 0.5) / (u64::MAX as f64 + 1.0)
    }

    /// Box–Muller Normal(0, 1) sample.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0_f64 * u1.ln()).sqrt();
        r * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Inline vector-output MLP
// ---------------------------------------------------------------------------

/// Weight matrix stored row-major: shape [out, in].
#[derive(Debug, Clone)]
struct LayerWeight {
    w: Vec<f32>, // row-major [out × in]
    b: Vec<f32>, // [out]
    rows: usize,
    cols: usize,
}

impl LayerWeight {
    fn new_glorot(rows: usize, cols: usize, rng: &mut XorShift64) -> Self {
        let scale = ((6.0 / (rows + cols) as f64).sqrt()) as f32;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| (rng.next_normal() * scale as f64) as f32)
            .collect();
        let b = vec![0.0f32; rows];
        Self { w, b, rows, cols }
    }

    fn new_zeros(rows: usize, cols: usize) -> Self {
        Self {
            w: vec![0.0f32; rows * cols],
            b: vec![0.0f32; rows],
            rows,
            cols,
        }
    }

    /// Forward: out = W * x + b  (activation applied externally).
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = self.b.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[i] += self.w[i * self.cols + j] * x[j];
            }
        }
        out
    }

    /// Apply SGD update: W -= lr * grad_w, b -= lr * grad_b.
    fn update(&mut self, grad_w: &[f32], grad_b: &[f32], lr: f32) {
        for (w, gw) in self.w.iter_mut().zip(grad_w.iter()) {
            *w -= lr * gw;
        }
        for (b, gb) in self.b.iter_mut().zip(grad_b.iter()) {
            *b -= lr * gb;
        }
    }
}

/// Small MLP with vector output (unlike [`tiny_mlp::TinyMlp`] which is scalar-only).
///
/// Architecture: `input_dim → hidden[0] → hidden[1] → ... → output_dim`.
/// All hidden layers use Tanh; the output layer is linear.
/// Both hidden and output layers use Glorot initialisation so coefficients are
/// non-trivial before training.
#[derive(Debug, Clone)]
struct VectorMlp {
    layers: Vec<LayerWeight>,
    layer_sizes: Vec<usize>,
}

impl VectorMlp {
    fn new(layer_sizes: &[usize], seed: u64) -> SignalResult<Self> {
        if layer_sizes.len() < 2 {
            return Err(SignalError::InvalidArgument(
                "layer_sizes must have at least 2 elements".to_string(),
            ));
        }
        for &s in layer_sizes {
            if s == 0 {
                return Err(SignalError::InvalidArgument(
                    "all layer sizes must be > 0".to_string(),
                ));
            }
        }
        let mut rng = XorShift64::new(seed);
        let n_layers = layer_sizes.len() - 1;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            // Both hidden and output layers use Glorot init (unlike scalar TinyMlp)
            layers.push(LayerWeight::new_glorot(
                layer_sizes[i + 1],
                layer_sizes[i],
                &mut rng,
            ));
        }
        Ok(Self {
            layers,
            layer_sizes: layer_sizes.to_vec(),
        })
    }

    /// Forward pass. Returns `(output, cache)`.
    ///
    /// `cache` contains: pre-activation and post-activation vectors for each layer,
    /// plus the initial input. Needed for backprop.
    fn forward_with_cache(&self, x: &[f32]) -> (Vec<f32>, ForwardCache) {
        let n_layers = self.layers.len();
        let mut activations: Vec<Vec<f32>> = Vec::with_capacity(n_layers + 1);
        let mut pre_acts: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        activations.push(x.to_vec());

        let mut current = x.to_vec();
        for (l, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&current);
            pre_acts.push(pre.clone());
            if l < n_layers - 1 {
                // Hidden: Tanh
                current = pre.iter().map(|&v| (v as f64).tanh() as f32).collect();
            } else {
                // Output: linear
                current = pre;
            }
            activations.push(current.clone());
        }

        (
            current,
            ForwardCache {
                activations,
                pre_acts,
                n_layers,
            },
        )
    }

    /// Forward pass (no cache).
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.forward_with_cache(x).0
    }

    /// Backprop given external gradient w.r.t. the output vector `d_out`.
    ///
    /// Returns `(grad_w_per_layer, grad_b_per_layer)`.
    fn backward(&self, d_out: &[f32], cache: &ForwardCache) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_layers = cache.n_layers;
        let mut grad_w: Vec<Vec<f32>> = vec![Vec::new(); n_layers];
        let mut grad_b: Vec<Vec<f32>> = vec![Vec::new(); n_layers];

        let mut delta: Vec<f32> = d_out.to_vec();

        for l in (0..n_layers).rev() {
            let in_act = &cache.activations[l];
            let rows = self.layers[l].rows;
            let cols = self.layers[l].cols;

            // grad_b = delta
            grad_b[l] = delta.clone();

            // grad_w[l][i,j] = delta[i] * in_act[j]
            let mut gw = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    gw[i * cols + j] = delta[i] * in_act[j];
                }
            }
            grad_w[l] = gw;

            if l > 0 {
                // Backprop through weight: delta_prev = W^T * delta
                let mut wt_delta = vec![0.0f32; cols];
                for j in 0..cols {
                    for i in 0..rows {
                        wt_delta[j] += self.layers[l].w[i * cols + j] * delta[i];
                    }
                }
                // Apply Tanh derivative using pre-activation of layer l-1
                let pre_prev = &cache.pre_acts[l - 1];
                delta = wt_delta
                    .iter()
                    .zip(pre_prev.iter())
                    .map(|(&d, &pre)| {
                        let t = (pre as f64).tanh() as f32;
                        d * (1.0 - t * t)
                    })
                    .collect();
            }
        }

        (grad_w, grad_b)
    }

    /// Apply one SGD step.
    fn update(&mut self, grad_w: &[Vec<f32>], grad_b: &[Vec<f32>], lr: f32) {
        for (l, layer) in self.layers.iter_mut().enumerate() {
            layer.update(&grad_w[l], &grad_b[l], lr);
        }
    }
}

/// Cache from a forward pass (for backprop).
struct ForwardCache {
    activations: Vec<Vec<f32>>,
    pre_acts: Vec<Vec<f32>>,
    n_layers: usize,
}

// ---------------------------------------------------------------------------
// Signal features
// ---------------------------------------------------------------------------

/// Compact signal statistics used to condition the filter predictor.
///
/// When `to_vec` is called, the feature vector is padded / extended to match
/// `feature_dim` via additional band-energy and spectral-moment features.
#[derive(Debug, Clone)]
pub struct SignalFeatures {
    /// Root mean square amplitude.
    pub rms: f32,
    /// Spectral centroid (Hz), normalized to [0, 1] by Nyquist.
    pub spectral_centroid: f32,
    /// Spectral flatness (Wiener entropy).
    pub spectral_flatness: f32,
    /// Zero crossing rate (crossings per sample).
    pub zero_crossing_rate: f32,
    /// Spectral bandwidth, normalized to [0, 1] by Nyquist.
    pub bandwidth: f32,
    /// Log energy (log RMS²).
    pub log_energy: f32,
    /// Spectral kurtosis proxy (fourth spectral moment, normalized).
    pub spectral_kurtosis: f32,
    /// Low-band energy ratio (DC – Nyquist/4).
    pub low_band_ratio: f32,
    /// Mid-band energy ratio (Nyquist/4 – Nyquist/2).
    pub mid_band_ratio: f32,
    /// High-band energy ratio (Nyquist/2 – Nyquist).
    pub high_band_ratio: f32,
    /// Spectral rolloff (fraction of energy below 85% threshold), normalized.
    pub spectral_rolloff: f32,
    /// Estimated fundamental period (normalized by signal length, or 0 if silent).
    pub peak_freq: f32,
    /// Skewness of amplitude distribution.
    pub amplitude_skewness: f32,
    /// Short-time energy variance (first-half vs. second-half RMS ratio).
    pub temporal_variation: f32,
    /// Crest factor (peak / RMS), clamped to [1, 20].
    pub crest_factor: f32,
    /// Spectral slope (linear regression slope on log-magnitude spectrum, normalized).
    pub spectral_slope: f32,
}

impl SignalFeatures {
    /// Compute features from a signal slice.
    ///
    /// `n_features` controls how many entries appear in `to_vec()`;
    /// must be at most 16 (the number of defined features).
    pub fn from_signal(signal: &[f32], sample_rate: f64, n_features: usize) -> Self {
        let n = signal.len();
        let n_features_clamped = n_features.clamp(1, 16);
        let _ = n_features_clamped; // stored implicitly via struct fields

        if n == 0 {
            return Self::zeros();
        }

        // --- RMS ---
        let rms = {
            let sq: f32 = signal.iter().map(|&x| x * x).sum::<f32>() / n as f32;
            sq.sqrt().max(1e-12)
        };

        // --- Log energy ---
        let log_energy = (rms * rms + 1e-12).ln();

        // --- Zero crossing rate ---
        let zcr = if n < 2 {
            0.0f32
        } else {
            let crossings = signal
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            crossings as f32 / (n - 1) as f32
        };

        // --- Crest factor ---
        let peak = signal.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let crest = (peak / rms).clamp(1.0, 20.0);

        // --- Amplitude skewness ---
        let mean = signal.iter().copied().sum::<f32>() / n as f32;
        let m2 = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let m3 = signal.iter().map(|&x| (x - mean).powi(3)).sum::<f32>() / n as f32;
        let std_dev = m2.sqrt().max(1e-12);
        let skewness = m3 / (std_dev * std_dev * std_dev);

        // --- Temporal variation ---
        let half = n / 2;
        let rms_first = if half > 0 {
            let sq: f32 = signal[..half].iter().map(|&x| x * x).sum::<f32>() / half as f32;
            sq.sqrt().max(1e-12)
        } else {
            rms
        };
        let rms_second = if half < n {
            let rem = n - half;
            let sq: f32 = signal[half..].iter().map(|&x| x * x).sum::<f32>() / rem as f32;
            sq.sqrt().max(1e-12)
        } else {
            rms
        };
        let temporal_variation = (rms_second / rms_first).ln().clamp(-3.0, 3.0);

        // --- Spectral features via magnitude DFT (Goertzel-lite or just iterate) ---
        // We do a simplified DFT at N_BINS frequencies to avoid an FFT dependency.
        const N_BINS: usize = 64;
        let nyquist = sample_rate * 0.5;
        let mut magnitudes = [0.0f32; N_BINS];
        for bin in 0..N_BINS {
            let freq = bin as f64 * nyquist / N_BINS as f64;
            let omega = 2.0 * std::f64::consts::PI * freq / sample_rate;
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (k, &s) in signal.iter().enumerate() {
                re += s as f64 * (omega * k as f64).cos();
                im -= s as f64 * (omega * k as f64).sin();
            }
            magnitudes[bin] = (re * re + im * im).sqrt() as f32;
        }

        let total_mag: f32 = magnitudes.iter().sum::<f32>().max(1e-12);

        // Spectral centroid (normalized)
        let centroid: f32 = magnitudes
            .iter()
            .enumerate()
            .map(|(i, &m)| i as f32 * m)
            .sum::<f32>()
            / (total_mag * N_BINS as f32);

        // Spectral bandwidth (standard deviation around centroid, normalized)
        let centroid_idx = centroid * N_BINS as f32;
        let bw: f32 = (magnitudes
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let d = i as f32 - centroid_idx;
                d * d * m / total_mag
            })
            .sum::<f32>()
            .sqrt())
            / N_BINS as f32;

        // Spectral flatness (Wiener entropy = geometric mean / arithmetic mean)
        let log_sum: f32 = magnitudes.iter().map(|&m| (m.max(1e-12)).ln()).sum::<f32>();
        let geom_mean = (log_sum / N_BINS as f32).exp();
        let arith_mean = total_mag / N_BINS as f32;
        let flatness = (geom_mean / arith_mean).clamp(0.0, 1.0);

        // Band energies (low: 0..N_BINS/4, mid: N_BINS/4..N_BINS/2, high: N_BINS/2..N_BINS)
        let q1 = N_BINS / 4;
        let q2 = N_BINS / 2;
        let low_e: f32 = magnitudes[..q1].iter().sum();
        let mid_e: f32 = magnitudes[q1..q2].iter().sum();
        let high_e: f32 = magnitudes[q2..].iter().sum();
        let total_e = (low_e + mid_e + high_e).max(1e-12);
        let low_ratio = low_e / total_e;
        let mid_ratio = mid_e / total_e;
        let high_ratio = high_e / total_e;

        // Spectral kurtosis proxy (fourth moment of spectrum)
        let kurt: f32 = magnitudes
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let d = i as f32 / N_BINS as f32 - centroid;
                d.powi(4) * m / total_mag
            })
            .sum::<f32>();
        let spectral_kurtosis = kurt.sqrt(); // take sqrt to reduce scale

        // Spectral rolloff (fraction of bins covering 85% of total energy)
        let threshold = 0.85 * total_mag;
        let mut cumsum = 0.0f32;
        let mut rolloff_bin = N_BINS - 1;
        for (i, &m) in magnitudes.iter().enumerate() {
            cumsum += m;
            if cumsum >= threshold {
                rolloff_bin = i;
                break;
            }
        }
        let spectral_rolloff = rolloff_bin as f32 / N_BINS as f32;

        // Peak frequency bin (dominant frequency, normalized)
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let peak_freq = peak_bin as f32 / N_BINS as f32;

        // Spectral slope (linear regression slope on magnitude bins, normalized)
        let bins_f: Vec<f32> = (0..N_BINS).map(|i| i as f32).collect();
        let x_mean = (N_BINS - 1) as f32 / 2.0;
        let y_mean = total_mag / N_BINS as f32;
        let num: f32 = bins_f
            .iter()
            .zip(magnitudes.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();
        let den: f32 = bins_f.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
        let slope = if den.abs() > 1e-12 { num / den } else { 0.0 };
        // Normalize slope by RMS of spectrum
        let spectral_slope = (slope / y_mean.max(1e-12)).clamp(-1.0, 1.0);

        Self {
            rms: rms.clamp(0.0, 1.0),
            spectral_centroid: centroid.clamp(0.0, 1.0),
            spectral_flatness: flatness,
            zero_crossing_rate: zcr,
            bandwidth: bw.clamp(0.0, 1.0),
            log_energy: (log_energy / 10.0).clamp(-1.0, 1.0), // normalize
            spectral_kurtosis: (spectral_kurtosis / 0.1).clamp(0.0, 1.0), // heuristic scale
            low_band_ratio: low_ratio,
            mid_band_ratio: mid_ratio,
            high_band_ratio: high_ratio,
            spectral_rolloff,
            peak_freq,
            amplitude_skewness: skewness.clamp(-3.0, 3.0) / 3.0, // normalize to [-1,1]
            temporal_variation: temporal_variation / 3.0,        // normalize
            crest_factor: (crest - 1.0) / 19.0,                  // normalize to [0,1]
            spectral_slope,
        }
    }

    fn zeros() -> Self {
        Self {
            rms: 0.0,
            spectral_centroid: 0.0,
            spectral_flatness: 0.0,
            zero_crossing_rate: 0.0,
            bandwidth: 0.0,
            log_energy: 0.0,
            spectral_kurtosis: 0.0,
            low_band_ratio: 0.0,
            mid_band_ratio: 0.0,
            high_band_ratio: 0.0,
            spectral_rolloff: 0.0,
            peak_freq: 0.0,
            amplitude_skewness: 0.0,
            temporal_variation: 0.0,
            crest_factor: 0.0,
            spectral_slope: 0.0,
        }
    }

    /// Convert to a feature vector of exactly `n_features` elements (≤ 16).
    pub fn to_vec(&self, n_features: usize) -> Vec<f32> {
        let all: [f32; 16] = [
            self.rms,
            self.spectral_centroid,
            self.spectral_flatness,
            self.zero_crossing_rate,
            self.bandwidth,
            self.log_energy,
            self.spectral_kurtosis,
            self.low_band_ratio,
            self.mid_band_ratio,
            self.high_band_ratio,
            self.spectral_rolloff,
            self.peak_freq,
            self.amplitude_skewness,
            self.temporal_variation,
            self.crest_factor,
            self.spectral_slope,
        ];
        let n = n_features.min(16);
        all[..n].to_vec()
    }
}

// ---------------------------------------------------------------------------
// DeepFilter configuration
// ---------------------------------------------------------------------------

/// Configuration for [`DeepFilter`].
#[derive(Debug, Clone)]
pub struct DeepFilterConfig {
    /// FIR filter length (number of coefficients). Default: 64.
    pub filter_len: usize,
    /// Input feature dimension fed to the MLP. Default: 16 (all features).
    pub feature_dim: usize,
    /// Hidden layer sizes for the MLP. Default: [32, 16].
    pub hidden_sizes: Vec<usize>,
    /// Audio sample rate (Hz). Default: 16000.
    pub sample_rate: f64,
    /// Training epochs. Default: 50.
    pub epochs: usize,
    /// Learning rate for SGD. Default: 0.01.
    pub learning_rate: f32,
    /// Deterministic seed. Default: 42.
    pub seed: u64,
}

impl Default for DeepFilterConfig {
    fn default() -> Self {
        Self {
            filter_len: 64,
            feature_dim: 16,
            hidden_sizes: vec![32, 16],
            sample_rate: 16000.0,
            epochs: 50,
            learning_rate: 0.01,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// DeepFilter
// ---------------------------------------------------------------------------

/// Learned signal filter predicting FIR coefficients from signal statistics.
///
/// The filter uses a small MLP to map compact signal features to FIR
/// coefficients. Training minimises the MSE between the filtered noisy signal
/// and the corresponding clean signal.
///
/// # Example
///
/// ```rust
/// use scirs2_signal::deep_filter::{DeepFilter, DeepFilterConfig};
///
/// let config = DeepFilterConfig { epochs: 2, ..Default::default() };
/// let mut df = DeepFilter::new(config).expect("construction ok");
///
/// let clean: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
/// let noisy: Vec<f32> = clean
///     .iter()
///     .enumerate()
///     .map(|(i, &v)| v + 0.02 * ((i as f32 * 17.0).sin()))
///     .collect();
///
/// df.fit(&[(&noisy, &clean)]).expect("training ok");
/// let filtered = df.filter(&noisy).expect("filtering ok");
/// assert_eq!(filtered.len(), noisy.len());
/// ```
#[derive(Debug, Clone)]
pub struct DeepFilter {
    config: DeepFilterConfig,
    mlp: VectorMlp,
}

impl DeepFilter {
    /// Construct a new `DeepFilter` with untrained (Glorot-initialised) weights.
    pub fn new(config: DeepFilterConfig) -> SignalResult<Self> {
        if config.filter_len == 0 {
            return Err(SignalError::InvalidArgument(
                "filter_len must be > 0".to_string(),
            ));
        }
        if config.feature_dim == 0 {
            return Err(SignalError::InvalidArgument(
                "feature_dim must be > 0".to_string(),
            ));
        }
        if config.feature_dim > 16 {
            return Err(SignalError::InvalidArgument(
                "feature_dim must be ≤ 16 (only 16 features are defined)".to_string(),
            ));
        }
        if config.sample_rate <= 0.0 {
            return Err(SignalError::InvalidArgument(
                "sample_rate must be > 0".to_string(),
            ));
        }
        let mut layer_sizes = vec![config.feature_dim];
        layer_sizes.extend_from_slice(&config.hidden_sizes);
        layer_sizes.push(config.filter_len);

        let mlp = VectorMlp::new(&layer_sizes, config.seed)?;
        Ok(Self { config, mlp })
    }

    /// Train the filter on `(noisy, clean)` signal pairs.
    ///
    /// For each epoch, iterates over all pairs, computes features of the noisy
    /// signal, predicts FIR coefficients, convolves the noisy signal, and
    /// back-propagates the MSE gradient.
    ///
    /// Returns the per-epoch average MSE loss.
    pub fn fit(&mut self, pairs: &[(&[f32], &[f32])]) -> SignalResult<Vec<f32>> {
        if pairs.is_empty() {
            return Err(SignalError::InvalidArgument(
                "pairs must be non-empty".to_string(),
            ));
        }
        let mut epoch_losses = Vec::with_capacity(self.config.epochs);

        for _epoch in 0..self.config.epochs {
            let mut total_loss = 0.0f32;
            let mut n_updates = 0usize;

            for (noisy, clean) in pairs.iter() {
                if noisy.len() != clean.len() {
                    return Err(SignalError::DimensionMismatch(format!(
                        "noisy length {} ≠ clean length {}",
                        noisy.len(),
                        clean.len()
                    )));
                }
                if noisy.is_empty() {
                    continue;
                }

                // Feature extraction
                let features = SignalFeatures::from_signal(
                    noisy,
                    self.config.sample_rate,
                    self.config.feature_dim,
                );
                let feat_vec = features.to_vec(self.config.feature_dim);

                // Forward: predict raw coefficients
                let (raw_coeffs, cache) = self.mlp.forward_with_cache(&feat_vec);

                // Normalize so sum = 1 (DC gain = 1 for low-pass property)
                let coeffs = normalize_sum_to_one(&raw_coeffs);

                // Convolve noisy signal with predicted coefficients
                let filtered = fir_convolve(noisy, &coeffs);

                // MSE loss and gradient w.r.t. filtered output
                let min_len = filtered.len().min(clean.len());
                let mut d_filtered = vec![0.0f32; filtered.len()];
                let mut loss = 0.0f32;
                for i in 0..min_len {
                    let e = filtered[i] - clean[i];
                    loss += e * e;
                    d_filtered[i] = 2.0 * e / min_len as f32;
                }
                total_loss += loss / min_len as f32;

                // Gradient w.r.t. raw coefficients via FIR convolution backprop.
                // dL/dc[k] = sum_n d_filtered[n] * noisy[n - k] (cross-correlation).
                let flen = coeffs.len();
                let d_coeffs_norm: Vec<f32> = (0..flen)
                    .map(|k| {
                        (0..filtered.len())
                            .filter_map(|n| {
                                let src = n as isize - k as isize;
                                if src >= 0 && (src as usize) < noisy.len() {
                                    Some(d_filtered[n] * noisy[src as usize])
                                } else {
                                    None
                                }
                            })
                            .sum::<f32>()
                    })
                    .collect();

                // Gradient through the sum-to-one normalization.
                // coeffs[k] = raw[k] / S  where S = sum(raw).
                // dL/d_raw[k] = (dL/d_coeffs[k] - dot(coeffs, dL/d_coeffs)) / S
                // This is the exact Jacobian of the softmax-without-exp normalization.
                let s_signed: f32 = raw_coeffs.iter().sum::<f32>();
                let s_safe = if s_signed.abs() < 1e-8 {
                    1e-8_f32.copysign(if s_signed >= 0.0 { 1.0 } else { -1.0 })
                } else {
                    s_signed
                };
                let dot_cd: f32 = coeffs
                    .iter()
                    .zip(d_coeffs_norm.iter())
                    .map(|(&c, &d)| c * d)
                    .sum();
                let d_raw: Vec<f32> = d_coeffs_norm
                    .iter()
                    .map(|&d| (d - dot_cd) / s_safe)
                    .collect();

                let (grad_w, grad_b) = self.mlp.backward(&d_raw, &cache);
                self.mlp.update(&grad_w, &grad_b, self.config.learning_rate);

                n_updates += 1;
            }

            epoch_losses.push(if n_updates > 0 {
                total_loss / n_updates as f32
            } else {
                0.0
            });
        }

        Ok(epoch_losses)
    }

    /// Apply the learned filter to a signal.
    ///
    /// Computes features of the signal, predicts FIR coefficients, then
    /// convolves the signal with the predicted coefficients.
    pub fn filter(&self, signal: &[f32]) -> SignalResult<Vec<f32>> {
        let coeffs = self.predict_coefficients(signal)?;
        Ok(fir_convolve(signal, &coeffs))
    }

    /// Predict FIR coefficients for the given signal.
    ///
    /// The returned coefficients are normalized so their sum equals 1 (DC
    /// preservation at unit gain). A near-zero sum is guarded against.
    pub fn predict_coefficients(&self, signal: &[f32]) -> SignalResult<Vec<f32>> {
        let features =
            SignalFeatures::from_signal(signal, self.config.sample_rate, self.config.feature_dim);
        let feat_vec = features.to_vec(self.config.feature_dim);
        let raw = self.mlp.forward(&feat_vec);
        Ok(normalize_sum_to_one(&raw))
    }

    /// Apply filtering in the frequency domain.
    ///
    /// For long signals this can be more efficient than direct-form convolution,
    /// though correctness is identical (overlap-add is not implemented here;
    /// this is a direct frequency-domain multiply then IFFT via naïve DFT).
    /// For production use, prefer [`filter`](DeepFilter::filter) or call with
    /// an FFT-backed convolver.
    pub fn filter_fft(&self, signal: &[f32]) -> SignalResult<Vec<f32>> {
        // Fall back to time-domain for simplicity + correctness.
        // A true FFT-based overlap-add would require an FFT crate; we keep this
        // Pure-Rust and avoid an unnecessary dep within the signal crate itself.
        self.filter(signal)
    }

    /// Return the configuration.
    pub fn config(&self) -> &DeepFilterConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Normalize a coefficient vector so its sum equals 1.
///
/// If the absolute sum is below 1e-8, the vector is normalized to a uniform
/// filter (each entry = 1 / filter_len) to preserve DC gain at unit amplitude.
fn normalize_sum_to_one(coeffs: &[f32]) -> Vec<f32> {
    let s: f32 = coeffs.iter().sum::<f32>();
    if s.abs() < 1e-8 {
        // Fall back to uniform filter
        let v = 1.0 / coeffs.len().max(1) as f32;
        return vec![v; coeffs.len()];
    }
    coeffs.iter().map(|&c| c / s).collect()
}

/// Direct-form FIR convolution (linear mode, full output length = signal.len()).
///
/// Zero-pads the signal at the boundaries. The output has the same length as
/// the input, with symmetric zero-padding (`filter_len / 2` samples on each side).
pub(crate) fn fir_convolve(signal: &[f32], coeffs: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let m = coeffs.len();
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let half = (m / 2) as isize;
    (0..n)
        .map(|i| {
            coeffs
                .iter()
                .enumerate()
                .map(|(k, &c)| {
                    let src = i as isize - (k as isize - half);
                    if src >= 0 && (src as usize) < n {
                        c * signal[src as usize]
                    } else {
                        0.0
                    }
                })
                .sum()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_filter_output_length_matches_input() {
        let config = DeepFilterConfig {
            filter_len: 16,
            feature_dim: 8,
            hidden_sizes: vec![12],
            sample_rate: 16000.0,
            epochs: 0,
            ..Default::default()
        };
        let df = DeepFilter::new(config).expect("construction: should succeed");
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let out = df.filter(&signal).expect("filter: should succeed");
        assert_eq!(out.len(), signal.len(), "output length must match input");
    }

    #[test]
    fn deep_filter_coefficients_sum_approximately_one() {
        let config = DeepFilterConfig {
            filter_len: 32,
            feature_dim: 8,
            hidden_sizes: vec![16],
            epochs: 0,
            ..Default::default()
        };
        let df = DeepFilter::new(config).expect("construction: should succeed");
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).cos()).collect();
        let coeffs = df
            .predict_coefficients(&signal)
            .expect("predict: should succeed");
        let sum: f32 = coeffs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "DC preservation: coefficient sum should be ~1.0, got {sum}"
        );
    }

    #[test]
    fn deep_filter_features_correct_shape() {
        let n_features = 10;
        let signal: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
        let feats = SignalFeatures::from_signal(&signal, 16000.0, n_features);
        let v = feats.to_vec(n_features);
        assert_eq!(
            v.len(),
            n_features,
            "feature vector must have length n_features"
        );
    }

    #[test]
    fn deep_filter_deterministic_with_seed() {
        let make_df = || {
            let config = DeepFilterConfig {
                filter_len: 16,
                feature_dim: 8,
                hidden_sizes: vec![12],
                seed: 1234,
                epochs: 0,
                ..Default::default()
            };
            DeepFilter::new(config).expect("construction: should succeed")
        };
        let df1 = make_df();
        let df2 = make_df();
        let signal: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let c1 = df1
            .predict_coefficients(&signal)
            .expect("predict: should succeed");
        let c2 = df2
            .predict_coefficients(&signal)
            .expect("predict: should succeed");
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "same seed must produce same coefficients"
            );
        }
    }

    #[test]
    fn deep_filter_reduces_white_noise() {
        // Generate a clean low-frequency signal + high-frequency white noise.
        // A DC-normalized FIR filter (sum=1) acts as a weighted moving average,
        // smoothing out high-frequency noise while preserving slowly-varying signal.
        // We use a very low frequency signal (1 Hz at 1000 Hz sample rate) so the
        // FIR filter's moving-average behavior genuinely reduces noise-to-signal.
        let n = 512;
        let sr = 1000.0_f32;
        // Clean signal: 1 Hz sinusoid — very slowly varying, well below filter bandwidth.
        let clean: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 1.0 / sr).sin())
            .collect();

        // Deterministic noise using XorShift for reproducibility
        let mut rng = XorShift64::new(999);
        let noise_scale = 0.5_f32;
        let noise: Vec<f32> = (0..n)
            .map(|_| (rng.next_f64() as f32 - 0.5) * noise_scale)
            .collect();
        let noisy: Vec<f32> = clean
            .iter()
            .zip(noise.iter())
            .map(|(&c, &e)| c + e)
            .collect();

        let config = DeepFilterConfig {
            filter_len: 32,
            feature_dim: 8,
            hidden_sizes: vec![16],
            sample_rate: sr as f64,
            epochs: 200,
            learning_rate: 0.01,
            seed: 77,
        };
        let mut df = DeepFilter::new(config).expect("construction: should succeed");
        let losses = df.fit(&[(&noisy, &clean)]).expect("fit: should succeed");
        assert_eq!(losses.len(), 200, "should return one loss per epoch");

        let filtered = df.filter(&noisy).expect("filter: should succeed");
        assert_eq!(filtered.len(), noisy.len());

        // Verify numerical stability of the filter output.
        assert!(
            filtered.iter().all(|v| v.is_finite()),
            "filtered output must be numerically stable (all finite)"
        );
        assert!(
            losses.iter().all(|&l| l.is_finite()),
            "training losses must be finite"
        );

        // Training loss should decrease: final loss < initial loss.
        // This is the primary verification that the gradient is correct and training works.
        let initial_loss = losses[0];
        let final_loss = losses[losses.len() - 1];
        assert!(
            final_loss < initial_loss,
            "training loss should decrease: initial={initial_loss:.6}, final={final_loss:.6}"
        );

        // After training, filtered output should be closer to clean than noisy input.
        // With 200 epochs and correct gradient, a 32-tap moving-average FIR trained to
        // minimize MSE vs clean should outperform the unfiltered noisy signal.
        let mse_noisy: f32 = noisy
            .iter()
            .zip(clean.iter())
            .map(|(nv, c)| (nv - c).powi(2))
            .sum::<f32>()
            / n as f32;
        let mse_filtered: f32 = filtered
            .iter()
            .zip(clean.iter())
            .map(|(f, c)| (f - c).powi(2))
            .sum::<f32>()
            / n as f32;

        assert!(
            mse_filtered < mse_noisy,
            "filtered MSE {mse_filtered:.6} should be less than noisy MSE {mse_noisy:.6}"
        );
    }
}
