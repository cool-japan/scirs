//! Machine learning-based feature detection
//!
//! This module provides ML-powered feature detection algorithms including
//! learned edge detectors, keypoint detectors, and semantic feature extraction.

use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{convolve, gaussian_filter};
use crate::interpolation::{zoom, InterpolationOrder};
use statrs::statistics::Statistics;

/// Pre-trained model weights for feature detection
#[derive(Clone, Debug)]
pub struct FeatureDetectorWeights {
    /// Convolutional kernels for each layer
    pub conv_kernels: Vec<Array4<f64>>,
    /// Bias terms for each layer
    pub biases: Vec<Array1<f64>>,
    /// Batch normalization parameters
    pub bn_params: Option<Vec<BatchNormParams>>,
}

/// Batch normalization parameters
#[derive(Clone, Debug)]
pub struct BatchNormParams {
    pub mean: Array1<f64>,
    pub variance: Array1<f64>,
    pub gamma: Array1<f64>,
    pub beta: Array1<f64>,
}

/// Configuration for ML-based feature detection
#[derive(Clone, Debug)]
pub struct MLDetectorConfig {
    /// Number of pyramid levels for multi-scale detection
    pub pyramid_levels: usize,
    /// Scale factor between pyramid levels
    pub scale_factor: f64,
    /// Non-maximum suppression threshold
    pub nms_threshold: f64,
    /// Minimum confidence score for detections
    pub confidence_threshold: f64,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
}

impl Default for MLDetectorConfig {
    fn default() -> Self {
        Self {
            pyramid_levels: 3,
            scale_factor: 1.5,
            nms_threshold: 0.3,
            confidence_threshold: 0.5,
            use_gpu: false,
        }
    }
}

/// Learned edge detector using convolutional filters
pub struct LearnedEdgeDetector {
    weights: FeatureDetectorWeights,
    config: MLDetectorConfig,
}

impl LearnedEdgeDetector {
    /// Create a new learned edge detector with pre-trained weights
    pub fn new(weights: Option<FeatureDetectorWeights>, config: Option<MLDetectorConfig>) -> Self {
        let weights = weights.unwrap_or_else(|| Self::default_weights());
        let config = config.unwrap_or_default();

        Self { weights, config }
    }

    /// Get default pre-trained weights (simplified example)
    fn default_weights() -> FeatureDetectorWeights {
        // Create learned filters that combine multiple edge detection approaches
        let mut kernels = Vec::new();

        // Layer 1: Basic edge filters (3x3x1x8)
        let mut layer1 = Array4::zeros((3, 3, 1, 8));

        // Sobel-like filters
        layer1
            .slice_mut(scirs2_core::ndarray::s![.., .., 0, 0])
            .assign(&scirs2_core::ndarray::arr2(&[
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ]));

        layer1
            .slice_mut(scirs2_core::ndarray::s![.., .., 0, 1])
            .assign(&scirs2_core::ndarray::arr2(&[
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ]));

        // Diagonal edge filters
        layer1
            .slice_mut(scirs2_core::ndarray::s![.., .., 0, 2])
            .assign(&scirs2_core::ndarray::arr2(&[
                [-2.0, -1.0, 0.0],
                [-1.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
            ]));

        layer1
            .slice_mut(scirs2_core::ndarray::s![.., .., 0, 3])
            .assign(&scirs2_core::ndarray::arr2(&[
                [0.0, -1.0, -2.0],
                [1.0, 0.0, -1.0],
                [2.0, 1.0, 0.0],
            ]));

        // Laplacian-like filters
        layer1
            .slice_mut(scirs2_core::ndarray::s![.., .., 0, 4])
            .assign(&scirs2_core::ndarray::arr2(&[
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ]));

        // Corner-like filters
        layer1
            .slice_mut(scirs2_core::ndarray::s![.., .., 0, 5])
            .assign(&scirs2_core::ndarray::arr2(&[
                [1.0, -2.0, 1.0],
                [-2.0, 4.0, -2.0],
                [1.0, -2.0, 1.0],
            ]));

        // Texture filters
        layer1
            .slice_mut(scirs2_core::ndarray::s![.., .., 0, 6])
            .assign(&scirs2_core::ndarray::arr2(&[
                [1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0],
            ]));

        layer1
            .slice_mut(scirs2_core::ndarray::s![.., .., 0, 7])
            .assign(&scirs2_core::ndarray::arr2(&[
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ]));

        kernels.push(layer1);

        // Layer 2: Combination filters (3x3x8x4)
        let mut layer2 = Array4::zeros((3, 3, 8, 4));

        // Learned combinations of previous features
        for i in 0..4 {
            for j in 0..8 {
                let weight = if i == j / 2 { 1.0 } else { 0.1 };
                layer2
                    .slice_mut(scirs2_core::ndarray::s![1, 1, j, i])
                    .fill(weight);
            }
        }

        kernels.push(layer2);

        // Biases
        let biases = vec![Array1::zeros(8), Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1])];

        FeatureDetectorWeights {
            conv_kernels: kernels,
            biases,
            bn_params: None,
        }
    }

    /// Detect edges using learned filters
    pub fn detect_edges<T>(&self, image: &ArrayView2<T>) -> NdimageResult<Array2<f64>>
    where
        T: Float + FromPrimitive + Debug + Send + Sync + 'static,
    {
        let (height, width) = image.dim();

        // Convert to f64 and normalize
        let mut features = image.mapv(|x| x.to_f64().unwrap_or(0.0));
        let max_val = features.iter().cloned().fold(0.0, f64::max);
        if max_val > 0.0 {
            features /= max_val;
        }

        // Add channel dimension
        let mut features_3d = Array3::zeros((height, width, 1));
        features_3d
            .slice_mut(scirs2_core::ndarray::s![.., .., 0])
            .assign(&features);

        // Apply convolutional layers
        for (layer_idx, (kernels, bias)) in self
            .weights
            .conv_kernels
            .iter()
            .zip(self.weights.biases.iter())
            .enumerate()
        {
            let in_channels = features_3d.dim().2;
            let out_channels = kernels.dim().3;
            let mut output = Array3::zeros((height, width, out_channels));

            // Apply convolutions
            for out_ch in 0..out_channels {
                let mut channel_sum = Array2::zeros((height, width));

                for in_ch in 0..in_channels {
                    let kernel = kernels.slice(scirs2_core::ndarray::s![.., .., in_ch, out_ch]);
                    let input = features_3d.slice(scirs2_core::ndarray::s![.., .., in_ch]);

                    // Simple 2D convolution
                    let conv_result = self.convolve_2d(&input, &kernel)?;
                    channel_sum += &conv_result;
                }

                // Add bias and apply ReLU activation
                channel_sum += bias[out_ch];
                channel_sum.mapv_inplace(|x| x.max(0.0));

                output
                    .slice_mut(scirs2_core::ndarray::s![.., .., out_ch])
                    .assign(&channel_sum);
            }

            features_3d = output;
        }

        // Combine all output channels into edge strength
        let mut edge_map = Array2::zeros((height, width));
        for ch in 0..features_3d.dim().2 {
            let channel = features_3d.slice(scirs2_core::ndarray::s![.., .., ch]);
            edge_map += &(&channel * &channel);
        }
        edge_map.mapv_inplace(|x| x.sqrt());

        // Apply non-maximum suppression
        let suppressed = self.non_max_suppression(&edge_map.view())?;

        Ok(suppressed)
    }

    /// Simple 2D convolution
    fn convolve_2d(
        &self,
        input: &ArrayView2<f64>,
        kernel: &ArrayView2<f64>,
    ) -> NdimageResult<Array2<f64>> {
        let (h, w) = input.dim();
        let (kh, kw) = kernel.dim();
        let pad_h = kh / 2;
        let pad_w = kw / 2;

        let mut output = Array2::zeros((h, w));

        for i in pad_h..h - pad_h {
            for j in pad_w..w - pad_w {
                let mut sum = 0.0;

                for ki in 0..kh {
                    for kj in 0..kw {
                        let ii = i + ki - pad_h;
                        let jj = j + kj - pad_w;
                        sum += input[[ii, jj]] * kernel[[ki, kj]];
                    }
                }

                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Non-maximum suppression for edge thinning
    fn non_max_suppression(&self, edgemap: &ArrayView2<f64>) -> NdimageResult<Array2<f64>> {
        let (height, width) = edgemap.dim();
        let mut suppressed = Array2::zeros((height, width));

        // Compute gradients
        let gx = crate::filters::sobel(&edgemap.to_owned(), 1, None)?; // axis 1 for x-direction
        let gy = crate::filters::sobel(&edgemap.to_owned(), 0, None)?; // axis 0 for y-direction

        for i in 1..height - 1 {
            for j in 1..width - 1 {
                let mag = edgemap[[i, j]];

                if mag < self.config.confidence_threshold {
                    continue;
                }

                // Compute gradient direction
                let angle = gy[[i, j]].atan2(gx[[i, j]]);

                // Discretize to 8 directions
                let direction = ((angle + std::f64::consts::PI) * 4.0 / std::f64::consts::PI)
                    .round() as i32
                    % 8;

                // Check neighbors based on gradient direction
                let (di1, dj1, di2, dj2) = match direction {
                    0 | 4 => (0, -1, 0, 1),  // Horizontal
                    1 | 5 => (-1, -1, 1, 1), // Diagonal
                    2 | 6 => (-1, 0, 1, 0),  // Vertical
                    3 | 7 => (-1, 1, 1, -1), // Anti-diagonal
                    _ => (0, -1, 0, 1),
                };

                let neighbor1 = edgemap[[(i as i32 + di1) as usize, (j as i32 + dj1) as usize]];
                let neighbor2 = edgemap[[(i as i32 + di2) as usize, (j as i32 + dj2) as usize]];

                // Keep only if local maximum
                if mag >= neighbor1 && mag >= neighbor2 {
                    suppressed[[i, j]] = mag;
                }
            }
        }

        Ok(suppressed)
    }
}

/// Keypoint descriptor using learned features
pub struct LearnedKeypointDescriptor {
    patch_size: usize,
    descriptor_size: usize,
    weights: Array2<f64>,
}

impl LearnedKeypointDescriptor {
    /// Create a new learned keypoint descriptor
    pub fn new(patch_size: usize, descriptor_size: usize) -> Self {
        // Initialize with random projection matrix (simplified)
        let weights =
            Array2::from_shape_fn((descriptor_size, patch_size * patch_size), |(i, j)| {
                // Simple deterministic "random" weights
                ((i * 7 + j * 13) % 11) as f64 / 11.0 - 0.5
            });

        Self {
            patch_size,
            descriptor_size,
            weights,
        }
    }

    /// Extract descriptors for given keypoints
    pub fn extract_descriptors<T>(
        &self,
        image: &ArrayView2<T>,
        keypoints: &[(f64, f64)],
    ) -> NdimageResult<Vec<Array1<f64>>>
    where
        T: Float + FromPrimitive + Debug,
    {
        let mut descriptors = Vec::new();
        let half_patch = self.patch_size / 2;

        for &(x, y) in keypoints {
            let xi = x.round() as i32;
            let yi = y.round() as i32;

            // Extract patch
            let mut patch = Array1::zeros(self.patch_size * self.patch_size);
            let mut idx = 0;

            for dy in -(half_patch as i32)..=(half_patch as i32) {
                for dx in -(half_patch as i32)..=(half_patch as i32) {
                    let px = xi + dx;
                    let py = yi + dy;

                    if px >= 0 && px < image.dim().1 as i32 && py >= 0 && py < image.dim().0 as i32
                    {
                        patch[idx] = image[[py as usize, px as usize]].to_f64().unwrap_or(0.0);
                    }
                    idx += 1;
                }
            }

            // Normalize patch
            let mean = patch.clone().mean();
            let std = patch.std(0.0);
            if std > 0.0 {
                patch = (patch - mean) / std;
            }

            // Apply learned projection
            let descriptor = self.weights.dot(&patch);

            // L2 normalize descriptor
            let norm = descriptor.dot(&descriptor).sqrt();
            let descriptor = if norm > 0.0 {
                descriptor / norm
            } else {
                descriptor
            };

            descriptors.push(descriptor);
        }

        Ok(descriptors)
    }
}

/// Semantic feature extractor using pre-trained deep features
pub struct SemanticFeatureExtractor {
    feature_maps: HashMap<String, Array4<f64>>,
    config: MLDetectorConfig,
}

impl SemanticFeatureExtractor {
    /// Create a new semantic feature extractor
    pub fn new(config: Option<MLDetectorConfig>) -> Self {
        Self {
            feature_maps: HashMap::new(),
            config: config.unwrap_or_default(),
        }
    }

    /// Extract semantic features at multiple scales
    pub fn extractfeatures<T>(
        &mut self,
        image: &ArrayView2<T>,
        feature_types: &[&str],
    ) -> NdimageResult<HashMap<String, Array3<f64>>>
    where
        T: Float + FromPrimitive + Debug + Send + Sync + 'static,
    {
        let mut results = HashMap::new();

        for &feature_type in feature_types {
            match feature_type {
                "texture" => {
                    let texturefeatures = self.extracttexturefeatures(image)?;
                    results.insert("texture".to_string(), texturefeatures);
                }
                "shape" => {
                    let shapefeatures = self.extractshapefeatures(image)?;
                    results.insert("shape".to_string(), shapefeatures);
                }
                "color" => {
                    let colorfeatures = self.extract_colorfeatures(image)?;
                    results.insert("color".to_string(), colorfeatures);
                }
                _ => {
                    return Err(NdimageError::InvalidInput(format!(
                        "Unknown feature type: {}",
                        feature_type
                    )));
                }
            }
        }

        Ok(results)
    }

    /// Extract texture features using Gabor-like filters
    fn extracttexturefeatures<T>(&self, image: &ArrayView2<T>) -> NdimageResult<Array3<f64>>
    where
        T: Float + FromPrimitive + Debug + Send + Sync + 'static,
    {
        let (height, width) = image.dim();
        let num_orientations = 4;
        let num_scales = 3;
        let num_features = num_orientations * num_scales;

        let mut features = Array3::zeros((height, width, num_features));
        let img_f64 = image.mapv(|x| x.to_f64().unwrap_or(0.0));

        let mut feature_idx = 0;
        for scale in 0..num_scales {
            let sigma = 2.0 * (scale + 1) as f64;

            for orientation in 0..num_orientations {
                let angle = orientation as f64 * std::f64::consts::PI / num_orientations as f64;

                // Create oriented filter (simplified Gabor-like)
                let filter_size = (sigma * 3.0) as usize | 1; // Ensure odd
                let mut filter = Array2::zeros((filter_size, filter_size));
                let center = filter_size / 2;

                for i in 0..filter_size {
                    for j in 0..filter_size {
                        let x = (j as f64 - center as f64) * angle.cos()
                            + (i as f64 - center as f64) * angle.sin();
                        let y = -(j as f64 - center as f64) * angle.sin()
                            + (i as f64 - center as f64) * angle.cos();

                        let gaussian = (-0.5 * (x * x + y * y) / (sigma * sigma)).exp();
                        let sinusoid = (2.0 * std::f64::consts::PI * x / (sigma * 2.0)).cos();

                        filter[[i, j]] = gaussian * sinusoid;
                    }
                }

                // Normalize filter
                let sum: f64 = filter.iter().map(|x| x.abs()).sum();
                if sum > 0.0 {
                    filter /= sum;
                }

                // Apply filter
                let response = convolve(&img_f64, &filter, None)?;
                features
                    .slice_mut(scirs2_core::ndarray::s![.., .., feature_idx])
                    .assign(&response);

                feature_idx += 1;
            }
        }

        Ok(features)
    }

    /// Extract shape features using morphological operations
    fn extractshapefeatures<T>(&self, image: &ArrayView2<T>) -> NdimageResult<Array3<f64>>
    where
        T: Float + FromPrimitive + Debug + Send + Sync + 'static,
    {
        let (height, width) = image.dim();
        let mut features = Array3::zeros((height, width, 4));

        // Convert to binary for shape analysis
        let img_f64 = image.mapv(|x| x.to_f64().unwrap_or(0.0));
        let threshold = img_f64.clone().mean();
        let binary = img_f64.mapv(|x| if x > threshold { 1.0 } else { 0.0 });

        // Feature 1: Distance to nearest edge
        let edges_x = crate::filters::sobel(&binary, 1, None)?; // x-direction gradient
        let edges_y = crate::filters::sobel(&binary, 0, None)?; // y-direction gradient
        let edge_magnitude = (edges_x.mapv(|x| x * x) + edges_y.mapv(|x| x * x)).mapv(|x| x.sqrt());
        features
            .slice_mut(scirs2_core::ndarray::s![.., .., 0])
            .assign(&edge_magnitude);

        // Feature 2: Local curvature (using Laplacian)
        let curvature = crate::filters::laplace(&binary, None, None)?;
        features
            .slice_mut(scirs2_core::ndarray::s![.., .., 1])
            .assign(&curvature.mapv(|x| x.abs()));

        // Feature 3: Local thickness (simplified)
        let smoothed = gaussian_filter(&binary, 3.0, None, None)?;
        features
            .slice_mut(scirs2_core::ndarray::s![.., .., 2])
            .assign(&smoothed);

        // Feature 4: Orientation strength
        let (gx, gy) = (&edges_x, &edges_y);
        let orientation_strength = gx.mapv(|x| x.abs()) + gy.mapv(|x| x.abs());
        features
            .slice_mut(scirs2_core::ndarray::s![.., .., 3])
            .assign(&orientation_strength);

        Ok(features)
    }

    /// Extract color-based features (for grayscale, extract intensity features)
    fn extract_colorfeatures<T>(&self, image: &ArrayView2<T>) -> NdimageResult<Array3<f64>>
    where
        T: Float + FromPrimitive + Debug + Send + Sync + 'static,
    {
        let (height, width) = image.dim();
        let mut features = Array3::zeros((height, width, 3));

        let img_f64 = image.mapv(|x| x.to_f64().unwrap_or(0.0));

        // Feature 1: Normalized intensity
        let max_val = img_f64.iter().cloned().fold(0.0, f64::max);
        let normalized = if max_val > 0.0 {
            &img_f64 / max_val
        } else {
            img_f64.clone()
        };
        features
            .slice_mut(scirs2_core::ndarray::s![.., .., 0])
            .assign(&normalized);

        // Feature 2: Local contrast
        let window_size = 5;
        let mut contrast = Array2::zeros((height, width));

        for i in window_size / 2..height - window_size / 2 {
            for j in window_size / 2..width - window_size / 2 {
                let window = img_f64.slice(scirs2_core::ndarray::s![
                    i - window_size / 2..=i + window_size / 2,
                    j - window_size / 2..=j + window_size / 2
                ]);

                let local_mean = window.mean();
                let local_std = window.std(0.0);
                let epsilon = T::from_f64(1e-6).unwrap_or_else(|| T::zero());
                contrast[[i, j]] = local_std / (local_mean + epsilon.to_f64().unwrap_or(1e-6));
            }
        }
        features
            .slice_mut(scirs2_core::ndarray::s![.., .., 1])
            .assign(&contrast);

        // Feature 3: Local entropy (simplified)
        let mut entropy = Array2::zeros((height, width));

        for i in window_size / 2..height - window_size / 2 {
            for j in window_size / 2..width - window_size / 2 {
                let window = img_f64.slice(scirs2_core::ndarray::s![
                    i - window_size / 2..=i + window_size / 2,
                    j - window_size / 2..=j + window_size / 2
                ]);

                // Simple entropy approximation
                let variance = window.variance();
                entropy[[i, j]] = (1.0 + variance).ln();
            }
        }
        features
            .slice_mut(scirs2_core::ndarray::s![.., .., 2])
            .assign(&entropy);

        Ok(features)
    }
}

/// Object proposal generator using learned objectness scores
pub struct ObjectProposalGenerator {
    min_size: usize,
    max_size: usize,
    stride: usize,
    aspect_ratios: Vec<f64>,
    config: MLDetectorConfig,
}

impl ObjectProposalGenerator {
    /// Create a new object proposal generator
    pub fn new(config: Option<MLDetectorConfig>) -> Self {
        Self {
            min_size: 16,
            max_size: 256,
            stride: 8,
            aspect_ratios: vec![0.5, 1.0, 2.0],
            config: config.unwrap_or_default(),
        }
    }

    /// Generate object proposals with objectness scores
    pub fn generate_proposals<T>(
        &self,
        image: &ArrayView2<T>,
        edge_map: Option<&ArrayView2<f64>>,
    ) -> NdimageResult<Vec<ObjectProposal>>
    where
        T: Float + FromPrimitive + Debug + Send + Sync + 'static,
    {
        let (height, width) = image.dim();
        let mut proposals = Vec::new();

        // Compute edge _map if not provided
        let edge_detector = LearnedEdgeDetector::new(None, None);
        let edges = if let Some(e) = edge_map {
            e.to_owned()
        } else {
            edge_detector.detect_edges(image)?
        };

        // Generate proposals at multiple scales
        for scale in 0..self.config.pyramid_levels {
            let scale_factor = self.config.scale_factor.powi(scale as i32);

            // Resize edge _map for current scale
            let scaled_height = ((height as f64) / scale_factor) as usize;
            let scaled_width = ((width as f64) / scale_factor) as usize;

            if scaled_height < self.min_size || scaled_width < self.min_size {
                continue;
            }

            let scaled_edges = zoom(
                &edges,
                1.0 / scale_factor,
                Some(InterpolationOrder::Linear),
                None,
                None,
                None,
            )?;

            // Sliding window with multiple sizes and aspect ratios
            for box_size in (self.min_size..=self.max_size).step_by(self.stride * 2) {
                for &aspect_ratio in &self.aspect_ratios {
                    let box_width = (box_size as f64 * aspect_ratio.sqrt()) as usize;
                    let box_height = (box_size as f64 / aspect_ratio.sqrt()) as usize;

                    if box_width > scaled_width || box_height > scaled_height {
                        continue;
                    }

                    for y in (0..=scaled_height - box_height).step_by(self.stride) {
                        for x in (0..=scaled_width - box_width).step_by(self.stride) {
                            // Compute objectness score
                            let roi = scaled_edges.slice(scirs2_core::ndarray::s![
                                y..y + box_height,
                                x..x + box_width
                            ]);

                            let objectness = self.compute_objectness(&roi);

                            if objectness > self.config.confidence_threshold {
                                // Convert back to original coordinates
                                let proposal = ObjectProposal {
                                    x: (x as f64 * scale_factor) as usize,
                                    y: (y as f64 * scale_factor) as usize,
                                    width: (box_width as f64 * scale_factor) as usize,
                                    height: (box_height as f64 * scale_factor) as usize,
                                    score: objectness,
                                    scale,
                                };

                                proposals.push(proposal);
                            }
                        }
                    }
                }
            }
        }

        // Apply non-maximum suppression
        let filtered_proposals = self.non_max_suppression_boxes(&mut proposals);

        Ok(filtered_proposals)
    }

    /// Compute objectness score for a region
    fn compute_objectness(&self, roi: &ArrayView2<f64>) -> f64 {
        // Simple objectness based on edge density and distribution
        let edge_sum: f64 = roi.sum();
        let num_pixels = (roi.dim().0 * roi.dim().1) as f64;
        let edge_density = edge_sum / num_pixels;

        // Check edge distribution (prefer edges near boundaries)
        let (h, w) = roi.dim();
        let border_width = 3;

        let mut border_sum = 0.0;
        let mut border_pixels = 0;

        for i in 0..h {
            for j in 0..w {
                if i < border_width
                    || i >= h - border_width
                    || j < border_width
                    || j >= w - border_width
                {
                    border_sum += roi[[i, j]];
                    border_pixels += 1;
                }
            }
        }

        let border_density = if border_pixels > 0 {
            border_sum / border_pixels as f64
        } else {
            0.0
        };

        // Combine scores
        let objectness = edge_density * 0.3 + border_density * 0.7;

        objectness.min(1.0)
    }

    /// Non-maximum suppression for object proposals
    fn non_max_suppression_boxes(
        &self,
        proposals: &mut Vec<ObjectProposal>,
    ) -> Vec<ObjectProposal> {
        // Sort by score in descending order
        proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let mut keep = Vec::new();
        let mut suppressed = vec![false; proposals.len()];

        for i in 0..proposals.len() {
            if suppressed[i] {
                continue;
            }

            keep.push(proposals[i].clone());

            // Suppress overlapping proposals
            for j in i + 1..proposals.len() {
                if suppressed[j] {
                    continue;
                }

                let iou = self.compute_iou(&proposals[i], &proposals[j]);
                if iou > self.config.nms_threshold {
                    suppressed[j] = true;
                }
            }
        }

        keep
    }

    /// Compute intersection over union for two boxes
    fn compute_iou(&self, box1: &ObjectProposal, box2: &ObjectProposal) -> f64 {
        let x1 = box1.x.max(box2.x);
        let y1 = box1.y.max(box2.y);
        let x2 = (box1.x + box1.width).min(box2.x + box2.width);
        let y2 = (box1.y + box1.height).min(box2.y + box2.height);

        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let area1 = box1.width * box1.height;
        let area2 = box2.width * box2.height;
        let union = area1 + area2 - intersection;

        intersection as f64 / union as f64
    }
}

/// Object proposal with location and score
#[derive(Clone, Debug)]
pub struct ObjectProposal {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub score: f64,
    pub scale: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_learned_edge_detector() {
        // Create a simple test image
        let image = arr2(&[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]);

        let detector = LearnedEdgeDetector::new(None, None);
        let edges = detector.detect_edges(&image.view()).unwrap();

        assert_eq!(edges.dim(), image.dim());
        // Should detect edges at boundaries
        assert!(edges[[1, 2]] > 0.0 || edges[[2, 1]] > 0.0);
    }

    #[test]
    fn test_keypoint_descriptor() {
        let image = arr2(&[
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.7, 0.8],
            [0.5, 0.6, 0.7, 0.8, 0.9],
        ]);

        let descriptor = LearnedKeypointDescriptor::new(3, 16);
        let keypoints = vec![(2.0, 2.0)];

        let descriptors = descriptor
            .extract_descriptors(&image.view(), &keypoints)
            .unwrap();

        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].len(), 16);

        // Check normalization
        let norm = descriptors[0].dot(&descriptors[0]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_semantic_feature_extractor() {
        let image = arr2(&[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]);

        let mut extractor = SemanticFeatureExtractor::new(None);
        let features = extractor
            .extractfeatures(&image.view(), &["texture", "shape", "color"])
            .unwrap();

        assert_eq!(features.len(), 3);
        assert!(features.contains_key("texture"));
        assert!(features.contains_key("shape"));
        assert!(features.contains_key("color"));

        let texturefeatures = &features["texture"];
        assert_eq!(texturefeatures.dim().0, 4);
        assert_eq!(texturefeatures.dim().1, 4);
        assert!(texturefeatures.dim().2 > 0);
    }

    #[test]
    fn test_object_proposal_generator() {
        let mut image = Array2::zeros((50, 50));

        // Create a simple rectangle
        for i in 10..30 {
            for j in 15..35 {
                image[[i, j]] = 1.0;
            }
        }

        let generator = ObjectProposalGenerator::new(None);
        let proposals = generator.generate_proposals(&image.view(), None).unwrap();

        assert!(!proposals.is_empty());

        // Check that proposals have valid dimensions
        for proposal in &proposals {
            assert!(proposal.x + proposal.width <= 50);
            assert!(proposal.y + proposal.height <= 50);
            assert!(proposal.score >= 0.0 && proposal.score <= 1.0);
        }
    }
}
