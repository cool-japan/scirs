//! Domain-specific imaging functions
//!
//! This module provides specialized image processing functions for different domains:
//! medical imaging, satellite/remote sensing, and microscopy.

use scirs2_core::ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};
use crate::utils::{safe_f64_to_float, safe_float_to_f64, safe_usize_to_float};

use crate::filters::{gaussian_filter, median_filter};
use crate::interpolation::{zoom, InterpolationOrder};
use crate::measurements::{center_of_mass, central_moments, moments};
use crate::morphology::label;
use crate::morphology::{binary_closing, binary_opening, grey_opening};

/// Medical imaging functions
pub mod medical {
    use super::*;

    /// Parameters for vessel enhancement
    #[derive(Clone, Debug)]
    pub struct VesselEnhancementParams {
        /// Scales at which to compute vesselness
        pub scales: Vec<f64>,
        /// Frangi filter parameters
        pub alpha: f64, // Plate-like structures suppression
        pub beta: f64,  // Blob-like structures suppression
        pub gamma: f64, // Background suppression
    }

    impl Default for VesselEnhancementParams {
        fn default() -> Self {
            Self {
                scales: vec![1.0, 2.0, 3.0, 4.0],
                alpha: 0.5,
                beta: 0.5,
                gamma: 15.0,
            }
        }
    }

    /// Enhance blood vessels using Frangi filter
    pub fn frangi_vesselness<T>(
        image: &ArrayView2<T>,
        params: Option<VesselEnhancementParams>,
    ) -> NdimageResult<Array2<f64>>
    where
        T: Float + FromPrimitive + Debug + Send + Sync + 'static,
    {
        let params = params.unwrap_or_default();
        let (height, width) = image.dim();
        let mut vesselness = Array2::<f64>::zeros((height, width));

        // Convert to f64
        let img = image.mapv(|x| x.to_f64().unwrap_or(0.0));

        // Compute vesselness at each scale
        for &scale in &params.scales {
            // Compute Hessian matrix components
            let smoothed = gaussian_filter(&img, scale, None, None)?;
            let hessian = compute_hessian_2d(&smoothed.view(), scale)?;

            // Compute eigenvalues at each pixel
            for i in 0..height {
                for j in 0..width {
                    let hxx = hessian.0[[i, j]];
                    let hxy = hessian.1[[i, j]];
                    let hyy = hessian.2[[i, j]];

                    // Eigenvalues of 2x2 symmetric matrix
                    let trace = hxx + hyy;
                    let det = hxx * hyy - hxy * hxy;
                    let discriminant = trace * trace - 4.0 * det;

                    if discriminant >= 0.0 {
                        let sqrt_disc = discriminant.sqrt();
                        let lambda1 = (trace + sqrt_disc) / 2.0;
                        let lambda2 = (trace - sqrt_disc) / 2.0;

                        // Order eigenvalues by magnitude
                        let (l1, l2) = if lambda1.abs() > lambda2.abs() {
                            (lambda1, lambda2)
                        } else {
                            (lambda2, lambda1)
                        };

                        // Frangi vesselness measure
                        if l2 < 0.0 {
                            // Dark vessels on bright background
                            let rb = l1.abs() / l2.abs().max(1e-10);
                            let s = (l1 * l1 + l2 * l2).sqrt();

                            let v = (1.0 - (-rb * rb / (2.0 * params.beta * params.beta)).exp())
                                * (-s * s / (2.0 * params.gamma * params.gamma)).exp();

                            vesselness[[i, j]] = vesselness[[i, j]].max(v);
                        }
                    }
                }
            }
        }

        Ok(vesselness)
    }

    /// Compute Hessian matrix components
    fn compute_hessian_2d(
        image: &ArrayView2<f64>,
        scale: f64,
    ) -> NdimageResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let (height, width) = image.dim();
        let mut hxx = Array2::zeros((height, width));
        let mut hxy = Array2::zeros((height, width));
        let mut hyy = Array2::zeros((height, width));

        // Scale-normalized second derivatives
        let norm = scale * scale;

        for i in 2..height - 2 {
            for j in 2..width - 2 {
                // Second derivatives using central differences
                hxx[[i, j]] = (image[[i, j + 1]] - 2.0 * image[[i, j]] + image[[i, j - 1]]) * norm;
                hyy[[i, j]] = (image[[i + 1, j]] - 2.0 * image[[i, j]] + image[[i - 1, j]]) * norm;
                hxy[[i, j]] =
                    (image[[i + 1, j + 1]] - image[[i + 1, j - 1]] - image[[i - 1, j + 1]]
                        + image[[i - 1, j - 1]])
                        * norm
                        / 4.0;
            }
        }

        Ok((hxx, hxy, hyy))
    }

    /// Bone structure enhancement using morphological operations
    pub fn enhance_bone_structure<T>(
        image: &ArrayView2<T>,
        kernel_size: usize,
    ) -> NdimageResult<Array2<T>>
    where
        T: Float
            + FromPrimitive
            + Debug
            + Send
            + Sync
            + std::ops::AddAssign
            + std::ops::DivAssign
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        // Top-hat transform to enhance bright structures
        let structure = crate::morphology::disk_structure(kernel_size as f64, None)?;
        let structure_2d = structure.into_dimensionality::<scirs2_core::ndarray::Ix2>()?;
        let opened = grey_opening(
            &image.to_owned(),
            None,
            Some(&structure_2d),
            None,
            None,
            None,
        )?;
        let top_hat = image.to_owned() - opened;

        // Enhance contrast
        let two = safe_f64_to_float::<T>(2.0)?;
        let enhanced = image.to_owned() + top_hat * two;

        Ok(enhanced)
    }

    /// Lung nodule detection (simplified)
    pub fn detect_lung_nodules<T>(
        ct_slice: &ArrayView2<T>,
        min_size: usize,
        max_size: usize,
    ) -> NdimageResult<Vec<Nodule>>
    where
        T: Float
            + FromPrimitive
            + Debug
            + Send
            + Sync
            + scirs2_core::numeric::NumAssign
            + std::ops::DivAssign
            + 'static,
    {
        let mut nodules = Vec::new();

        // Threshold to segment lung tissue
        let threshold = safe_f64_to_float::<T>(-500.0)?; // Typical HU value for lung tissue
        let lung_mask = ct_slice.mapv(|x| x > threshold);

        // Apply morphological operations to clean up
        let cleaned = binary_closing(&lung_mask, None, Some(3), None, None, None, None)?;
        let cleaned = binary_opening(&cleaned, None, Some(2), None, None, None, None)?;

        // Find connected components
        let (labels, num_features) = label(&cleaned, None, None, None)?;

        // Analyze each component
        for i in 1..=num_features {
            let component_mask = labels.mapv(|x| x == i);
            let size = component_mask.iter().filter(|&&x| x).count();

            if size >= min_size && size <= max_size {
                // Compute properties
                let com = center_of_mass(&ct_slice.to_owned())?;

                // Simple circularity measure
                let coords: Vec<(usize, usize)> = component_mask
                    .indexed_iter()
                    .filter(|(_, &val)| val)
                    .map(|((y, x), _)| (y, x))
                    .collect();

                let cy = com[0].to_f64().unwrap_or(0.0);
                let cx = com[1].to_f64().unwrap_or(0.0);
                let mean_radius = coords
                    .iter()
                    .map(|&(y, x)| {
                        let dy = y as f64 - cy;
                        let dx = x as f64 - cx;
                        (dy * dy + dx * dx).sqrt()
                    })
                    .sum::<f64>()
                    / coords.len() as f64;

                let radius_variance = coords
                    .iter()
                    .map(|&(y, x)| {
                        let dy = y as f64 - cy;
                        let dx = x as f64 - cx;
                        let r = (dy * dy + dx * dx).sqrt();
                        (r - mean_radius).powi(2)
                    })
                    .sum::<f64>()
                    / coords.len() as f64;

                let circularity = 1.0 / (1.0 + radius_variance / mean_radius.powi(2));

                nodules.push(Nodule {
                    center: (cy, cx),
                    size,
                    circularity,
                    mean_intensity: ct_slice
                        .indexed_iter()
                        .filter(|((y, x), _)| component_mask[[*y, *x]])
                        .map(|(_, &val)| safe_float_to_f64(val).unwrap_or(0.0))
                        .sum::<f64>()
                        / size as f64,
                });
            }
        }

        Ok(nodules)
    }

    /// Detected nodule information
    #[derive(Clone, Debug)]
    pub struct Nodule {
        pub center: (f64, f64),
        pub size: usize,
        pub circularity: f64,
        pub mean_intensity: f64,
    }
}

/// Satellite and remote sensing imaging functions
pub mod satellite {
    use super::*;

    /// Compute Normalized Difference Vegetation Index (NDVI)
    pub fn compute_ndvi<T>(
        red_band: &ArrayView2<T>,
        nir_band: &ArrayView2<T>,
    ) -> NdimageResult<Array2<f64>>
    where
        T: Float + FromPrimitive,
    {
        if red_band.dim() != nir_band.dim() {
            return Err(NdimageError::DimensionError(
                "Red and NIR bands must have same dimensions".into(),
            ));
        }

        let (height, width) = red_band.dim();
        let mut ndvi = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let red = red_band[[i, j]].to_f64().unwrap_or(0.0);
                let nir = nir_band[[i, j]].to_f64().unwrap_or(0.0);

                let denominator = nir + red;
                if denominator.abs() > 1e-10 {
                    ndvi[[i, j]] = (nir - red) / denominator;
                } else {
                    ndvi[[i, j]] = 0.0;
                }
            }
        }

        Ok(ndvi)
    }

    /// Detect water bodies using spectral indices
    pub fn detect_water_bodies<T>(
        green_band: &ArrayView2<T>,
        nir_band: &ArrayView2<T>,
        threshold: Option<f64>,
    ) -> NdimageResult<Array2<bool>>
    where
        T: Float + FromPrimitive,
    {
        // Compute Normalized Difference Water Index (NDWI)
        let ndwi = compute_ndwi(green_band, nir_band)?;

        // Apply threshold
        let threshold = threshold.unwrap_or(0.3);
        let water_mask = ndwi.mapv(|x| x > threshold);

        // Clean up small patches
        let cleaned = binary_opening(&water_mask, None, Some(2), None, None, None, None)?;
        let cleaned = binary_closing(&cleaned, None, Some(3), None, None, None, None)?;

        Ok(cleaned)
    }

    /// Compute Normalized Difference Water Index (NDWI)
    fn compute_ndwi<T>(
        green_band: &ArrayView2<T>,
        nir_band: &ArrayView2<T>,
    ) -> NdimageResult<Array2<f64>>
    where
        T: Float + FromPrimitive,
    {
        if green_band.dim() != nir_band.dim() {
            return Err(NdimageError::DimensionError(
                "Green and NIR bands must have same dimensions".into(),
            ));
        }

        let (height, width) = green_band.dim();
        let mut ndwi = Array2::zeros((height, width));

        for i in 0..height {
            for j in 0..width {
                let green = green_band[[i, j]].to_f64().unwrap_or(0.0);
                let nir = nir_band[[i, j]].to_f64().unwrap_or(0.0);

                let denominator = green + nir;
                if denominator.abs() > 1e-10 {
                    ndwi[[i, j]] = (green - nir) / denominator;
                } else {
                    ndwi[[i, j]] = 0.0;
                }
            }
        }

        Ok(ndwi)
    }

    /// Cloud detection in satellite imagery
    pub fn detect_clouds<T>(
        image: &ArrayView3<T>, // Multi-spectral image
        brightness_threshold: f64,
        temperature_threshold: Option<f64>,
    ) -> NdimageResult<Array2<bool>>
    where
        T: Float + FromPrimitive,
    {
        if image.dim().2 < 3 {
            return Err(NdimageError::InvalidInput(
                "Image must have at least 3 spectral bands".into(),
            ));
        }

        let (height, width, _) = image.dim();
        let mut cloud_mask = Array2::default((height, width));

        // Simple brightness test (clouds are bright in visible bands)
        for i in 0..height {
            for j in 0..width {
                let brightness = (0..3)
                    .map(|k| image[[i, j, k]].to_f64().unwrap_or(0.0))
                    .sum::<f64>()
                    / 3.0;

                if brightness > brightness_threshold {
                    cloud_mask[[i, j]] = true;
                }
            }
        }

        // Thermal test if thermal band is available
        if let Some(temp_thresh) = temperature_threshold {
            if image.dim().2 > 3 {
                // Assume 4th band is thermal
                for i in 0..height {
                    for j in 0..width {
                        let temp = image[[i, j, 3]].to_f64().unwrap_or(0.0);
                        if cloud_mask[[i, j]] && temp > temp_thresh {
                            cloud_mask[[i, j]] = false; // Not a cloud if too warm
                        }
                    }
                }
            }
        }

        // Morphological cleaning
        let cleaned = binary_closing(&cloud_mask, None, Some(5), None, None, None, None)?;

        Ok(cleaned)
    }

    /// Pan-sharpening: merge high-resolution panchromatic with low-resolution multispectral
    pub fn pan_sharpen<T>(
        panimage: &ArrayView2<T>,
        multi_spectral: &ArrayView3<T>,
        method: PanSharpenMethod,
    ) -> NdimageResult<Array3<T>>
    where
        T: Float
            + FromPrimitive
            + Debug
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + std::ops::Mul<Output = T>
            + std::ops::AddAssign
            + std::ops::DivAssign
            + 'static,
    {
        let (pan_h, pan_w) = panimage.dim();
        let (ms_h, ms_w, num_bands) = multi_spectral.dim();

        // Compute scale factor
        let scale_y = pan_h as f64 / ms_h as f64;
        let scale_x = pan_w as f64 / ms_w as f64;

        match method {
            PanSharpenMethod::IHS => {
                // Intensity-Hue-Saturation method
                let mut sharpened = Array3::zeros((pan_h, pan_w, num_bands));

                // Upsample multispectral to pan resolution
                for band in 0..num_bands {
                    let ms_band = multi_spectral.slice(scirs2_core::ndarray::s![.., .., band]);
                    let upsampled = zoom(
                        &ms_band.to_owned(),
                        T::from_f64(scale_x).ok_or_else(|| {
                            NdimageError::InvalidInput("Failed to convert scale factor".into())
                        })?, // Use single scale factor
                        Some(InterpolationOrder::Cubic),
                        None,
                        None,
                        None,
                    )?;
                    sharpened
                        .slice_mut(scirs2_core::ndarray::s![.., .., band])
                        .assign(&upsampled);
                }

                // Compute intensity from multispectral
                let mut intensity = Array2::zeros((pan_h, pan_w));
                for i in 0..pan_h {
                    for j in 0..pan_w {
                        let sum: T = (0..num_bands)
                            .map(|k| sharpened[[i, j, k]])
                            .fold(T::zero(), |a, b| a + b);
                        intensity[[i, j]] = sum / safe_usize_to_float(num_bands)?;
                    }
                }

                // Replace intensity with pan
                for i in 0..pan_h {
                    for j in 0..pan_w {
                        let ratio = if intensity[[i, j]] > safe_f64_to_float::<T>(1e-10)? {
                            panimage[[i, j]] / intensity[[i, j]]
                        } else {
                            T::one()
                        };

                        for k in 0..num_bands {
                            sharpened[[i, j, k]] = sharpened[[i, j, k]] * ratio;
                        }
                    }
                }

                Ok(sharpened)
            }

            PanSharpenMethod::Brovey => {
                // Brovey transform
                let mut sharpened = Array3::zeros((pan_h, pan_w, num_bands));

                // Upsample and apply Brovey transform
                for band in 0..num_bands {
                    let ms_band = multi_spectral.slice(scirs2_core::ndarray::s![.., .., band]);
                    let upsampled = zoom(
                        &ms_band.to_owned(),
                        T::from_f64(scale_x).ok_or_else(|| {
                            NdimageError::InvalidInput("Failed to convert scale factor".into())
                        })?, // Use single scale factor
                        Some(InterpolationOrder::Cubic),
                        None,
                        None,
                        None,
                    )?;

                    // Compute sum of all bands at low resolution
                    let mut ms_sum = Array2::zeros((ms_h, ms_w));
                    for k in 0..num_bands {
                        ms_sum += &multi_spectral.slice(scirs2_core::ndarray::s![.., .., k]);
                    }

                    // Upsample sum
                    let sum_upsampled = zoom(
                        &ms_sum.to_owned(),
                        T::from_f64(scale_x).ok_or_else(|| {
                            NdimageError::InvalidInput("Failed to convert scale factor".into())
                        })?,
                        Some(InterpolationOrder::Cubic),
                        None,
                        None,
                        None,
                    )?;

                    // Apply Brovey transform
                    for i in 0..pan_h {
                        for j in 0..pan_w {
                            if sum_upsampled[[i, j]] > safe_f64_to_float::<T>(1e-10)? {
                                sharpened[[i, j, band]] =
                                    upsampled[[i, j]] * panimage[[i, j]] / sum_upsampled[[i, j]];
                            } else {
                                sharpened[[i, j, band]] = upsampled[[i, j]];
                            }
                        }
                    }
                }

                Ok(sharpened)
            }
        }
    }

    /// Pan-sharpening method
    #[derive(Clone, Debug)]
    pub enum PanSharpenMethod {
        IHS,    // Intensity-Hue-Saturation
        Brovey, // Brovey transform
    }
}

/// Microscopy imaging functions
pub mod microscopy {
    use super::*;

    /// Parameters for cell segmentation
    #[derive(Clone, Debug)]
    pub struct CellSegmentationParams {
        /// Minimum cell area in pixels
        pub min_area: usize,
        /// Maximum cell area in pixels
        pub max_area: usize,
        /// Threshold method
        pub threshold_method: ThresholdMethod,
        /// Morphological cleanup iterations
        pub cleanup_iterations: usize,
    }

    impl Default for CellSegmentationParams {
        fn default() -> Self {
            Self {
                min_area: 50,
                max_area: 5000,
                threshold_method: ThresholdMethod::Otsu,
                cleanup_iterations: 2,
            }
        }
    }

    #[derive(Clone, Debug)]
    pub enum ThresholdMethod {
        Otsu,
        Adaptive,
        Fixed(f64),
    }

    /// Segment cells in microscopy images
    pub fn segment_cells<T>(
        image: &ArrayView2<T>,
        params: Option<CellSegmentationParams>,
    ) -> NdimageResult<(Array2<i32>, Vec<CellInfo>)>
    where
        T: Float + FromPrimitive + Debug + Send + Sync + scirs2_core::numeric::NumAssign + 'static,
    {
        let params = params.unwrap_or_default();

        // Apply threshold
        let binary = match params.threshold_method {
            ThresholdMethod::Otsu => {
                let (_thresholded, threshold_val) =
                    crate::segmentation::otsu_threshold(&image.to_owned(), 256)?;
                image.mapv(|x| x > threshold_val)
            }
            ThresholdMethod::Adaptive => crate::segmentation::adaptive_threshold(
                &image.to_owned(),
                21,
                crate::segmentation::AdaptiveMethod::Gaussian,
                safe_f64_to_float::<T>(5.0)?,
            )?,
            ThresholdMethod::Fixed(thresh) => {
                let thresh_t = safe_f64_to_float::<T>(thresh)?;
                image.mapv(|x| x > thresh_t)
            }
        };

        // Morphological cleanup
        let mut cleaned = binary;
        for _ in 0..params.cleanup_iterations {
            cleaned = binary_opening(&cleaned, None, Some(3), None, None, None, None)?;
            cleaned = binary_closing(&cleaned, None, Some(3), None, None, None, None)?;
        }

        // Label connected components
        let (labels, num_cells) = label(&cleaned, None, None, None)?;

        // Analyze each cell
        let mut cell_info = Vec::new();
        let mut filtered_labels = Array2::zeros(labels.dim());
        let mut new_label = 1;

        for i in 1..=num_cells {
            let mask = labels.mapv(|x| x == i);
            let area = mask.iter().filter(|&&x| x).count();

            if area >= params.min_area && area <= params.max_area {
                // Compute cell properties
                let com = center_of_mass(&image.to_owned())?;
                let central_moments_result = central_moments(
                    &mask.mapv(|x| {
                        if x {
                            safe_f64_to_float::<T>(1.0).unwrap_or(T::one())
                        } else {
                            T::zero()
                        }
                    }),
                    2,
                    None,
                )?;

                // Compute eccentricity from central moments
                // For 2D with order=2: indices are M_00(0), M_01(1), M_02(2), M_10(3), M_11(4), M_12(5), M_20(6), M_21(7), M_22(8)
                let m00 = central_moments_result[0]; // μ_00 (total mass)
                let m20 = central_moments_result[6]; // μ_20
                let m02 = central_moments_result[2]; // μ_02
                let m11 = central_moments_result[4]; // μ_11

                let a = m20 / m00;
                let b = safe_f64_to_float::<T>(2.0)? * m11 / m00;
                let c = m02 / m00;

                let discriminant = (a - c) * (a - c) + b * b;
                let zero_t = T::zero();
                let eccentricity = if discriminant > zero_t {
                    let sqrt_disc = discriminant.sqrt();
                    let two_t = safe_f64_to_float::<T>(2.0)?;
                    let lambda1 = (a + c + sqrt_disc) / two_t;
                    let lambda2 = (a + c - sqrt_disc) / two_t;

                    if lambda1 > zero_t {
                        let one_t = T::one();
                        (one_t - lambda2 / lambda1).sqrt()
                    } else {
                        zero_t
                    }
                } else {
                    zero_t
                };

                // Update filtered labels
                for ((y, x), &val) in labels.indexed_iter() {
                    if val == i {
                        filtered_labels[[y, x]] = new_label;
                    }
                }

                let center_tuple = if com.len() >= 2 {
                    (
                        safe_float_to_f64(com[0]).unwrap_or(0.0),
                        safe_float_to_f64(com[1]).unwrap_or(0.0),
                    )
                } else {
                    (0.0, 0.0)
                };

                cell_info.push(CellInfo {
                    label: new_label,
                    area,
                    center: center_tuple,
                    eccentricity: safe_float_to_f64(eccentricity).unwrap_or(0.0),
                    mean_intensity: image
                        .indexed_iter()
                        .filter(|((y, x), _)| mask[[*y, *x]])
                        .map(|(_, &val)| safe_float_to_f64(val).unwrap_or(0.0))
                        .sum::<f64>()
                        / area as f64,
                });

                new_label += 1;
            }
        }

        Ok((filtered_labels, cell_info))
    }

    /// Information about a segmented cell
    #[derive(Clone, Debug)]
    pub struct CellInfo {
        pub label: i32,
        pub area: usize,
        pub center: (f64, f64),
        pub eccentricity: f64,
        pub mean_intensity: f64,
    }

    /// Detect and count nuclei in fluorescence microscopy
    pub fn detect_nuclei<T>(
        dapi_channel: &ArrayView2<T>,
        min_size: usize,
        max_size: usize,
    ) -> NdimageResult<(Array2<i32>, usize)>
    where
        T: Float
            + FromPrimitive
            + Debug
            + Send
            + Sync
            + std::ops::AddAssign
            + std::ops::DivAssign
            + scirs2_core::numeric::NumAssign
            + 'static,
    {
        // Preprocess with median filter to reduce noise
        let denoised = median_filter(&dapi_channel.to_owned(), &[3, 3], None)?;

        // Enhance nuclei using top-hat transform
        let structure = crate::morphology::disk_structure(10.0, None)?;
        let structure_2d = structure.into_dimensionality::<scirs2_core::ndarray::Ix2>()?;
        let background = grey_opening(&denoised, None, Some(&structure_2d), None, None, None)?;
        let enhanced = &denoised - &background;

        // Threshold using Otsu's method
        let (binary_t, threshold_value) = crate::segmentation::otsu_threshold(&enhanced, 256)?;

        // Convert to bool array
        let binary = binary_t.mapv(|x| x > threshold_value);

        // Fill holes in nuclei
        let filled = crate::morphology::binary_fill_holes(&binary, None, None)?;

        // Remove small objects
        let cleaned = crate::morphology::remove_small_objects(&filled, min_size, None)?;

        // Label nuclei
        let (labels_usize, num_features) = label(&cleaned, None, None, None)?;

        // Convert usize labels to i32 and filter by size
        let mut labels = Array2::<i32>::zeros(labels_usize.dim());
        let mut valid_count = 0;

        for i in 1..=num_features {
            let nucleus_size = labels_usize.iter().filter(|&&x| x == i).count();
            if nucleus_size >= min_size && nucleus_size <= max_size {
                valid_count += 1;
                // Copy this nucleus to the output with i32 label
                for ((y, x), &val) in labels_usize.indexed_iter() {
                    if val == i {
                        labels[[y, x]] = i as i32;
                    }
                }
            }
        }

        Ok((labels, valid_count))
    }

    /// Colocalization analysis for multi-channel microscopy
    pub fn colocalization_analysis<T>(
        channel1: &ArrayView2<T>,
        channel2: &ArrayView2<T>,
        threshold1: Option<T>,
        threshold2: Option<T>,
    ) -> NdimageResult<ColocalizationMetrics>
    where
        T: Float + FromPrimitive,
    {
        if channel1.dim() != channel2.dim() {
            return Err(NdimageError::DimensionError(
                "Channels must have same dimensions".into(),
            ));
        }

        // Apply thresholds
        let thresh1 = threshold1.unwrap_or_else(|| {
            let mean = channel1.sum() / safe_usize_to_float(channel1.len()).unwrap_or(T::one());
            let std = channel1.std(T::zero());
            mean + std
        });

        let thresh2 = threshold2.unwrap_or_else(|| {
            let mean = channel2.sum() / safe_usize_to_float(channel2.len()).unwrap_or(T::one());
            let std = channel2.std(T::zero());
            mean + std
        });

        // Create masks
        let mask1 = channel1.mapv(|x| x > thresh1);
        let mask2 = channel2.mapv(|x| x > thresh2);

        // Compute overlap
        let overlap = mask1
            .iter()
            .zip(mask2.iter())
            .filter(|(&a, &b)| a && b)
            .count();

        let area1 = mask1.iter().filter(|&&x| x).count();
        let area2 = mask2.iter().filter(|&&x| x).count();

        // Compute Manders coefficients
        let mut m1 = 0.0;
        let mut m2 = 0.0;
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        for ((y, x), &val1) in channel1.indexed_iter() {
            let val2 = channel2[[y, x]];

            if mask1[[y, x]] {
                sum1 += safe_float_to_f64(val1).unwrap_or(0.0);
                if mask2[[y, x]] {
                    m1 += safe_float_to_f64(val1).unwrap_or(0.0);
                }
            }

            if mask2[[y, x]] {
                sum2 += safe_float_to_f64(val2).unwrap_or(0.0);
                if mask1[[y, x]] {
                    m2 += safe_float_to_f64(val2).unwrap_or(0.0);
                }
            }
        }

        let manders_m1 = if sum1 > 0.0 { m1 / sum1 } else { 0.0 };
        let manders_m2 = if sum2 > 0.0 { m2 / sum2 } else { 0.0 };

        // Compute Pearson correlation
        let mean1 = safe_float_to_f64(
            channel1.sum() / safe_usize_to_float(channel1.len()).unwrap_or(T::one()),
        )
        .unwrap_or(0.0);
        let mean2 = safe_float_to_f64(
            channel2.sum() / safe_usize_to_float(channel2.len()).unwrap_or(T::one()),
        )
        .unwrap_or(0.0);

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for ((y, x), &val1) in channel1.indexed_iter() {
            if mask1[[y, x]] || mask2[[y, x]] {
                let v1 = safe_float_to_f64(val1).unwrap_or(0.0) - mean1;
                let v2 = safe_float_to_f64(channel2[[y, x]]).unwrap_or(0.0) - mean2;

                cov += v1 * v2;
                var1 += v1 * v1;
                var2 += v2 * v2;
            }
        }

        let pearson = if var1 > 0.0 && var2 > 0.0 {
            cov / (var1.sqrt() * var2.sqrt())
        } else {
            0.0
        };

        Ok(ColocalizationMetrics {
            overlap_coefficient: overlap as f64 / (area1.min(area2) as f64).max(1.0),
            manders_m1,
            manders_m2,
            pearson_correlation: pearson,
            overlap_area: overlap,
        })
    }

    /// Colocalization analysis results
    #[derive(Clone, Debug)]
    pub struct ColocalizationMetrics {
        pub overlap_coefficient: f64,
        pub manders_m1: f64,
        pub manders_m2: f64,
        pub pearson_correlation: f64,
        pub overlap_area: usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_ndvi() {
        let red = arr2(&[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]);

        let nir = arr2(&[[0.5, 0.6, 0.7], [0.6, 0.7, 0.8], [0.7, 0.8, 0.9]]);

        let ndvi =
            satellite::compute_ndvi(&red.view(), &nir.view()).expect("compute_ndvi should succeed");

        // Check NDVI values are in expected range
        for &val in ndvi.iter() {
            assert!(val >= -1.0 && val <= 1.0);
            assert!(val > 0.0); // Should be positive for healthy vegetation
        }
    }

    #[test]
    fn test_frangi_vesselness() {
        // Create a simple vessel-like structure
        let mut image = Array2::zeros((50, 50));

        // Horizontal vessel
        for i in 24..26 {
            for j in 10..40 {
                image[[i, j]] = 1.0;
            }
        }

        // Vertical vessel
        for i in 10..40 {
            for j in 24..26 {
                image[[i, j]] = 1.0;
            }
        }

        let vesselness = medical::frangi_vesselness(&image.view(), None)
            .expect("frangi_vesselness should succeed");

        // Check that vessel regions have high response
        assert!(vesselness[[25, 25]] > 0.0);
    }

    #[test]
    fn test_cell_segmentation() {
        // Create synthetic cell image
        let mut image = Array2::zeros((100, 100));

        // Add some circular "cells"
        for cy in [25, 75] {
            for cx in [25, 75] {
                for i in 0..100 {
                    for j in 0..100 {
                        let dy = i as f64 - cy as f64;
                        let dx = j as f64 - cx as f64;
                        let r = (dy * dy + dx * dx).sqrt();

                        if r < 10.0 {
                            image[[i, j]] = 1.0;
                        }
                    }
                }
            }
        }

        let (labels, cells) =
            microscopy::segment_cells(&image.view(), None).expect("segment_cells should succeed");

        assert_eq!(cells.len(), 4); // Should detect 4 cells
        assert!(labels.into_iter().max() == Some(4));
    }
}
