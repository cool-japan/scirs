//! SciPy ndimage API compatibility layer for seamless migration
//!
//! This module provides a drop-in replacement API that closely mirrors SciPy's
//! ndimage module, making it easy for users to migrate existing code with minimal
//! changes. It includes parameter mapping, behavior matching, and compatibility
//! warnings for any differences.

use ndarray::{Array, ArrayView2, Ix2, Ix3};
use num_traits::{Float, FromPrimitive};

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{gaussian_filter as internal_gaussian_filter, BorderMode};
use crate::interpolation::BoundaryMode;
use crate::measurements::center_of_mass as internal_center_of_mass;
use crate::morphology::{
    binary_dilation as internal_binary_dilation, binary_erosion as internal_binary_erosion,
};

/// SciPy ndimage compatibility layer
pub struct SciPyCompatLayer {
    /// Configuration for compatibility behavior
    config: CompatibilityConfig,
    /// Migration warnings
    warnings: Vec<MigrationWarning>,
}

#[derive(Debug, Clone)]
pub struct CompatibilityConfig {
    /// Enable strict SciPy compatibility mode
    pub strict_compatibility: bool,
    /// Show migration warnings
    pub show_warnings: bool,
    /// Default data type for operations
    pub default_dtype: String,
    /// Default boundary mode
    pub default_mode: String,
    /// Enable performance optimizations that may differ from SciPy
    pub enable_optimizations: bool,
}

impl Default for CompatibilityConfig {
    fn default() -> Self {
        Self {
            strict_compatibility: true,
            show_warnings: true,
            default_dtype: "float64".to_string(),
            default_mode: "reflect".to_string(),
            enable_optimizations: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MigrationWarning {
    /// Function name
    pub function: String,
    /// Warning message
    pub message: String,
    /// Suggested solution
    pub suggestion: Option<String>,
}

impl SciPyCompatLayer {
    /// Create a new SciPy compatibility layer
    pub fn new(config: CompatibilityConfig) -> Self {
        Self {
            config,
            warnings: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(CompatibilityConfig::default())
    }

    /// Get migration warnings
    pub fn get_warnings(&self) -> &[MigrationWarning] {
        &self.warnings
    }

    /// Clear warnings
    pub fn clear_warnings(&mut self) {
        self.warnings.clear();
    }
}

/// SciPy-compatible filter functions
impl SciPyCompatLayer {
    /// Gaussian filter with SciPy-compatible interface
    ///
    /// ```python
    /// # SciPy usage:
    /// scipy.ndimage.gaussian_filter(input, sigma, order=0, output=None,
    ///                               mode='reflect', cval=0.0, truncate=4.0)
    /// ```
    pub fn gaussian_filter<T>(
        &mut self,
        input: ArrayView2<T>,
        sigma: SigmaParam,
        order: Option<OrderParam>,
        mode: Option<&str>,
        cval: Option<f64>,
        truncate: Option<f64>,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync,
    {
        // Handle parameter conversion
        let sigma_tuple = self.convert_sigma_param(sigma)?;
        let boundary_mode = self.convert_mode_param(mode)?;

        // Check for unsupported parameters
        if order.is_some() && order != Some(OrderParam::Single(0)) {
            self.add_warning(
                "gaussian_filter",
                "Non-zero order parameter not yet supported, using order=0",
            );
        }

        if cval.is_some() && cval != Some(0.0) {
            self.add_warning(
                "gaussian_filter",
                "Custom cval not fully supported, using default boundary handling",
            );
        }

        if truncate.is_some() && truncate != Some(4.0) {
            self.add_warning(
                "gaussian_filter",
                "Custom truncate parameter not supported, using default value",
            );
        }

        // Convert input to f64 array and sigma tuple to single value
        let input_f64 = input.mapv(|x| x.to_f64().unwrap_or(0.0)).to_owned();
        let sigma_single = (sigma_tuple.0 + sigma_tuple.1) / 2.0; // Use average of sigma values

        // Convert BoundaryMode to BorderMode
        let border_mode = match boundary_mode {
            BoundaryMode::Constant => BorderMode::Constant,
            BoundaryMode::Reflect => BorderMode::Reflect,
            BoundaryMode::Mirror => BorderMode::Mirror,
            BoundaryMode::Wrap => BorderMode::Wrap,
            BoundaryMode::Nearest => BorderMode::Nearest,
        };

        // Call internal implementation
        let result_f64 =
            internal_gaussian_filter(&input_f64, sigma_single, Some(border_mode), None)?;

        // Convert back to original type
        let result = result_f64.mapv(|x| T::from_f64(x).unwrap_or(T::zero()));
        Ok(result)
    }

    /// Median filter with SciPy-compatible interface
    ///
    /// ```python
    /// # SciPy usage:
    /// scipy.ndimage.median_filter(input, size=None, footprint=None, output=None,
    ///                             mode='reflect', cval=0.0, origin=0)
    /// ```
    pub fn median_filter<T>(
        &mut self,
        input: ArrayView2<T>,
        size: Option<SizeParam>,
        footprint: Option<ArrayView2<bool>>,
        mode: Option<&str>,
        cval: Option<f64>,
        origin: Option<OriginParam>,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float
            + FromPrimitive
            + std::fmt::Debug
            + Clone
            + Send
            + Sync
            + PartialOrd
            + std::ops::AddAssign
            + std::ops::DivAssign
            + 'static,
    {
        let filter_size = self.convert_size_param(size, (3, 3))?;
        let boundary_mode = self.convert_mode_param(mode)?;

        if footprint.is_some() {
            self.add_warning(
                "median_filter",
                "Custom footprint not yet supported, using rectangular window",
            );
        }

        if origin.is_some() {
            self.add_warning(
                "median_filter",
                "Origin parameter not supported, using center origin",
            );
        }

        // Convert input to owned array and convert parameters
        let input_owned = input.to_owned();
        let size_slice = [filter_size.0, filter_size.1];

        // Convert BoundaryMode to BorderMode
        let border_mode = match boundary_mode {
            BoundaryMode::Constant => BorderMode::Constant,
            BoundaryMode::Reflect => BorderMode::Reflect,
            BoundaryMode::Mirror => BorderMode::Mirror,
            BoundaryMode::Wrap => BorderMode::Wrap,
            BoundaryMode::Nearest => BorderMode::Nearest,
        };

        crate::filters::median_filter(&input_owned, &size_slice, Some(border_mode))
    }

    /// Uniform filter with SciPy-compatible interface
    ///
    /// ```python
    /// # SciPy usage:
    /// scipy.ndimage.uniform_filter(input, size=3, output=None, mode='reflect',
    ///                              cval=0.0, origin=0)
    /// ```
    pub fn uniform_filter<T>(
        &mut self,
        input: ArrayView2<T>,
        size: Option<SizeParam>,
        mode: Option<&str>,
        cval: Option<f64>,
        origin: Option<OriginParam>,
    ) -> NdimageResult<Array<T, Ix2>>
    where
        T: Float
            + FromPrimitive
            + std::fmt::Debug
            + Clone
            + Send
            + Sync
            + std::ops::AddAssign
            + std::ops::DivAssign
            + 'static,
    {
        let filter_size = self.convert_size_param(size, (3, 3))?;
        let boundary_mode = self.convert_mode_param(mode)?;

        if origin.is_some() {
            self.add_warning(
                "uniform_filter",
                "Origin parameter not supported, using center origin",
            );
        }

        // Convert input to owned array and convert parameters
        let input_owned = input.to_owned();
        let size_slice = [filter_size.0, filter_size.1];

        // Convert BoundaryMode to BorderMode
        let border_mode = match boundary_mode {
            BoundaryMode::Constant => BorderMode::Constant,
            BoundaryMode::Reflect => BorderMode::Reflect,
            BoundaryMode::Mirror => BorderMode::Mirror,
            BoundaryMode::Wrap => BorderMode::Wrap,
            BoundaryMode::Nearest => BorderMode::Nearest,
        };

        crate::filters::uniform_filter(&input_owned, &size_slice, Some(border_mode), None)
    }
}

/// SciPy-compatible morphology functions
impl SciPyCompatLayer {
    /// Binary erosion with SciPy-compatible interface
    ///
    /// ```python
    /// # SciPy usage:
    /// scipy.ndimage.binary_erosion(input, structure=None, iterations=1,
    ///                              mask=None, output=None, border_value=0,
    ///                              origin=0, brute_force=False)
    /// ```
    pub fn binary_erosion<T>(
        &mut self,
        input: ArrayView2<T>,
        structure: Option<ArrayView2<bool>>,
        iterations: Option<usize>,
        mask: Option<ArrayView2<bool>>,
        border_value: Option<bool>,
        origin: Option<OriginParam>,
        brute_force: Option<bool>,
    ) -> NdimageResult<Array<bool, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
    {
        // Convert input to binary
        let binary_input = self.convert_to_binary(input);

        // Use default 3x3 cross structure if none provided
        let default_structure = Array::from_shape_vec(
            (3, 3),
            vec![false, true, false, true, true, true, false, true, false],
        )
        .unwrap();

        let structure_elem = match structure {
            Some(s) => s.to_owned(),
            None => default_structure,
        };

        if iterations.is_some() && iterations != Some(1) {
            self.add_warning(
                "binary_erosion",
                "Multiple iterations not yet optimized, using single iteration",
            );
        }

        if mask.is_some() {
            self.add_warning("binary_erosion", "Mask parameter not yet supported");
        }

        if origin.is_some() {
            self.add_warning("binary_erosion", "Origin parameter not supported");
        }

        if brute_force.is_some() {
            self.add_warning(
                "binary_erosion",
                "brute_force parameter not supported, using optimized algorithm",
            );
        }

        // Apply multiple iterations if requested
        let mut result = internal_binary_erosion(
            &binary_input,
            Some(&structure_elem),
            None,
            None,
            None,
            None,
            None,
        )?;

        for _ in 1..iterations.unwrap_or(1) {
            result = internal_binary_erosion(
                &result,
                Some(&structure_elem),
                None,
                None,
                None,
                None,
                None,
            )?;
        }

        Ok(result)
    }

    /// Binary dilation with SciPy-compatible interface
    ///
    /// ```python
    /// # SciPy usage:
    /// scipy.ndimage.binary_dilation(input, structure=None, iterations=1,
    ///                               mask=None, output=None, border_value=0,
    ///                               origin=0, brute_force=False)
    /// ```
    pub fn binary_dilation<T>(
        &mut self,
        input: ArrayView2<T>,
        structure: Option<ArrayView2<bool>>,
        iterations: Option<usize>,
        mask: Option<ArrayView2<bool>>,
        border_value: Option<bool>,
        origin: Option<OriginParam>,
        brute_force: Option<bool>,
    ) -> NdimageResult<Array<bool, Ix2>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
    {
        let binary_input = self.convert_to_binary(input);

        let default_structure = Array::from_shape_vec(
            (3, 3),
            vec![false, true, false, true, true, true, false, true, false],
        )
        .unwrap();

        let structure_elem = match structure {
            Some(s) => s.to_owned(),
            None => default_structure,
        };

        if iterations.is_some() && iterations != Some(1) {
            self.add_warning(
                "binary_dilation",
                "Multiple iterations not yet optimized, using single iteration",
            );
        }

        if mask.is_some() {
            self.add_warning("binary_dilation", "Mask parameter not yet supported");
        }

        // Apply multiple iterations if requested
        let mut result = internal_binary_dilation(
            &binary_input,
            Some(&structure_elem),
            None,
            None,
            None,
            None,
            None,
        )?;

        for _ in 1..iterations.unwrap_or(1) {
            result = internal_binary_dilation(
                &result,
                Some(&structure_elem),
                None,
                None,
                None,
                None,
                None,
            )?;
        }

        Ok(result)
    }

    /// Distance transform with SciPy-compatible interface
    ///
    /// ```python
    /// # SciPy usage:
    /// scipy.ndimage.distance_transform_edt(input, sampling=None, return_distances=True,
    ///                                      return_indices=False, distances=None, indices=None)
    /// ```
    pub fn distance_transform_edt<T>(
        &mut self,
        input: ArrayView2<T>,
        sampling: Option<SamplingParam>,
        return_distances: Option<bool>,
        return_indices: Option<bool>,
    ) -> NdimageResult<DistanceTransformResult<T>>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
    {
        let binary_input = self.convert_to_binary(input);

        if sampling.is_some() {
            self.add_warning(
                "distance_transform_edt",
                "Sampling parameter not yet supported, using unit sampling",
            );
        }

        if return_indices == Some(true) {
            self.add_warning(
                "distance_transform_edt",
                "Returning _indices not yet supported",
            );
        }

        let binary_input_dyn = binary_input.into_dyn();
        let (distances_opt, _indices_opt) = crate::morphology::distance_transform_edt(
            &binary_input_dyn,
            None,  // sampling
            true,  // return_distances
            false, // return_indices
        )?;

        let _distances = distances_opt.ok_or_else(|| {
            NdimageError::ComputationError("Failed to compute distances".to_string())
        })?;

        // Convert back to original type and convert to 2D
        let result_2d = _distances
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| {
                NdimageError::ComputationError("Failed to convert distances back to 2D".to_string())
            })?;
        let result_array = result_2d.mapv(|v| T::from_f64(v).unwrap_or(T::zero()));

        Ok(DistanceTransformResult {
            distances: Some(result_array),
            indices: None,
        })
    }
}

/// SciPy-compatible measurement functions
impl SciPyCompatLayer {
    /// Center of mass with SciPy-compatible interface
    ///
    /// ```python
    /// # SciPy usage:
    /// scipy.ndimage.center_of_mass(input, labels=None, index=None)
    /// ```
    pub fn center_of_mass<T>(
        &mut self,
        input: ArrayView2<T>,
        labels: Option<ArrayView2<i32>>,
        index: Option<IndexParam>,
    ) -> NdimageResult<CenterOfMassResult>
    where
        T: Float
            + FromPrimitive
            + Clone
            + Send
            + Sync
            + std::fmt::Debug
            + std::ops::DivAssign
            + num_traits::NumAssign
            + 'static,
    {
        if labels.is_some() {
            self.add_warning("center_of_mass", "Labels parameter not yet fully supported");
        }

        if index.is_some() {
            self.add_warning("center_of_mass", "Index parameter not yet supported");
        }

        let com = internal_center_of_mass(&input.to_owned())?;
        // Convert Vec<T> to (f64, f64)
        if com.len() >= 2 {
            let com_tuple = (
                com[0].to_f64().unwrap_or(0.0),
                com[1].to_f64().unwrap_or(0.0),
            );
            Ok(CenterOfMassResult::Single(com_tuple))
        } else {
            Err(NdimageError::ComputationError(
                "Center of mass computation failed".to_string(),
            ))
        }
    }

    /// Label connected components with SciPy-compatible interface
    ///
    /// ```python
    /// # SciPy usage:
    /// scipy.ndimage.label(input, structure=None, output=None)
    /// ```
    pub fn label<T>(
        &mut self,
        input: ArrayView2<T>,
        structure: Option<ArrayView2<bool>>,
    ) -> NdimageResult<LabelResult>
    where
        T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
    {
        let binary_input = self.convert_to_binary(input);

        if structure.is_some() {
            self.add_warning(
                "label",
                "Custom structure not yet supported, using default connectivity",
            );
        }

        let (labeled, num_labels) = crate::morphology::label(
            &binary_input,
            None, // structure
            None, // connectivity
            None, // background
        )?;

        Ok(LabelResult {
            labeled_array: labeled.mapv(|v| v as i32),
            num_features: num_labels as i32,
        })
    }
}

// Parameter types and conversion functions
impl SciPyCompatLayer {
    fn convert_sigma_param(&self, sigma: SigmaParam) -> NdimageResult<(f64, f64)> {
        match sigma {
            SigmaParam::Single(s) => Ok((s, s)),
            SigmaParam::Tuple(sx, sy) => Ok((sx, sy)),
            SigmaParam::Array(arr) => {
                if arr.len() == 1 {
                    Ok((arr[0], arr[0]))
                } else if arr.len() == 2 {
                    Ok((arr[0], arr[1]))
                } else {
                    Err(NdimageError::InvalidInput(
                        "Sigma must be scalar or 2-element array".to_string(),
                    ))
                }
            }
        }
    }

    fn convert_size_param(
        &self,
        size: Option<SizeParam>,
        default: (usize, usize),
    ) -> NdimageResult<(usize, usize)> {
        match size {
            None => Ok(default),
            Some(SizeParam::Single(s)) => Ok((s, s)),
            Some(SizeParam::Tuple(sx, sy)) => Ok((sx, sy)),
            Some(SizeParam::Array(arr)) => {
                if arr.len() == 1 {
                    Ok((arr[0], arr[0]))
                } else if arr.len() == 2 {
                    Ok((arr[0], arr[1]))
                } else {
                    Err(NdimageError::InvalidInput(
                        "Size must be scalar or 2-element array".to_string(),
                    ))
                }
            }
        }
    }

    fn convert_mode_param(&self, mode: Option<&str>) -> NdimageResult<BoundaryMode> {
        let mode_str = mode.unwrap_or(&self.config.default_mode);
        match mode_str {
            "reflect" => Ok(BoundaryMode::Reflect),
            "constant" => Ok(BoundaryMode::Constant),
            "nearest" => Ok(BoundaryMode::Nearest),
            "mirror" => Ok(BoundaryMode::Mirror),
            "wrap" => Ok(BoundaryMode::Wrap),
            _ => {
                self.add_warning_const(
                    "parameter_conversion",
                    &format!("Unknown mode '{}', using default 'reflect'", mode_str),
                );
                Ok(BoundaryMode::Reflect)
            }
        }
    }

    fn convert_to_binary<T>(&self, input: ArrayView2<T>) -> Array<bool, Ix2>
    where
        T: Float + FromPrimitive + PartialOrd,
    {
        input.mapv(|x| x > T::zero())
    }

    fn add_warning(&mut self, function: &str, message: &str) {
        if self.config.show_warnings {
            self.warnings.push(MigrationWarning {
                function: function.to_string(),
                message: message.to_string(),
                suggestion: None,
            });
        }
    }

    fn add_warning_const(&self, function: &str, message: &str) {
        if self.config.show_warnings {
            eprintln!("Warning in {}: {}", function, message);
        }
    }
}

// Parameter types for SciPy compatibility

#[derive(Debug, Clone)]
pub enum SigmaParam {
    Single(f64),
    Tuple(f64, f64),
    Array(Vec<f64>),
}

#[derive(Debug, Clone)]
pub enum SizeParam {
    Single(usize),
    Tuple(usize, usize),
    Array(Vec<usize>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderParam {
    Single(usize),
    Tuple(usize, usize),
    Array(Vec<usize>),
}

#[derive(Debug, Clone)]
pub enum OriginParam {
    Single(isize),
    Tuple(isize, isize),
    Array(Vec<isize>),
}

#[derive(Debug, Clone)]
pub enum SamplingParam {
    Single(f64),
    Tuple(f64, f64),
    Array(Vec<f64>),
}

#[derive(Debug, Clone)]
pub enum IndexParam {
    Single(i32),
    Array(Vec<i32>),
}

// Result types

#[derive(Debug, Clone)]
pub struct DistanceTransformResult<T> {
    pub distances: Option<Array<T, Ix2>>,
    pub indices: Option<Array<usize, Ix3>>, // (2, height, width) for 2D
}

#[derive(Debug, Clone)]
pub enum CenterOfMassResult {
    Single((f64, f64)),
    Multiple(Vec<(f64, f64)>),
}

#[derive(Debug, Clone)]
pub struct LabelResult {
    pub labeled_array: Array<i32, Ix2>,
    pub num_features: i32,
}

/// Global SciPy compatibility instance
static mut SCIPY_COMPAT: Option<SciPyCompatLayer> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Initialize global SciPy compatibility layer
#[allow(dead_code)]
pub fn init_scipy_compat() {
    INIT.call_once(|| unsafe {
        SCIPY_COMPAT = Some(SciPyCompatLayer::default());
    });
}

/// Get global SciPy compatibility layer
#[allow(dead_code)]
#[allow(static_mut_refs)]
fn get_scipy_compat() -> &'static mut SciPyCompatLayer {
    init_scipy_compat();
    unsafe { SCIPY_COMPAT.as_mut().unwrap() }
}

// Global convenience functions that match SciPy API exactly

/// Gaussian filter (global function matching SciPy API)
#[allow(dead_code)]
pub fn gaussian_filter<T>(
    input: ArrayView2<T>,
    sigma: SigmaParam,
    order: Option<OrderParam>,
    mode: Option<&str>,
    cval: Option<f64>,
    truncate: Option<f64>,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float + FromPrimitive + Clone + Send + Sync,
{
    get_scipy_compat().gaussian_filter(input, sigma, order, mode, cval, truncate)
}

/// Median filter (global function matching SciPy API)
#[allow(dead_code)]
pub fn median_filter<T>(
    input: ArrayView2<T>,
    size: Option<SizeParam>,
    footprint: Option<ArrayView2<bool>>,
    mode: Option<&str>,
    cval: Option<f64>,
    origin: Option<OriginParam>,
) -> NdimageResult<Array<T, Ix2>>
where
    T: Float
        + FromPrimitive
        + std::fmt::Debug
        + Clone
        + Send
        + Sync
        + PartialOrd
        + std::ops::AddAssign
        + std::ops::DivAssign
        + 'static,
{
    get_scipy_compat().median_filter(input, size, footprint, mode, cval, origin)
}

/// Binary erosion (global function matching SciPy API)
#[allow(dead_code)]
pub fn binary_erosion<T>(
    input: ArrayView2<T>,
    structure: Option<ArrayView2<bool>>,
    iterations: Option<usize>,
    mask: Option<ArrayView2<bool>>,
    border_value: Option<bool>,
    origin: Option<OriginParam>,
    brute_force: Option<bool>,
) -> NdimageResult<Array<bool, Ix2>>
where
    T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
{
    get_scipy_compat().binary_erosion(
        input,
        structure,
        iterations,
        mask,
        border_value,
        origin,
        brute_force,
    )
}

/// Binary erosion for boolean arrays (global function matching SciPy API)
#[allow(dead_code)]
pub fn binary_erosion_bool(
    input: ArrayView2<bool>,
    structure: Option<ArrayView2<bool>>,
    iterations: Option<usize>,
    mask: Option<ArrayView2<bool>>,
    border_value: Option<bool>,
    origin: Option<OriginParam>,
    brute_force: Option<bool>,
) -> NdimageResult<Array<bool, Ix2>> {
    // Convert boolean array to f64 for compatibility with existing implementation
    let input_f64 = input.map(|&x| if x { 1.0f64 } else { 0.0f64 });
    get_scipy_compat().binary_erosion(
        input_f64.view(),
        structure,
        iterations,
        mask,
        border_value,
        origin,
        brute_force,
    )
}

/// Distance transform EDT (global function matching SciPy API)
#[allow(dead_code)]
pub fn distance_transform_edt<T>(
    input: ArrayView2<T>,
    sampling: Option<SamplingParam>,
    return_distances: Option<bool>,
    return_indices: Option<bool>,
) -> NdimageResult<DistanceTransformResult<T>>
where
    T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
{
    get_scipy_compat().distance_transform_edt(input, sampling, return_distances, return_indices)
}

/// Center of mass (global function matching SciPy API)
#[allow(dead_code)]
pub fn center_of_mass<T>(
    input: ArrayView2<T>,
    labels: Option<ArrayView2<i32>>,
    index: Option<IndexParam>,
) -> NdimageResult<CenterOfMassResult>
where
    T: Float
        + FromPrimitive
        + Clone
        + Send
        + Sync
        + std::fmt::Debug
        + std::ops::DivAssign
        + num_traits::NumAssign
        + 'static,
{
    get_scipy_compat().center_of_mass(input, labels, index)
}

/// Label connected components (global function matching SciPy API)
#[allow(dead_code)]
pub fn label<T>(
    input: ArrayView2<T>,
    structure: Option<ArrayView2<bool>>,
) -> NdimageResult<LabelResult>
where
    T: Float + FromPrimitive + Clone + Send + Sync + PartialOrd,
{
    get_scipy_compat().label(input, structure)
}

/// Get migration warnings
#[allow(dead_code)]
pub fn get_migration_warnings() -> Vec<MigrationWarning> {
    get_scipy_compat().get_warnings().to_vec()
}

/// Clear migration warnings
#[allow(dead_code)]
pub fn clear_migration_warnings() {
    get_scipy_compat().clear_warnings();
}

/// Display migration guide
#[allow(dead_code)]
pub fn display_migration_guide() {
    println!(
        r#"
=== SciRS2 NDImage Migration Guide ===

This compatibility layer provides SciPy-compatible APIs for easy migration.

Basic Usage:
    use scirs2_ndimage::scipy_migration_layer as ndimage;
    
    // Same API as SciPy
    let result = ndimage::gaussian_filter(input, sigma, None, None, None, None)?;

Key Differences:
1. All functions return Result<T, NdimageError> for error handling
2. Some advanced parameters may not be fully supported yet
3. Performance characteristics may differ due to Rust optimizations

Migration Steps:
1. Replace `import scipy.ndimage` with `use scirs2_ndimage::scipy_migration_layer as ndimage;`
2. Add error handling for function calls
3. Check migration warnings for any unsupported features
4. Test thoroughly and report any compatibility issues

For full scirs2 performance, consider using the native APIs in other modules.
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_scipy_compat_creation() {
        let compat = SciPyCompatLayer::default();
        assert!(compat.warnings.is_empty());
    }

    #[test]
    fn test_gaussian_filter_compat() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = gaussian_filter(
            input.view(),
            SigmaParam::Single(1.0),
            None,
            None,
            None,
            None,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_binary_erosion_compat() {
        let input = array![
            [true, false, true],
            [false, true, false],
            [true, false, true]
        ];

        let result = binary_erosion_bool(input.view(), None, None, None, None, None, None);

        assert!(result.is_ok());
    }

    #[test]
    fn test_parameter_conversion() {
        let compat = SciPyCompatLayer::default();

        // Test sigma parameter conversion
        let sigma1 = compat.convert_sigma_param(SigmaParam::Single(2.0)).unwrap();
        assert_eq!(sigma1, (2.0, 2.0));

        let sigma2 = compat
            .convert_sigma_param(SigmaParam::Tuple(1.0, 2.0))
            .unwrap();
        assert_eq!(sigma2, (1.0, 2.0));

        // Test size parameter conversion
        let size1 = compat
            .convert_size_param(Some(SizeParam::Single(5)), (3, 3))
            .unwrap();
        assert_eq!(size1, (5, 5));

        let size2 = compat.convert_size_param(None, (3, 3)).unwrap();
        assert_eq!(size2, (3, 3));
    }

    #[test]
    fn test_migration_warnings() {
        init_scipy_compat();
        clear_migration_warnings();

        let input = array![[1.0, 2.0], [3.0, 4.0]];

        // This should generate a warning for unsupported parameter
        let _ = gaussian_filter(
            input.view(),
            SigmaParam::Single(1.0),
            Some(OrderParam::Single(1)), // Non-zero order should warn
            None,
            None,
            None,
        );

        let warnings = get_migration_warnings();
        assert!(!warnings.is_empty());
    }
}
