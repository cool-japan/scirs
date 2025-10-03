//! Module Documentation Builders
//!
//! This module contains functions to build comprehensive documentation
//! for all the different modules in the SciRS2 NDImage library.

use crate::documentation::types::{DocumentationSite, FunctionDoc, ModuleDoc, Parameter, Result};

impl DocumentationSite {
    /// Build comprehensive module documentation for all library modules
    pub fn build_module_documentation(&mut self) -> Result<()> {
        self.modules = vec![
            build_filters_documentation(),
            build_morphology_documentation(),
            build_interpolation_documentation(),
            build_measurements_documentation(),
        ];
        Ok(())
    }
}

/// Build documentation for the Filters module
pub fn build_filters_documentation() -> ModuleDoc {
    let mut module = ModuleDoc::new(
        "Filters",
        "Image filtering operations including Gaussian, median, rank, and edge detection filters",
    );

    // Gaussian filter documentation
    let mut gaussian_filter = FunctionDoc::new(
        "gaussian_filter",
        "pub fn gaussian_filter<T>(input: &ArrayD<T>, sigma: f64) -> ArrayD<T>",
        "Apply Gaussian filter to n-dimensional array",
        "ArrayD<T> - Filtered array",
    );

    gaussian_filter.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<T>",
        "Input n-dimensional array",
    ));
    gaussian_filter.add_parameter(Parameter::required(
        "sigma",
        "f64",
        "Standard deviation for Gaussian kernel",
    ));

    gaussian_filter.add_example(
        r#"use scirs2_ndimage::filters::gaussian_filter;
use scirs2_core::ndarray::Array2;

let image = Array2::from_elem((100, 100), 1.0f64);
let filtered = gaussian_filter(&image, 2.0);
assert_eq!(filtered.shape(), image.shape());"#,
    );

    gaussian_filter.add_note("Uses separable convolution for efficiency");
    gaussian_filter.add_note("Supports all numeric types");

    module.add_function(gaussian_filter);

    // Median filter documentation
    let mut median_filter = FunctionDoc::new(
        "median_filter",
        "pub fn median_filter<T>(input: &ArrayD<T>, size: usize) -> ArrayD<T>",
        "Apply median filter to remove noise while preserving edges",
        "ArrayD<T> - Filtered array",
    );

    median_filter.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<T>",
        "Input n-dimensional array",
    ));
    median_filter.add_parameter(Parameter::required(
        "size",
        "usize",
        "Size of the median filter window",
    ));

    median_filter.add_example(
        r#"use scirs2_ndimage::filters::median_filter;
use scirs2_core::ndarray::Array2;

let noisyimage = Array2::from_elem((50, 50), 128.0f64);
let filtered = median_filter(&noisyimage, 3);
// Median filter removes salt-and-pepper noise"#,
    );

    median_filter.add_note("Excellent for removing salt-and-pepper noise");
    median_filter.add_note("Preserves edges better than linear filters");

    module.add_function(median_filter);

    // Module examples
    module.add_example("Basic filtering operations");
    module.add_example("Edge detection pipeline");
    module.add_example("Noise reduction techniques");

    module
}

/// Build documentation for the Morphology module
pub fn build_morphology_documentation() -> ModuleDoc {
    let mut module = ModuleDoc::new(
        "Morphology",
        "Mathematical morphology operations for binary and grayscale images",
    );

    // Binary erosion documentation
    let mut binary_erosion = FunctionDoc::new(
        "binary_erosion",
        "pub fn binary_erosion(input: &ArrayD<bool>, structure: &ArrayD<bool>) -> ArrayD<bool>",
        "Perform binary erosion operation",
        "ArrayD<bool> - Eroded binary array",
    );

    binary_erosion.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<bool>",
        "Input binary array",
    ));
    binary_erosion.add_parameter(Parameter::required(
        "structure",
        "&ArrayD<bool>",
        "Structuring element",
    ));

    binary_erosion.add_example(
        r#"use scirs2_ndimage::morphology::binary_erosion;
use scirs2_core::ndarray::Array2;

let binary_image = Array2::from_elem((10, 10), true);
let structure = Array2::from_elem((3, 3), true);
let eroded = binary_erosion(&binary_image, &structure);"#,
    );

    binary_erosion.add_note("Shrinks white regions in binary images");
    binary_erosion.add_note("Useful for separating connected objects");

    module.add_function(binary_erosion);

    // Distance transform documentation
    let mut distance_transform = FunctionDoc::new(
        "distance_transform_edt",
        "pub fn distance_transform_edt(input: &ArrayD<bool>) -> ArrayD<f64>",
        "Compute Euclidean distance transform using optimized algorithm",
        "ArrayD<f64> - Distance transform",
    );

    distance_transform.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<bool>",
        "Input binary array",
    ));

    distance_transform.add_example(
        r#"use scirs2_ndimage::morphology::distance_transform_edt;
use scirs2_core::ndarray::Array2;

let binary_image = Array2::from_elem((100, 100), false);
let distances = distance_transform_edt(&binary_image);
// Each pixel contains distance to nearest background pixel"#,
    );

    distance_transform.add_note("Uses Felzenszwalb & Huttenlocher separable algorithm for O(n) performance");
    distance_transform.add_note("Supports arbitrary dimensions");

    module.add_function(distance_transform);

    // Module examples
    module.add_example("Object size analysis");
    module.add_example("Shape decomposition");
    module.add_example("Skeletonization");

    module
}

/// Build documentation for the Interpolation module
pub fn build_interpolation_documentation() -> ModuleDoc {
    let mut module = ModuleDoc::new(
        "Interpolation",
        "Geometric transformations and interpolation operations",
    );

    // Affine transform documentation
    let mut affine_transform = FunctionDoc::new(
        "affine_transform",
        "pub fn affine_transform<T>(input: &ArrayD<T>, matrix: &Array2<f64>) -> ArrayD<T>",
        "Apply affine transformation to n-dimensional array",
        "ArrayD<T> - Transformed array",
    );

    affine_transform.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<T>",
        "Input array to transform",
    ));
    affine_transform.add_parameter(Parameter::required(
        "matrix",
        "&Array2<f64>",
        "Affine transformation matrix",
    ));

    affine_transform.add_example(
        r#"use scirs2_ndimage::interpolation::affine_transform;
use scirs2_core::ndarray::{Array2, array};

let image = Array2::from_elem((50, 50), 1.0f64);
let rotation_matrix = array![[0.866, -0.5], [0.5, 0.866]]; // 30 degree rotation
let rotated = affine_transform(&image, &rotation_matrix);"#,
    );

    affine_transform.add_note("Supports rotation, scaling, shearing, and translation");
    affine_transform.add_note("Uses spline interpolation for high quality results");

    module.add_function(affine_transform);

    // Module examples
    module.add_example("Image registration");
    module.add_example("Geometric correction");
    module.add_example("Multi-resolution analysis");

    module
}

/// Build documentation for the Measurements module
pub fn build_measurements_documentation() -> ModuleDoc {
    let mut module = ModuleDoc::new(
        "Measurements",
        "Statistical measurements and region analysis",
    );

    // Center of mass documentation
    let mut center_of_mass = FunctionDoc::new(
        "center_of_mass",
        "pub fn center_of_mass<T>(input: &ArrayD<T>) -> Vec<f64>",
        "Calculate center of mass of n-dimensional array",
        "Vec<f64> - Center of mass coordinates",
    );

    center_of_mass.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<T>",
        "Input array",
    ));

    center_of_mass.add_example(
        r#"use scirs2_ndimage::measurements::center_of_mass;
use scirs2_core::ndarray::Array2;

let image = Array2::from_elem((100, 100), 1.0f64);
let com = center_of_mass(&image);
println!("Center of mass: {:?}", com);"#,
    );

    center_of_mass.add_note("Works with any numeric type");
    center_of_mass.add_note("Returns coordinates in array index order");

    module.add_function(center_of_mass);

    // Module examples
    module.add_example("Object property analysis");
    module.add_example("Region statistics");
    module.add_example("Feature extraction");

    module
}

/// Helper function to add common filter functions to a module
pub fn add_common_filter_functions(module: &mut ModuleDoc) {
    // Add sobel filter
    let mut sobel_filter = FunctionDoc::new(
        "sobel_filter",
        "pub fn sobel_filter<T>(input: &ArrayD<T>) -> ArrayD<T>",
        "Apply Sobel edge detection filter",
        "ArrayD<T> - Edge-detected array",
    );

    sobel_filter.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<T>",
        "Input n-dimensional array",
    ));

    sobel_filter.add_example(
        r#"use scirs2_ndimage::filters::sobel_filter;
use scirs2_core::ndarray::Array2;

let image = Array2::from_elem((100, 100), 1.0f64);
let edges = sobel_filter(&image);"#,
    );

    sobel_filter.add_note("Detects edges using Sobel operators");
    sobel_filter.add_note("Returns magnitude of gradient");

    module.add_function(sobel_filter);

    // Add laplacian filter
    let mut laplacian_filter = FunctionDoc::new(
        "laplacian_filter",
        "pub fn laplacian_filter<T>(input: &ArrayD<T>) -> ArrayD<T>",
        "Apply Laplacian edge detection filter",
        "ArrayD<T> - Edge-detected array",
    );

    laplacian_filter.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<T>",
        "Input n-dimensional array",
    ));

    laplacian_filter.add_example(
        r#"use scirs2_ndimage::filters::laplacian_filter;
use scirs2_core::ndarray::Array2;

let image = Array2::from_elem((100, 100), 1.0f64);
let edges = laplacian_filter(&image);"#,
    );

    laplacian_filter.add_note("Uses second derivative for edge detection");
    laplacian_filter.add_note("More sensitive to noise than Sobel");

    module.add_function(laplacian_filter);
}

/// Helper function to add common morphology functions to a module
pub fn add_common_morphology_functions(module: &mut ModuleDoc) {
    // Add binary dilation
    let mut binary_dilation = FunctionDoc::new(
        "binary_dilation",
        "pub fn binary_dilation(input: &ArrayD<bool>, structure: &ArrayD<bool>) -> ArrayD<bool>",
        "Perform binary dilation operation",
        "ArrayD<bool> - Dilated binary array",
    );

    binary_dilation.add_parameter(Parameter::required(
        "_input",
        "&ArrayD<bool>",
        "Input binary array",
    ));
    binary_dilation.add_parameter(Parameter::required(
        "structure",
        "&ArrayD<bool>",
        "Structuring element",
    ));

    binary_dilation.add_example(
        r#"use scirs2_ndimage::morphology::binary_dilation;
use scirs2_core::ndarray::Array2;

let binary_image = Array2::from_elem((10, 10), true);
let structure = Array2::from_elem((3, 3), true);
let dilated = binary_dilation(&binary_image, &structure);"#,
    );

    binary_dilation.add_note("Expands white regions in binary images");
    binary_dilation.add_note("Useful for connecting nearby objects");

    module.add_function(binary_dilation);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filters_documentation() {
        let filters_doc = build_filters_documentation();
        assert_eq!(filters_doc.name, "Filters");
        assert!(!filters_doc.functions.is_empty());
        assert!(!filters_doc.examples.is_empty());

        // Check that Gaussian filter is documented
        let gaussian = filters_doc
            .functions
            .iter()
            .find(|f| f.name == "gaussian_filter");
        assert!(gaussian.is_some());

        let gaussian = gaussian.unwrap();
        assert_eq!(gaussian.parameters.len(), 2);
        assert!(!gaussian.examples.is_empty());
        assert!(!gaussian.notes.is_empty());
    }

    #[test]
    fn test_morphology_documentation() {
        let morphology_doc = build_morphology_documentation();
        assert_eq!(morphology_doc.name, "Morphology");
        assert!(!morphology_doc.functions.is_empty());

        // Check that binary erosion is documented
        let erosion = morphology_doc
            .functions
            .iter()
            .find(|f| f.name == "binary_erosion");
        assert!(erosion.is_some());

        let erosion = erosion.unwrap();
        assert_eq!(erosion.parameters.len(), 2);
    }

    #[test]
    fn test_interpolation_documentation() {
        let interpolation_doc = build_interpolation_documentation();
        assert_eq!(interpolation_doc.name, "Interpolation");
        assert!(!interpolation_doc.functions.is_empty());

        // Check that affine transform is documented
        let affine = interpolation_doc
            .functions
            .iter()
            .find(|f| f.name == "affine_transform");
        assert!(affine.is_some());
    }

    #[test]
    fn test_measurements_documentation() {
        let measurements_doc = build_measurements_documentation();
        assert_eq!(measurements_doc.name, "Measurements");
        assert!(!measurements_doc.functions.is_empty());

        // Check that center of mass is documented
        let com = measurements_doc
            .functions
            .iter()
            .find(|f| f.name == "center_of_mass");
        assert!(com.is_some());
    }

    #[test]
    fn test_module_documentation_builder() {
        let mut site = DocumentationSite::new();
        let result = site.build_module_documentation();

        assert!(result.is_ok());
        assert_eq!(site.modules.len(), 4);

        // Check all expected modules are present
        let module_names: Vec<_> = site.modules.iter().map(|m| &m.name).collect();
        assert!(module_names.contains(&&"Filters".to_string()));
        assert!(module_names.contains(&&"Morphology".to_string()));
        assert!(module_names.contains(&&"Interpolation".to_string()));
        assert!(module_names.contains(&&"Measurements".to_string()));
    }
}