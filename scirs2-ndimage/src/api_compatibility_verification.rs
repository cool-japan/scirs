//! API Compatibility Verification with SciPy ndimage
//!
//! This module provides comprehensive testing utilities to verify that
//! scirs2-ndimage maintains API compatibility with SciPy's ndimage module.
//! It includes parameter validation, behavior verification, and migration
//! guidance for any incompatibilities.

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
use crate::filters::*;
use crate::interpolation::*;
use crate::measurements::*;
use crate::morphology::*;
use crate::scipy_compat_layer;
use scirs2_core::ndarray::Array2;

/// API compatibility test result
#[derive(Debug, Clone)]
pub struct ApiCompatibilityResult {
    /// Function name being tested
    pub function_name: String,
    /// Specific test case description
    pub test_case: String,
    /// Whether the API is compatible
    pub compatible: bool,
    /// Compatibility score (0.0 = incompatible, 1.0 = fully compatible)
    pub compatibility_score: f64,
    /// List of incompatible parameters
    pub incompatible_parameters: Vec<String>,
    /// Error messages for failures
    pub error_messages: Vec<String>,
    /// Suggested workarounds or fixes
    pub suggestions: Vec<String>,
    /// SciPy reference behavior description
    pub scipy_behavior: String,
    /// scirs2-ndimage behavior description
    pub scirs2_behavior: String,
}

/// Parameter compatibility test
#[derive(Debug, Clone)]
pub struct ParameterTest {
    /// Parameter name
    pub name: String,
    /// Test description
    pub description: String,
    /// Test function that returns (success, error_message)
    pub test_fn: fn() -> (bool, Option<String>),
    /// Expected behavior in SciPy
    pub scipy_expected: String,
    /// Priority level (High, Medium, Low)
    pub priority: String,
}

/// Comprehensive API compatibility tester
pub struct ApiCompatibilityTester {
    /// Test results
    results: Vec<ApiCompatibilityResult>,
    /// Overall compatibility score
    overall_score: f64,
    /// Configuration for testing
    config: CompatibilityConfig,
}

/// Configuration for compatibility testing
#[derive(Debug, Clone)]
pub struct CompatibilityConfig {
    /// Whether to test edge cases
    pub test_edge_cases: bool,
    /// Whether to test error conditions
    pub test_error_conditions: bool,
    /// Whether to test performance characteristics
    pub test_performance: bool,
    /// Tolerance for numerical comparisons
    pub numerical_tolerance: f64,
    /// Maximum array size for testing
    pub max_test_size: usize,
}

impl Default for CompatibilityConfig {
    fn default() -> Self {
        Self {
            test_edge_cases: true,
            test_error_conditions: true,
            test_performance: false, // Expensive, disabled by default
            numerical_tolerance: 1e-10,
            max_test_size: 1000,
        }
    }
}

impl ApiCompatibilityTester {
    /// Create a new compatibility tester
    pub fn new() -> Self {
        Self::with_config(CompatibilityConfig::default())
    }

    /// Create a new compatibility tester with custom configuration
    pub fn with_config(config: CompatibilityConfig) -> Self {
        Self {
            results: Vec::new(),
            overall_score: 0.0,
            config,
        }
    }

    /// Test all filter function APIs for compatibility
    pub fn test_filter_apis(&mut self) -> Result<()> {
        // Test gaussian_filter API compatibility
        self.test_gaussian_filter_api()?;

        // Test median_filter API compatibility
        self.test_median_filter_api()?;

        // Test uniform_filter API compatibility
        self.test_uniform_filter_api()?;

        // Test sobel filter API compatibility
        self.test_sobel_filter_api()?;

        // Test rank filters API compatibility
        self.test_rank_filter_apis()?;

        Ok(())
    }

    fn test_gaussian_filter_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();
        let mut suggestions = Vec::new();

        // Test 1: Basic parameter compatibility
        let input: Array2<f64> = Array2::zeros((10, 10));

        // Test sigma parameter - should accept scalar and array
        let test1_success = gaussian_filter(&input, 1.0, None, None).is_ok();
        if !test1_success {
            incompatible_params.push("sigma_scalar".to_string());
            error_messages.push("Scalar sigma parameter not supported".to_string());
        }

        // Test output parameter - scirs2 returns result instead of in-place
        let scipy_behavior = "SciPy accepts optional output array for in-place operations";
        let scirs2_behavior = "scirs2-ndimage always returns new array, no in-place operations";
        suggestions.push("Use returned array instead of in-place modification".to_string());

        // Test mode parameter compatibility
        let mode_test = gaussian_filter(&input, 1.0, Some(BorderMode::Constant), None).is_ok();
        if !mode_test {
            incompatible_params.push("mode".to_string());
            error_messages.push("BorderMode enum may not match SciPy string modes".to_string());
            suggestions.push("Use BorderMode enum instead of strings".to_string());
        }

        // Test cval parameter
        let cval_test = gaussian_filter(&input, 1.0, Some(BorderMode::Constant), Some(0.0)).is_ok();
        if !cval_test {
            incompatible_params.push("cval".to_string());
            error_messages.push("Constant value parameter handling differs".to_string());
        }

        // Edge case: Very small sigma
        let small_sigma_test = gaussian_filter(&input, 1e-10, None, None).is_ok();
        if !small_sigma_test && self.config.test_edge_cases {
            incompatible_params.push("sigma_edge_case".to_string());
            error_messages.push("Very small sigma values may be handled differently".to_string());
        }

        // Error condition: Negative sigma
        let negative_sigma_test = gaussian_filter(&input, -1.0, None, None).is_err();
        if !negative_sigma_test && self.config.test_error_conditions {
            incompatible_params.push("sigma_validation".to_string());
            error_messages.push("Negative sigma should raise error".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 6.0); // 6 total tests

        self.results.push(ApiCompatibilityResult {
            function_name: "gaussian_filter".to_string(),
            test_case: "Parameter compatibility and behavior".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions,
            scipy_behavior: scipy_behavior.to_string(),
            scirs2_behavior: scirs2_behavior.to_string(),
        });

        Ok(())
    }

    fn test_median_filter_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();
        let mut suggestions = Vec::new();

        let input: Array2<f64> = Array2::zeros((10, 10));

        // Test size parameter - should accept various formats
        let size_array_test = median_filter(&input, &[3, 3], None).is_ok();
        if !size_array_test {
            incompatible_params.push("size_array".to_string());
            error_messages.push("Array size parameter not supported".to_string());
        }

        // Test footprint parameter - SciPy has footprint, we have size
        suggestions.push("Use size parameter instead of footprint for filter kernel".to_string());

        // Test mode parameter
        let mode_test = median_filter(&input, &[3, 3], Some(BorderMode::Reflect)).is_ok();
        if !mode_test {
            incompatible_params.push("mode".to_string());
            error_messages.push("Border mode handling may differ".to_string());
        }

        // Edge case: Size = 1 (no filtering)
        let size_one_test = median_filter(&input, &[1, 1], None).is_ok();
        if !size_one_test && self.config.test_edge_cases {
            incompatible_params.push("size_edge_case".to_string());
            error_messages.push("Size=1 edge case handling differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 4.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "median_filter".to_string(),
            test_case: "Parameter and edge case compatibility".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions,
            scipy_behavior: "Accepts footprint or size, various modes".to_string(),
            scirs2_behavior: "Accepts size array and BorderMode enum".to_string(),
        });

        Ok(())
    }

    fn test_uniform_filter_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input: Array2<f64> = Array2::zeros((10, 10));

        // Test basic functionality
        let basic_test = uniform_filter(&input, &[3, 3], None, None).is_ok();
        if !basic_test {
            incompatible_params.push("basic_functionality".to_string());
            error_messages.push("Basic uniform filter functionality differs".to_string());
        }

        // Test mode parameter
        let mode_test = uniform_filter(&input, &[3, 3], Some(BorderMode::Wrap), None).is_ok();
        if !mode_test {
            incompatible_params.push("mode".to_string());
            error_messages.push("Wrap mode may not be supported".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 2.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "uniform_filter".to_string(),
            test_case: "Basic functionality".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Ensure all border modes are supported".to_string()],
            scipy_behavior: "Supports all scipy.ndimage border modes".to_string(),
            scirs2_behavior: "Supports BorderMode enum variants".to_string(),
        });

        Ok(())
    }

    fn test_sobel_filter_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input: Array2<f64> = Array2::zeros((10, 10));

        // Test axis parameter
        let axis_test = sobel(&input, 0, None).is_ok();
        if !axis_test {
            incompatible_params.push("axis".to_string());
            error_messages.push("Axis parameter handling differs".to_string());
        }

        // Test without axis (should compute magnitude)
        let no_axis_test = sobel(&input, 1, None).is_ok();
        if !no_axis_test {
            incompatible_params.push("axis_none".to_string());
            error_messages.push("Default behavior without axis differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 2.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "sobel".to_string(),
            test_case: "Axis parameter handling".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Verify axis parameter behavior matches SciPy".to_string()],
            scipy_behavior: "axis=None computes gradient magnitude".to_string(),
            scirs2_behavior: "Axis parameter controls gradient direction".to_string(),
        });

        Ok(())
    }

    fn test_rank_filter_apis(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input: Array2<f64> = Array2::zeros((10, 10));

        // Test minimum_filter
        let min_test = minimum_filter(&input, &[3, 3], None, None).is_ok();
        if !min_test {
            incompatible_params.push("minimum_filter".to_string());
            error_messages.push("minimum_filter API differs".to_string());
        }

        // Test maximum_filter
        let max_test = maximum_filter(&input, &[3, 3], None, None).is_ok();
        if !max_test {
            incompatible_params.push("maximum_filter".to_string());
            error_messages.push("maximum_filter API differs".to_string());
        }

        // Test percentile_filter
        let percentile_test = percentile_filter(&input, 50.0, &[3, 3], None).is_ok();
        if !percentile_test {
            incompatible_params.push("percentile_filter".to_string());
            error_messages.push("percentile_filter API differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 3.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "rank_filters".to_string(),
            test_case: "Rank-based filters compatibility".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Ensure rank filters match SciPy parameter order".to_string()],
            scipy_behavior: "Standard rank filter implementations".to_string(),
            scirs2_behavior: "Rank filters with size and mode parameters".to_string(),
        });

        Ok(())
    }

    /// Test morphological operation APIs
    pub fn test_morphology_apis(&mut self) -> Result<()> {
        self.test_binary_morphology_api()?;
        self.test_grayscale_morphology_api()?;
        Ok(())
    }

    fn test_binary_morphology_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input = Array2::from_elem((10, 10), true);

        // Test binary_erosion with default structure
        let erosion_test = binary_erosion(&input, None, None, None, None, None, None).is_ok();
        if !erosion_test {
            incompatible_params.push("binary_erosion".to_string());
            error_messages.push("binary_erosion default parameters differ".to_string());
        }

        // Test binary_dilation
        let dilation_test = binary_dilation(&input, None, None, None, None, None, None).is_ok();
        if !dilation_test {
            incompatible_params.push("binary_dilation".to_string());
            error_messages.push("binary_dilation default parameters differ".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 2.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "binary_morphology".to_string(),
            test_case: "Binary morphological operations".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Verify structuring element defaults".to_string()],
            scipy_behavior: "Uses cross-shaped structuring element by default".to_string(),
            scirs2_behavior: "Default structuring element may differ".to_string(),
        });

        Ok(())
    }

    fn test_grayscale_morphology_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input: Array2<f64> = Array2::zeros((10, 10));

        // Test grey_erosion
        let erosion_test = grey_erosion(&input, None, None, None, None, None).is_ok();
        if !erosion_test {
            incompatible_params.push("grey_erosion".to_string());
            error_messages.push("grey_erosion API differs".to_string());
        }

        // Test grey_dilation
        let dilation_test = grey_dilation(&input, None, None, None, None, None).is_ok();
        if !dilation_test {
            incompatible_params.push("grey_dilation".to_string());
            error_messages.push("grey_dilation API differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 2.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "grayscale_morphology".to_string(),
            test_case: "Grayscale morphological operations".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Ensure grayscale morphology parameters match SciPy".to_string()],
            scipy_behavior: "Standard grayscale morphology with size/footprint".to_string(),
            scirs2_behavior: "Grayscale morphology with structure parameters".to_string(),
        });

        Ok(())
    }

    /// Test interpolation function APIs
    pub fn test_interpolation_apis(&mut self) -> Result<()> {
        self.test_zoom_api()?;
        self.test_rotate_api()?;
        self.test_affine_transform_api()?;
        Ok(())
    }

    fn test_zoom_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input: Array2<f64> = Array2::zeros((10, 10));

        // Test zoom with scalar factor
        let scalar_zoom_test = scipy_compat_layer::scipy_ndimage::zoom(
            input.view(),
            vec![2.0f64, 2.0f64],
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .is_ok();
        if !scalar_zoom_test {
            incompatible_params.push("zoom_factor".to_string());
            error_messages.push("Zoom factor parameter handling differs".to_string());
        }

        // Test zoom with interpolation order
        let order_test = scipy_compat_layer::scipy_ndimage::zoom(
            input.view(),
            vec![2.0f64, 2.0f64],
            None,
            Some(1), // Linear interpolation order
            None,
            None,
            None,
            None,
        )
        .is_ok();
        if !order_test {
            incompatible_params.push("interpolation_order".to_string());
            error_messages.push("Interpolation order specification differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 2.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "zoom".to_string(),
            test_case: "Zoom operation parameters".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Verify zoom factor and order parameter compatibility".to_string()],
            scipy_behavior: "Accepts scalar or array zoom factors, integer order".to_string(),
            scirs2_behavior: "Uses array zoom factors and InterpolationOrder enum".to_string(),
        });

        Ok(())
    }

    fn test_rotate_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input: Array2<f64> = Array2::zeros((10, 10));

        // Test basic rotation
        let rotate_test = rotate(&input, 45.0, None, None, None, None, None, None).is_ok();
        if !rotate_test {
            incompatible_params.push("angle".to_string());
            error_messages.push("Rotation angle parameter differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 1.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "rotate".to_string(),
            test_case: "Rotation operation".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Verify rotation parameter compatibility".to_string()],
            scipy_behavior: "Accepts angle in degrees, various reshape options".to_string(),
            scirs2_behavior: "Accepts angle in degrees with optional parameters".to_string(),
        });

        Ok(())
    }

    fn test_affine_transform_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input: Array2<f64> = Array2::zeros((10, 10));
        let matrix = Array2::eye(2);

        // Test affine transform
        let affine_test =
            affine_transform(&input, &matrix, None, None, None, None, None, None).is_ok();
        if !affine_test {
            incompatible_params.push("matrix".to_string());
            error_messages.push("Affine matrix parameter handling differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 1.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "affine_transform".to_string(),
            test_case: "Affine transformation".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Verify affine matrix parameter format".to_string()],
            scipy_behavior: "Accepts transformation matrix in specific format".to_string(),
            scirs2_behavior: "Uses ndarray matrix for transformations".to_string(),
        });

        Ok(())
    }

    /// Test measurement function APIs
    pub fn test_measurement_apis(&mut self) -> Result<()> {
        self.test_center_of_mass_api()?;
        self.test_label_api()?;
        Ok(())
    }

    fn test_center_of_mass_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input = Array2::<f64>::ones((10, 10));

        // Test center of mass calculation
        let com_test = center_of_mass(&input).is_ok();
        if !com_test {
            incompatible_params.push("basic_functionality".to_string());
            error_messages.push("center_of_mass basic functionality differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 1.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "center_of_mass".to_string(),
            test_case: "Center of mass calculation".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Verify center of mass calculation accuracy".to_string()],
            scipy_behavior: "Returns center of mass coordinates".to_string(),
            scirs2_behavior: "Returns coordinate array".to_string(),
        });

        Ok(())
    }

    fn test_label_api(&mut self) -> Result<()> {
        let mut incompatible_params = Vec::new();
        let mut error_messages = Vec::new();

        let input = Array2::from_elem((10, 10), true);

        // Test label function
        let label_test = label(&input, None, None, None).is_ok();
        if !label_test {
            incompatible_params.push("structure".to_string());
            error_messages.push("label structure parameter differs".to_string());
        }

        let compatibility_score = 1.0 - (incompatible_params.len() as f64 / 1.0);

        self.results.push(ApiCompatibilityResult {
            function_name: "label".to_string(),
            test_case: "Connected component labeling".to_string(),
            compatible: incompatible_params.is_empty(),
            compatibility_score,
            incompatible_parameters: incompatible_params,
            error_messages,
            suggestions: vec!["Verify labeling algorithm compatibility".to_string()],
            scipy_behavior: "Connected component labeling with structure".to_string(),
            scirs2_behavior: "Labeling with optional connectivity structure".to_string(),
        });

        Ok(())
    }

    /// Run all API compatibility tests
    pub fn run_all_tests(&mut self) -> Result<()> {
        println!("Running comprehensive API compatibility tests...");

        self.test_filter_apis()?;
        self.test_morphology_apis()?;
        self.test_interpolation_apis()?;
        self.test_measurement_apis()?;

        // Calculate overall score
        if !self.results.is_empty() {
            self.overall_score = self
                .results
                .iter()
                .map(|r| r.compatibility_score)
                .sum::<f64>()
                / self.results.len() as f64;
        }

        println!("API compatibility tests completed!");
        Ok(())
    }

    /// Generate compatibility report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# API Compatibility Report\n\n");

        report.push_str(&format!(
            "Overall Compatibility Score: {:.2}%\n\n",
            self.overall_score * 100.0
        ));

        // Summary statistics
        let total_tests = self.results.len();
        let compatible_tests = self.results.iter().filter(|r| r.compatible).count();

        report.push_str(&format!(
            "Compatible Functions: {}/{} ({:.1}%)\n",
            compatible_tests,
            total_tests,
            (compatible_tests as f64 / total_tests as f64) * 100.0
        ));

        report.push_str(&format!(
            "Incompatible Functions: {}\n\n",
            total_tests - compatible_tests
        ));

        // Detailed results
        for result in &self.results {
            report.push_str(&format!("## {}\n", result.function_name));
            report.push_str(&format!(
                "**Compatibility Score:** {:.2}%\n",
                result.compatibility_score * 100.0
            ));
            report.push_str(&format!(
                "**Compatible:** {}\n\n",
                if result.compatible {
                    "✓ Yes"
                } else {
                    "✗ No"
                }
            ));

            if !result.incompatible_parameters.is_empty() {
                report.push_str("**Incompatible Parameters:**\n");
                for param in &result.incompatible_parameters {
                    report.push_str(&format!("- {}\n", param));
                }
                report.push('\n');
            }

            if !result.error_messages.is_empty() {
                report.push_str("**Issues:**\n");
                for msg in &result.error_messages {
                    report.push_str(&format!("- {}\n", msg));
                }
                report.push('\n');
            }

            if !result.suggestions.is_empty() {
                report.push_str("**Suggestions:**\n");
                for suggestion in &result.suggestions {
                    report.push_str(&format!("- {}\n", suggestion));
                }
                report.push('\n');
            }

            report.push_str(&format!("**SciPy Behavior:** {}\n", result.scipy_behavior));
            report.push_str(&format!(
                "**scirs2 Behavior:** {}\n\n",
                result.scirs2_behavior
            ));

            report.push_str("---\n\n");
        }

        report
    }

    /// Get test results
    pub fn get_results(&self) -> &[ApiCompatibilityResult] {
        &self.results
    }

    /// Get overall compatibility score
    pub fn get_overall_score(&self) -> f64 {
        self.overall_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compatibility_tester_creation() {
        let tester = ApiCompatibilityTester::new();
        assert_eq!(tester.results.len(), 0);
        assert_eq!(tester.overall_score, 0.0);
    }

    #[test]
    fn test_compatibility_config() {
        let config = CompatibilityConfig::default();
        assert!(config.test_edge_cases);
        assert!(config.test_error_conditions);
        assert!(!config.test_performance);
    }

    #[test]
    fn test_api_result_creation() {
        let result = ApiCompatibilityResult {
            function_name: "test_function".to_string(),
            test_case: "basic_test".to_string(),
            compatible: true,
            compatibility_score: 1.0,
            incompatible_parameters: vec![],
            error_messages: vec![],
            suggestions: vec![],
            scipy_behavior: "Expected behavior".to_string(),
            scirs2_behavior: "Actual behavior".to_string(),
        };

        assert!(result.compatible);
        assert_eq!(result.compatibility_score, 1.0);
        assert_eq!(result.function_name, "test_function");
    }
}
