//! Core Documentation Types and Structures
//!
//! This module contains all the fundamental data structures used for documentation generation,
//! including site structure, module documentation, function documentation, tutorials, and examples.

use std::collections::HashMap;

/// Result type alias for documentation operations
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Main documentation site structure
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentationSite {
    /// Site title
    pub title: String,
    /// Site description
    pub description: String,
    /// Documentation version
    pub version: String,
    /// Base URL for the documentation
    pub base_url: String,
    /// Module documentation entries
    pub modules: Vec<ModuleDoc>,
    /// Tutorial entries
    pub tutorials: Vec<Tutorial>,
    /// Code examples
    pub examples: Vec<Example>,
}

/// Documentation for a single module
#[derive(Debug, Clone)]
pub struct ModuleDoc {
    /// Module name
    pub name: String,
    /// Module description
    pub description: String,
    /// Function documentation for this module
    pub functions: Vec<FunctionDoc>,
    /// Module-level examples
    pub examples: Vec<String>,
}

/// Documentation for a single function
#[derive(Debug, Clone)]
pub struct FunctionDoc {
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: String,
    /// Function description
    pub description: String,
    /// Function parameters
    pub parameters: Vec<Parameter>,
    /// Return type and description
    pub returns: String,
    /// Usage examples
    pub examples: Vec<String>,
    /// Additional notes
    pub notes: Vec<String>,
}

/// Function parameter documentation
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Parameter description
    pub description: String,
    /// Whether parameter is optional
    pub optional: bool,
}

/// Tutorial content structure
#[derive(Debug, Clone)]
pub struct Tutorial {
    /// Tutorial title
    pub title: String,
    /// Tutorial description
    pub description: String,
    /// Tutorial content (markdown format)
    pub content: String,
    /// Code examples within the tutorial
    pub code_examples: Vec<String>,
    /// Difficulty level (Beginner, Intermediate, Advanced)
    pub difficulty: String,
}

/// Code example structure
#[derive(Debug, Clone)]
pub struct Example {
    /// Example title
    pub title: String,
    /// Example description
    pub description: String,
    /// Example code
    pub code: String,
    /// Expected output (if applicable)
    pub expected_output: Option<String>,
    /// Example category
    pub category: String,
}

impl DocumentationSite {
    /// Create a new documentation site with default values
    pub fn new() -> Self {
        Self {
            title: "SciRS2 NDImage Documentation".to_string(),
            description:
                "Comprehensive documentation for SciRS2 N-dimensional image processing library"
                    .to_string(),
            version: "0.1.0-beta.2".to_string(),
            base_url: "https://scirs2.github.io/ndimage".to_string(),
            modules: Vec::new(),
            tutorials: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// Build comprehensive documentation by calling all builders
    pub fn build_comprehensive_documentation(&mut self) -> Result<()> {
        self.build_module_documentation()?;
        self.build_tutorials()?;
        self.build_examples()?;
        Ok(())
    }
}

impl Default for DocumentationSite {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDoc {
    /// Create a new module documentation entry
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            functions: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// Add a function to this module's documentation
    pub fn add_function(&mut self, function: FunctionDoc) {
        self.functions.push(function);
    }

    /// Add an example to this module
    pub fn add_example(&mut self, example: impl Into<String>) {
        self.examples.push(example.into());
    }
}

impl FunctionDoc {
    /// Create a new function documentation entry
    pub fn new(
        name: impl Into<String>,
        signature: impl Into<String>,
        description: impl Into<String>,
        returns: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            signature: signature.into(),
            description: description.into(),
            returns: returns.into(),
            parameters: Vec::new(),
            examples: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Add a parameter to this function
    pub fn add_parameter(&mut self, parameter: Parameter) {
        self.parameters.push(parameter);
    }

    /// Add an example to this function
    pub fn add_example(&mut self, example: impl Into<String>) {
        self.examples.push(example.into());
    }

    /// Add a note to this function
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
}

impl Parameter {
    /// Create a new parameter documentation entry
    pub fn new(
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
        optional: bool,
    ) -> Self {
        Self {
            name: name.into(),
            param_type: param_type.into(),
            description: description.into(),
            optional,
        }
    }

    /// Create a required parameter
    pub fn required(
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self::new(name, param_type, description, false)
    }

    /// Create an optional parameter
    pub fn optional(
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self::new(name, param_type, description, true)
    }
}

impl Tutorial {
    /// Create a new tutorial
    pub fn new(
        title: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
        difficulty: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            content: content.into(),
            difficulty: difficulty.into(),
            code_examples: Vec::new(),
        }
    }

    /// Add a code example to this tutorial
    pub fn add_code_example(&mut self, example: impl Into<String>) {
        self.code_examples.push(example.into());
    }

    /// Create a beginner tutorial
    pub fn beginner(
        title: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::new(title, description, content, "Beginner")
    }

    /// Create an intermediate tutorial
    pub fn intermediate(
        title: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::new(title, description, content, "Intermediate")
    }

    /// Create an advanced tutorial
    pub fn advanced(
        title: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::new(title, description, content, "Advanced")
    }
}

impl Example {
    /// Create a new code example
    pub fn new(
        title: impl Into<String>,
        description: impl Into<String>,
        code: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            code: code.into(),
            category: category.into(),
            expected_output: None,
        }
    }

    /// Create a new code example with expected output
    pub fn with_output(
        title: impl Into<String>,
        description: impl Into<String>,
        code: impl Into<String>,
        category: impl Into<String>,
        expected_output: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            code: code.into(),
            category: category.into(),
            expected_output: Some(expected_output.into()),
        }
    }

    /// Set the expected output for this example
    pub fn set_expected_output(&mut self, output: impl Into<String>) {
        self.expected_output = Some(output.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_documentation_site_creation() {
        let site = DocumentationSite::new();
        assert_eq!(site.title, "SciRS2 NDImage Documentation");
        assert_eq!(site.version, "0.1.0-beta.2");
        assert!(site.modules.is_empty());
        assert!(site.tutorials.is_empty());
        assert!(site.examples.is_empty());
    }

    #[test]
    fn test_module_doc_creation() {
        let mut module = ModuleDoc::new("filters", "Image filtering operations");
        assert_eq!(module.name, "filters");
        assert_eq!(module.description, "Image filtering operations");
        assert!(module.functions.is_empty());

        module.add_example("Basic filtering example");
        assert_eq!(module.examples.len(), 1);
    }

    #[test]
    fn test_function_doc_creation() {
        let mut func = FunctionDoc::new(
            "gaussian_filter",
            "pub fn gaussian_filter<T>(input: &ArrayD<T>, sigma: f64) -> ArrayD<T>",
            "Apply Gaussian filter to n-dimensional array",
            "ArrayD<T> - Filtered array",
        );

        let param = Parameter::required("input", "&ArrayD<T>", "Input n-dimensional array");
        func.add_parameter(param);
        func.add_note("Uses separable convolution for efficiency");

        assert_eq!(func.name, "gaussian_filter");
        assert_eq!(func.parameters.len(), 1);
        assert_eq!(func.notes.len(), 1);
    }

    #[test]
    fn test_parameter_creation() {
        let required_param = Parameter::required("input", "&ArrayD<T>", "Input array");
        assert!(!required_param.optional);

        let optional_param = Parameter::optional("sigma", "f64", "Standard deviation");
        assert!(optional_param.optional);
    }

    #[test]
    fn test_tutorial_creation() {
        let tutorial = Tutorial::beginner(
            "Getting Started",
            "Introduction to image processing",
            "# Getting Started\n\nThis tutorial covers basic operations...",
        );

        assert_eq!(tutorial.title, "Getting Started");
        assert_eq!(tutorial.difficulty, "Beginner");
        assert!(tutorial.code_examples.is_empty());
    }

    #[test]
    fn test_example_creation() {
        let example = Example::new(
            "Basic Filtering",
            "Simple Gaussian filter example",
            "let filtered = gaussian_filter(&image, 2.0);",
            "filters",
        );

        assert_eq!(example.title, "Basic Filtering");
        assert_eq!(example.category, "filters");
        assert!(example.expected_output.is_none());

        let example_with_output = Example::with_output(
            "Math Example",
            "Simple math operation",
            "2 + 2",
            "math",
            "4",
        );

        assert!(example_with_output.expected_output.is_some());
        assert_eq!(example_with_output.expected_output.unwrap(), "4");
    }
}