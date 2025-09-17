//! Documentation Generation System
//!
//! This module provides a comprehensive documentation generation system for SciRS2 NDImage,
//! including HTML generation, tutorials, examples, API documentation, and styling.
//!
//! This module has been refactored into focused components for better maintainability.
//! See the submodules for specific functionality.
//!
//! # Features
//!
//! - **Comprehensive API Documentation**: Auto-generated documentation for all modules and functions
//! - **Interactive Tutorials**: Step-by-step guides with executable examples
//! - **Code Examples**: Real-world examples for different domains (medical, satellite, scientific)
//! - **Modern Web Interface**: Responsive design with search functionality
//! - **Syntax Highlighting**: Code syntax highlighting with Prism.js
//! - **Mobile-Friendly**: Responsive design that works on all devices
//!
//! # Usage
//!
//! ```rust
//! use scirs2_ndimage::documentation::DocumentationSite;
//!
//! let mut site = DocumentationSite::new();
//! site.build_comprehensive_documentation()?;
//! site.generate_html_documentation("./docs")?;
//! ```

// Re-export all module components for backward compatibility
pub use self::{
    html_generation::*, modules::*, styling::*, tutorials::*, types::*,
};

// Module declarations
pub mod html_generation;
pub mod modules;
pub mod styling;
pub mod tutorials;
pub mod types;

// Import for conditional compilation
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Quick builder function for creating a complete documentation site
pub fn create_documentation_site() -> types::Result<DocumentationSite> {
    let mut site = DocumentationSite::new();
    site.build_comprehensive_documentation()?;
    Ok(site)
}

/// Generate complete HTML documentation to a directory
pub fn generate_complete_documentation(output_dir: &str) -> types::Result<()> {
    let site = create_documentation_site()?;
    site.generate_html_documentation(output_dir)?;
    Ok(())
}

/// Builder pattern for customizing documentation generation
pub struct DocumentationBuilder {
    site: DocumentationSite,
    include_tutorials: bool,
    include_examples: bool,
    include_search: bool,
    custom_css: Option<String>,
    custom_js: Option<String>,
}

impl DocumentationBuilder {
    /// Create a new documentation builder
    pub fn new() -> Self {
        Self {
            site: DocumentationSite::new(),
            include_tutorials: true,
            include_examples: true,
            include_search: true,
            custom_css: None,
            custom_js: None,
        }
    }

    /// Set the site title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.site.title = title.into();
        self
    }

    /// Set the site description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.site.description = description.into();
        self
    }

    /// Set the site version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.site.version = version.into();
        self
    }

    /// Set the base URL
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.site.base_url = url.into();
        self
    }

    /// Include or exclude tutorials
    pub fn with_tutorials(mut self, include: bool) -> Self {
        self.include_tutorials = include;
        self
    }

    /// Include or exclude examples
    pub fn with_examples(mut self, include: bool) -> Self {
        self.include_examples = include;
        self
    }

    /// Include or exclude search functionality
    pub fn with_search(mut self, include: bool) -> Self {
        self.include_search = include;
        self
    }

    /// Add custom CSS
    pub fn custom_css(mut self, css: impl Into<String>) -> Self {
        self.custom_css = Some(css.into());
        self
    }

    /// Add custom JavaScript
    pub fn custom_js(mut self, js: impl Into<String>) -> Self {
        self.custom_js = Some(js.into());
        self
    }

    /// Build the documentation site
    pub fn build(mut self) -> types::Result<DocumentationSite> {
        // Build modules (always included)
        self.site.build_module_documentation()?;

        // Build tutorials if requested
        if self.include_tutorials {
            self.site.build_tutorials()?;
        }

        // Build examples if requested
        if self.include_examples {
            self.site.build_examples()?;
        }

        Ok(self.site)
    }

    /// Build and generate HTML documentation
    pub fn generate(self, output_dir: &str) -> types::Result<()> {
        let site = self.build()?;
        site.generate_html_documentation(output_dir)?;
        Ok(())
    }
}

impl Default for DocumentationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for common documentation tasks
pub mod utils {
    use super::*;

    /// Extract function signatures from Rust code
    pub fn extract_function_signatures(rust_code: &str) -> Vec<String> {
        let mut signatures = Vec::new();

        // Simple regex-based extraction (in practice, you'd want a proper parser)
        if let Ok(re) = regex::Regex::new(r"pub fn (\w+).*?{") {
            for cap in re.captures_iter(rust_code) {
                if let Some(sig) = cap.get(0) {
                    signatures.push(sig.as_str().to_string());
                }
            }
        }

        signatures
    }

    /// Generate API documentation from source files
    pub fn generate_api_from_source(source_dir: &str) -> types::Result<Vec<types::ModuleDoc>> {
        use std::fs;
        use std::path::Path;

        let mut modules = Vec::new();
        let source_path = Path::new(source_dir);

        if source_path.exists() {
            for entry in fs::read_dir(source_path)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    let content = fs::read_to_string(&path)?;
                    let module_name = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let mut module = types::ModuleDoc::new(
                        module_name,
                        "Auto-generated documentation"
                    );

                    // Extract functions (simplified)
                    let signatures = extract_function_signatures(&content);
                    for sig in signatures {
                        let func = types::FunctionDoc::new(
                            "extracted_function",
                            sig,
                            "Auto-extracted function",
                            "Return type"
                        );
                        module.add_function(func);
                    }

                    modules.push(module);
                }
            }
        }

        Ok(modules)
    }

    /// Validate HTML output for common issues
    pub fn validate_html_output(html: &str) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for unclosed tags (simplified)
        let open_tags = html.matches('<').count();
        let close_tags = html.matches("</").count() + html.matches("/>").count();

        if open_tags != close_tags {
            issues.push("Potential unclosed HTML tags detected".to_string());
        }

        // Check for missing alt attributes on images
        if html.contains("<img") && !html.contains("alt=") {
            issues.push("Images missing alt attributes".to_string());
        }

        // Check for missing title
        if !html.contains("<title>") {
            issues.push("HTML missing title tag".to_string());
        }

        issues
    }

    /// Generate sitemap.xml for the documentation
    pub fn generate_sitemap(base_url: &str, pages: &[&str]) -> String {
        let mut sitemap = String::from(r#"<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">"#);

        for page in pages {
            sitemap.push_str(&format!(
                r#"
  <url>
    <loc>{}{}</loc>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>"#,
                base_url.trim_end_matches('/'),
                page
            ));
        }

        sitemap.push_str("\n</urlset>");
        sitemap
    }
}

/// Configuration for documentation themes and styling
#[derive(Debug, Clone)]
pub struct DocumentationTheme {
    /// Primary color (hex)
    pub primary_color: String,
    /// Secondary color (hex)
    pub secondary_color: String,
    /// Background color (hex)
    pub background_color: String,
    /// Text color (hex)
    pub text_color: String,
    /// Font family
    pub font_family: String,
    /// Code font family
    pub code_font_family: String,
}

impl Default for DocumentationTheme {
    fn default() -> Self {
        Self {
            primary_color: "#3498db".to_string(),
            secondary_color: "#2c3e50".to_string(),
            background_color: "#f8f9fa".to_string(),
            text_color: "#333333".to_string(),
            font_family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif".to_string(),
            code_font_family: "Consolas, Monaco, 'Andale Mono', monospace".to_string(),
        }
    }
}

impl DocumentationTheme {
    /// Create a dark theme variant
    pub fn dark() -> Self {
        Self {
            primary_color: "#61dafb".to_string(),
            secondary_color: "#282c34".to_string(),
            background_color: "#20232a".to_string(),
            text_color: "#ffffff".to_string(),
            ..Default::default()
        }
    }

    /// Create a minimal theme variant
    pub fn minimal() -> Self {
        Self {
            primary_color: "#000000".to_string(),
            secondary_color: "#666666".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#333333".to_string(),
            ..Default::default()
        }
    }

    /// Generate CSS variables for this theme
    pub fn to_css_variables(&self) -> String {
        format!(
            r#":root {{
    --primary-color: {};
    --secondary-color: {};
    --background-color: {};
    --text-color: {};
    --font-family: {};
    --code-font-family: {};
}}"#,
            self.primary_color,
            self.secondary_color,
            self.background_color,
            self.text_color,
            self.font_family,
            self.code_font_family
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_documentation_site() {
        let result = create_documentation_site();
        assert!(result.is_ok());

        let site = result.unwrap();
        assert_eq!(site.title, "SciRS2 NDImage Documentation");
        assert!(!site.modules.is_empty());
        assert!(!site.tutorials.is_empty());
        assert!(!site.examples.is_empty());
    }

    #[test]
    fn test_documentation_builder() {
        let site = DocumentationBuilder::new()
            .title("Custom Documentation")
            .description("Custom description")
            .version("1.0.0")
            .with_tutorials(true)
            .with_examples(false)
            .build()
            .unwrap();

        assert_eq!(site.title, "Custom Documentation");
        assert_eq!(site.description, "Custom description");
        assert_eq!(site.version, "1.0.0");
        assert!(!site.tutorials.is_empty());
        assert!(site.examples.is_empty()); // Examples disabled
    }

    #[test]
    fn test_documentation_theme() {
        let default_theme = DocumentationTheme::default();
        assert_eq!(default_theme.primary_color, "#3498db");

        let dark_theme = DocumentationTheme::dark();
        assert_eq!(dark_theme.background_color, "#20232a");

        let css_vars = default_theme.to_css_variables();
        assert!(css_vars.contains("--primary-color"));
        assert!(css_vars.contains("--text-color"));
    }

    #[test]
    fn test_utils_extract_functions() {
        let rust_code = r#"
            pub fn example_function() {
                // Some code
            }

            pub fn another_function(param: i32) -> String {
                // More code
            }
        "#;

        let signatures = utils::extract_function_signatures(rust_code);
        assert_eq!(signatures.len(), 2);
        assert!(signatures[0].contains("example_function"));
        assert!(signatures[1].contains("another_function"));
    }

    #[test]
    fn test_utils_validate_html() {
        let good_html = "<html><head><title>Test</title></head><body><img src='test.png' alt='test'/></body></html>";
        let issues = utils::validate_html_output(good_html);
        assert!(issues.is_empty());

        let bad_html = "<html><head></head><body><img src='test.png'/></body>";
        let issues = utils::validate_html_output(bad_html);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_utils_generate_sitemap() {
        let pages = vec!["/index.html", "/api/index.html", "/tutorials/getting-started.html"];
        let sitemap = utils::generate_sitemap("https://example.com", &pages);

        assert!(sitemap.contains("<?xml version"));
        assert!(sitemap.contains("https://example.com/index.html"));
        assert!(sitemap.contains("https://example.com/api/index.html"));
        assert!(sitemap.contains("https://example.com/tutorials/getting-started.html"));
    }

    #[test]
    fn test_module_integration() {
        let mut site = DocumentationSite::new();

        // Test that all modules integrate properly
        assert!(site.build_module_documentation().is_ok());
        assert!(site.build_tutorials().is_ok());
        assert!(site.build_examples().is_ok());

        // Verify content was built
        assert!(!site.modules.is_empty());
        assert!(!site.tutorials.is_empty());
        assert!(!site.examples.is_empty());

        // Test HTML generation methods exist
        let _ = site.generate_module_cards();
        let _ = site.generate_api_module_list();
        let _ = site.generate_tutorial_cards();
        let _ = site.generate_default_css();
        let _ = site.generate_default_js();
    }
}