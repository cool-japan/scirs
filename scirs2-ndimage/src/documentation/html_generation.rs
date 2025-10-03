//! HTML Generation Functionality
//!
//! This module contains functions to generate static HTML documentation
//! including index pages, API documentation, tutorials, and examples.

use crate::documentation::types::{DocumentationSite, FunctionDoc, ModuleDoc, Result, Tutorial};
use std::fs;
use std::io::Write;
use std::path::Path;

impl DocumentationSite {
    /// Generate complete HTML documentation
    pub fn generate_html_documentation(&self, output_dir: &str) -> Result<()> {
        let output_path = Path::new(output_dir);
        fs::create_dir_all(output_path)?;

        // Generate main index page
        self.generate_index_page(output_path)?;

        // Generate API documentation
        self.generate_api_documentation(output_path)?;

        // Generate tutorials
        self.generate_tutorials(output_path)?;

        // Generate examples
        self.generate_examples(output_path)?;

        // Generate search functionality
        self.generate_search_index(output_path)?;

        // Copy CSS and JavaScript files
        self.generate_static_files(output_path)?;

        Ok(())
    }

    /// Generate the main index page
    pub fn generate_index_page(&self, output_path: &Path) -> Result<()> {
        let index_html = format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <link rel="stylesheet" href="static/style.css">
    <link rel="stylesheet" href="static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo">{}</h1>
            <nav class="nav">
                <a href="index.html">Home</a>
                <a href="api/index.html">API Reference</a>
                <a href="tutorials/index.html">Tutorials</a>
                <a href="examples/index.html">Examples</a>
                <a href="search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <section class="hero">
            <div class="container">
                <h2>High-Performance N-Dimensional Image Processing for Rust</h2>
                <p class="hero-description">{}</p>
                <div class="hero-buttons">
                    <a href="tutorials/getting-started.html" class="btn btn-primary">Get Started</a>
                    <a href="api/index.html" class="btn btn-secondary">API Reference</a>
                </div>
            </div>
        </section>

        <section class="features">
            <div class="container">
                <h3>Key Features</h3>
                <div class="feature-grid">
                    <div class="feature-card">
                        <h4>🚀 High Performance</h4>
                        <p>SIMD-optimized algorithms with parallel processing and GPU acceleration support</p>
                    </div>
                    <div class="feature-card">
                        <h4>📐 N-Dimensional</h4>
                        <p>Work seamlessly with 1D, 2D, 3D, and higher-dimensional arrays</p>
                    </div>
                    <div class="feature-card">
                        <h4>🔬 Scientific</h4>
                        <p>Domain-specific functions for medical imaging, satellite analysis, and microscopy</p>
                    </div>
                    <div class="feature-card">
                        <h4>🐍 SciPy Compatible</h4>
                        <p>API compatible with SciPy's ndimage module for easy migration</p>
                    </div>
                    <div class="feature-card">
                        <h4>🛡️ Memory Safe</h4>
                        <p>Rust's ownership system ensures memory safety without runtime overhead</p>
                    </div>
                    <div class="feature-card">
                        <h4>⚡ Zero-Copy</h4>
                        <p>Efficient memory usage with zero-copy operations where possible</p>
                    </div>
                </div>
            </div>
        </section>

        <section class="modules">
            <div class="container">
                <h3>Core Modules</h3>
                <div class="module-grid">
                    {}
                </div>
            </div>
        </section>

        <section class="quick-start">
            <div class="container">
                <h3>Quick Start</h3>
                <pre><code class="language-toml">[dependencies]
scirs2-ndimage = "{}"
ndarray = "0.16"</code></pre>
                <pre><code class="language-rust">use scirs2_ndimage::filters::gaussian_filter;
use scirs2_core::ndarray::Array2;

let image = Array2::from_elem((100, 100), 1.0f64);
let filtered = gaussian_filter(&image, 2.0);
println!("Filtered image shape: {{:?}}", filtered.shape());</code></pre>
            </div>
        </section>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
            <p>Version {} | <a href="https://github.com/scirs2/ndimage">GitHub</a> | <a href="https://docs.rs/scirs2-ndimage">docs.rs</a></p>
        </div>
    </footer>

    <script src="static/prism.js"></script>
    <script src="static/main.js"></script>
</body>
</html>
"#,
            self.title,
            self.title,
            self.description,
            self.generate_module_cards(),
            self.version,
            self.version
        );

        let mut index_file = fs::File::create(output_path.join("index.html"))?;
        index_file.write_all(index_html.as_bytes())?;
        Ok(())
    }

    /// Generate module cards for the index page
    pub fn generate_module_cards(&self) -> String {
        self.modules
            .iter()
            .map(|module| {
                format!(
                    r#"
                <div class="module-card">
                    <h4>{}</h4>
                    <p>{}</p>
                    <div class="module-functions">
                        <span class="function-count">{} functions</span>
                        <a href="api/{}.html" class="module-link">View API →</a>
                    </div>
                </div>
            "#,
                    module.name,
                    module.description,
                    module.functions.len(),
                    module.name.to_lowercase()
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Generate API documentation pages
    pub fn generate_api_documentation(&self, output_path: &Path) -> Result<()> {
        let api_dir = output_path.join("api");
        fs::create_dir_all(&api_dir)?;

        // Generate API index
        let api_index = self.generate_api_index();
        let mut api_index_file = fs::File::create(api_dir.join("index.html"))?;
        api_index_file.write_all(api_index.as_bytes())?;

        // Generate individual module pages
        for module in &self.modules {
            let module_html = self.generate_module_page(module);
            let module_filename = format!("{}.html", module.name.to_lowercase());
            let mut module_file = fs::File::create(api_dir.join(module_filename))?;
            module_file.write_all(module_html.as_bytes())?;
        }

        Ok(())
    }

    /// Generate API index page
    pub fn generate_api_index(&self) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Reference - {}</title>
    <link rel="stylesheet" href="../static/style.css">
    <link rel="stylesheet" href="../static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="index.html" class="active">API Reference</a>
                <a href="../tutorials/index.html">Tutorials</a>
                <a href="../examples/index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>API Reference</h2>
            <p>Complete reference for all modules and functions in SciRS2 NDImage.</p>

            <div class="api-modules">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="../static/prism.js"></script>
</body>
</html>
        "#,
            self.title,
            self.title,
            self.generate_api_module_list()
        )
    }

    /// Generate API module list
    pub fn generate_api_module_list(&self) -> String {
        self.modules
            .iter()
            .map(|module| {
                format!(
                    r#"
                <div class="api-module">
                    <h3><a href="{}.html">{}</a></h3>
                    <p>{}</p>
                    <div class="function-list">
                        {}
                    </div>
                </div>
            "#,
                    module.name.to_lowercase(),
                    module.name,
                    module.description,
                    module
                        .functions
                        .iter()
                        .map(|func| {
                            format!(r#"<span class="function-name">{}</span>"#, func.name)
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Generate individual module page
    pub fn generate_module_page(&self, module: &ModuleDoc) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} Module - {}</title>
    <link rel="stylesheet" href="../static/style.css">
    <link rel="stylesheet" href="../static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="index.html">API Reference</a>
                <a href="../tutorials/index.html">Tutorials</a>
                <a href="../examples/index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>{} Module</h2>
            <p class="module-description">{}</p>

            <div class="functions">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="../static/prism.js"></script>
</body>
</html>
        "#,
            module.name,
            self.title,
            self.title,
            module.name,
            module.description,
            self.generate_function_documentation(&module.functions)
        )
    }

    /// Generate function documentation HTML
    pub fn generate_function_documentation(&self, functions: &[FunctionDoc]) -> String {
        functions
            .iter()
            .map(|func| {
                format!(
                    r#"
                <div class="function">
                    <h3 class="function-name">{}</h3>
                    <pre class="function-signature"><code class="language-rust">{}</code></pre>
                    <p class="function-description">{}</p>

                    <div class="parameters">
                        <h4>Parameters</h4>
                        <ul>
                            {}
                        </ul>
                    </div>

                    <div class="returns">
                        <h4>Returns</h4>
                        <p>{}</p>
                    </div>

                    {}

                    {}
                </div>
            "#,
                    func.name,
                    func.signature,
                    func.description,
                    func.parameters
                        .iter()
                        .map(|param| {
                            format!(
                                r#"<li><code>{}</code> ({}): {}{}</li>"#,
                                param.name,
                                param.param_type,
                                param.description,
                                if param.optional { " (optional)" } else { "" }
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                    func.returns,
                    if !func.examples.is_empty() {
                        format!(
                            r#"
                        <div class="examples">
                            <h4>Examples</h4>
                            {}
                        </div>
                    "#,
                            func.examples
                                .iter()
                                .map(|example| {
                                    format!(
                                        r#"<pre><code class="language-rust">{}</code></pre>"#,
                                        example
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("")
                        )
                    } else {
                        String::new()
                    },
                    if !func.notes.is_empty() {
                        format!(
                            r#"
                        <div class="notes">
                            <h4>Notes</h4>
                            <ul>
                                {}
                            </ul>
                        </div>
                    "#,
                            func.notes
                                .iter()
                                .map(|note| { format!(r#"<li>{}</li>"#, note) })
                                .collect::<Vec<_>>()
                                .join("")
                        )
                    } else {
                        String::new()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Generate tutorial pages
    pub fn generate_tutorials(&self, output_path: &Path) -> Result<()> {
        let tutorials_dir = output_path.join("tutorials");
        fs::create_dir_all(&tutorials_dir)?;

        // Generate tutorials index
        let tutorials_index = self.generate_tutorials_index();
        let mut tutorials_index_file = fs::File::create(tutorials_dir.join("index.html"))?;
        tutorials_index_file.write_all(tutorials_index.as_bytes())?;

        // Generate individual tutorial pages
        for tutorial in &self.tutorials {
            let tutorial_html = self.generate_tutorial_page(tutorial);
            let tutorial_filename =
                format!("{}.html", tutorial.title.to_lowercase().replace(" ", "-"));
            let mut tutorial_file = fs::File::create(tutorials_dir.join(tutorial_filename))?;
            tutorial_file.write_all(tutorial_html.as_bytes())?;
        }

        Ok(())
    }

    /// Generate tutorials index page
    pub fn generate_tutorials_index(&self) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tutorials - {}</title>
    <link rel="stylesheet" href="../static/style.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="../api/index.html">API Reference</a>
                <a href="index.html" class="active">Tutorials</a>
                <a href="../examples/index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>Tutorials</h2>
            <p>Step-by-step guides to master SciRS2 NDImage features.</p>

            <div class="tutorial-grid">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>
</body>
</html>
        "#,
            self.title,
            self.title,
            self.generate_tutorial_cards()
        )
    }

    /// Generate tutorial cards for index
    pub fn generate_tutorial_cards(&self) -> String {
        self.tutorials
            .iter()
            .map(|tutorial| {
                let difficulty_class = match tutorial.difficulty.as_str() {
                    "Beginner" => "difficulty-beginner",
                    "Intermediate" => "difficulty-intermediate",
                    "Advanced" => "difficulty-advanced",
                    _ => "difficulty-beginner",
                };

                format!(
                    r#"
                <div class="tutorial-card">
                    <h3><a href="{}.html">{}</a></h3>
                    <p>{}</p>
                    <div class="tutorial-meta">
                        <span class="difficulty {}">📖 {}</span>
                    </div>
                </div>
            "#,
                    tutorial.title.to_lowercase().replace(" ", "-"),
                    tutorial.title,
                    tutorial.description,
                    difficulty_class,
                    tutorial.difficulty
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Generate individual tutorial page
    pub fn generate_tutorial_page(&self, tutorial: &Tutorial) -> String {
        // Convert markdown-like content to HTML
        let html_content = self.markdown_to_html(&tutorial.content);

        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - {}</title>
    <link rel="stylesheet" href="../static/style.css">
    <link rel="stylesheet" href="../static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="../api/index.html">API Reference</a>
                <a href="index.html">Tutorials</a>
                <a href="../examples/index.html">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <div class="tutorial-content">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="../static/prism.js"></script>
</body>
</html>
        "#,
            tutorial.title, self.title, self.title, html_content
        )
    }

    /// Generate examples pages
    pub fn generate_examples(&self, output_path: &Path) -> Result<()> {
        let examples_dir = output_path.join("examples");
        fs::create_dir_all(&examples_dir)?;

        // Generate examples index
        let examples_index = self.generate_examples_index();
        let mut examples_index_file = fs::File::create(examples_dir.join("index.html"))?;
        examples_index_file.write_all(examples_index.as_bytes())?;

        Ok(())
    }

    /// Generate examples index page
    pub fn generate_examples_index(&self) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Examples - {}</title>
    <link rel="stylesheet" href="../static/style.css">
    <link rel="stylesheet" href="../static/prism.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo"><a href="../index.html">{}</a></h1>
            <nav class="nav">
                <a href="../index.html">Home</a>
                <a href="../api/index.html">API Reference</a>
                <a href="../tutorials/index.html">Tutorials</a>
                <a href="index.html" class="active">Examples</a>
                <a href="../search.html">Search</a>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            <h2>Code Examples</h2>
            <p>Practical examples demonstrating SciRS2 NDImage capabilities.</p>

            <div class="examples-grid">
                {}
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 SciRS2 Project. Licensed under MIT License.</p>
        </div>
    </footer>

    <script src="../static/prism.js"></script>
</body>
</html>
        "#,
            self.title,
            self.title,
            self.generate_example_cards()
        )
    }

    /// Generate example cards
    pub fn generate_example_cards(&self) -> String {
        self.examples
            .iter()
            .map(|example| {
                format!(
                    r#"
                <div class="example-card">
                    <h3>{}</h3>
                    <p>{}</p>
                    <div class="example-meta">
                        <span class="category">{}</span>
                    </div>
                    <pre><code class="language-rust">{}</code></pre>
                    {}
                </div>
            "#,
                    example.title,
                    example.description,
                    example.category,
                    example.code,
                    if let Some(output) = &example.expected_output {
                        format!(r#"<div class="expected-output"><strong>Expected output:</strong> {}</div>"#, output)
                    } else {
                        String::new()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Generate search index for JavaScript search functionality
    pub fn generate_search_index(&self, output_path: &Path) -> Result<()> {
        // Create a simple search index as JSON
        let search_data = format!(
            r#"{{
    "modules": {},
    "functions": {},
    "tutorials": {}
}}"#,
            serde_json::to_string(&self.modules).unwrap_or_default(),
            serde_json::to_string(
                &self
                    .modules
                    .iter()
                    .flat_map(|m| &m.functions)
                    .collect::<Vec<_>>()
            )
            .unwrap_or_default(),
            serde_json::to_string(&self.tutorials).unwrap_or_default()
        );

        let mut search_file = fs::File::create(output_path.join("search-index.json"))?;
        search_file.write_all(search_data.as_bytes())?;

        Ok(())
    }

    /// Generate static files (CSS, JavaScript)
    pub fn generate_static_files(&self, output_path: &Path) -> Result<()> {
        let static_dir = output_path.join("static");
        fs::create_dir_all(&static_dir)?;

        // Note: In a real implementation, you would copy actual CSS and JS files
        // For this example, we'll create placeholder files

        let css_content = "/* CSS styles would go here */";
        let mut css_file = fs::File::create(static_dir.join("style.css"))?;
        css_file.write_all(css_content.as_bytes())?;

        let js_content = "// JavaScript functionality would go here";
        let mut js_file = fs::File::create(static_dir.join("main.js"))?;
        js_file.write_all(js_content.as_bytes())?;

        let prism_css = "/* Prism.js CSS would go here */";
        let mut prism_css_file = fs::File::create(static_dir.join("prism.css"))?;
        prism_css_file.write_all(prism_css.as_bytes())?;

        let prism_js = "// Prism.js would go here";
        let mut prism_js_file = fs::File::create(static_dir.join("prism.js"))?;
        prism_js_file.write_all(prism_js.as_bytes())?;

        Ok(())
    }

    /// Simple markdown to HTML conversion
    pub fn markdown_to_html(&self, markdown: &str) -> String {
        let mut html = markdown.to_string();

        // Headers
        html = html.replace("# ", "<h1>").replace("\n\n", "</h1>\n\n");
        html = html.replace("## ", "<h2>").replace("\n\n", "</h2>\n\n");
        html = html.replace("### ", "<h3>").replace("\n\n", "</h3>\n\n");

        // Code blocks (simplified)
        if html.contains("```") {
            // For now, just wrap code blocks in <pre><code>
            html = html.replace("```rust\n", "<pre><code class=\"language-rust\">");
            html = html.replace("```toml\n", "<pre><code class=\"language-toml\">");
            html = html.replace("```\n", "</code></pre>");
        }

        // Inline code
        html = html.replace("`", "<code>").replace("`", "</code>");

        // Paragraphs (simple implementation)
        html = html.replace("\n\n", "</p><p>");
        if !html.starts_with("<h") && !html.starts_with("<pre") {
            html = format!("<p>{}", html);
        }
        if !html.ends_with("</h>") && !html.ends_with("</pre>") {
            html = format!("{}</p>", html);
        }

        html
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documentation::types::{FunctionDoc, Parameter, Tutorial};

    #[test]
    fn test_module_cards_generation() {
        let mut site = DocumentationSite::new();
        site.modules = vec![ModuleDoc {
            name: "Filters".to_string(),
            description: "Image filtering operations".to_string(),
            functions: vec![FunctionDoc::new(
                "gaussian_filter",
                "pub fn gaussian_filter<T>(...)",
                "Apply Gaussian filter",
                "ArrayD<T>",
            )],
            examples: vec![],
        }];

        let cards = site.generate_module_cards();
        assert!(cards.contains("Filters"));
        assert!(cards.contains("Image filtering operations"));
        assert!(cards.contains("1 functions"));
    }

    #[test]
    fn test_function_documentation_generation() {
        let site = DocumentationSite::new();
        let func = FunctionDoc {
            name: "test_func".to_string(),
            signature: "pub fn test_func(x: i32) -> i32".to_string(),
            description: "A test function".to_string(),
            parameters: vec![Parameter::required("x", "i32", "Input value")],
            returns: "i32 - Output value".to_string(),
            examples: vec!["let result = test_func(42);".to_string()],
            notes: vec!["This is a test function".to_string()],
        };

        let html = site.generate_function_documentation(&[func]);
        assert!(html.contains("test_func"));
        assert!(html.contains("A test function"));
        assert!(html.contains("Input value"));
        assert!(html.contains("This is a test function"));
    }

    #[test]
    fn test_tutorial_cards_generation() {
        let mut site = DocumentationSite::new();
        site.tutorials = vec![Tutorial::beginner(
            "Getting Started",
            "Introduction to the library",
            "# Getting Started\n\nWelcome to the tutorial.",
        )];

        let cards = site.generate_tutorial_cards();
        assert!(cards.contains("Getting Started"));
        assert!(cards.contains("Introduction to the library"));
        assert!(cards.contains("difficulty-beginner"));
    }

    #[test]
    fn test_markdown_to_html() {
        let site = DocumentationSite::new();

        // Test header conversion
        let markdown = "# Title\n\nSome content";
        let html = site.markdown_to_html(markdown);
        assert!(html.contains("<h1>Title"));

        // Test code block conversion
        let markdown_with_code = "```rust\nlet x = 42;\n```";
        let html_with_code = site.markdown_to_html(markdown_with_code);
        assert!(html_with_code.contains("<pre><code class=\"language-rust\">"));
    }

    #[test]
    fn test_api_module_list_generation() {
        let mut site = DocumentationSite::new();
        site.modules = vec![ModuleDoc {
            name: "TestModule".to_string(),
            description: "A test module".to_string(),
            functions: vec![
                FunctionDoc::new("func1", "signature1", "desc1", "ret1"),
                FunctionDoc::new("func2", "signature2", "desc2", "ret2"),
            ],
            examples: vec![],
        }];

        let list = site.generate_api_module_list();
        assert!(list.contains("TestModule"));
        assert!(list.contains("A test module"));
        assert!(list.contains("func1"));
        assert!(list.contains("func2"));
    }
}