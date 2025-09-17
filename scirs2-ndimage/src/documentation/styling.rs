//! CSS Styling and JavaScript Resources
//!
//! This module contains all the CSS styles, JavaScript functionality,
//! and static resources needed for the documentation website.

use crate::documentation::types::{DocumentationSite, Result};

impl DocumentationSite {
    /// Generate default CSS styles for the documentation
    pub fn generate_default_css(&self) -> String {
        r#"
/* Reset and base styles */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background: #f8f9fa;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Header */
.header {
    background: #2c3e50;
    color: white;
    padding: 1rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
}

.logo a {
    color: white;
    text-decoration: none;
}

.nav {
    display: flex;
    gap: 2rem;
}

.nav a {
    color: white;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    transition: background-color 0.3s;
}

.nav a:hover,
.nav a.active {
    background-color: #34495e;
}

/* Main content */
.main {
    min-height: calc(100vh - 120px);
    padding: 2rem 0;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 4rem 0;
    text-align: center;
}

.hero h2 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.hero-description {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    opacity: 0.9;
}

.hero-buttons {
    display: flex;
    gap: 1rem;
    justify-content: center;
}

.btn {
    padding: 0.75rem 1.5rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 500;
    transition: all 0.3s;
}

.btn-primary {
    background: #3498db;
    color: white;
}

.btn-primary:hover {
    background: #2980b9;
}

.btn-secondary {
    background: transparent;
    color: white;
    border: 2px solid white;
}

.btn-secondary:hover {
    background: white;
    color: #667eea;
}

/* Feature grid */
.features {
    padding: 4rem 0;
    background: #f8f9fa;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.feature-card {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.feature-card h4 {
    font-size: 1.2rem;
    margin-bottom: 1rem;
    color: #2c3e50;
}

/* Module grid */
.modules {
    padding: 4rem 0;
}

.module-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.module-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1.5rem;
    background: white;
}

.module-card h4 {
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.module-functions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #eee;
}

.function-count {
    color: #666;
    font-size: 0.9rem;
}

.module-link {
    color: #3498db;
    text-decoration: none;
    font-weight: 500;
}

.module-link:hover {
    text-decoration: underline;
}

/* Quick start */
.quick-start {
    background: #2c3e50;
    color: white;
    padding: 3rem 0;
}

.quick-start pre {
    background: #34495e;
    padding: 1rem;
    border-radius: 6px;
    margin: 1rem 0;
    overflow-x: auto;
}

/* Footer */
.footer {
    background: #2c3e50;
    color: white;
    text-align: center;
    padding: 2rem 0;
}

.footer a {
    color: #3498db;
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}

/* Function documentation */
.function {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2rem;
}

.function-name {
    color: #2c3e50;
    margin-bottom: 1rem;
}

.function-signature {
    background: #f8f9fa;
    border-left: 4px solid #3498db;
    padding: 1rem;
    margin: 1rem 0;
}

.function-description {
    margin-bottom: 1.5rem;
}

.parameters,
.returns,
.examples,
.notes {
    margin-bottom: 1.5rem;
}

.parameters h4,
.returns h4,
.examples h4,
.notes h4 {
    color: #2c3e50;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
}

.parameters ul {
    list-style: none;
    padding-left: 0;
}

.parameters li {
    background: #f8f9fa;
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-radius: 4px;
}

.examples pre {
    background: #f8f9fa;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 1rem;
    overflow-x: auto;
}

/* Search */
.search-container {
    max-width: 600px;
    margin: 2rem auto;
}

#search-input {
    width: 100%;
    padding: 1rem;
    border: 2px solid #ddd;
    border-radius: 8px;
    font-size: 1.1rem;
}

#search-input:focus {
    border-color: #3498db;
    outline: none;
}

#search-results {
    margin-top: 2rem;
}

.search-result {
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.search-result h4 {
    margin-bottom: 0.5rem;
}

.search-result a {
    color: #3498db;
    text-decoration: none;
}

.search-result a:hover {
    text-decoration: underline;
}

.result-type,
.result-module {
    display: inline-block;
    background: #ecf0f1;
    color: #2c3e50;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    margin-left: 0.5rem;
}

/* Tutorial styles */
.tutorial-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.tutorial-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1.5rem;
    transition: box-shadow 0.3s;
}

.tutorial-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.tutorial-card h3 {
    margin-bottom: 1rem;
}

.tutorial-card a {
    color: #2c3e50;
    text-decoration: none;
}

.tutorial-card a:hover {
    color: #3498db;
}

.tutorial-meta {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #eee;
}

.difficulty {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 500;
}

.difficulty-beginner {
    background: #d4edda;
    color: #155724;
}

.difficulty-intermediate {
    background: #fff3cd;
    color: #856404;
}

.difficulty-advanced {
    background: #f8d7da;
    color: #721c24;
}

/* Example styles */
.examples-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 2rem;
    margin-top: 2rem;
}

.example-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1.5rem;
}

.example-card h3 {
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.example-card .category {
    display: inline-block;
    background: #3498db;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    margin-bottom: 1rem;
}

.example-card pre {
    margin: 1rem 0;
}

.expected-output {
    background: #f8f9fa;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin-top: 1rem;
    border-radius: 4px;
}

/* Tutorial content */
.tutorial-content {
    background: white;
    padding: 3rem;
    border-radius: 8px;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.8;
}

.tutorial-content h1,
.tutorial-content h2,
.tutorial-content h3 {
    color: #2c3e50;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.tutorial-content h1 {
    margin-top: 0;
    font-size: 2.2rem;
}

.tutorial-content p {
    margin-bottom: 1rem;
}

.tutorial-content pre {
    background: #f8f9fa;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 1rem;
    overflow-x: auto;
    margin: 1rem 0;
}

.tutorial-content code {
    background: #f8f9fa;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-size: 0.9rem;
}

/* API styles */
.api-modules {
    display: grid;
    gap: 2rem;
    margin-top: 2rem;
}

.api-module {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 2rem;
}

.api-module h3 {
    color: #2c3e50;
    margin-bottom: 1rem;
}

.api-module a {
    color: #3498db;
    text-decoration: none;
}

.api-module a:hover {
    text-decoration: underline;
}

.function-list {
    margin-top: 1rem;
}

.function-name {
    display: inline-block;
    background: #f8f9fa;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    margin: 0.25rem 0.25rem 0.25rem 0;
    font-family: monospace;
    font-size: 0.9rem;
}

/* Copy button for code blocks */
.copy-button {
    position: absolute;
    top: 10px;
    right: 10px;
    padding: 5px 10px;
    background: #3498db;
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.3s;
}

.copy-button:hover {
    background: #2980b9;
}

/* Responsive design */
@media (max-width: 768px) {
    .hero h2 {
        font-size: 2rem;
    }

    .hero-buttons {
        flex-direction: column;
        align-items: center;
    }

    .feature-grid,
    .module-grid,
    .tutorial-grid,
    .examples-grid {
        grid-template-columns: 1fr;
    }

    .nav {
        flex-direction: column;
        gap: 0.5rem;
    }

    .tutorial-content {
        padding: 1.5rem;
    }

    .container {
        padding: 0 15px;
    }
}

@media (max-width: 480px) {
    .hero {
        padding: 2rem 0;
    }

    .hero h2 {
        font-size: 1.8rem;
    }

    .features,
    .modules,
    .quick-start {
        padding: 2rem 0;
    }

    .function,
    .tutorial-card,
    .example-card,
    .api-module {
        padding: 1rem;
    }
}
        "#
        .to_string()
    }

    /// Generate default JavaScript for the documentation
    pub fn generate_default_js(&self) -> String {
        r##"
// SciRS2 NDImage Documentation JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('SciRS2 NDImage Documentation loaded successfully');

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add copy functionality to code blocks
    document.querySelectorAll('pre code').forEach(codeBlock => {
        const button = document.createElement('button');
        button.textContent = 'Copy';
        button.className = 'copy-button';

        const pre = codeBlock.parentNode;
        pre.style.position = 'relative';
        pre.appendChild(button);

        button.addEventListener('click', function() {
            navigator.clipboard.writeText(codeBlock.textContent).then(function() {
                button.textContent = 'Copied!';
                button.style.background = '#28a745';
                setTimeout(function() {
                    button.textContent = 'Copy';
                    button.style.background = '#3498db';
                }, 2000);
            }).catch(function() {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = codeBlock.textContent;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);

                button.textContent = 'Copied!';
                button.style.background = '#28a745';
                setTimeout(function() {
                    button.textContent = 'Copy';
                    button.style.background = '#3498db';
                }, 2000);
            });
        });
    });

    // Search functionality
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    if (searchInput && searchResults) {
        let searchData = null;

        // Load search index
        fetch('search-index.json')
            .then(response => response.json())
            .then(data => {
                searchData = data;
            })
            .catch(error => {
                console.warn('Search index not available:', error);
            });

        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase().trim();

            if (query.length < 2) {
                searchResults.innerHTML = '';
                return;
            }

            if (!searchData) {
                searchResults.innerHTML = '<p>Search index not available.</p>';
                return;
            }

            const results = performSearch(query, searchData);
            displaySearchResults(results, searchResults);
        });
    }

    // Table of contents generation for tutorial pages
    generateTableOfContents();

    // Highlight current navigation item
    highlightCurrentNav();
});

function performSearch(query, data) {
    const results = [];

    // Search modules
    if (data.modules) {
        data.modules.forEach(module => {
            if (module.name.toLowerCase().includes(query) ||
                module.description.toLowerCase().includes(query)) {
                results.push({
                    type: 'module',
                    title: module.name,
                    description: module.description,
                    url: `api/${module.name.toLowerCase()}.html`
                });
            }
        });
    }

    // Search functions
    if (data.functions) {
        data.functions.forEach(func => {
            if (func.name.toLowerCase().includes(query) ||
                func.description.toLowerCase().includes(query)) {
                results.push({
                    type: 'function',
                    title: func.name,
                    description: func.description,
                    url: `api/index.html#${func.name}`
                });
            }
        });
    }

    // Search tutorials
    if (data.tutorials) {
        data.tutorials.forEach(tutorial => {
            if (tutorial.title.toLowerCase().includes(query) ||
                tutorial.description.toLowerCase().includes(query)) {
                results.push({
                    type: 'tutorial',
                    title: tutorial.title,
                    description: tutorial.description,
                    url: `tutorials/${tutorial.title.toLowerCase().replace(/\s+/g, '-')}.html`
                });
            }
        });
    }

    return results;
}

function displaySearchResults(results, container) {
    if (results.length === 0) {
        container.innerHTML = '<p>No results found.</p>';
        return;
    }

    const html = results.map(result => `
        <div class="search-result">
            <h4><a href="${result.url}">${result.title}</a></h4>
            <p>${result.description}</p>
            <span class="result-type">${result.type}</span>
        </div>
    `).join('');

    container.innerHTML = html;
}

function generateTableOfContents() {
    const content = document.querySelector('.tutorial-content');
    const headings = content ? content.querySelectorAll('h2, h3') : [];

    if (headings.length === 0) return;

    const toc = document.createElement('div');
    toc.className = 'table-of-contents';
    toc.innerHTML = '<h3>Table of Contents</h3>';

    const list = document.createElement('ul');

    headings.forEach((heading, index) => {
        const id = `heading-${index}`;
        heading.id = id;

        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = `#${id}`;
        link.textContent = heading.textContent;
        link.className = heading.tagName.toLowerCase();

        item.appendChild(link);
        list.appendChild(item);
    });

    toc.appendChild(list);

    // Insert TOC after the first heading
    const firstHeading = content.querySelector('h1');
    if (firstHeading && firstHeading.nextSibling) {
        content.insertBefore(toc, firstHeading.nextSibling);
    }
}

function highlightCurrentNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav a');

    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (currentPath.includes(href)) {
            link.classList.add('active');
        }
    });
}
        "##
        .to_string()
    }

    /// Generate Prism.js CSS for syntax highlighting
    pub fn generate_default_prism_css(&self) -> String {
        r#"
/* Prism.js Default Theme */
code[class*="language-"],
pre[class*="language-"] {
    color: #333;
    background: none;
    font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
    font-size: 1em;
    text-align: left;
    white-space: pre;
    word-spacing: normal;
    word-break: normal;
    word-wrap: normal;
    line-height: 1.5;
    -moz-tab-size: 4;
    -o-tab-size: 4;
    tab-size: 4;
    -webkit-hyphens: none;
    -moz-hyphens: none;
    -ms-hyphens: none;
    hyphens: none;
}

pre[class*="language-"] {
    position: relative;
    margin: .5em 0;
    overflow: visible;
    padding: 1em;
    background: #f5f2f0;
    border-radius: 6px;
}

:not(pre) > code[class*="language-"],
pre[class*="language-"] {
    background: #f5f2f0;
}

:not(pre) > code[class*="language-"] {
    padding: .1em;
    border-radius: .3em;
    white-space: normal;
}

.token.comment,
.token.prolog,
.token.doctype,
.token.cdata {
    color: slategray;
}

.token.punctuation {
    color: #999;
}

.token.namespace {
    opacity: .7;
}

.token.property,
.token.tag,
.token.boolean,
.token.number,
.token.constant,
.token.symbol,
.token.deleted {
    color: #905;
}

.token.selector,
.token.attr-name,
.token.string,
.token.char,
.token.builtin,
.token.inserted {
    color: #690;
}

.token.operator,
.token.entity,
.token.url,
.language-css .token.string,
.style .token.string {
    color: #9a6e3a;
}

.token.atrule,
.token.attr-value,
.token.keyword {
    color: #07a;
}

.token.function,
.token.class-name {
    color: #dd4a68;
}

.token.regex,
.token.important,
.token.variable {
    color: #e90;
}

.token.important,
.token.bold {
    font-weight: bold;
}

.token.italic {
    font-style: italic;
}

.token.entity {
    cursor: help;
}

/* Language-specific highlighting */
.language-rust .token.keyword {
    color: #8959a8;
}

.language-rust .token.string {
    color: #718c00;
}

.language-rust .token.function {
    color: #4271ae;
}

.language-rust .token.macro {
    color: #f5871f;
}

.language-toml .token.key {
    color: #4271ae;
}

.language-toml .token.string {
    color: #718c00;
}

/* Copy button integration */
pre[class*="language-"] {
    position: relative;
}

pre[class*="language-"] .copy-button {
    position: absolute;
    top: 10px;
    right: 10px;
    opacity: 0;
    transition: opacity 0.3s;
}

pre[class*="language-"]:hover .copy-button {
    opacity: 1;
}
        "#
        .to_string()
    }

    /// Enhanced markdown to HTML conversion with better support
    pub fn enhanced_markdown_to_html(&self, markdown: &str) -> String {
        let mut html = markdown.to_string();

        // Headers with proper closing
        html = self.process_headers(&html);

        // Code blocks with language support
        html = self.process_code_blocks(&html);

        // Inline code
        html = self.process_inline_code(&html);

        // Links
        html = self.process_links(&html);

        // Bold and italic text
        html = self.process_emphasis(&html);

        // Lists
        html = self.process_lists(&html);

        // Paragraphs
        html = self.process_paragraphs(&html);

        html
    }

    /// Process markdown headers
    fn process_headers(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Process headers from h6 to h1 to avoid conflicts
        for level in (1..=6).rev() {
            let marker = "#".repeat(level);
            let regex_pattern = format!(r"(?m)^{} (.+)$", marker);

            if let Ok(re) = regex::Regex::new(&regex_pattern) {
                result = re
                    .replace_all(&result, |caps: &regex::Captures| {
                        format!("<h{}>{}</h{}>", level, &caps[1], level)
                    })
                    .to_string();
            }
        }

        result
    }

    /// Process markdown code blocks
    fn process_code_blocks(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Language-specific code blocks
        if let Ok(re) = regex::Regex::new(r"```(\w+)\n([\s\S]*?)\n```") {
            result = re
                .replace_all(&result, |caps: &regex::Captures| {
                    format!(
                        r#"<pre><code class="language-{}">{}</code></pre>"#,
                        &caps[1], &caps[2]
                    )
                })
                .to_string();
        }

        // Generic code blocks
        if let Ok(re) = regex::Regex::new(r"```\n([\s\S]*?)\n```") {
            result = re
                .replace_all(&result, |caps: &regex::Captures| {
                    format!(r#"<pre><code>{}</code></pre>"#, &caps[1])
                })
                .to_string();
        }

        result
    }

    /// Process inline code
    fn process_inline_code(&self, content: &str) -> String {
        if let Ok(re) = regex::Regex::new(r"`([^`]+)`") {
            re.replace_all(content, r#"<code>$1</code>"#).to_string()
        } else {
            content.to_string()
        }
    }

    /// Process markdown links
    fn process_links(&self, content: &str) -> String {
        if let Ok(re) = regex::Regex::new(r"\[([^\]]+)\]\(([^)]+)\)") {
            re.replace_all(content, r#"<a href="$2">$1</a>"#)
                .to_string()
        } else {
            content.to_string()
        }
    }

    /// Process bold and italic text
    fn process_emphasis(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Bold text
        if let Ok(re) = regex::Regex::new(r"\*\*([^*]+)\*\*") {
            result = re.replace_all(&result, r#"<strong>$1</strong>"#).to_string();
        }

        // Italic text
        if let Ok(re) = regex::Regex::new(r"\*([^*]+)\*") {
            result = re.replace_all(&result, r#"<em>$1</em>"#).to_string();
        }

        result
    }

    /// Process markdown lists
    fn process_lists(&self, content: &str) -> String {
        let mut result = content.to_string();

        // Unordered lists
        if let Ok(re) = regex::Regex::new(r"(?m)^- (.+)$") {
            result = re.replace_all(&result, r#"<li>$1</li>"#).to_string();
            result = result.replace("<li>", "<ul><li>").replace("</li>", "</li></ul>");
        }

        // Numbered lists
        if let Ok(re) = regex::Regex::new(r"(?m)^\d+\. (.+)$") {
            result = re.replace_all(&result, r#"<li>$1</li>"#).to_string();
            result = result.replace("<li>", "<ol><li>").replace("</li>", "</li></ol>");
        }

        result
    }

    /// Process paragraphs
    fn process_paragraphs(&self, content: &str) -> String {
        let lines: Vec<&str> = content.split('\n').collect();
        let mut result = Vec::new();
        let mut in_paragraph = false;

        for line in lines {
            let trimmed = line.trim();

            if trimmed.is_empty() {
                if in_paragraph {
                    result.push("</p>");
                    in_paragraph = false;
                }
            } else if !trimmed.starts_with('<') {
                // Not HTML, treat as paragraph content
                if !in_paragraph {
                    result.push("<p>");
                    in_paragraph = true;
                }
                result.push(line);
            } else {
                // HTML tag, close paragraph if open
                if in_paragraph {
                    result.push("</p>");
                    in_paragraph = false;
                }
                result.push(line);
            }
        }

        if in_paragraph {
            result.push("</p>");
        }

        result.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_css_generation() {
        let site = DocumentationSite::new();
        let css = site.generate_default_css();

        assert!(css.contains("body {"));
        assert!(css.contains(".header {"));
        assert!(css.contains(".function {"));
        assert!(css.contains("@media"));
    }

    #[test]
    fn test_js_generation() {
        let site = DocumentationSite::new();
        let js = site.generate_default_js();

        assert!(js.contains("DOMContentLoaded"));
        assert!(js.contains("copy-button"));
        assert!(js.contains("performSearch"));
    }

    #[test]
    fn test_prism_css_generation() {
        let site = DocumentationSite::new();
        let prism_css = site.generate_default_prism_css();

        assert!(prism_css.contains("code[class*=\"language-\"]"));
        assert!(prism_css.contains(".token."));
        assert!(prism_css.contains("language-rust"));
    }

    #[test]
    fn test_enhanced_markdown_processing() {
        let site = DocumentationSite::new();

        // Test headers
        let markdown_headers = "# Title\n## Subtitle";
        let html_headers = site.enhanced_markdown_to_html(markdown_headers);
        assert!(html_headers.contains("<h1>Title</h1>"));
        assert!(html_headers.contains("<h2>Subtitle</h2>"));

        // Test code blocks
        let markdown_code = "```rust\nlet x = 42;\n```";
        let html_code = site.enhanced_markdown_to_html(markdown_code);
        assert!(html_code.contains("<pre><code class=\"language-rust\">"));

        // Test inline code
        let markdown_inline = "Use the `function_name` here.";
        let html_inline = site.enhanced_markdown_to_html(markdown_inline);
        assert!(html_inline.contains("<code>function_name</code>"));

        // Test links
        let markdown_link = "[Documentation](https://example.com)";
        let html_link = site.enhanced_markdown_to_html(markdown_link);
        assert!(html_link.contains("<a href=\"https://example.com\">Documentation</a>"));
    }
}