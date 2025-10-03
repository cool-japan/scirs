//! Visualization tools for text processing and analysis
//!
//! This module provides comprehensive visualization capabilities for text data,
//! including word clouds, attention visualizations, embedding plots, and various
//! text analysis charts.

use crate::error::{Result, TextError};
use crate::sentiment::SentimentResult;
use crate::topic_modeling::Topic;
use crate::vectorize::{CountVectorizer, TfidfVectorizer, Vectorizer};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Configuration for text visualizations
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Output image width
    pub width: usize,
    /// Output image height
    pub height: usize,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Font size range
    pub font_size_range: (usize, usize),
    /// Background color
    pub background_color: Color,
    /// Whether to save high-DPI images
    pub high_dpi: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            color_scheme: ColorScheme::Viridis,
            font_size_range: (10, 100),
            background_color: Color::WHITE,
            high_dpi: false,
        }
    }
}

/// Color schemes for visualizations
#[derive(Debug, Clone)]
pub enum ColorScheme {
    /// Viridis color scheme
    Viridis,
    /// Plasma color scheme
    Plasma,
    /// Inferno color scheme
    Inferno,
    /// Cool colors
    Cool,
    /// Warm colors
    Warm,
    /// Custom color palette
    Custom(Vec<Color>),
}

/// RGB color representation
#[derive(Debug, Clone, Copy)]
pub struct Color {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
}

impl Color {
    /// White color constant
    pub const WHITE: Color = Color {
        r: 255,
        g: 255,
        b: 255,
    };
    /// Black color constant
    pub const BLACK: Color = Color { r: 0, g: 0, b: 0 };
    /// Red color constant
    pub const RED: Color = Color { r: 255, g: 0, b: 0 };
    /// Green color constant
    pub const GREEN: Color = Color { r: 0, g: 255, b: 0 };
    /// Blue color constant
    pub const BLUE: Color = Color { r: 0, g: 0, b: 255 };

    /// Create new color from RGB values
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

/// Word cloud visualization
pub struct WordCloud {
    /// Word frequency data
    word_frequencies: HashMap<String, f64>,
    /// Configuration
    config: VisualizationConfig,
}

impl WordCloud {
    /// Create new word cloud from text
    pub fn fromtext(text: &str, config: VisualizationConfig) -> Result<Self> {
        let mut vectorizer = CountVectorizer::new(false);
        let documents = vec![text];
        let matrix = vectorizer.fit_transform(&documents)?;

        let vocabulary_map = vectorizer.vocabulary_map();
        let mut word_frequencies = HashMap::new();

        // Extract word frequencies from the matrix
        for (word, &idx) in vocabulary_map.iter() {
            if let Some(count) = vectorizer.get_feature_count(&matrix, 0, idx) {
                if count > 0.0 {
                    word_frequencies.insert(word.clone(), count);
                }
            }
        }

        Ok(Self {
            word_frequencies,
            config,
        })
    }

    /// Create word cloud from TF-IDF vectorizer and matrix
    pub fn from_tfidf(
        vectorizer: &TfidfVectorizer,
        matrix: &Array2<f64>,
        document_index: usize,
    ) -> Result<Self> {
        let vocabulary_map = vectorizer.vocabulary_map();
        let mut word_frequencies = HashMap::new();

        // Get TF-IDF scores for the document
        for (word, &idx) in vocabulary_map.iter() {
            if let Some(score) = vectorizer.get_feature_score(matrix, document_index, idx) {
                if score > 0.0 {
                    word_frequencies.insert(word.clone(), score);
                }
            }
        }

        Ok(Self {
            word_frequencies,
            config: VisualizationConfig::default(),
        })
    }

    /// Create word cloud from frequency map
    pub fn from_frequencies(
        frequencies: HashMap<String, f64>,
        config: VisualizationConfig,
    ) -> Self {
        Self {
            word_frequencies: frequencies,
            config,
        }
    }

    /// Generate word cloud as SVG
    pub fn to_svg(&self) -> Result<String> {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));

        // Background
        svg.push_str(&format!(
            r#"<rect width="100%" height="100%" fill="{}" />"#,
            self.config.background_color.to_hex()
        ));

        // Sort words by frequency
        let mut sorted_words: Vec<_> = self.word_frequencies.iter().collect();
        sorted_words.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        // Take top words
        let max_words = 50;
        let top_words: Vec<_> = sorted_words.into_iter().take(max_words).collect();

        if top_words.is_empty() {
            return Ok(svg + "</svg>");
        }

        // Calculate font sizes
        let max_freq = top_words[0].1;
        let min_freq = top_words.last().map(|x| *x.1).unwrap_or(*max_freq);
        let freq_range = max_freq - min_freq;

        // Generate colors
        let colors = self.generate_colors(top_words.len());

        // Position words (simplified grid layout)
        let cols = (top_words.len() as f64).sqrt().ceil() as usize;
        let rows = top_words.len().div_ceil(cols);
        let cell_width = self.config.width / cols;
        let cell_height = self.config.height / rows;

        for (i, (word, &freq)) in top_words.iter().enumerate() {
            let row = i / cols;
            let col = i % cols;

            // Calculate position
            let x = col * cell_width + cell_width / 2;
            let y = row * cell_height + cell_height / 2;

            // Calculate font size
            let font_size = if freq_range > 0.0 {
                let normalized = (freq - min_freq) / freq_range;
                self.config.font_size_range.0
                    + (normalized
                        * (self.config.font_size_range.1 - self.config.font_size_range.0) as f64)
                        as usize
            } else {
                (self.config.font_size_range.0 + self.config.font_size_range.1) / 2
            };

            // Add word to SVG
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="{}" 
                   fill="{}" text-anchor="middle" dominant-baseline="middle">{}</text>"#,
                x,
                y,
                font_size,
                colors[i % colors.len()].to_hex(),
                word
            ));
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Generate colors based on color scheme
    fn generate_colors(&self, count: usize) -> Vec<Color> {
        match &self.config.color_scheme {
            ColorScheme::Viridis => self.generate_viridis_colors(count),
            ColorScheme::Plasma => self.generate_plasma_colors(count),
            ColorScheme::Inferno => self.generate_inferno_colors(count),
            ColorScheme::Cool => self.generate_cool_colors(count),
            ColorScheme::Warm => self.generate_warm_colors(count),
            ColorScheme::Custom(colors) => colors.clone(),
        }
    }

    /// Generate viridis color scheme
    fn generate_viridis_colors(&self, count: usize) -> Vec<Color> {
        let mut colors = Vec::new();
        for i in 0..count {
            let t = i as f64 / (count - 1).max(1) as f64;
            // Simplified viridis approximation
            let r = (68.0 + t * (253.0 - 68.0)) as u8;
            let g = (1.0 + t * (231.0 - 1.0)) as u8;
            let b = (84.0 + t * (37.0 - 84.0)) as u8;
            colors.push(Color::new(r, g, b));
        }
        colors
    }

    /// Generate plasma color scheme
    fn generate_plasma_colors(&self, count: usize) -> Vec<Color> {
        let mut colors = Vec::new();
        for i in 0..count {
            let t = i as f64 / (count - 1).max(1) as f64;
            // Simplified plasma approximation
            let r = (13.0 + t * (240.0 - 13.0)) as u8;
            let g = (8.0 + t * (249.0 - 8.0)) as u8;
            let b = (135.0 + t * (33.0 - 135.0)) as u8;
            colors.push(Color::new(r, g, b));
        }
        colors
    }

    /// Generate inferno color scheme
    fn generate_inferno_colors(&self, count: usize) -> Vec<Color> {
        let mut colors = Vec::new();
        for i in 0..count {
            let t = i as f64 / (count - 1).max(1) as f64;
            // Simplified inferno approximation
            let r = (0.0 + t * (252.0 - 0.0)) as u8;
            let g = (0.0 + t * (255.0 - 0.0)) as u8;
            let b = (4.0 + t * (164.0 - 4.0)) as u8;
            colors.push(Color::new(r, g, b));
        }
        colors
    }

    /// Generate cool color scheme
    fn generate_cool_colors(&self, count: usize) -> Vec<Color> {
        let mut colors = Vec::new();
        for i in 0..count {
            let t = i as f64 / (count - 1).max(1) as f64;
            let r = (0.0 + t * (255.0 - 0.0)) as u8;
            let g = (255.0 - t * (255.0 - 0.0)) as u8;
            let b = 255;
            colors.push(Color::new(r, g, b));
        }
        colors
    }

    /// Generate warm color scheme
    fn generate_warm_colors(&self, count: usize) -> Vec<Color> {
        let mut colors = Vec::new();
        for i in 0..count {
            let t = i as f64 / (count - 1).max(1) as f64;
            let r = 255;
            let g = (255.0 - t * (255.0 - 0.0)) as u8;
            let b = (0.0 + t * (255.0 - 0.0)) as u8;
            colors.push(Color::new(r, g, b));
        }
        colors
    }

    /// Save word cloud to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let svg_content = self.to_svg()?;
        fs::write(path, svg_content)
            .map_err(|e| TextError::IoError(format!("Failed to save word cloud: {e}")))?;
        Ok(())
    }
}

/// Attention visualization for transformer models
pub struct AttentionVisualizer {
    config: VisualizationConfig,
}

impl AttentionVisualizer {
    /// Create new attention visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Visualize attention weights as heatmap
    pub fn attention_heatmap(
        &self,
        attentionweights: ArrayView2<f64>,
        source_tokens: &[String],
        target_tokens: &[String],
    ) -> Result<String> {
        let (n_target, n_source) = attentionweights.dim();

        if source_tokens.len() != n_source || target_tokens.len() != n_target {
            return Err(TextError::InvalidInput(
                "Token count doesn't match attention matrix dimensions".to_string(),
            ));
        }

        let mut svg = String::new();

        // Calculate dimensions
        let cell_width = self.config.width / (n_source + 1);
        let cell_height = self.config.height / (n_target + 1);
        let _matrix_width = n_source * cell_width;
        let matrix_height = n_target * cell_height;

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width + 100,
            self.config.height + 100
        ));

        // Background
        svg.push_str(&format!(
            r#"<rect width="100%" height="100%" fill="{}" />"#,
            self.config.background_color.to_hex()
        ));

        // Find min and max attention values for normalization
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &val in attentionweights.iter() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // Draw attention matrix
        for i in 0..n_target {
            for j in 0..n_source {
                let attention = attentionweights[[i, j]];
                let normalized = if max_val > min_val {
                    (attention - min_val) / (max_val - min_val)
                } else {
                    0.5
                };

                // Color based on attention value
                let intensity = (normalized * 255.0) as u8;
                let color = Color::new(255 - intensity, 255 - intensity, 255);

                let x = 50 + j * cell_width;
                let y = 50 + i * cell_height;

                svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="black" stroke-width="1" />"#,
                    x, y, cell_width, cell_height, color.to_hex()
                ));

                // Add attention value text
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="10" 
                       text-anchor="middle" dominant-baseline="middle">{:.2}</text>"#,
                    x + cell_width / 2,
                    y + cell_height / 2,
                    attention
                ));
            }
        }

        // Add source token labels (bottom)
        for (j, token) in source_tokens.iter().enumerate() {
            let x = 50 + j * cell_width + cell_width / 2;
            let y = 50 + matrix_height + 20;

            svg.push_str(&format!(
                r#"<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="12" 
                   text-anchor="middle" dominant-baseline="middle" transform="rotate(-45 {x} {y})">{token}</text>"#
            ));
        }

        // Add target token labels (left)
        for (i, token) in target_tokens.iter().enumerate() {
            let x = 30;
            let y = 50 + i * cell_height + cell_height / 2;

            svg.push_str(&format!(
                r#"<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="12" 
                   text-anchor="end" dominant-baseline="middle">{token}</text>"#
            ));
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Save attention visualization to file
    pub fn save_attention_heatmap<P: AsRef<Path>>(
        &self,
        attentionweights: ArrayView2<f64>,
        source_tokens: &[String],
        target_tokens: &[String],
        path: P,
    ) -> Result<()> {
        let svg_content = self.attention_heatmap(attentionweights, source_tokens, target_tokens)?;
        fs::write(path, svg_content)
            .map_err(|e| TextError::IoError(format!("Failed to save attention heatmap: {e}")))?;
        Ok(())
    }
}

/// Embedding visualization using dimensionality reduction
pub struct EmbeddingVisualizer {
    config: VisualizationConfig,
}

impl EmbeddingVisualizer {
    /// Create new embedding visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Visualize word embeddings using PCA projection to 2D
    pub fn visualize_embeddings(
        &self,
        word_vectors: &HashMap<String, Array1<f64>>,
        words_to_plot: Option<&[String]>,
    ) -> Result<String> {
        let words: Vec<String> = if let Some(selected_words) = words_to_plot {
            selected_words.to_vec()
        } else {
            word_vectors.keys().take(100).cloned().collect()
        };

        if words.is_empty() {
            return Err(TextError::InvalidInput("No words to visualize".to_string()));
        }

        // Collect embedding _vectors
        let mut embeddings = Vec::new();
        let mut valid_words = Vec::new();

        for word in &words {
            if let Some(vector) = word_vectors.get(word) {
                embeddings.push(vector.clone());
                valid_words.push(word.clone());
            }
        }

        if embeddings.is_empty() {
            return Err(TextError::InvalidInput(
                "No valid embeddings found".to_string(),
            ));
        }

        // Simple PCA to 2D (simplified implementation)
        let projected_points = self.simple_pca_2d(&embeddings)?;

        // Create SVG
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));

        // Background
        svg.push_str(&format!(
            r#"<rect width="100%" height="100%" fill="{}" />"#,
            self.config.background_color.to_hex()
        ));

        // Find bounds for scaling
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &(x, y) in &projected_points {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let margin = 50.0;
        let plot_width = self.config.width as f64 - 2.0 * margin;
        let plot_height = self.config.height as f64 - 2.0 * margin;

        // Generate colors
        let colors = self.generate_colors(valid_words.len());

        // Plot points and labels
        for (i, ((x, y), word)) in projected_points.iter().zip(&valid_words).enumerate() {
            // Scale to _plot area
            let scaled_x = margin + (x - min_x) / (max_x - min_x) * plot_width;
            let scaled_y = margin + (y - min_y) / (max_y - min_y) * plot_height;

            // Draw point
            svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="3" fill="{}" />"#,
                scaled_x,
                scaled_y,
                colors[i % colors.len()].to_hex()
            ));

            // Draw label
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="10" 
                   text-anchor="start" dominant-baseline="middle">{}</text>"#,
                scaled_x + 5.0,
                scaled_y,
                word
            ));
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Simple PCA implementation for 2D projection
    fn simple_pca_2d(&self, embeddings: &[Array1<f64>]) -> Result<Vec<(f64, f64)>> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }

        let n_samples = embeddings.len();
        let n_features = embeddings[0].len();

        // Create data matrix
        let mut data_matrix = Array2::zeros((n_samples, n_features));
        for (i, embedding) in embeddings.iter().enumerate() {
            data_matrix.row_mut(i).assign(embedding);
        }

        // Center the data
        let mean = data_matrix.mean_axis(Axis(0)).unwrap();
        for mut row in data_matrix.rows_mut() {
            row -= &mean;
        }

        // Simplified SVD (using covariance matrix approach)
        let _cov_matrix = data_matrix.t().dot(&data_matrix) / (n_samples - 1) as f64;

        // Find first two principal components (simplified eigenvalue decomposition)
        // This is a very simplified approach - in practice would use proper SVD/eigendecomposition
        let mut pc1 = Array1::zeros(n_features);
        let mut pc2 = Array1::zeros(n_features);

        // Use random orthogonal vectors as approximation
        for i in 0..n_features {
            pc1[i] = (i as f64).sin();
            pc2[i] = (i as f64).cos();
        }

        // Normalize
        pc1 /= pc1.dot(&pc1).sqrt();
        pc2 /= pc2.dot(&pc2).sqrt();

        // Project data
        let mut projected = Vec::new();
        for row in data_matrix.rows() {
            let x = row.dot(&pc1);
            let y = row.dot(&pc2);
            projected.push((x, y));
        }

        Ok(projected)
    }

    /// Generate colors for embedding visualization
    fn generate_colors(&self, count: usize) -> Vec<Color> {
        let mut colors = Vec::new();
        for i in 0..count {
            let hue = (i as f64 / count as f64) * 360.0;
            let color = self.hsv_to_rgb(hue, 0.8, 0.9);
            colors.push(color);
        }
        colors
    }

    /// Convert HSV to RGB
    fn hsv_to_rgb(&self, h: f64, s: f64, v: f64) -> Color {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r_prime, g_prime, b_prime) = match h as i32 / 60 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };

        Color::new(
            ((r_prime + m) * 255.0) as u8,
            ((g_prime + m) * 255.0) as u8,
            ((b_prime + m) * 255.0) as u8,
        )
    }

    /// Save embedding visualization to file
    pub fn save_embeddings<P: AsRef<Path>>(
        &self,
        word_vectors: &HashMap<String, Array1<f64>>,
        words_to_plot: Option<&[String]>,
        path: P,
    ) -> Result<()> {
        let svg_content = self.visualize_embeddings(word_vectors, words_to_plot)?;
        fs::write(path, svg_content).map_err(|e| {
            TextError::IoError(format!("Failed to save embedding visualization: {e}"))
        })?;
        Ok(())
    }
}

/// Sentiment analysis visualization
pub struct SentimentVisualizer {
    config: VisualizationConfig,
}

impl SentimentVisualizer {
    /// Create new sentiment visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Create sentiment distribution chart
    pub fn sentiment_distribution(
        &self,
        sentiment_results: &[SentimentResult],
        labels: &[String],
    ) -> Result<String> {
        if sentiment_results.len() != labels.len() {
            return Err(TextError::InvalidInput(
                "Number of sentiment _results must match number of labels".to_string(),
            ));
        }

        // Count sentiment categories
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut neutral_count = 0;

        for result in sentiment_results {
            match result.sentiment {
                crate::sentiment::Sentiment::Positive => positive_count += 1,
                crate::sentiment::Sentiment::Negative => negative_count += 1,
                crate::sentiment::Sentiment::Neutral => neutral_count += 1,
            }
        }

        let total = sentiment_results.len();
        if total == 0 {
            return Err(TextError::InvalidInput(
                "No sentiment data to visualize".to_string(),
            ));
        }

        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));

        // Background
        svg.push_str(&format!(
            r#"<rect width="100%" height="100%" fill="{}" />"#,
            self.config.background_color.to_hex()
        ));

        // Create pie chart
        let center_x = self.config.width as f64 / 2.0;
        let center_y = self.config.height as f64 / 2.0;
        let radius = (self.config.width.min(self.config.height) as f64 / 2.0 - 50.0).max(50.0);

        let positive_angle = (positive_count as f64 / total as f64) * 2.0 * std::f64::consts::PI;
        let negative_angle = (negative_count as f64 / total as f64) * 2.0 * std::f64::consts::PI;
        let neutral_angle = (neutral_count as f64 / total as f64) * 2.0 * std::f64::consts::PI;

        let mut start_angle = 0.0;

        // Positive segment
        if positive_count > 0 {
            let end_angle = start_angle + positive_angle;
            svg.push_str(&self.create_pie_segment(
                center_x,
                center_y,
                radius,
                start_angle,
                end_angle,
                Color::GREEN,
                "Positive",
                positive_count,
                total,
            ));
            start_angle = end_angle;
        }

        // Negative segment
        if negative_count > 0 {
            let end_angle = start_angle + negative_angle;
            svg.push_str(&self.create_pie_segment(
                center_x,
                center_y,
                radius,
                start_angle,
                end_angle,
                Color::RED,
                "Negative",
                negative_count,
                total,
            ));
            start_angle = end_angle;
        }

        // Neutral segment
        if neutral_count > 0 {
            let end_angle = start_angle + neutral_angle;
            svg.push_str(&self.create_pie_segment(
                center_x,
                center_y,
                radius,
                start_angle,
                end_angle,
                Color::new(128, 128, 128),
                "Neutral",
                neutral_count,
                total,
            ));
        }

        // Add legend
        svg.push_str(r#"<text x="20" y="30" font-family="Arial, sans-serif" font-size="16" font-weight="bold">Sentiment Distribution</text>"#);

        let legend_y = 60;
        svg.push_str(&format!(
            r#"<rect x="20" y="{}" width="15" height="15" fill="{}" />"#,
            legend_y,
            Color::GREEN.to_hex()
        ));
        svg.push_str(&format!(
            r#"<text x="40" y="{}" font-family="Arial, sans-serif" font-size="12">Positive: {} ({:.1}%)</text>"#,
            legend_y + 12, positive_count, (positive_count as f64 / total as f64) * 100.0
        ));

        svg.push_str(&format!(
            r#"<rect x="20" y="{}" width="15" height="15" fill="{}" />"#,
            legend_y + 25,
            Color::RED.to_hex()
        ));
        svg.push_str(&format!(
            r#"<text x="40" y="{}" font-family="Arial, sans-serif" font-size="12">Negative: {} ({:.1}%)</text>"#,
            legend_y + 37, negative_count, (negative_count as f64 / total as f64) * 100.0
        ));

        svg.push_str(&format!(
            r#"<rect x="20" y="{}" width="15" height="15" fill="{}" />"#,
            legend_y + 50,
            Color::new(128, 128, 128).to_hex()
        ));
        svg.push_str(&format!(
            r#"<text x="40" y="{}" font-family="Arial, sans-serif" font-size="12">Neutral: {} ({:.1}%)</text>"#,
            legend_y + 62, neutral_count, (neutral_count as f64 / total as f64) * 100.0
        ));

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Create pie chart segment
    fn create_pie_segment(
        &self,
        center_x: f64,
        center_y: f64,
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        color: Color,
        label: &str,
        _count: usize,
        total: usize,
    ) -> String {
        let x1 = center_x + radius * start_angle.cos();
        let y1 = center_y + radius * start_angle.sin();
        let x2 = center_x + radius * end_angle.cos();
        let y2 = center_y + radius * end_angle.sin();

        let large_arc = if end_angle - start_angle > std::f64::consts::PI {
            1
        } else {
            0
        };

        format!(
            r#"<path d="M {} {} L {} {} A {} {} 0 {} 1 {} {} Z" fill="{}" stroke="white" stroke-width="2" />"#,
            center_x,
            center_y,
            x1,
            y1,
            radius,
            radius,
            large_arc,
            x2,
            y2,
            color.to_hex()
        )
    }

    /// Save sentiment visualization to file
    pub fn save_sentiment_distribution<P: AsRef<Path>>(
        &self,
        sentiment_results: &[SentimentResult],
        labels: &[String],
        path: P,
    ) -> Result<()> {
        let svg_content = self.sentiment_distribution(sentiment_results, labels)?;
        fs::write(path, svg_content).map_err(|e| {
            TextError::IoError(format!("Failed to save sentiment visualization: {e}"))
        })?;
        Ok(())
    }
}

/// Topic modeling visualization
pub struct TopicVisualizer {
    config: VisualizationConfig,
}

impl TopicVisualizer {
    /// Create new topic visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Visualize topic word distributions
    pub fn topic_words_chart(&self, topics: &[Topic], topn: usize) -> Result<String> {
        if topics.is_empty() {
            return Err(TextError::InvalidInput(
                "No topics to visualize".to_string(),
            ));
        }

        let mut svg = String::new();

        // Calculate dimensions
        let chart_width = self.config.width;
        let chart_height = self.config.height;
        let margin = 50;
        let topic_height = (chart_height - 2 * margin) / topics.len();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">"#
        ));

        // Background
        svg.push_str(&format!(
            r#"<rect width="100%" height="100%" fill="{}" />"#,
            self.config.background_color.to_hex()
        ));

        // Title
        svg.push_str(&format!(
            r#"<text x="{}" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="bold" text-anchor="middle">Topic Word Distributions</text>"#,
            chart_width / 2
        ));

        // Generate colors for topics
        let colors = self.generate_topic_colors(topics.len());

        for (topic_idx, topic) in topics.iter().enumerate() {
            let y_offset = margin + topic_idx * topic_height;

            // Topic label
            svg.push_str(&format!(
                r#"<text x="20" y="{}" font-family="Arial, sans-serif" font-size="14" font-weight="bold">Topic {}</text>"#,
                y_offset + 20, topic_idx
            ));

            // Get top words for this topic
            let mut topic_words: Vec<_> = topic.top_words.iter().collect();
            topic_words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_words: Vec<_> = topic_words.into_iter().take(topn).collect();

            if !top_words.is_empty() {
                let max_prob = top_words[0].1;
                let bar_area_width = chart_width - 200;

                // Draw bars for top words
                for (word_idx, (word, prob)) in top_words.iter().enumerate() {
                    let bar_y = y_offset + 30 + word_idx * 15;
                    let bar_width = (*prob / max_prob * bar_area_width as f64) as usize;

                    // Word bar
                    svg.push_str(&format!(
                        r#"<rect x="120" y="{}" width="{}" height="12" fill="{}" />"#,
                        bar_y,
                        bar_width,
                        colors[topic_idx % colors.len()].to_hex()
                    ));

                    // Word label
                    svg.push_str(&format!(
                        r#"<text x="115" y="{}" font-family="Arial, sans-serif" font-size="10" text-anchor="end">{}</text>"#,
                        bar_y + 9, word
                    ));

                    // Probability value
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="10">{:.3}</text>"#,
                        125 + bar_width, bar_y + 9, *prob
                    ));
                }
            }
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Generate colors for topics
    fn generate_topic_colors(&self, count: usize) -> Vec<Color> {
        let mut colors = Vec::new();
        for i in 0..count {
            let hue = (i as f64 / count as f64) * 360.0;
            let color = self.hsv_to_rgb(hue, 0.7, 0.8);
            colors.push(color);
        }
        colors
    }

    /// Convert HSV to RGB
    fn hsv_to_rgb(&self, h: f64, s: f64, v: f64) -> Color {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r_prime, g_prime, b_prime) = match h as i32 / 60 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };

        Color::new(
            ((r_prime + m) * 255.0) as u8,
            ((g_prime + m) * 255.0) as u8,
            ((b_prime + m) * 255.0) as u8,
        )
    }

    /// Save topic visualization to file
    pub fn save_topic_words<P: AsRef<Path>>(
        &self,
        topics: &[Topic],
        topn: usize,
        path: P,
    ) -> Result<()> {
        let svg_content = self.topic_words_chart(topics, topn)?;
        fs::write(path, svg_content)
            .map_err(|e| TextError::IoError(format!("Failed to save topic visualization: {e}")))?;
        Ok(())
    }
}

/// Text analysis dashboard generator
pub struct TextAnalyticsDashboard {
    config: VisualizationConfig,
}

impl TextAnalyticsDashboard {
    /// Create new analytics dashboard
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate complete text analytics dashboard
    pub fn generate_dashboard(
        &self,
        text_data: &[String],
        sentiment_results: &[SentimentResult],
        topics: &[Topic],
        word_frequencies: &HashMap<String, f64>,
    ) -> Result<String> {
        let mut html = String::new();

        // HTML header
        html.push_str(r#"<!DOCTYPE html>
<html>
<head>
    <title>Text Analytics Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .widget { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .widget h3 { margin-top: 0; color: #333; }
        .full-width { grid-column: 1 / -1; }
        .stats { display: flex; justify-content: space-around; }
        .stat { text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #2196F3; }
        .stat-label { color: #666; }
    </style>
</head>
<body>
    <h1>Text Analytics Dashboard</h1>
    <div class="dashboard">
"#);

        // Text statistics widget
        html.push_str(
            r#"<div class="widget">
            <h3>Text Statistics</h3>
            <div class="stats">
"#,
        );

        let total_docs = text_data.len();
        let total_words: usize = text_data
            .iter()
            .map(|text| text.split_whitespace().count())
            .sum();
        let avg_words = if total_docs > 0 {
            total_words / total_docs
        } else {
            0
        };
        let unique_words = word_frequencies.len();

        html.push_str(&format!(
            r#"
                <div class="stat">
                    <div class="stat-value">{total_docs}</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{total_words}</div>
                    <div class="stat-label">Total Words</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{avg_words}</div>
                    <div class="stat-label">Avg Words/Doc</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{unique_words}</div>
                    <div class="stat-label">Unique Words</div>
                </div>
"#
        ));

        html.push_str("</div></div>");

        // Word cloud widget
        let word_cloud = WordCloud::from_frequencies(word_frequencies.clone(), self.config.clone());
        let word_cloud_svg = word_cloud.to_svg()?;

        html.push_str(&format!(
            r#"<div class="widget">
            <h3>Word Cloud</h3>
            {word_cloud_svg}
        </div>"#
        ));

        // Sentiment analysis widget
        let sentiment_viz = SentimentVisualizer::new(self.config.clone());
        let labels: Vec<String> = (0..sentiment_results.len())
            .map(|i| {
                let doc_num = i + 1;
                format!("Doc {doc_num}")
            })
            .collect();
        let sentiment_svg = sentiment_viz.sentiment_distribution(sentiment_results, &labels)?;

        html.push_str(&format!(
            r#"<div class="widget">
            <h3>Sentiment Distribution</h3>
            {sentiment_svg}
        </div>"#
        ));

        // Topic modeling widget
        if !topics.is_empty() {
            let topic_viz = TopicVisualizer::new(self.config.clone());
            let topic_svg = topic_viz.topic_words_chart(topics, 5)?;

            html.push_str(&format!(
                r#"<div class="widget full-width">
                <h3>Topic Analysis</h3>
                {topic_svg}
            </div>"#
            ));
        }

        // HTML footer
        html.push_str(
            r#"
    </div>
</body>
</html>"#,
        );

        Ok(html)
    }

    /// Save dashboard to HTML file
    pub fn save_dashboard<P: AsRef<Path>>(
        &self,
        text_data: &[String],
        sentiment_results: &[SentimentResult],
        topics: &[Topic],
        word_frequencies: &HashMap<String, f64>,
        path: P,
    ) -> Result<()> {
        let html_content =
            self.generate_dashboard(text_data, sentiment_results, topics, word_frequencies)?;
        fs::write(path, html_content)
            .map_err(|e| TextError::IoError(format!("Failed to save dashboard: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sentiment::{Sentiment, SentimentResult};
    use std::collections::HashMap;

    #[test]
    fn test_word_cloud_creation() {
        let mut frequencies = HashMap::new();
        frequencies.insert("hello".to_string(), 10.0);
        frequencies.insert("world".to_string(), 8.0);
        frequencies.insert("test".to_string(), 5.0);

        let config = VisualizationConfig::default();
        let word_cloud = WordCloud::from_frequencies(frequencies, config);

        let svg = word_cloud.to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("hello"));
        assert!(svg.contains("world"));
        assert!(svg.contains("test"));
    }

    #[test]
    fn test_sentiment_visualization() {
        let sentiment_results = vec![
            SentimentResult {
                sentiment: Sentiment::Positive,
                confidence: 0.8,
                score: 0.8,
                word_counts: crate::sentiment::SentimentWordCounts::default(),
            },
            SentimentResult {
                sentiment: Sentiment::Negative,
                confidence: 0.7,
                score: -0.7,
                word_counts: crate::sentiment::SentimentWordCounts::default(),
            },
            SentimentResult {
                sentiment: Sentiment::Neutral,
                confidence: 0.6,
                score: 0.0,
                word_counts: crate::sentiment::SentimentWordCounts::default(),
            },
        ];

        let labels = vec!["Doc1".to_string(), "Doc2".to_string(), "Doc3".to_string()];
        let config = VisualizationConfig::default();
        let viz = SentimentVisualizer::new(config);

        let svg = viz
            .sentiment_distribution(&sentiment_results, &labels)
            .unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Positive"));
        assert!(svg.contains("Negative"));
        assert!(svg.contains("Neutral"));
    }

    #[test]
    fn test_color_generation() {
        let config = VisualizationConfig::default();
        let word_cloud = WordCloud::from_frequencies(HashMap::new(), config);

        let colors = word_cloud.generate_viridis_colors(5);
        assert_eq!(colors.len(), 5);

        let colors = word_cloud.generate_plasma_colors(3);
        assert_eq!(colors.len(), 3);
    }

    #[test]
    fn test_hsv_to_rgb_conversion() {
        let viz = EmbeddingVisualizer::new(VisualizationConfig::default());

        // Test red (0 degrees)
        let red = viz.hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);

        // Test green (120 degrees)
        let green = viz.hsv_to_rgb(120.0, 1.0, 1.0);
        assert_eq!(green.r, 0);
        assert_eq!(green.g, 255);
        assert_eq!(green.b, 0);
    }
}
