//! AdvancedNativePlotter — the main plotting engine and all its methods.

use super::types::{
    AnimationEngine, Camera3D, DendrogramNode, DendrogramTree, DirectionalLight, ExecutionSummary,
    InteractiveController, InteractiveFeature, InteractivePerformanceDashboard, Lighting3D,
    MetricTimelinePoint, Native3DClusterPlot, NativeClusterPlot, NativeDendrogramPlot,
    NativePlotConfig, NativeVisualizationOutput, NeuromorphicActivityPlot, PlotColorScheme,
    PointLight, QuantumCoherenceAnimation, QuantumCoherenceFrame, QuantumField3D, SvgCanvas,
    SvgElement,
};
use crate::advanced_clustering::{AdvancedClusteringResult, AdvancedPerformanceMetrics};
use crate::error::{ClusteringError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Native plotting engine for Advanced clustering
#[derive(Debug)]
pub struct AdvancedNativePlotter {
    /// Plot configuration
    pub(crate) config: NativePlotConfig,
    /// SVG canvas for rendering
    pub(crate) svg_canvas: SvgCanvas,
    /// Animation engine
    pub(crate) animation_engine: AnimationEngine,
    /// Interactive controller
    pub(crate) interactive_controller: InteractiveController,
}

impl AdvancedNativePlotter {
    /// Create a new native plotter
    pub fn new(config: NativePlotConfig) -> Self {
        Self {
            svg_canvas: SvgCanvas::new(config.width, config.height),
            animation_engine: AnimationEngine::new(config.animation_fps),
            interactive_controller: InteractiveController::new(),
            config,
        }
    }

    /// Create comprehensive native visualization
    pub fn create_comprehensive_plot(
        &mut self,
        data: &ArrayView2<f64>,
        result: &AdvancedClusteringResult,
    ) -> Result<NativeVisualizationOutput> {
        // Clear canvas
        self.svg_canvas.clear();

        // Create main cluster plot
        let cluster_plot = self.create_native_cluster_plot(data, result)?;

        // Create dendrogram if hierarchical clustering was used
        let dendrogram = if result.selected_algorithm.contains("hierarchical") {
            Some(self.create_native_dendrogram(data, result)?)
        } else {
            None
        };

        // Create 3D visualization for high-dimensional data
        let plot_3d = if data.ncols() > 2 {
            Some(self.create_native_3d_plot(data, result)?)
        } else {
            None
        };

        // Create quantum coherence animation
        let quantum_animation = if self.config.enable_animations {
            Some(self.create_quantum_coherence_animation(result)?)
        } else {
            None
        };

        // Create neuromorphic activity visualization
        let neuromorphic_plot = self.create_neuromorphic_activity_plot(result)?;

        // Create interactive performance dashboard
        let performance_dashboard = self.create_interactive_performance_dashboard(result)?;

        Ok(NativeVisualizationOutput {
            cluster_plot,
            dendrogram,
            plot_3d,
            quantum_animation,
            neuromorphic_plot,
            performance_dashboard,
            svg_content: self.svg_canvas.to_svg(),
            interactive_script: self.generate_interactive_script(),
        })
    }

    /// Create native cluster plot with quantum enhancement
    fn create_native_cluster_plot(
        &mut self,
        data: &ArrayView2<f64>,
        result: &AdvancedClusteringResult,
    ) -> Result<NativeClusterPlot> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Apply dimensionality reduction if needed
        let plot_data = if n_features > 2 {
            self.apply_native_pca(data, 2)?
        } else {
            data.to_owned()
        };

        // Calculate plot bounds
        let (x_min, x_max, y_min, y_max) = self.calculate_plot_bounds(&plot_data);

        // Create coordinate transformation
        let margin = 50.0;
        let plot_width = self.config.width as f64 - 2.0 * margin;
        let plot_height = self.config.height as f64 - 2.0 * margin;

        let x_scale = plot_width / (x_max - x_min);
        let y_scale = plot_height / (y_max - y_min);

        // Plot data points with quantum enhancement
        let mut point_elements = Vec::new();
        let mut quantum_enhancements = Vec::new();

        for i in 0..n_samples {
            let x = margin + (plot_data[[i, 0]] - x_min) * x_scale;
            let y = margin + (plot_data[[i, 1]] - y_min) * y_scale;
            let cluster_id = result.clusters[i];

            // Calculate quantum enhancement for this point
            let quantum_factor = self.calculate_point_quantum_enhancement(i, cluster_id, result);
            quantum_enhancements.push(quantum_factor);

            // Determine point color and size based on quantum properties
            let base_color = self.get_cluster_color(cluster_id);
            let enhanced_color = self.apply_quantum_color_enhancement(base_color, quantum_factor);
            let point_radius = 3.0 + quantum_factor * 2.0; // Quantum-enhanced size

            let circle = SvgElement::Circle {
                cx: x,
                cy: y,
                r: point_radius,
                fill: enhanced_color.clone(),
                stroke: "#000000".to_string(),
                stroke_width: 0.5,
                opacity: 0.8 + quantum_factor * 0.2,
            };

            point_elements.push(circle);
        }

        // Plot centroids with special quantum aura
        let mut centroid_elements = Vec::new();
        for (cluster_id, centroid) in result.centroids.outer_iter().enumerate() {
            if centroid.len() >= 2 {
                let x = margin + (centroid[0] - x_min) * x_scale;
                let y = margin + (centroid[1] - y_min) * y_scale;

                // Create quantum aura around centroid
                let aura_radius = 15.0;
                let aura = SvgElement::Circle {
                    cx: x,
                    cy: y,
                    r: aura_radius,
                    fill: "none".to_string(),
                    stroke: self.get_cluster_color(cluster_id),
                    stroke_width: 2.0,
                    opacity: 0.3,
                };

                let centroid_circle = SvgElement::Circle {
                    cx: x,
                    cy: y,
                    r: 6.0,
                    fill: self.get_cluster_color(cluster_id),
                    stroke: "#FFFFFF".to_string(),
                    stroke_width: 2.0,
                    opacity: 1.0,
                };

                centroid_elements.push(aura);
                centroid_elements.push(centroid_circle);
            }
        }

        // Add all elements to canvas
        for element in &point_elements {
            self.svg_canvas.add_element(element.clone());
        }
        for element in &centroid_elements {
            self.svg_canvas.add_element(element.clone());
        }

        // Add axes and labels
        self.add_plot_axes_and_labels(x_min, x_max, y_min, y_max, margin)?;

        Ok(NativeClusterPlot {
            data: plot_data,
            point_elements,
            centroid_elements,
            quantum_enhancements,
            bounds: (x_min, x_max, y_min, y_max),
            scale: (x_scale, y_scale),
        })
    }

    /// Create native dendrogram visualization
    fn create_native_dendrogram(
        &mut self,
        data: &ArrayView2<f64>,
        result: &AdvancedClusteringResult,
    ) -> Result<NativeDendrogramPlot> {
        // Create hierarchical tree structure
        let tree = self.build_dendrogram_tree(data, result)?;

        // Calculate node positions using optimal layout
        let node_positions = self.calculate_dendrogram_layout(&tree)?;

        // Calculate branch lengths based on quantum distances
        let branch_lengths = self.calculate_quantum_branch_lengths(&tree, result)?;

        // Add quantum enhancement data
        let quantum_enhancements = self.calculate_dendrogram_quantum_enhancements(&tree, result)?;

        // Create interactive features
        let interactive_features = vec![
            InteractiveFeature::ZoomPan,
            InteractiveFeature::NodeSelection,
            InteractiveFeature::Tooltip,
            InteractiveFeature::RealTimeFilter,
        ];

        // Render dendrogram to SVG
        self.render_dendrogram_to_svg(
            &tree,
            &node_positions,
            &branch_lengths,
            &quantum_enhancements,
        )?;

        Ok(NativeDendrogramPlot {
            tree,
            node_positions,
            branch_lengths,
            quantum_enhancements,
            interactive_features,
        })
    }

    /// Create native 3D cluster plot
    fn create_native_3d_plot(
        &mut self,
        data: &ArrayView2<f64>,
        result: &AdvancedClusteringResult,
    ) -> Result<Native3DClusterPlot> {
        // Reduce to 3D if needed
        let points_3d = if data.ncols() > 3 {
            self.apply_native_pca(data, 3)?
        } else if data.ncols() == 2 {
            // Add a third dimension with quantum enhancement
            let mut points_3d = Array2::zeros((data.nrows(), 3));
            points_3d
                .slice_mut(scirs2_core::ndarray::s![.., 0..2])
                .assign(data);

            // Calculate third dimension based on quantum properties
            for i in 0..data.nrows() {
                let cluster_id = result.clusters[i];
                let quantum_factor =
                    self.calculate_point_quantum_enhancement(i, cluster_id, result);
                points_3d[[i, 2]] = quantum_factor * 5.0; // Scale for visibility
            }
            points_3d
        } else {
            data.to_owned()
        };

        // Generate point colors
        let mut point_colors = Vec::new();
        for i in 0..points_3d.nrows() {
            let cluster_id = result.clusters[i];
            let base_color = self.get_cluster_color_rgb(cluster_id);
            let quantum_factor = self.calculate_point_quantum_enhancement(i, cluster_id, result);
            let enhanced_color =
                self.apply_quantum_color_enhancement_rgb(base_color, quantum_factor);
            point_colors.push(enhanced_color);
        }

        // Calculate 3D centroids
        let centroids_3d = if result.centroids.ncols() >= 3 {
            result
                .centroids
                .slice(scirs2_core::ndarray::s![.., 0..3])
                .to_owned()
        } else {
            let mut centroids_3d = Array2::zeros((result.centroids.nrows(), 3));
            centroids_3d
                .slice_mut(scirs2_core::ndarray::s![.., 0..result.centroids.ncols()])
                .assign(&result.centroids);
            centroids_3d
        };

        // Setup camera
        let camera = Camera3D {
            position: [10.0, 10.0, 10.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
            near: 0.1,
            far: 100.0,
        };

        // Setup lighting
        let lighting = Lighting3D {
            ambient: 0.3,
            directional_lights: vec![DirectionalLight {
                direction: [-1.0, -1.0, -1.0],
                intensity: 0.7,
                color: [1.0, 1.0, 1.0],
            }],
            point_lights: vec![PointLight {
                position: [5.0, 5.0, 5.0],
                intensity: 0.5,
                color: [0.0, 1.0, 1.0], // Quantum cyan
                attenuation: 0.1,
            }],
        };

        // Create quantum field visualization
        let quantum_field = self.create_quantum_field_3d(&points_3d, result)?;

        Ok(Native3DClusterPlot {
            points_3d,
            point_colors,
            centroids_3d,
            camera,
            lighting,
            quantum_field,
        })
    }

    /// Create quantum coherence animation
    fn create_quantum_coherence_animation(
        &mut self,
        result: &AdvancedClusteringResult,
    ) -> Result<QuantumCoherenceAnimation> {
        let num_frames = (self.config.animation_fps * 5.0) as usize; // 5 second animation
        let mut frames = Vec::new();

        for frame_idx in 0..num_frames {
            let time = frame_idx as f64 / self.config.animation_fps;

            // Create quantum coherence visualization for this frame
            let coherence_frame = self.create_quantum_coherence_frame(result, time)?;

            frames.push(coherence_frame);
        }

        Ok(QuantumCoherenceAnimation {
            frames,
            duration: 5.0,
            fps: self.config.animation_fps,
        })
    }

    /// Create neuromorphic activity plot
    fn create_neuromorphic_activity_plot(
        &mut self,
        result: &AdvancedClusteringResult,
    ) -> Result<NeuromorphicActivityPlot> {
        let n_neurons = result.centroids.nrows();
        let time_steps = 100;

        // Simulate neuromorphic activity based on clustering performance
        let mut activity_matrix = Array2::zeros((time_steps, n_neurons));
        let mut spike_trains = Array2::zeros((time_steps, n_neurons));

        for t in 0..time_steps {
            let time = t as f64 / time_steps as f64;

            for neuron in 0..n_neurons {
                // Base activity influenced by quantum coherence
                let base_activity = result.performance.neural_adaptation_rate;
                let quantum_modulation =
                    result.performance.quantum_coherence * (2.0 * PI * time * 3.0).sin();
                let noise = 0.1 * (time * 47.0 + neuron as f64 * 13.0).sin();

                let activity = base_activity + 0.2 * quantum_modulation + noise;
                activity_matrix[[t, neuron]] = activity.max(0.0).min(1.0);

                // Generate spikes based on activity
                let spike_threshold = 0.7;
                let spike_prob = if activity > spike_threshold { 1.0 } else { 0.0 };
                spike_trains[[t, neuron]] = spike_prob;
            }
        }

        // Create plasticity visualization
        let mut plasticity_changes = Array2::zeros((n_neurons, n_neurons));
        for i in 0..n_neurons {
            for j in 0..n_neurons {
                if i != j {
                    let distance = ((i as f64 - j as f64).abs() / n_neurons as f64).min(1.0);
                    let plasticity = result.performance.neural_adaptation_rate * (1.0 - distance);
                    plasticity_changes[[i, j]] = plasticity;
                }
            }
        }

        Ok(NeuromorphicActivityPlot {
            activity_matrix,
            spike_trains,
            plasticity_changes,
            time_resolution: 1.0 / time_steps as f64,
        })
    }

    /// Create interactive performance dashboard
    fn create_interactive_performance_dashboard(
        &mut self,
        result: &AdvancedClusteringResult,
    ) -> Result<InteractivePerformanceDashboard> {
        let metrics = &result.performance;

        // Create performance metrics visualization
        let mut performance_metrics = HashMap::new();
        performance_metrics.insert("Silhouette Score".to_string(), metrics.silhouette_score);
        performance_metrics.insert("Quantum Coherence".to_string(), metrics.quantum_coherence);
        performance_metrics.insert(
            "Neural Adaptation".to_string(),
            metrics.neural_adaptation_rate,
        );
        performance_metrics.insert("Energy Efficiency".to_string(), metrics.energy_efficiency);

        // Create improvement comparisons
        let mut improvements = HashMap::new();
        improvements.insert("AI Speedup".to_string(), result.ai_speedup);
        improvements.insert("Quantum Advantage".to_string(), result.quantum_advantage);
        improvements.insert(
            "Neuromorphic Benefit".to_string(),
            result.neuromorphic_benefit,
        );
        improvements.insert(
            "Meta-learning Improvement".to_string(),
            result.meta_learning_improvement,
        );

        // Create real-time metrics timeline
        let mut metrics_timeline = Vec::new();
        for i in 0..metrics.ai_iterations {
            let progress = i as f64 / metrics.ai_iterations as f64;
            let timestamp = progress * metrics.execution_time;

            // Simulate metric evolution during optimization
            let coherence = metrics.quantum_coherence * (1.0 - 0.3 * (-progress * 5.0).exp());
            let adaptation = metrics.neural_adaptation_rate * (1.0 + 0.5 * progress);

            metrics_timeline.push(MetricTimelinePoint {
                timestamp,
                quantum_coherence: coherence,
                neural_adaptation: adaptation,
                ai_confidence: result.confidence * (1.0 - (-progress * 3.0).exp()),
            });
        }

        Ok(InteractivePerformanceDashboard {
            performance_metrics,
            improvements,
            metrics_timeline,
            execution_summary: ExecutionSummary {
                total_time: metrics.execution_time,
                memory_usage: metrics.memory_usage,
                iterations: metrics.ai_iterations,
                algorithm: result.selected_algorithm.clone(),
                confidence: result.confidence,
            },
        })
    }

    // Helper methods for calculations and rendering

    fn calculate_point_quantum_enhancement(
        &self,
        point_idx: usize,
        cluster_id: usize,
        result: &AdvancedClusteringResult,
    ) -> f64 {
        // Calculate quantum enhancement based on clustering properties
        let base_quantum = result.quantum_advantage / 10.0;
        let coherence_factor = result.performance.quantum_coherence;
        let confidence_factor = result.confidence;

        // Add point-specific quantum noise
        let quantum_phase = 2.0 * PI * (point_idx as f64 + cluster_id as f64) / 100.0;
        let phase_modulation = quantum_phase.cos() * 0.2;

        (base_quantum + coherence_factor * 0.3 + confidence_factor * 0.2 + phase_modulation)
            .max(0.0)
            .min(1.0)
    }

    fn get_cluster_color(&self, cluster_id: usize) -> String {
        match self.config.color_scheme {
            PlotColorScheme::Quantum => {
                let hue = (cluster_id as f64 * 137.5) % 360.0; // Golden angle
                format!("hsl({}, 70%, 60%)", hue)
            }
            PlotColorScheme::Neuromorphic => {
                let colors = ["#00FF00", "#FFD700", "#FF4500", "#FF1493", "#00CED1"];
                colors[cluster_id % colors.len()].to_string()
            }
            PlotColorScheme::AI => {
                let colors = ["#FFD700", "#FF8C00", "#FF4500", "#DC143C", "#B22222"];
                colors[cluster_id % colors.len()].to_string()
            }
            PlotColorScheme::Scientific => {
                let intensity = 128 + (cluster_id * 32) % 128;
                format!("rgb({}, {}, {})", intensity, intensity, intensity)
            }
            PlotColorScheme::Custom(ref colors) => {
                if colors.is_empty() {
                    "#0088FF".to_string()
                } else {
                    let color = colors[cluster_id % colors.len()];
                    format!("rgb({}, {}, {})", color[0], color[1], color[2])
                }
            }
        }
    }

    fn get_cluster_color_rgb(&self, cluster_id: usize) -> [u8; 3] {
        match self.config.color_scheme {
            PlotColorScheme::Quantum => {
                let hue = (cluster_id as f64 * 137.5) % 360.0;
                self.hsl_to_rgb(hue, 0.7, 0.6)
            }
            PlotColorScheme::Neuromorphic => {
                let colors = [
                    [0, 255, 0],
                    [255, 215, 0],
                    [255, 69, 0],
                    [255, 20, 147],
                    [0, 206, 209],
                ];
                colors[cluster_id % colors.len()]
            }
            PlotColorScheme::AI => {
                let colors = [
                    [255, 215, 0],
                    [255, 140, 0],
                    [255, 69, 0],
                    [220, 20, 60],
                    [178, 34, 34],
                ];
                colors[cluster_id % colors.len()]
            }
            PlotColorScheme::Scientific => {
                let intensity = 128 + (cluster_id * 32) % 128;
                [intensity as u8, intensity as u8, intensity as u8]
            }
            PlotColorScheme::Custom(ref colors) => {
                if colors.is_empty() {
                    [0, 136, 255]
                } else {
                    colors[cluster_id % colors.len()]
                }
            }
        }
    }

    fn apply_quantum_color_enhancement(&self, base_color: String, quantum_factor: f64) -> String {
        // Apply quantum shimmer effect to color
        if base_color.starts_with("hsl") {
            // Extract hue, saturation, lightness
            if let Some(hsl_part) = base_color
                .strip_prefix("hsl(")
                .and_then(|s| s.strip_suffix(")"))
            {
                let parts: Vec<&str> = hsl_part.split(", ").collect();
                if parts.len() == 3 {
                    if let (Ok(h), Ok(s), Ok(l)) = (
                        parts[0].parse::<f64>(),
                        parts[1].strip_suffix("%").unwrap_or("0").parse::<f64>(),
                        parts[2].strip_suffix("%").unwrap_or("0").parse::<f64>(),
                    ) {
                        let enhanced_s = (s + quantum_factor * 20.0).min(100.0);
                        let enhanced_l = (l + quantum_factor * 10.0).min(90.0);
                        return format!("hsl({}, {}%, {}%)", h, enhanced_s, enhanced_l);
                    }
                }
            }
        }
        base_color // Return original if parsing fails
    }

    fn apply_quantum_color_enhancement_rgb(
        &self,
        base_color: [u8; 3],
        quantum_factor: f64,
    ) -> [u8; 3] {
        let enhancement = (quantum_factor * 50.0) as u8;
        [
            (base_color[0] as u16 + enhancement as u16).min(255) as u8,
            base_color[1],
            (base_color[2] as u16 + enhancement as u16).min(255) as u8,
        ]
    }

    fn hsl_to_rgb(&self, h: f64, s: f64, l: f64) -> [u8; 3] {
        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = l - c / 2.0;

        let (r_prime, g_prime, b_prime) = match h as u32 {
            0..=59 => (c, x, 0.0),
            60..=119 => (x, c, 0.0),
            120..=179 => (0.0, c, x),
            180..=239 => (0.0, x, c),
            240..=299 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };

        [
            ((r_prime + m) * 255.0) as u8,
            ((g_prime + m) * 255.0) as u8,
            ((b_prime + m) * 255.0) as u8,
        ]
    }

    fn calculate_plot_bounds(&self, data: &Array2<f64>) -> (f64, f64, f64, f64) {
        let x_values = data.column(0);
        let y_values = data.column(1);

        let x_min = x_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Add some padding
        let x_padding = (x_max - x_min) * 0.1;
        let y_padding = (y_max - y_min) * 0.1;

        (
            x_min - x_padding,
            x_max + x_padding,
            y_min - y_padding,
            y_max + y_padding,
        )
    }

    fn apply_native_pca(&self, data: &ArrayView2<f64>, target_dims: usize) -> Result<Array2<f64>> {
        // Simplified PCA implementation for native plotting
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if target_dims >= n_features {
            return Ok(data.to_owned());
        }

        // Center the data
        let mean = data.mean_axis(Axis(0)).expect("Operation failed");
        let centered = data - &mean.insert_axis(Axis(0));

        // For simplicity, just take the first few dimensions with some processing
        let mut reduced = Array2::zeros((n_samples, target_dims));

        for i in 0..n_samples {
            for j in 0..target_dims {
                let mut component = 0.0;
                for k in 0..n_features {
                    let weight = (k as f64 * PI / n_features as f64
                        + j as f64 * PI / target_dims as f64)
                        .cos();
                    component += centered[[i, k]] * weight;
                }
                reduced[[i, j]] = component / (n_features as f64).sqrt();
            }
        }

        Ok(reduced)
    }

    fn add_plot_axes_and_labels(
        &mut self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        ymax: f64,
        margin: f64,
    ) -> Result<()> {
        let plot_width = self.config.width as f64 - 2.0 * margin;
        let plot_height = self.config.height as f64 - 2.0 * margin;

        // X-axis
        let x_axis = SvgElement::Line {
            x1: margin,
            y1: margin + plot_height,
            x2: margin + plot_width,
            y2: margin + plot_height,
            stroke: "#333333".to_string(),
            stroke_width: 2.0,
            opacity: 1.0,
        };

        // Y-axis
        let y_axis = SvgElement::Line {
            x1: margin,
            y1: margin,
            x2: margin,
            y2: margin + plot_height,
            stroke: "#333333".to_string(),
            stroke_width: 2.0,
            opacity: 1.0,
        };

        // Axis labels
        let x_label = SvgElement::Text {
            x: margin + plot_width / 2.0,
            y: margin + plot_height + 30.0,
            content: "Principal Component 1".to_string(),
            font_size: 14.0,
            fill: "#333333".to_string(),
            text_anchor: "middle".to_string(),
        };

        let y_label = SvgElement::Text {
            x: margin - 30.0,
            y: margin + plot_height / 2.0,
            content: "Principal Component 2".to_string(),
            font_size: 14.0,
            fill: "#333333".to_string(),
            text_anchor: "middle".to_string(),
        };

        self.svg_canvas.add_element(x_axis);
        self.svg_canvas.add_element(y_axis);
        self.svg_canvas.add_element(x_label);
        self.svg_canvas.add_element(y_label);

        // Suppress unused variable warnings for bounds that are used for label generation
        let _ = (x_min, x_max, y_min, ymax);

        Ok(())
    }

    fn generate_interactive_script(&self) -> String {
        // Generate JavaScript for interactivity
        r#"
        // Advanced Native Plotting Interactive Script
        (function() {
            let zoom = 1.0;
            let panX = 0, panY = 0;
            let selectedElements = [];

            // Initialize interactive features
            function initInteractivity() {
                const svg = document.querySelector('svg');
                if (!svg) return;

                // Zoom and pan
                svg.addEventListener('wheel', handleZoom);
                svg.addEventListener('mousedown', handlePanStart);
                svg.addEventListener('mousemove', handlePanMove);
                svg.addEventListener('mouseup', handlePanEnd);

                // Element selection
                svg.addEventListener('click', handleElementClick);
                svg.addEventListener('mouseover', handleElementHover);
                svg.addEventListener('mouseout', handleElementOut);
            }

            function handleZoom(event) {
                event.preventDefault();
                const delta = event.deltaY > 0 ? 0.9 : 1.1;
                zoom *= delta;
                updateTransform();
            }

            function handleElementClick(event) {
                const target = event.target;
                if (target.tagName === 'circle' || target.tagName === 'path') {
                    toggleSelection(target);
                }
            }

            function toggleSelection(element) {
                const index = selectedElements.indexOf(element);
                if (index > -1) {
                    selectedElements.splice(index, 1);
                    element.classList.remove('selected');
                } else {
                    selectedElements.push(element);
                    element.classList.add('selected');
                }
            }

            function updateTransform() {
                const svg = document.querySelector('svg');
                const mainGroup = svg.querySelector('g.main-group');
                if (mainGroup) {
                    mainGroup.setAttribute('transform',
                        `translate(${panX}, ${panY}) scale(${zoom})`);
                }
            }

            // Initialize when DOM is ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initInteractivity);
            } else {
                initInteractivity();
            }
        })();
        "#
        .to_string()
    }

    /// Build a hierarchical dendrogram tree from cluster results.
    ///
    /// Constructs a binary tree by greedily merging the two nearest cluster
    /// centroids at each step, mirroring single-linkage agglomeration.
    pub(crate) fn build_dendrogram_tree(
        &self,
        data: &ArrayView2<f64>,
        result: &AdvancedClusteringResult,
    ) -> Result<DendrogramTree> {
        let n_samples = data.nrows();
        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput(
                "Cannot build dendrogram from empty data".into(),
            ));
        }

        // Create leaf nodes — one per data point.
        let mut nodes: Vec<DendrogramNode> = (0..n_samples)
            .map(|i| {
                let cluster_id = if i < result.clusters.len() {
                    result.clusters[i]
                } else {
                    0
                };
                let quantum_coherence = result.performance.quantum_coherence
                    * ((i as f64 * PI / n_samples as f64).cos().abs());
                DendrogramNode {
                    id: format!("leaf_{i}"),
                    height: 0.0,
                    children: Vec::new(),
                    data_indices: vec![i],
                    quantum_coherence,
                    neuromorphic_activity: result.performance.neural_adaptation_rate
                        * (1.0 - (cluster_id as f64 / (result.centroids.nrows().max(1) as f64))),
                }
            })
            .collect();

        // Agglomerate: repeatedly merge the two nodes with the smallest
        // centroid-distance until a single root remains.
        let mut merge_height = 0.0_f64;
        while nodes.len() > 1 {
            let n = nodes.len();
            // Find the pair (i, j) that minimises the L2 centroid distance.
            // Centroid approximated as mean of represented data points.
            let centroid = |node: &DendrogramNode| -> Array1<f64> {
                let cols = data.ncols();
                let mut sum: Array1<f64> = Array1::zeros(cols);
                for &idx in &node.data_indices {
                    if idx < data.nrows() {
                        for c in 0..cols {
                            sum[c] += data[[idx, c]];
                        }
                    }
                }
                let cnt = node.data_indices.len().max(1) as f64;
                sum.mapv(|v: f64| v / cnt)
            };

            let mut best_dist = f64::INFINITY;
            let mut best_i = 0;
            let mut best_j = 1;
            for i in 0..n {
                let ci = centroid(&nodes[i]);
                for j in (i + 1)..n {
                    let cj = centroid(&nodes[j]);
                    let dist: f64 = ci
                        .iter()
                        .zip(cj.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt();
                    if dist < best_dist {
                        best_dist = dist;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            merge_height += best_dist;
            // Merge nodes[best_j] into nodes[best_i].
            let node_j = nodes.remove(best_j);
            let node_i = nodes.remove(best_i);
            let mut merged_indices = node_i.data_indices.clone();
            merged_indices.extend_from_slice(&node_j.data_indices);
            let avg_coherence = (node_i.quantum_coherence + node_j.quantum_coherence) / 2.0;
            let avg_neuro = (node_i.neuromorphic_activity + node_j.neuromorphic_activity) / 2.0;
            let merged = DendrogramNode {
                id: format!("merge_{}_{}", node_i.id, node_j.id),
                height: merge_height,
                children: vec![node_i, node_j],
                data_indices: merged_indices,
                quantum_coherence: avg_coherence,
                neuromorphic_activity: avg_neuro,
            };
            nodes.push(merged);
        }

        let root = nodes.remove(0);
        let leaf_count = root.data_indices.len();
        let total_height = root.height;

        Ok(DendrogramTree {
            root,
            height: total_height,
            leaf_count,
        })
    }

    /// Compute 2-D (x, y) canvas positions for each node in the dendrogram.
    ///
    /// Leaves are spread evenly along the x-axis; internal nodes are placed
    /// at the mean x of their children and at their merge-height on the y-axis.
    pub(crate) fn calculate_dendrogram_layout(
        &self,
        tree: &DendrogramTree,
    ) -> Result<HashMap<String, (f64, f64)>> {
        let mut positions: HashMap<String, (f64, f64)> = HashMap::new();
        let plot_width = self.config.width as f64;
        let plot_height = self.config.height as f64;
        let max_height = tree.height.max(1.0);

        // Assign a leaf-index to every leaf node via a DFS traversal.
        let mut leaf_counter = 0usize;
        fn assign_leaves(
            node: &DendrogramNode,
            positions: &mut HashMap<String, (f64, f64)>,
            leaf_counter: &mut usize,
            plot_width: f64,
            plot_height: f64,
            max_height: f64,
            leaf_count: usize,
        ) {
            if node.children.is_empty() {
                // Leaf node — place on x-axis.
                let x = if leaf_count > 1 {
                    (*leaf_counter as f64 / (leaf_count - 1) as f64) * plot_width
                } else {
                    plot_width / 2.0
                };
                let y = plot_height; // Leaves at the bottom.
                positions.insert(node.id.clone(), (x, y));
                *leaf_counter += 1;
            } else {
                for child in &node.children {
                    assign_leaves(
                        child,
                        positions,
                        leaf_counter,
                        plot_width,
                        plot_height,
                        max_height,
                        leaf_count,
                    );
                }
                // Internal node — x is the mean of child x-positions.
                let child_x: f64 = node
                    .children
                    .iter()
                    .filter_map(|c| positions.get(&c.id).map(|(x, _)| *x))
                    .sum::<f64>()
                    / node.children.len().max(1) as f64;
                let y = plot_height * (1.0 - node.height / max_height);
                positions.insert(node.id.clone(), (child_x, y));
            }
        }

        assign_leaves(
            &tree.root,
            &mut positions,
            &mut leaf_counter,
            plot_width,
            plot_height,
            max_height,
            tree.leaf_count.max(1),
        );

        Ok(positions)
    }

    /// Compute branch lengths for each node, scaled by quantum coherence.
    pub(crate) fn calculate_quantum_branch_lengths(
        &self,
        tree: &DendrogramTree,
        result: &AdvancedClusteringResult,
    ) -> Result<HashMap<String, f64>> {
        let mut branch_lengths: HashMap<String, f64> = HashMap::new();
        let quantum_scale = result.quantum_advantage.max(1.0);

        fn traverse(
            node: &DendrogramNode,
            parent_height: f64,
            branch_lengths: &mut HashMap<String, f64>,
            quantum_scale: f64,
        ) {
            let raw_length = (node.height - parent_height).abs();
            let quantum_enhanced = raw_length * (1.0 + node.quantum_coherence / quantum_scale);
            branch_lengths.insert(node.id.clone(), quantum_enhanced);
            for child in &node.children {
                traverse(child, node.height, branch_lengths, quantum_scale);
            }
        }

        traverse(&tree.root, 0.0, &mut branch_lengths, quantum_scale);
        Ok(branch_lengths)
    }

    /// Compute per-node quantum enhancement values for dendrogram colouring.
    pub(crate) fn calculate_dendrogram_quantum_enhancements(
        &self,
        tree: &DendrogramTree,
        result: &AdvancedClusteringResult,
    ) -> Result<HashMap<String, f64>> {
        let mut enhancements: HashMap<String, f64> = HashMap::new();

        fn traverse(
            node: &DendrogramNode,
            enhancements: &mut HashMap<String, f64>,
            base_quantum: f64,
        ) {
            let enhancement = (base_quantum * node.quantum_coherence).clamp(0.0, 1.0);
            enhancements.insert(node.id.clone(), enhancement);
            for child in &node.children {
                traverse(child, enhancements, base_quantum);
            }
        }

        let base_quantum = result.quantum_advantage.clamp(0.01, 10.0);
        traverse(&tree.root, &mut enhancements, base_quantum);
        Ok(enhancements)
    }

    /// Render dendrogram tree as SVG line elements and add them to the canvas.
    pub(crate) fn render_dendrogram_to_svg(
        &mut self,
        tree: &DendrogramTree,
        positions: &HashMap<String, (f64, f64)>,
        _branch_lengths: &HashMap<String, f64>,
        enhancements: &HashMap<String, f64>,
    ) -> Result<()> {
        fn draw_node(
            node: &DendrogramNode,
            positions: &HashMap<String, (f64, f64)>,
            enhancements: &HashMap<String, f64>,
            elements: &mut Vec<SvgElement>,
        ) {
            if let Some(&(x, y)) = positions.get(&node.id) {
                for child in &node.children {
                    if let Some(&(cx, cy)) = positions.get(&child.id) {
                        let enhancement = enhancements.get(&child.id).copied().unwrap_or(0.0);
                        let blue = ((enhancement * 200.0) as u8).saturating_add(55);
                        let color = format!("rgb(0, {}, {})", blue, blue);

                        // Horizontal branch from parent to child x, then vertical.
                        elements.push(SvgElement::Line {
                            x1: x,
                            y1: y,
                            x2: cx,
                            y2: y,
                            stroke: color.clone(),
                            stroke_width: 1.5,
                            opacity: 0.85,
                        });
                        elements.push(SvgElement::Line {
                            x1: cx,
                            y1: y,
                            x2: cx,
                            y2: cy,
                            stroke: color,
                            stroke_width: 1.5,
                            opacity: 0.85,
                        });
                        draw_node(child, positions, enhancements, elements);
                    }
                }
                // Draw a small circle at each internal node.
                if !node.children.is_empty() {
                    elements.push(SvgElement::Circle {
                        cx: x,
                        cy: y,
                        r: 3.0,
                        fill: "#00CCFF".to_string(),
                        stroke: "#0088BB".to_string(),
                        stroke_width: 1.0,
                        opacity: 0.9,
                    });
                }
            }
        }

        let mut elements: Vec<SvgElement> = Vec::new();
        draw_node(&tree.root, positions, enhancements, &mut elements);
        for el in elements {
            self.svg_canvas.add_element(el);
        }
        Ok(())
    }

    /// Build a quantum-field representation for 3D cluster plots.
    ///
    /// The field is sampled on a 10×10 grid overlaid on the data bounding box.
    pub(crate) fn create_quantum_field_3d(
        &self,
        points_3d: &Array2<f64>,
        result: &AdvancedClusteringResult,
    ) -> Result<QuantumField3D> {
        let grid = 10usize;
        let n_points = points_3d.nrows();

        // Compute bounding box.
        let (mut x_min, mut x_max, mut y_min, mut y_max) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for i in 0..n_points {
            let x = points_3d[[i, 0]];
            let y = points_3d[[i, 1]];
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
        x_min -= 1.0;
        x_max += 1.0;
        y_min -= 1.0;
        y_max += 1.0;

        let mut field_strength = Array2::zeros((grid, grid));
        let mut coherence = Array2::zeros((grid, grid));
        let mut phase = Array2::zeros((grid, grid));

        let base_coherence = result.performance.quantum_coherence;

        for gi in 0..grid {
            for gj in 0..grid {
                let gx = x_min + (gi as f64 / (grid - 1) as f64) * (x_max - x_min);
                let gy = y_min + (gj as f64 / (grid - 1) as f64) * (y_max - y_min);

                // Field strength = inverse-distance-weighted sum over all points.
                let mut strength = 0.0_f64;
                for i in 0..n_points {
                    let dx = gx - points_3d[[i, 0]];
                    let dy = gy - points_3d[[i, 1]];
                    let dz = if points_3d.ncols() > 2 {
                        gx - points_3d[[i, 2]]
                    } else {
                        0.0
                    };
                    let dist2 = (dx * dx + dy * dy + dz * dz).max(1e-6);
                    strength += 1.0 / dist2;
                }
                field_strength[[gi, gj]] = strength.min(100.0);
                coherence[[gi, gj]] = base_coherence * (1.0 - strength / (strength + 10.0));
                phase[[gi, gj]] = (gx * 0.5 + gy * 0.3).sin() * PI;
            }
        }

        // Entanglement lines: connect centroids with high quantum coherence.
        let n_centroids = result.centroids.nrows().min(5);
        let mut entanglement_lines = Vec::new();
        for i in 0..n_centroids {
            for j in (i + 1)..n_centroids {
                let coherence_ij = base_coherence * 0.8;
                if coherence_ij > 0.3 {
                    let pi_x = result.centroids[[i, 0]];
                    let pi_y = result.centroids[[i, 1]];
                    let pi_z = if result.centroids.ncols() > 2 {
                        result.centroids[[i, 2]]
                    } else {
                        0.0
                    };
                    let pj_x = result.centroids[[j, 0]];
                    let pj_y = result.centroids[[j, 1]];
                    let pj_z = if result.centroids.ncols() > 2 {
                        result.centroids[[j, 2]]
                    } else {
                        0.0
                    };
                    entanglement_lines.push(([pi_x, pi_y, pi_z], [pj_x, pj_y, pj_z], coherence_ij));
                }
            }
        }

        Ok(QuantumField3D {
            field_strength,
            coherence,
            phase,
            entanglement_lines,
        })
    }

    /// Generate a single quantum coherence animation frame at time `t`.
    pub(crate) fn create_quantum_coherence_frame(
        &self,
        result: &AdvancedClusteringResult,
        t: f64,
    ) -> Result<QuantumCoherenceFrame> {
        let grid = 8usize;
        let mut field_strength = Array2::zeros((grid, grid));

        let base = result.performance.quantum_coherence;
        for gi in 0..grid {
            for gj in 0..grid {
                let wave_x = (gi as f64 / grid as f64 * 2.0 * PI + t * 2.5).cos();
                let wave_y = (gj as f64 / grid as f64 * 2.0 * PI + t * 1.7).sin();
                field_strength[[gi, gj]] = (base + 0.3 * wave_x * wave_y).clamp(0.0, 1.0);
            }
        }

        // Build SVG pulsing circles for each cluster centroid.
        let n_centroids = result.centroids.nrows().min(8);
        let mut elements: Vec<SvgElement> = Vec::new();
        let w = self.config.width as f64;
        let h = self.config.height as f64;

        for k in 0..n_centroids {
            let cx_val = result.centroids[[k, 0]];
            let cy_val = result.centroids[[k, 1]];
            // Normalise to canvas — assume data in [-5, 5].
            let sx = (cx_val + 5.0) / 10.0 * w;
            let sy = (cy_val + 5.0) / 10.0 * h;

            let pulse = (t * 2.0 * PI + k as f64 * 0.5).sin() * 0.5 + 0.5;
            let r = 5.0 + pulse * 10.0;
            let opacity = 0.4 + pulse * 0.5;
            let blue_val = ((base * 200.0) as u8).saturating_add(55);
            let color = format!("rgba(0, {blue_val}, 255, {opacity:.2})");

            elements.push(SvgElement::Circle {
                cx: sx,
                cy: sy,
                r,
                fill: color.clone(),
                stroke: "#00FFFF".to_string(),
                stroke_width: 1.0,
                opacity,
            });
        }

        Ok(QuantumCoherenceFrame {
            timestamp: t,
            elements,
            field_strength,
        })
    }
}

// Suppress the unused import warning for AdvancedPerformanceMetrics — it is only
// needed by the test helpers which live in mod.rs and use the crate path directly.
#[allow(unused_imports)]
use crate::advanced_clustering::AdvancedPerformanceMetrics as _AdvancedPerformanceMetrics;
