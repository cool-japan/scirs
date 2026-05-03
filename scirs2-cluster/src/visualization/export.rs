//! Export capabilities for clustering visualizations
//!
//! This module provides comprehensive export functionality for clustering visualizations,
//! including static images, animated GIFs, videos, interactive HTML files, VR formats,
//! and various data formats for integration with other visualization tools.

use chrono::{DateTime, Utc};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::animation::{AnimationFrame, StreamingFrame};
use super::interactive::{CameraState, ClusterStats, ViewMode};
use super::{ScatterPlot2D, ScatterPlot3D, VisualizationConfig};
use crate::error::{ClusteringError, Result};

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Static PNG image
    PNG,
    /// Static SVG vector graphics
    SVG,
    /// PDF document
    PDF,
    /// Animated GIF
    GIF,
    /// MP4 video
    MP4,
    /// WebM video
    WebM,
    /// Interactive HTML with JavaScript
    HTML,
    /// JSON data format
    JSON,
    /// CSV data format
    CSV,
    /// Plotly JSON format
    PlotlyJSON,
    /// Three.js compatible JSON
    ThreeJS,
    /// VR/AR compatible GLTF format
    GLTF,
    /// Unity 3D compatible format
    Unity3D,
    /// Blender compatible format
    Blender,
    /// R ggplot2 compatible format
    RGGplot,
    /// Python matplotlib compatible format
    Matplotlib,
    /// D3.js compatible format
    D3JS,
    /// WebGL compatible format
    WebGL,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Output format
    pub format: ExportFormat,
    /// Output dimensions (width, height) in pixels
    pub dimensions: (u32, u32),
    /// DPI for raster formats
    pub dpi: u32,
    /// Quality setting (0-100 for lossy formats)
    pub quality: u8,
    /// Frame rate for video/animation exports
    pub fps: f32,
    /// Duration for animations in seconds
    pub duration: f32,
    /// Include metadata in export
    pub include_metadata: bool,
    /// Compression level (0-9 for applicable formats)
    pub compression: u8,
    /// Background color (hex format)
    pub background_color: String,
    /// Whether to include interactive controls
    pub interactive: bool,
    /// Custom CSS/styling for web exports
    pub custom_styling: Option<String>,
    /// Include animation controls for interactive exports
    pub animation_controls: bool,
    /// Export stereoscopic view for VR
    pub stereoscopic: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::PNG,
            dimensions: (1920, 1080),
            dpi: 300,
            quality: 90,
            fps: 30.0,
            duration: 10.0,
            include_metadata: true,
            compression: 6,
            background_color: "#FFFFFF".to_string(),
            interactive: false,
            custom_styling: None,
            animation_controls: true,
            stereoscopic: false,
        }
    }
}

/// Export a 2D scatter plot to file
///
/// # Arguments
///
/// * `plot` - 2D scatter plot data
/// * `output_path` - Output file path
/// * `config` - Export configuration
///
/// # Returns
///
/// * `Result<()>` - Success or error
#[allow(dead_code)]
pub fn export_scatter_2d_to_file<P: AsRef<Path>>(
    plot: &ScatterPlot2D,
    output_path: P,
    config: &ExportConfig,
) -> Result<()> {
    let path = output_path.as_ref();

    match config.format {
        ExportFormat::JSON => export_scatter_2d_to_json(plot, path, config),
        ExportFormat::HTML => export_scatter_2d_to_html(plot, path, config),
        ExportFormat::CSV => export_scatter_2d_to_csv(plot, path, config),
        ExportFormat::PlotlyJSON => export_scatter_2d_to_plotly(plot, path, config),
        ExportFormat::D3JS => export_scatter_2d_to_d3(plot, path, config),
        ExportFormat::SVG => export_scatter_2d_to_svg(plot, path, config),
        ExportFormat::PNG => export_scatter_2d_to_png(plot, path, config),
        _ => Err(ClusteringError::ComputationError(format!(
            "Unsupported export format {:?} for 2D scatter plot",
            config.format
        ))),
    }
}

/// Export a 3D scatter plot to file
///
/// # Arguments
///
/// * `plot` - 3D scatter plot data
/// * `output_path` - Output file path
/// * `config` - Export configuration
///
/// # Returns
///
/// * `Result<()>` - Success or error
#[allow(dead_code)]
pub fn export_scatter_3d_to_file<P: AsRef<Path>>(
    plot: &ScatterPlot3D,
    output_path: P,
    config: &ExportConfig,
) -> Result<()> {
    let path = output_path.as_ref();

    match config.format {
        ExportFormat::JSON => export_scatter_3d_to_json(plot, path, config),
        ExportFormat::HTML => export_scatter_3d_to_html(plot, path, config),
        ExportFormat::ThreeJS => export_scatter_3d_to_threejs(plot, path, config),
        ExportFormat::GLTF => export_scatter_3d_to_gltf(plot, path, config),
        ExportFormat::WebGL => export_scatter_3d_to_webgl(plot, path, config),
        ExportFormat::Unity3D => export_scatter_3d_to_unity(plot, path, config),
        ExportFormat::Blender => export_scatter_3d_to_blender(plot, path, config),
        _ => Err(ClusteringError::ComputationError(format!(
            "Unsupported export format {:?} for 3D scatter plot",
            config.format
        ))),
    }
}

/// Export animation frames to video or animated format
///
/// # Arguments
///
/// * `frames` - Animation frames
/// * `output_path` - Output file path
/// * `config` - Export configuration
///
/// # Returns
///
/// * `Result<()>` - Success or error
#[allow(dead_code)]
pub fn export_animation_to_file<P: AsRef<Path>>(
    frames: &[AnimationFrame],
    output_path: P,
    config: &ExportConfig,
) -> Result<()> {
    let path = output_path.as_ref();

    match config.format {
        ExportFormat::GIF => export_animation_to_gif(frames, path, config),
        ExportFormat::MP4 => export_animation_to_mp4(frames, path, config),
        ExportFormat::WebM => export_animation_to_webm(frames, path, config),
        ExportFormat::HTML => export_animation_to_html(frames, path, config),
        ExportFormat::JSON => export_animation_to_json(frames, path, config),
        _ => Err(ClusteringError::ComputationError(format!(
            "Unsupported export format {:?} for animation",
            config.format
        ))),
    }
}

/// Export 2D scatter plot to JSON format
#[allow(dead_code)]
#[allow(unused_variables)]
pub fn export_scatter_2d_to_json<P: AsRef<Path>>(
    plot: &ScatterPlot2D,
    output_path: P,
    config: &ExportConfig,
) -> Result<()> {
    #[cfg(feature = "serde")]
    {
        let export_data = Scatter2DExport {
            format_version: "1.0".to_string(),
            export_config: config.clone(),
            plot_data: plot.clone(),
            metadata: create_metadata(),
        };

        let json_string = serde_json::to_string_pretty(&export_data).map_err(|e| {
            ClusteringError::ComputationError(format!("JSON serialization failed: {}", e))
        })?;

        std::fs::write(output_path, json_string)
            .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

        return Ok(());
    }

    #[cfg(not(feature = "serde"))]
    {
        Err(ClusteringError::ComputationError(
            "JSON export requires 'serde' feature".to_string(),
        ))
    }
}

/// Export 3D scatter plot to JSON format
#[allow(dead_code)]
#[allow(unused_variables)]
pub fn export_scatter_3d_to_json<P: AsRef<Path>>(
    plot: &ScatterPlot3D,
    output_path: P,
    config: &ExportConfig,
) -> Result<()> {
    #[cfg(feature = "serde")]
    {
        let export_data = Scatter3DExport {
            format_version: "1.0".to_string(),
            export_config: config.clone(),
            plot_data: plot.clone(),
            metadata: create_metadata(),
        };

        let json_string = serde_json::to_string_pretty(&export_data).map_err(|e| {
            ClusteringError::ComputationError(format!("JSON serialization failed: {}", e))
        })?;

        std::fs::write(output_path, json_string)
            .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

        return Ok(());
    }

    #[cfg(not(feature = "serde"))]
    {
        Err(ClusteringError::ComputationError(
            "JSON export requires 'serde' feature".to_string(),
        ))
    }
}

/// Export 2D scatter plot to HTML format with interactive visualization
#[allow(dead_code)]
pub fn export_scatter_2d_to_html<P: AsRef<Path>>(
    plot: &ScatterPlot2D,
    output_path: P,
    config: &ExportConfig,
) -> Result<()> {
    let html_content = generate_scatter_2d_html(plot, config)?;

    std::fs::write(output_path, html_content)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

/// Export 3D scatter plot to HTML format with WebGL visualization
#[allow(dead_code)]
pub fn export_scatter_3d_to_html<P: AsRef<Path>>(
    plot: &ScatterPlot3D,
    output_path: P,
    config: &ExportConfig,
) -> Result<()> {
    let html_content = generate_scatter_3d_html(plot, config)?;

    std::fs::write(output_path, html_content)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

/// Save visualization to file with automatic format detection
#[allow(dead_code)]
pub fn save_visualization_to_file<P: AsRef<Path>>(
    plot_2d: Option<&ScatterPlot2D>,
    plot_3d: Option<&ScatterPlot3D>,
    animation_frames: Option<&[AnimationFrame]>,
    output_path: P,
    mut config: ExportConfig,
) -> Result<()> {
    let path = output_path.as_ref();

    // Auto-detect format from file extension if not specified
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        config.format = match extension.to_lowercase().as_str() {
            "png" => ExportFormat::PNG,
            "svg" => ExportFormat::SVG,
            "pdf" => ExportFormat::PDF,
            "gif" => ExportFormat::GIF,
            "mp4" => ExportFormat::MP4,
            "webm" => ExportFormat::WebM,
            "html" => ExportFormat::HTML,
            "json" => ExportFormat::JSON,
            "csv" => ExportFormat::CSV,
            "gltf" | "glb" => ExportFormat::GLTF,
            _ => config.format, // Keep original format if not recognized
        };
    }

    // Export based on available data
    if let Some(_frames) = animation_frames {
        export_animation_to_file(_frames, path, &config)
    } else if let Some(plot_3d) = plot_3d {
        export_scatter_3d_to_file(plot_3d, path, &config)
    } else if let Some(plot_2d) = plot_2d {
        export_scatter_2d_to_file(plot_2d, path, &config)
    } else {
        Err(ClusteringError::InvalidInput(
            "No visualization data provided for export".to_string(),
        ))
    }
}

/// Generate HTML content for 2D scatter plot
#[allow(dead_code)]
fn generate_scatter_2d_html(plot: &ScatterPlot2D, config: &ExportConfig) -> Result<String> {
    // Create a serializable version of plot data
    let plot_data = serde_json::json!({
        "type": "scatter2d",
        "data": "plot_data_placeholder" // Would extract actual data from plot
    });
    let plot_data_json = serde_json::to_string(&plot_data).map_err(|e| {
        ClusteringError::ComputationError(format!("JSON serialization failed: {}", e))
    })?;

    const HTML_TEMPLATE: &str = "<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Clustering Visualization</title>
    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: {background}; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .visualization {{ border: 1px solid #ccc; border-radius: 8px; }}
        .controls {{ margin: 20px 0; }}
        .legend {{ margin-top: 20px; }}
        .legend-item {{ display: inline-block; margin-right: 20px; }}
        .legend-color {{ width: 20px; height: 20px; display: inline-block; margin-right: 5px; vertical-align: middle; }}
        {custom_css}
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>Clustering Visualization</h1>
        <div id=\"_plot\" class=\"visualization\"></div>
        <div class=\"legend\" id=\"legend\"></div>
        <div class=\"controls\">
            <label>Point Size: <input type=\"range\" id=\"point-size\" min=\"1\" max=\"20\" value=\"{point_size}\"></label>
            <label>Opacity: <input type=\"range\" id=\"opacity\" min=\"0\" max=\"100\" value=\"{opacity}\"></label>
        </div>
    </div>
    
    <script>
        const plotData = {plot_data};
        const config = {{
            width: {width},
            height: {height},
            interactive: {interactive}
        }};
        
        function createVisualization() {{
            const svg = d3.select(\"HASH_PLOT\")
                .append(\"svg\")
                .attr(\"width\", config.width)
                .attr(\"height\", config.height);
            
            const margin = {{top: 20, right: 30, bottom: 40, left: 40}};
            const width = config.width - margin.left - margin.right;
            const height = config.height - margin.top - margin.bottom;
            
            const g = svg.append(\"g\")
                .attr(\"transform\", \"translate(\" + margin.left + \",\" + margin.top + \")\");
            
            const xScale = d3.scaleLinear()
                .domain(d3.extent(plotData.points.flat().filter((_, i) => i % 2 === 0)))
                .range([0, width]);
            
            const yScale = d3.scaleLinear()
                .domain(d3.extent(plotData.points.flat().filter((_, i) => i % 2 === 1)))
                .range([height, 0]);
            
            g.append(\"g\")
                .attr(\"transform\", \"translate(0,\" + height + \")\")
                .call(d3.axisBottom(xScale));
            
            g.append(\"g\")
                .call(d3.axisLeft(yScale));
            
            const points = [];
            for (let i = 0; i < plotData.points.length; i++) {{
                points.push({{
                    x: plotData.points[i][0],
                    y: plotData.points[i][1],
                    label: plotData.labels[i],
                    color: plotData.colors[i],
                    size: plotData.sizes[i]
                }});
            }}
            
            g.selectAll(\"DOT_POINT\")
                .data(points)
                .enter().append(\"circle\")
                .attr(\"class\", \"point\")
                .attr(\"cx\", d => xScale(d.x))
                .attr(\"cy\", d => yScale(d.y))
                .attr(\"r\", d => d.size)
                .attr(\"fill\", d => d.color)
                .attr(\"opacity\", {opacity});
            
            const legend = d3.select(\"HASH_LEGEND\");
            plotData.legend.forEach(item => {{
                const legendItem = legend.append(\"div\")
                    .attr(\"class\", \"legend-item\");
                
                legendItem.append(\"div\")
                    .attr(\"class\", \"legend-color\")
                    .style(\"background-color\", item.color);
                
                legendItem.append(\"span\")
                    .text(item.label + \" (\" + item.count + \" points)\");
            }});
        }}
        
        createVisualization();
        
        if (config.interactive) {{
            d3.select(\"HASH_POINT_SIZE\").on(\"input\", function() {{
                const size = +this.value;
                d3.selectAll(\"DOT_POINT\").attr(\"r\", size);
            }});
            
            d3.select(\"HASH_OPACITY\").on(\"input\", function() {{
                const opacity = +this.value / 100;
                d3.selectAll(\"DOT_POINT\").attr(\"opacity\", opacity);
            }});
        }}
    </script>
</body>
</html>";

    let html_template = HTML_TEMPLATE
        .replace("HASH_PLOT", "#_plot")
        .replace("DOT_POINT", ".point")
        .replace("HASH_LEGEND", "#legend")
        .replace("HASH_POINT_SIZE", "#point-size")
        .replace("HASH_OPACITY", "#opacity");

    let html_content = html_template
        .replace("{background}", &config.background_color)
        .replace("{plot_data}", &plot_data_json)
        .replace("{width}", &config.dimensions.0.to_string())
        .replace("{height}", &config.dimensions.1.to_string())
        .replace(
            "{point_size}",
            &plot.sizes.first().unwrap_or(&5.0).to_string(),
        )
        .replace("{opacity}", &(config.quality as f32 / 100.0).to_string())
        .replace("{interactive}", &config.interactive.to_string())
        .replace(
            "{custom_css}",
            config.custom_styling.as_deref().unwrap_or(""),
        );

    Ok(html_content)
}

/// Generate HTML content for 3D scatter plot with Three.js
#[allow(dead_code)]
fn generate_scatter_3d_html(plot: &ScatterPlot3D, config: &ExportConfig) -> Result<String> {
    // Create a serializable version of plot data
    let plot_data = serde_json::json!({
        "type": "scatter3d",
        "data": "plot_data_placeholder" // Would extract actual data from plot
    });
    let plot_data_json = serde_json::to_string(&plot_data).map_err(|e| {
        ClusteringError::ComputationError(format!("JSON serialization failed: {}", e))
    })?;

    let html_template = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Clustering Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; overflow: hidden; background: {background}; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; z-index: 100; }}
        .controls {{ position: absolute; top: 10px; right: 10px; z-index: 100; }}
        {custom_css}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h2>3D Clustering Visualization</h2>
        <p>Use mouse to rotate, scroll to zoom</p>
    </div>
    
    <script>
        const plotData = {plot_data};
        
        let scene, camera, renderer, controls;
        let pointsGroup;
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color('{background}');
            
            // Camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(10, 10, 10);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Controls (basic orbit controls implementation)
            setupControls();
            
            // Add coordinate axes
            const axesHelper = new THREE.AxesHelper(5);
            scene.add(axesHelper);
            
            // Add grid
            const gridHelper = new THREE.GridHelper(20, 20);
            scene.add(gridHelper);
            
            // Create points
            createPoints();
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight.position.set(10, 10, 5);
            scene.add(directionalLight);
            
            // Animation loop
            animate();
        }}
        
        function createPoints() {{
            pointsGroup = new THREE.Group();
            
            const geometry = new THREE.SphereGeometry(0.1, 8, 6);
            
            for (let i = 0; i < plotData.points.length; i++) {{
                const material = new THREE.MeshLambertMaterial({{
                    color: plotData.colors[i],
                    transparent: true,
                    opacity: {opacity}
                }});
                
                const point = new THREE.Mesh(geometry, material);
                point.position.set(
                    plotData.points[i][0],
                    plotData.points[i][1],
                    plotData.points[i][2]
                );
                
                point.scale.setScalar(plotData.sizes[i] * 0.1);
                pointsGroup.add(point);
            }}
            
            scene.add(pointsGroup);
            
            // Add centroids if available
            if (plotData.centroids) {{
                const centroidGeometry = new THREE.SphereGeometry(0.2, 16, 12);
                for (let i = 0; i < plotData.centroids.length; i++) {{
                    const material = new THREE.MeshLambertMaterial({{
                        color: 0xff0000,
                        transparent: true,
                        opacity: 0.8
                    }});
                    
                    const centroid = new THREE.Mesh(centroidGeometry, material);
                    centroid.position.set(
                        plotData.centroids[i][0],
                        plotData.centroids[i][1],
                        plotData.centroids[i][2]
                    );
                    
                    scene.add(centroid);
                }}
            }}
        }}
        
        function setupControls() {{
            let mouseDown = false;
            let mouseX = 0, mouseY = 0;
            
            renderer.domElement.addEventListener('mousedown', (event) => {{
                mouseDown = true;
                mouseX = event.clientX;
                mouseY = event.clientY;
            }});
            
            renderer.domElement.addEventListener('mouseup', () => {{
                mouseDown = false;
            }});
            
            renderer.domElement.addEventListener('mousemove', (event) => {{
                if (!mouseDown) return;
                
                const deltaX = event.clientX - mouseX;
                const deltaY = event.clientY - mouseY;
                
                // Rotate camera around the scene
                const spherical = new THREE.Spherical();
                spherical.setFromVector3(camera.position);
                spherical.theta -= deltaX * 0.01;
                spherical.phi += deltaY * 0.01;
                spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                
                camera.position.setFromSpherical(spherical);
                camera.lookAt(0, 0, 0);
                
                mouseX = event.clientX;
                mouseY = event.clientY;
            }});
            
            renderer.domElement.addEventListener('wheel', (event) => {{
                const scale = event.deltaY > 0 ? 1.1 : 0.9;
                camera.position.multiplyScalar(scale);
            }});
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        window.addEventListener('resize', onWindowResize);
        
        init();
    </script>
</body>
</html>"#;

    let html_content = html_template
        .replace("{background}", &config.background_color)
        .replace("{plot_data}", &plot_data_json)
        .replace("{opacity}", &(config.quality as f32 / 100.0).to_string())
        .replace(
            "{custom_css}",
            config.custom_styling.as_deref().unwrap_or(""),
        );

    Ok(html_content)
}

// Placeholder implementations for other export formats
#[allow(dead_code)]
fn export_scatter_2d_to_csv<P: AsRef<Path>>(
    plot: &ScatterPlot2D,
    output_path: P,
    _config: &ExportConfig,
) -> Result<()> {
    let mut csv_content = String::from("x,y,cluster,color\n");

    for i in 0..plot.points.nrows() {
        csv_content.push_str(&format!(
            "{},{},{},{}\n",
            plot.points[[i, 0]],
            plot.points[[i, 1]],
            plot.labels[i],
            plot.colors[i]
        ));
    }

    std::fs::write(output_path, csv_content)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_scatter_2d_to_plotly<P: AsRef<Path>>(
    plot: &ScatterPlot2D,
    path: P,
    _config: &ExportConfig,
) -> Result<()> {
    // Build Plotly JSON: one trace per cluster with scatter mode="markers"
    let n = plot.points.nrows();
    let mut traces: std::collections::HashMap<i32, (Vec<f64>, Vec<f64>, String)> =
        std::collections::HashMap::new();

    for i in 0..n {
        let label = plot.labels[i];
        let entry = traces.entry(label).or_insert_with(|| {
            let color = plot
                .colors
                .get(i)
                .cloned()
                .unwrap_or_else(|| "#888888".to_string());
            (Vec::new(), Vec::new(), color)
        });
        entry.0.push(plot.points[[i, 0]]);
        entry.1.push(plot.points[[i, 1]]);
    }

    let mut trace_array = Vec::new();
    let mut labels_sorted: Vec<i32> = traces.keys().copied().collect();
    labels_sorted.sort();
    for label in labels_sorted {
        let (xs, ys, color) = &traces[&label];
        trace_array.push(serde_json::json!({
            "type": "scatter",
            "mode": "markers",
            "name": format!("Cluster {}", label),
            "x": xs,
            "y": ys,
            "marker": {
                "color": color,
                "size": 8
            }
        }));
    }

    let layout = serde_json::json!({
        "title": "Clustering Visualization",
        "xaxis": { "title": "X" },
        "yaxis": { "title": "Y" }
    });

    let plotly_doc = serde_json::json!({
        "data": trace_array,
        "layout": layout
    });

    let json_string = serde_json::to_string_pretty(&plotly_doc).map_err(|e| {
        ClusteringError::ComputationError(format!("Plotly JSON serialization failed: {}", e))
    })?;

    std::fs::write(path, json_string)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_scatter_2d_to_d3<P: AsRef<Path>>(
    plot: &ScatterPlot2D,
    path: P,
    config: &ExportConfig,
) -> Result<()> {
    // Build an HTML file with embedded D3.js and inline data
    let n = plot.points.nrows();
    let mut points_json = String::from("[");
    for i in 0..n {
        if i > 0 {
            points_json.push(',');
        }
        let color = plot.colors.get(i).map(|s| s.as_str()).unwrap_or("#888888");
        let size = plot.sizes.get(i).copied().unwrap_or(5.0);
        points_json.push_str(&format!(
            "{{\"x\":{},\"y\":{},\"label\":{},\"color\":\"{}\",\"r\":{}}}",
            plot.points[[i, 0]],
            plot.points[[i, 1]],
            plot.labels[i],
            color,
            size
        ));
    }
    points_json.push(']');

    let (width, height) = config.dimensions;
    let background = &config.background_color;

    // Build HTML via concatenation to avoid format!() conflicts with JS tokens
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("<meta charset=\"UTF-8\">\n");
    html.push_str("<title>D3.js Clustering Visualization</title>\n");
    html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
    html.push_str(&format!(
        "<style>body{{margin:0;background:{};}}svg{{display:block;}}</style>\n",
        background
    ));
    html.push_str("</head>\n<body>\n");
    html.push_str(&format!(
        "<svg id=\"chart\" width=\"{}\" height=\"{}\"></svg>\n",
        width, height
    ));
    html.push_str("<script>\n");
    html.push_str(&format!("const data = {};\n", points_json));
    html.push_str("const svg = d3.select(\"#chart\");\n");
    html.push_str("const margin = {top:20,right:20,bottom:40,left:40};\n");
    html.push_str(&format!(
        "const w = {} - margin.left - margin.right;\n",
        width
    ));
    html.push_str(&format!(
        "const h = {} - margin.top - margin.bottom;\n",
        height
    ));
    html.push_str("const g = svg.append(\"g\").attr(\"transform\",\"translate(\"+margin.left+\",\"+margin.top+\")\");\n");
    html.push_str("const xExt = d3.extent(data, d => d.x);\n");
    html.push_str("const yExt = d3.extent(data, d => d.y);\n");
    html.push_str("const xSc = d3.scaleLinear().domain(xExt).range([0,w]);\n");
    html.push_str("const ySc = d3.scaleLinear().domain(yExt).range([h,0]);\n");
    html.push_str(
        "g.append(\"g\").attr(\"transform\",\"translate(0,\"+h+\")\").call(d3.axisBottom(xSc));\n",
    );
    html.push_str("g.append(\"g\").call(d3.axisLeft(ySc));\n");
    html.push_str("g.selectAll(\"circle\").data(data).enter().append(\"circle\")\n");
    html.push_str("  .attr(\"cx\", d => xSc(d.x))\n");
    html.push_str("  .attr(\"cy\", d => ySc(d.y))\n");
    html.push_str("  .attr(\"r\", d => d.r)\n");
    html.push_str("  .attr(\"fill\", d => d.color)\n");
    html.push_str("  .attr(\"opacity\", 0.8);\n");
    html.push_str("</script>\n</body>\n</html>\n");

    std::fs::write(path, html)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_scatter_2d_to_svg<P: AsRef<Path>>(
    plot: &ScatterPlot2D,
    path: P,
    config: &ExportConfig,
) -> Result<()> {
    let (width, height) = config.dimensions;
    let margin = 60u32;
    let plot_w = width.saturating_sub(2 * margin) as f64;
    let plot_h = height.saturating_sub(2 * margin) as f64;

    let (min_x, max_x, min_y, max_y) = plot.bounds;
    let range_x = (max_x - min_x).max(1e-10);
    let range_y = (max_y - min_y).max(1e-10);

    let scale_x = |v: f64| -> f64 { (v - min_x) / range_x * plot_w + margin as f64 };
    let scale_y =
        |v: f64| -> f64 { height as f64 - margin as f64 - (v - min_y) / range_y * plot_h };

    let mut svg = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="{width}" height="{height}" fill="{bg}"/>
"#,
        width = width,
        height = height,
        bg = config.background_color
    );

    // Axis lines
    svg.push_str(&format!(
        r#"<line x1="{m}" y1="{m}" x2="{m}" y2="{bot}" stroke="black" stroke-width="1"/>
<line x1="{m}" y1="{bot}" x2="{right}" y2="{bot}" stroke="black" stroke-width="1"/>
"#,
        m = margin,
        bot = height - margin,
        right = width - margin
    ));

    // Points
    let n = plot.points.nrows();
    for i in 0..n {
        let cx = scale_x(plot.points[[i, 0]]);
        let cy = scale_y(plot.points[[i, 1]]);
        let r = plot.sizes.get(i).copied().unwrap_or(5.0);
        let color = plot.colors.get(i).map(|s| s.as_str()).unwrap_or("#888888");
        svg.push_str(&format!(
            r#"<circle cx="{cx:.2}" cy="{cy:.2}" r="{r:.1}" fill="{color}" opacity="0.8"/>
"#,
            cx = cx,
            cy = cy,
            r = r,
            color = color
        ));
    }

    svg.push_str("</svg>\n");

    std::fs::write(path, svg)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_scatter_2d_to_png<P: AsRef<Path>>(
    _plot: &ScatterPlot2D,
    _path: P,
    _config: &ExportConfig,
) -> Result<()> {
    Err(ClusteringError::ComputationError(
        "PNG export requires image rendering library".to_string(),
    ))
}

#[allow(dead_code)]
fn export_scatter_3d_to_threejs<P: AsRef<Path>>(
    plot: &ScatterPlot3D,
    path: P,
    config: &ExportConfig,
) -> Result<()> {
    // Emit a Three.js-compatible BufferGeometry JSON descriptor
    let n = plot.points.nrows();
    let mut positions: Vec<f64> = Vec::with_capacity(n * 3);
    let mut colors_rgb: Vec<f64> = Vec::with_capacity(n * 3);

    for i in 0..n {
        positions.push(plot.points[[i, 0]]);
        positions.push(plot.points[[i, 1]]);
        positions.push(plot.points[[i, 2]]);

        // Parse hex color to r,g,b in [0,1]
        let hex = plot
            .colors
            .get(i)
            .map(|s| s.trim_start_matches('#'))
            .unwrap_or("888888");
        let (r, g, b) = parse_hex_color(hex);
        colors_rgb.push(r);
        colors_rgb.push(g);
        colors_rgb.push(b);
    }

    let threejs_json = serde_json::json!({
        "metadata": {
            "version": 4.5,
            "type": "BufferGeometry",
            "generator": "scirs2-cluster"
        },
        "uuid": "scirs2-points",
        "type": "BufferGeometry",
        "data": {
            "attributes": {
                "position": {
                    "itemSize": 3,
                    "type": "Float32Array",
                    "array": positions,
                    "normalized": false
                },
                "color": {
                    "itemSize": 3,
                    "type": "Float32Array",
                    "array": colors_rgb,
                    "normalized": false
                }
            }
        },
        "pointCount": n,
        "config": {
            "background": config.background_color,
            "opacity": config.quality as f64 / 100.0
        }
    });

    let json_string = serde_json::to_string_pretty(&threejs_json).map_err(|e| {
        ClusteringError::ComputationError(format!("Three.js JSON serialization failed: {}", e))
    })?;

    std::fs::write(path, json_string)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_scatter_3d_to_gltf<P: AsRef<Path>>(
    plot: &ScatterPlot3D,
    path: P,
    _config: &ExportConfig,
) -> Result<()> {
    // Build a minimal valid glTF 2.0 JSON with POINTS primitive and base64-encoded buffer
    let n = plot.points.nrows();

    // Pack positions as f32 LE bytes
    let mut pos_bytes: Vec<u8> = Vec::with_capacity(n * 3 * 4);
    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];

    for i in 0..n {
        for c in 0..3 {
            let v = plot.points[[i, c]] as f32;
            pos_bytes.extend_from_slice(&v.to_le_bytes());
            if v < min_pos[c] {
                min_pos[c] = v;
            }
            if v > max_pos[c] {
                max_pos[c] = v;
            }
        }
    }

    // Pack color as f32 LE bytes (r,g,b per vertex)
    let mut col_bytes: Vec<u8> = Vec::with_capacity(n * 3 * 4);
    for i in 0..n {
        let hex = plot
            .colors
            .get(i)
            .map(|s| s.trim_start_matches('#'))
            .unwrap_or("888888");
        let (r, g, b) = parse_hex_color(hex);
        col_bytes.extend_from_slice(&(r as f32).to_le_bytes());
        col_bytes.extend_from_slice(&(g as f32).to_le_bytes());
        col_bytes.extend_from_slice(&(b as f32).to_le_bytes());
    }

    let pos_byte_len = pos_bytes.len();
    let col_byte_len = col_bytes.len();
    let mut combined = pos_bytes;
    combined.extend(col_bytes);

    let mut b64 = String::new();
    // Simple base64 encoding without external crate
    {
        const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let data = &combined;
        let mut i = 0;
        while i + 2 < data.len() {
            let v = (data[i] as usize) << 16 | (data[i + 1] as usize) << 8 | (data[i + 2] as usize);
            b64.push(TABLE[(v >> 18) & 63] as char);
            b64.push(TABLE[(v >> 12) & 63] as char);
            b64.push(TABLE[(v >> 6) & 63] as char);
            b64.push(TABLE[v & 63] as char);
            i += 3;
        }
        let rem = data.len() - i;
        if rem == 1 {
            let v = (data[i] as usize) << 16;
            b64.push(TABLE[(v >> 18) & 63] as char);
            b64.push(TABLE[(v >> 12) & 63] as char);
            b64.push_str("==");
        } else if rem == 2 {
            let v = (data[i] as usize) << 16 | (data[i + 1] as usize) << 8;
            b64.push(TABLE[(v >> 18) & 63] as char);
            b64.push(TABLE[(v >> 12) & 63] as char);
            b64.push(TABLE[(v >> 6) & 63] as char);
            b64.push('=');
        }
    }

    let data_uri = format!("data:application/octet-stream;base64,{}", b64);
    let total_byte_len = combined.len();

    let gltf = serde_json::json!({
        "asset": { "version": "2.0", "generator": "scirs2-cluster" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "name": "ClusterPoints",
            "primitives": [{
                "attributes": { "POSITION": 0, "COLOR_0": 1 },
                "mode": 0
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": n,
                "type": "VEC3",
                "min": [min_pos[0], min_pos[1], min_pos[2]],
                "max": [max_pos[0], max_pos[1], max_pos[2]]
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": n,
                "type": "VEC3"
            }
        ],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": pos_byte_len },
            { "buffer": 0, "byteOffset": pos_byte_len, "byteLength": col_byte_len }
        ],
        "buffers": [{ "uri": data_uri, "byteLength": total_byte_len }]
    });

    let json_string = serde_json::to_string_pretty(&gltf).map_err(|e| {
        ClusteringError::ComputationError(format!("GLTF JSON serialization failed: {}", e))
    })?;

    std::fs::write(path, json_string)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_scatter_3d_to_webgl<P: AsRef<Path>>(
    plot: &ScatterPlot3D,
    path: P,
    config: &ExportConfig,
) -> Result<()> {
    // Emit a self-contained HTML with inline WebGL vertex/fragment shaders
    let n = plot.points.nrows();

    // Build flat JS arrays
    let mut pos_vals = Vec::with_capacity(n * 3);
    let mut col_vals = Vec::with_capacity(n * 3);

    for i in 0..n {
        pos_vals.push(plot.points[[i, 0]]);
        pos_vals.push(plot.points[[i, 1]]);
        pos_vals.push(plot.points[[i, 2]]);

        let hex = plot
            .colors
            .get(i)
            .map(|s| s.trim_start_matches('#'))
            .unwrap_or("888888");
        let (r, g, b) = parse_hex_color(hex);
        col_vals.push(r);
        col_vals.push(g);
        col_vals.push(b);
    }

    let pos_json = serde_json::to_string(&pos_vals).unwrap_or_else(|_| "[]".to_string());
    let col_json = serde_json::to_string(&col_vals).unwrap_or_else(|_| "[]".to_string());

    let (width, height) = config.dimensions;
    let bg = &config.background_color;

    // Parse background hex to float triplet for WebGL clearColor
    let bg_hex = bg.trim_start_matches('#');
    let (bg_r, bg_g, bg_b) = parse_hex_color(bg_hex);

    // Build HTML by concatenation to avoid format!() conflicts with JS template literals
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("<meta charset=\"UTF-8\">\n");
    html.push_str("<title>WebGL Clustering Visualization</title>\n");
    html.push_str(&format!(
        "<style>body{{margin:0;background:{};}}canvas{{display:block;}}</style>\n",
        bg
    ));
    html.push_str("</head>\n<body>\n");
    html.push_str(&format!(
        "<canvas id=\"gl\" width=\"{}\" height=\"{}\"></canvas>\n",
        width, height
    ));
    html.push_str("<script>\n");
    html.push_str(&format!("const posData = {};\n", pos_json));
    html.push_str(&format!("const colData = {};\n", col_json));
    html.push_str("const canvas = document.getElementById(\"gl\");\n");
    html.push_str(
        "const gl = canvas.getContext(\"webgl\") || canvas.getContext(\"experimental-webgl\");\n",
    );
    html.push_str("if (!gl) { alert(\"WebGL not supported\"); throw new Error(\"no webgl\"); }\n");
    html.push_str(&format!(
        "gl.clearColor({:.4}, {:.4}, {:.4}, 1.0);\n",
        bg_r, bg_g, bg_b
    ));
    html.push_str("gl.enable(gl.DEPTH_TEST);\n");
    // Vertex shader as a regular string (no template literals needed)
    html.push_str("const vsrc = \"attribute vec3 aPos; attribute vec3 aCol; varying vec3 vCol; uniform mat4 uMVP; void main() { gl_Position = uMVP * vec4(aPos, 1.0); gl_PointSize = 6.0; vCol = aCol; }\";\n");
    html.push_str("const fsrc = \"precision mediump float; varying vec3 vCol; void main() { float d = length(gl_PointCoord - vec2(0.5)); if (d > 0.5) discard; gl_FragColor = vec4(vCol, 0.85); }\";\n");
    html.push_str("function compileShader(type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; }\n");
    html.push_str("const prog = gl.createProgram();\n");
    html.push_str("gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, vsrc));\n");
    html.push_str("gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, fsrc));\n");
    html.push_str("gl.linkProgram(prog);\ngl.useProgram(prog);\n");
    html.push_str("function makeBuffer(data, attr, size) { const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW); const loc = gl.getAttribLocation(prog, attr); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); }\n");
    html.push_str("makeBuffer(posData, \"aPos\", 3);\nmakeBuffer(colData, \"aCol\", 3);\n");
    html.push_str("function mat4ortho(l,r,b,t,n,f) { return [2/(r-l),0,0,0, 0,2/(t-b),0,0, 0,0,-2/(f-n),0, -(r+l)/(r-l),-(t+b)/(t-b),-(f+n)/(f-n),1]; }\n");
    html.push_str("const mvpLoc = gl.getUniformLocation(prog, \"uMVP\");\n");
    html.push_str("const mvp = mat4ortho(-10,10,-10,10,-10,10);\n");
    html.push_str("gl.uniformMatrix4fv(mvpLoc, false, mvp);\n");
    html.push_str("gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);\n");
    html.push_str(&format!("gl.drawArrays(gl.POINTS, 0, {});\n", n));
    html.push_str("</script>\n</body>\n</html>\n");

    std::fs::write(path, html)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_scatter_3d_to_unity<P: AsRef<Path>>(
    plot: &ScatterPlot3D,
    path: P,
    _config: &ExportConfig,
) -> Result<()> {
    // Emit a Unity-importable JSON descriptor with positions and colors
    // Unity scripts can load this via JsonUtility or Newtonsoft.Json
    let n = plot.points.nrows();
    let mut points_json_arr = Vec::with_capacity(n);
    for i in 0..n {
        let hex = plot
            .colors
            .get(i)
            .map(|s| s.trim_start_matches('#'))
            .unwrap_or("888888");
        let (r, g, b) = parse_hex_color(hex);
        points_json_arr.push(serde_json::json!({
            "x": plot.points[[i, 0]],
            "y": plot.points[[i, 1]],
            "z": plot.points[[i, 2]],
            "cluster": plot.labels[i],
            "color": { "r": r, "g": g, "b": b, "a": 1.0 },
            "size": plot.sizes.get(i).copied().unwrap_or(1.0)
        }));
    }

    let doc = serde_json::json!({
        "format": "scirs2-unity3d-v1",
        "pointCount": n,
        "points": points_json_arr
    });

    let json_string = serde_json::to_string_pretty(&doc).map_err(|e| {
        ClusteringError::ComputationError(format!("Unity JSON serialization failed: {}", e))
    })?;

    std::fs::write(path, json_string)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_scatter_3d_to_blender<P: AsRef<Path>>(
    plot: &ScatterPlot3D,
    path: P,
    _config: &ExportConfig,
) -> Result<()> {
    // Emit a Blender-importable Python script that creates a mesh of instanced spheres
    let n = plot.points.nrows();
    let mut script = String::from(
        "import bpy\nimport mathutils\n\n# Generated by scirs2-cluster\n# Run via: blender --background --python this_file.py\n\n",
    );

    script.push_str("def create_cluster_points():\n");
    script.push_str("    bpy.ops.object.select_all(action='DESELECT')\n");
    script.push_str("    mesh_data = []\n");

    for i in 0..n {
        let hex = plot
            .colors
            .get(i)
            .map(|s| s.trim_start_matches('#'))
            .unwrap_or("888888");
        let (r, g, b) = parse_hex_color(hex);
        let size = plot.sizes.get(i).copied().unwrap_or(1.0) * 0.05;
        script.push_str(&format!(
            "    mesh_data.append(({:.6}, {:.6}, {:.6}, {:.4}, {:.4}, {:.4}, {:.4}))\n",
            plot.points[[i, 0]],
            plot.points[[i, 1]],
            plot.points[[i, 2]],
            r,
            g,
            b,
            size
        ));
    }

    script.push_str(
        r#"
    for (x, y, z, r, g, b, s) in mesh_data:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=s, location=(x, y, z))
        obj = bpy.context.active_object
        mat = bpy.data.materials.new(name=f"cluster_mat_{len(bpy.data.materials)}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
        obj.data.materials.append(mat)

create_cluster_points()
print("Done: created cluster points")
"#,
    );

    std::fs::write(path, script)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
fn export_animation_to_gif<P: AsRef<Path>>(
    _frames: &[AnimationFrame],
    _output_path: P,
    _config: &ExportConfig,
) -> Result<()> {
    Err(ClusteringError::ComputationError(
        "GIF export requires animation library".to_string(),
    ))
}

#[allow(dead_code)]
fn export_animation_to_mp4<P: AsRef<Path>>(
    _frames: &[AnimationFrame],
    _output_path: P,
    _config: &ExportConfig,
) -> Result<()> {
    Err(ClusteringError::ComputationError(
        "MP4 export requires video encoding library".to_string(),
    ))
}

#[allow(dead_code)]
fn export_animation_to_webm<P: AsRef<Path>>(
    _frames: &[AnimationFrame],
    _output_path: P,
    _config: &ExportConfig,
) -> Result<()> {
    Err(ClusteringError::ComputationError(
        "WebM export requires video encoding library".to_string(),
    ))
}

#[allow(dead_code)]
fn export_animation_to_html<P: AsRef<Path>>(
    frames: &[AnimationFrame],
    output_path: P,
    config: &ExportConfig,
) -> Result<()> {
    if frames.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "No animation frames provided".to_string(),
        ));
    }

    // Serialize all frames to JSON for embedding in the HTML animation loop
    let frames_json = serde_json::to_string(frames).map_err(|e| {
        ClusteringError::ComputationError(format!("Frame JSON serialization failed: {}", e))
    })?;

    let (width, height) = config.dimensions;
    let fps = config.fps.max(1.0);
    let bg = &config.background_color;

    // Build HTML by concatenation to avoid format!() conflicts with JS hex tokens
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("<meta charset=\"UTF-8\">\n");
    html.push_str("<title>Clustering Animation</title>\n");
    html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
    html.push_str("<style>\n");
    html.push_str(&format!(
        "body{{margin:0;background:{};font-family:Arial,sans-serif;}}\n",
        bg
    ));
    html.push_str("svg{display:block;margin:auto;}\n");
    html.push_str("#controls{text-align:center;padding:10px;}\n");
    html.push_str("</style>\n</head>\n<body>\n");
    html.push_str("<div id=\"controls\">\n");
    html.push_str("  <button id=\"play\">Play</button>\n");
    html.push_str("  <button id=\"pause\">Pause</button>\n");
    html.push_str("  <button id=\"reset\">Reset</button>\n");
    html.push_str("  <span id=\"frame-info\"> Frame: 0 / 0 </span>\n");
    html.push_str("</div>\n");
    html.push_str(&format!(
        "<svg id=\"chart\" width=\"{}\" height=\"{}\"></svg>\n",
        width, height
    ));
    html.push_str("<script>\n");
    html.push_str(&format!("const frames = {};\n", frames_json));
    html.push_str(&format!("const FPS = {:.1};\n", fps));
    html.push_str("const svg = d3.select(\"#chart\");\n");
    html.push_str("const margin = {top:20,right:20,bottom:40,left:40};\n");
    html.push_str(&format!(
        "const w = {} - margin.left - margin.right;\n",
        width
    ));
    html.push_str(&format!(
        "const h = {} - margin.top - margin.bottom;\n",
        height
    ));
    html.push_str("const g = svg.append(\"g\").attr(\"transform\",\"translate(\"+margin.left+\",\"+margin.top+\")\");\n");
    html.push_str("let currentFrame = 0;\n");
    html.push_str("let interval = null;\n");
    html.push_str("function allPoints() {\n");
    html.push_str("  let pts = [];\n");
    html.push_str("  for (const f of frames) {\n");
    html.push_str("    for (let i=0;i<f.points.length;i++) pts.push(f.points[i]);\n");
    html.push_str("  }\n  return pts;\n}\n");
    html.push_str("const allPts = allPoints();\n");
    html.push_str("const xExt = d3.extent(allPts, d => d[0]);\n");
    html.push_str("const yExt = d3.extent(allPts, d => d[1]);\n");
    html.push_str("const xSc = d3.scaleLinear().domain(xExt).range([0,w]);\n");
    html.push_str("const ySc = d3.scaleLinear().domain(yExt).range([h,0]);\n");
    html.push_str(
        "g.append(\"g\").attr(\"transform\",\"translate(0,\"+h+\")\").call(d3.axisBottom(xSc));\n",
    );
    html.push_str("g.append(\"g\").call(d3.axisLeft(ySc));\n");
    // Color palette as a const - use string concatenation to avoid Rust 2021 prefix issues
    let palette_str = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]
    .iter()
    .map(|c| format!("\"{}\"", c))
    .collect::<Vec<_>>()
    .join(",");
    html.push_str(&format!("const palette = [{}];\n", palette_str));
    html.push_str("function colorForLabel(label) {\n");
    html.push_str("  return palette[Math.abs(label) % palette.length];\n}\n");
    html.push_str("function renderFrame(idx) {\n");
    html.push_str("  if (idx >= frames.length) return;\n");
    html.push_str("  const f = frames[idx];\n");
    html.push_str("  const pts = f.points.map((p,i) => ({x:p[0],y:p[1],label:f.labels[i]}));\n");
    html.push_str("  const circles = g.selectAll(\"circle.point\").data(pts, (_,i) => i);\n");
    html.push_str("  circles.enter().append(\"circle\").attr(\"class\",\"point\")\n");
    html.push_str("    .merge(circles)\n");
    html.push_str("    .attr(\"cx\", d => xSc(d.x))\n");
    html.push_str("    .attr(\"cy\", d => ySc(d.y))\n");
    html.push_str("    .attr(\"r\", 5)\n");
    html.push_str("    .attr(\"fill\", d => colorForLabel(d.label))\n");
    html.push_str("    .attr(\"opacity\", 0.8);\n");
    html.push_str("  circles.exit().remove();\n");
    html.push_str("  document.getElementById(\"frame-info\").textContent =\n");
    html.push_str("    \" Frame: \" + (idx+1) + \" / \" + frames.length;\n}\n");
    html.push_str("renderFrame(0);\n");
    html.push_str("document.getElementById(\"play\").addEventListener(\"click\", () => {\n");
    html.push_str("  if (interval) return;\n");
    html.push_str("  interval = setInterval(() => {\n");
    html.push_str("    currentFrame = (currentFrame + 1) % frames.length;\n");
    html.push_str("    renderFrame(currentFrame);\n");
    html.push_str(&format!("  }}, 1000 / FPS);\n}}){}\n", ";"));
    html.push_str("document.getElementById(\"pause\").addEventListener(\"click\", () => {\n");
    html.push_str("  clearInterval(interval);\n  interval = null;\n});\n");
    html.push_str("document.getElementById(\"reset\").addEventListener(\"click\", () => {\n");
    html.push_str("  clearInterval(interval);\n  interval = null;\n");
    html.push_str("  currentFrame = 0;\n  renderFrame(0);\n});\n");
    html.push_str("</script>\n</body>\n</html>\n");

    std::fs::write(output_path, html)
        .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

    Ok(())
}

#[allow(dead_code)]
#[allow(unused_variables)]
fn export_animation_to_json<P: AsRef<Path>>(
    frames: &[AnimationFrame],
    output_path: P,
    _config: &ExportConfig,
) -> Result<()> {
    #[cfg(feature = "serde")]
    {
        let json_string = serde_json::to_string_pretty(frames).map_err(|e| {
            ClusteringError::ComputationError(format!("JSON serialization failed: {}", e))
        })?;

        std::fs::write(output_path, json_string)
            .map_err(|e| ClusteringError::ComputationError(format!("File write failed: {}", e)))?;

        return Ok(());
    }

    #[cfg(not(feature = "serde"))]
    {
        Err(ClusteringError::ComputationError(
            "JSON export requires 'serde' feature".to_string(),
        ))
    }
}

/// Create metadata for exports
#[allow(dead_code)]
fn create_metadata() -> ExportMetadata {
    ExportMetadata {
        created_at: chrono::Utc::now().to_rfc3339(),
        software: "scirs2-cluster".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        format_version: "1.0".to_string(),
    }
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportMetadata {
    created_at: String,
    software: String,
    version: String,
    format_version: String,
}

/// 2D scatter plot export wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Scatter2DExport {
    format_version: String,
    export_config: ExportConfig,
    plot_data: ScatterPlot2D,
    metadata: ExportMetadata,
}

/// 3D scatter plot export wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Scatter3DExport {
    format_version: String,
    export_config: ExportConfig,
    plot_data: ScatterPlot3D,
    metadata: ExportMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_export_config_defaults() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::PNG);
        assert_eq!(config.dimensions, (1920, 1080));
        assert_eq!(config.dpi, 300);
    }

    #[test]
    fn test_scatter_2d_csv_export() {
        let plot = ScatterPlot2D {
            points: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
                .expect("Operation failed"),
            labels: Array1::from_vec(vec![0, 1]),
            centroids: None,
            colors: vec!["#FF0000".to_string(), "#00FF00".to_string()],
            sizes: vec![5.0, 5.0],
            point_labels: None,
            bounds: (0.0, 4.0, 0.0, 4.0),
            legend: Vec::new(),
        };

        let config = ExportConfig {
            format: ExportFormat::CSV,
            ..Default::default()
        };

        let temp_file = tempfile::NamedTempFile::new().expect("Operation failed");
        export_scatter_2d_to_csv(&plot, temp_file.path(), &config).expect("Operation failed");

        let content = std::fs::read_to_string(temp_file.path()).expect("Operation failed");
        assert!(content.contains("x,y,cluster,color"));
        assert!(content.contains("1,2,0,#FF0000"));
    }

    fn make_plot_2d() -> ScatterPlot2D {
        ScatterPlot2D {
            points: Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 1.0, 5.0, 4.0])
                .expect("shape"),
            labels: Array1::from_vec(vec![0, 0, 1]),
            centroids: None,
            colors: vec![
                "#FF0000".to_string(),
                "#FF0000".to_string(),
                "#0000FF".to_string(),
            ],
            sizes: vec![4.0, 4.0, 6.0],
            point_labels: None,
            bounds: (1.0, 5.0, 1.0, 4.0),
            legend: Vec::new(),
        }
    }

    fn make_plot_3d() -> ScatterPlot3D {
        use scirs2_core::ndarray::Array1;
        ScatterPlot3D {
            points: Array2::from_shape_vec(
                (3, 3),
                vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0],
            )
            .expect("shape"),
            labels: Array1::from_vec(vec![0, 0, 1]),
            centroids: None,
            colors: vec![
                "#FF0000".to_string(),
                "#FF0000".to_string(),
                "#00FF00".to_string(),
            ],
            sizes: vec![4.0, 4.0, 6.0],
            point_labels: None,
            bounds: (1.0, 4.0, 1.0, 4.0, 1.0, 3.0),
            legend: Vec::new(),
        }
    }

    #[test]
    fn test_plotly_export_writes_valid_json() {
        let plot = make_plot_2d();
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_plotly_export.json");
        export_scatter_2d_to_plotly(&plot, &tmp, &config).expect("plotly export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        let v: serde_json::Value = serde_json::from_str(&content).expect("valid json");
        assert!(v["data"].is_array());
        assert!(v["layout"].is_object());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_d3_export_writes_html() {
        let plot = make_plot_2d();
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_d3_export.html");
        export_scatter_2d_to_d3(&plot, &tmp, &config).expect("d3 export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("d3.v7.min.js"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_svg_export_contains_circles() {
        let plot = make_plot_2d();
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_svg_export.svg");
        export_scatter_2d_to_svg(&plot, &tmp, &config).expect("svg export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        assert!(content.contains("<svg"));
        assert!(content.contains("<circle"));
        assert!(content.contains("</svg>"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_threejs_export_writes_valid_json() {
        let plot = make_plot_3d();
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_threejs_export.json");
        export_scatter_3d_to_threejs(&plot, &tmp, &config).expect("threejs export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        let v: serde_json::Value = serde_json::from_str(&content).expect("valid json");
        assert_eq!(v["type"], "BufferGeometry");
        assert!(v["data"]["attributes"]["position"].is_object());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_gltf_export_writes_valid_gltf2() {
        let plot = make_plot_3d();
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_gltf_export.gltf");
        export_scatter_3d_to_gltf(&plot, &tmp, &config).expect("gltf export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        let v: serde_json::Value = serde_json::from_str(&content).expect("valid json");
        assert_eq!(v["asset"]["version"], "2.0");
        assert!(v["buffers"].is_array());
        assert!(v["meshes"].is_array());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_webgl_export_writes_html_with_shaders() {
        let plot = make_plot_3d();
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_webgl_export.html");
        export_scatter_3d_to_webgl(&plot, &tmp, &config).expect("webgl export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("aPos"));
        assert!(content.contains("gl.drawArrays"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_unity_export_writes_valid_json() {
        let plot = make_plot_3d();
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_unity_export.json");
        export_scatter_3d_to_unity(&plot, &tmp, &config).expect("unity export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        let v: serde_json::Value = serde_json::from_str(&content).expect("valid json");
        assert_eq!(v["format"], "scirs2-unity3d-v1");
        assert_eq!(v["pointCount"], 3);
        assert!(v["points"].is_array());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_blender_export_writes_python_script() {
        let plot = make_plot_3d();
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_blender_export.py");
        export_scatter_3d_to_blender(&plot, &tmp, &config).expect("blender export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        assert!(content.contains("import bpy"));
        assert!(content.contains("create_cluster_points"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_animation_html_export_writes_interactive_html() {
        use crate::visualization::animation::AnimationFrame;
        use scirs2_core::ndarray::Array1;
        let frame = AnimationFrame {
            frame_number: 0,
            iteration: 0,
            timestamp: 0.0,
            points: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape"),
            labels: Array1::from_vec(vec![0, 1]),
            centroids: None,
            previous_centroids: None,
            convergence_info: None,
            annotations: Vec::new(),
        };
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_anim_export.html");
        export_animation_to_html(&[frame], &tmp, &config).expect("animation html export");
        let content = std::fs::read_to_string(&tmp).expect("read");
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("renderFrame"));
        assert!(content.contains("play"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_parse_hex_color_white() {
        let (r, g, b) = parse_hex_color("FFFFFF");
        assert!((r - 1.0).abs() < 1e-6);
        assert!((g - 1.0).abs() < 1e-6);
        assert!((b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_hex_color_shorthand() {
        let (r, g, b) = parse_hex_color("F00");
        assert!((r - 1.0).abs() < 1e-6);
        assert!(g < 0.01);
        assert!(b < 0.01);
    }

    #[test]
    fn test_animation_html_empty_frames_error() {
        let config = ExportConfig::default();
        let tmp = std::env::temp_dir().join("test_anim_empty.html");
        let result = export_animation_to_html(&[], &tmp, &config);
        assert!(result.is_err());
    }
}

/// Parse a hex color string (without `#` prefix) into (r,g,b) floats in [0.0, 1.0].
/// Falls back to `(0.533, 0.533, 0.533)` on malformed input.
fn parse_hex_color(hex: &str) -> (f64, f64, f64) {
    let h = if hex.len() == 3 {
        // Expand shorthand #RGB → #RRGGBB
        format!(
            "{}{}{}{}{}{}",
            &hex[0..1],
            &hex[0..1],
            &hex[1..2],
            &hex[1..2],
            &hex[2..3],
            &hex[2..3]
        )
    } else {
        hex.to_string()
    };

    if h.len() != 6 {
        return (0.533, 0.533, 0.533);
    }

    let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(136) as f64 / 255.0;
    let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(136) as f64 / 255.0;
    let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(136) as f64 / 255.0;
    (r, g, b)
}

// Conditionally include chrono for metadata timestamps

use chrono;
