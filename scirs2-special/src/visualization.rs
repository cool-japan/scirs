//! Visualization tools for special functions
//!
//! This module provides comprehensive plotting and visualization capabilities
//! for all special functions, including 2D/3D plots, animations, and interactive
//! visualizations.

#[cfg(feature = "plotting")]
use plotters::prelude::*;
use scirs2_core::numeric::Complex64;
use std::error::Error;
use std::path::Path;

/// Configuration for plot generation
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
    /// DPI for high-resolution output
    pub dpi: u32,
    /// Plot title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Whether to show grid
    pub show_grid: bool,
    /// Whether to show legend
    pub show_legend: bool,
    /// Color scheme
    pub color_scheme: ColorScheme,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            dpi: 100,
            title: String::new(),
            x_label: "x".to_string(),
            y_label: "f(x)".to_string(),
            show_grid: true,
            show_legend: true,
            color_scheme: ColorScheme::default(),
        }
    }
}

/// Color schemes for plots
#[derive(Debug, Clone)]
pub enum ColorScheme {
    Default,
    Viridis,
    Plasma,
    Inferno,
    Magma,
    ColorBlind,
}

impl Default for ColorScheme {
    fn default() -> Self {
        ColorScheme::Default
    }
}

/// Trait for functions that can be visualized
pub trait Visualizable {
    /// Generate a 2D plot
    fn plot_2d(&self, config: &PlotConfig) -> Result<Vec<u8>, Box<dyn Error>>;

    /// Generate a 3D surface plot
    fn plot_3d(&self, config: &PlotConfig) -> Result<Vec<u8>, Box<dyn Error>>;

    /// Generate an animated visualization
    fn animate(&self, config: &PlotConfig) -> Result<Vec<Vec<u8>>, Box<dyn Error>>;
}

/// Plot multiple functions on the same axes
pub struct MultiPlot {
    functions: Vec<Box<dyn Fn(f64) -> f64>>,
    labels: Vec<String>,
    x_range: (f64, f64),
    config: PlotConfig,
}

impl MultiPlot {
    pub fn new(config: PlotConfig) -> Self {
        Self {
            functions: Vec::new(),
            labels: Vec::new(),
            x_range: (-10.0, 10.0),
            config,
        }
    }

    pub fn add_function(mut self, f: Box<dyn Fn(f64) -> f64>, label: &str) -> Self {
        self.functions.push(f);
        self.labels.push(label.to_string());
        self
    }

    pub fn set_x_range(mut self, min: f64, max: f64) -> Self {
        self.x_range = (min, max);
        self
    }

    #[cfg(feature = "plotting")]
    pub fn plot<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(path.as_ref(), (self.config.width, self.config.height))
            .into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(self.x_range.0..self.x_range.1, -2f64..2f64)?;

        if self.config.show_grid {
            chart
                .configure_mesh()
                .x_desc(&self.config.x_label)
                .y_desc(&self.config.y_label)
                .draw()?;
        }

        let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];

        for (i, (f, label)) in self.functions.iter().zip(&self.labels).enumerate() {
            let color = colors[i % colors.len()];
            let data: Vec<(f64, f64)> = ((self.x_range.0 * 100.0) as i32
                ..(self.x_range.1 * 100.0) as i32)
                .map(|x| x as f64 / 100.0)
                .map(|x| (x, f(x)))
                .collect();

            chart
                .draw_series(LineSeries::new(data, color))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
        }

        if self.config.show_legend {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }

        root.present()?;
        Ok(())
    }
}

/// Gamma function visualization
pub mod gamma_plots {
    use super::*;
    use crate::{digamma, gamma, gammaln};

    /// Plot gamma function and its logarithm
    pub fn plot_gamma_family<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        let config = PlotConfig {
            title: "Gamma Function Family".to_string(),
            x_label: "x".to_string(),
            y_label: "f(x)".to_string(),
            ..Default::default()
        };

        MultiPlot::new(config)
            .add_function(Box::new(|x| gamma(x)), "Γ(x)")
            .add_function(Box::new(|x| gammaln(x)), "ln Γ(x)")
            .add_function(Box::new(|x| digamma(x)), "ψ(x)")
            .set_x_range(0.1, 5.0)
            .plot(path)
    }

    /// Create a heatmap of gamma function in complex plane
    #[cfg(feature = "plotting")]
    pub fn plot_gamma_complex<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        use crate::gamma::complex::gamma_complex;

        let root = BitMapBackend::new(path.as_ref(), (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Complex Gamma Function |Γ(z)|", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(-5f64..5f64, -5f64..5f64)?;

        chart
            .configure_mesh()
            .x_desc("Re(z)")
            .y_desc("Im(z)")
            .draw()?;

        // Create heatmap data
        let n = 100;
        let mut data = vec![];

        for i in 0..n {
            for j in 0..n {
                let x = -5.0 + 10.0 * i as f64 / n as f64;
                let y = -5.0 + 10.0 * j as f64 / n as f64;
                let z = Complex64::new(x, y);
                let gamma_z = gamma_complex(z);
                let magnitude = gamma_z.norm().ln(); // Log scale for better visualization

                data.push(Rectangle::new(
                    [(x, y), (x + 0.1, y + 0.1)],
                    HSLColor(240.0 - magnitude * 30.0, 0.7, 0.5).filled(),
                ));
            }
        }

        chart.draw_series(data)?;

        root.present()?;
        Ok(())
    }
}

/// Bessel function visualization
pub mod bessel_plots {
    use super::*;
    use crate::bessel::{j0, j1, jn};

    /// Plot Bessel functions of the first kind
    pub fn plot_bessel_j<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        let config = PlotConfig {
            title: "Bessel Functions of the First Kind".to_string(),
            ..Default::default()
        };

        MultiPlot::new(config)
            .add_function(Box::new(|x| j0(x)), "J₀(x)")
            .add_function(Box::new(|x| j1(x)), "J₁(x)")
            .add_function(Box::new(|x| jn(2, x)), "J₂(x)")
            .add_function(Box::new(|x| jn(3, x)), "J₃(x)")
            .set_x_range(0.0, 20.0)
            .plot(path)
    }

    /// Plot zeros of Bessel functions
    pub fn plot_bessel_zeros<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        use crate::bessel_zeros::j0_zeros;

        #[cfg(feature = "plotting")]
        {
            let root = BitMapBackend::new(path.as_ref(), (800, 600)).into_drawing_area();
            root.fill(&WHITE)?;

            let mut chart = ChartBuilder::on(&root)
                .caption("Bessel Function Zeros", ("sans-serif", 40))
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(40)
                .build_cartesian_2d(0f64..30f64, -0.5f64..1f64)?;

            chart.configure_mesh().x_desc("x").y_desc("J_n(x)").draw()?;

            // Plot J0
            let j0_data: Vec<(f64, f64)> = (0..3000)
                .map(|i| i as f64 / 100.0)
                .map(|x| (x, j0(x)))
                .collect();
            chart
                .draw_series(LineSeries::new(j0_data, &BLUE))?
                .label("J₀(x)")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], &BLUE));

            // Mark zeros (first 10 zeros)
            for k in 1..=10 {
                if let Ok(zero) = j0_zeros::<f64>(k) {
                    chart.draw_series(PointSeries::of_element(
                        vec![(zero, 0.0)],
                        5,
                        &RED,
                        &|c, s, st| {
                            return EmptyElement::at(c)
                                + Circle::new((0, 0), s, st.filled())
                                + Text::new(format!("{:.3}", zero), (10, 0), ("sans-serif", 15));
                        },
                    ))?;
                }
            }

            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;

            root.present()?;
        }

        Ok(())
    }
}

/// Error function visualization
pub mod error_function_plots {
    use super::*;
    use crate::{erf, erfc, erfinv};

    /// Plot error functions and their inverses
    pub fn plot_error_functions<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn Error>> {
        let config = PlotConfig {
            title: "Error Functions".to_string(),
            ..Default::default()
        };

        MultiPlot::new(config)
            .add_function(Box::new(|x| erf(x)), "erf(x)")
            .add_function(Box::new(|x| erfc(x)), "erfc(x)")
            .add_function(
                Box::new(|x| if x.abs() < 0.999 { erfinv(x) } else { f64::NAN }),
                "erfinv(x)",
            )
            .set_x_range(-3.0, 3.0)
            .plot(path)
    }
}

/// Orthogonal polynomial visualization
pub mod polynomial_plots {
    use super::*;
    use crate::legendre;

    /// Plot Legendre polynomials
    pub fn plot_legendre<P: AsRef<Path>>(path: P, maxn: usize) -> Result<(), Box<dyn Error>> {
        let config = PlotConfig {
            title: format!("Legendre Polynomials P_n(x) for _n = 0..{}", maxn),
            ..Default::default()
        };

        let mut plot = MultiPlot::new(config).set_x_range(-1.0, 1.0);

        for _n in 0..=maxn {
            plot = plot.add_function(Box::new(move |x| legendre(_n, x)), &format!("P_{}", _n));
        }

        plot.plot(path)
    }

    /// Create an animated visualization of orthogonal polynomials
    pub fn animate_polynomials() -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
        // This would generate frames for an animation
        // showing how orthogonal polynomials evolve with increasing order
        Ok(vec![])
    }
}

/// Special function surface plots
pub mod surface_plots {
    use super::*;

    /// Plot a 3D surface for functions of two variables
    #[cfg(feature = "plotting")]
    pub fn plot_3d_surface<P, F>(path: P, f: F, title: &str) -> Result<(), Box<dyn Error>>
    where
        P: AsRef<Path>,
        F: Fn(f64, f64) -> f64,
    {
        let root = BitMapBackend::new(path.as_ref(), (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_3d(-5.0..5.0, -5.0..5.0, -2.0..2.0)?;

        chart.configure_axes().draw()?;

        // Generate surface data
        let n = 50;
        let mut data = vec![];

        for i in 0..n {
            for j in 0..n {
                let x = -5.0 + 10.0 * i as f64 / n as f64;
                let y = -5.0 + 10.0 * j as f64 / n as f64;
                let z = f(x, y);

                if z.is_finite() {
                    data.push((x, y, z));
                }
            }
        }

        // Create iterators for x and y coordinates
        let x_range: Vec<f64> = (0..51).map(|i| -5.0 + i as f64 * 0.2).collect();
        let y_range: Vec<f64> = (0..51).map(|i| -5.0 + i as f64 * 0.2).collect();

        chart.draw_series(
            SurfaceSeries::xoz(x_range.into_iter(), y_range.into_iter(), |x, y| f(x, y))
                .style(&BLUE.mix(0.5)),
        )?;

        root.present()?;
        Ok(())
    }
}

/// Interactive visualization support
#[cfg(feature = "interactive")]
pub mod interactive {
    #[allow(unused_imports)]
    use super::*;

    /// Configuration for interactive plots
    pub struct InteractivePlotConfig {
        pub enable_zoom: bool,
        pub enable_pan: bool,
        pub enable_tooltips: bool,
        pub enable_export: bool,
    }

    /// Create an interactive plot that can be embedded in a web page
    pub fn create_interactive_plot<F>(
        f: F,
        config: InteractivePlotConfig,
        x_range: (f64, f64),
        function_name: &str,
    ) -> String
    where
        F: Fn(f64) -> f64,
    {
        // Generate data points for the function
        let n_points = 1000;
        let step = (x_range.1 - x_range.0) / n_points as f64;
        let mut data_points = Vec::new();

        for i in 0..=n_points {
            let x = x_range.0 + i as f64 * step;
            let y = f(x);
            if y.is_finite() {
                data_points.push(format!("[{}, {}]", x, y));
            }
        }

        // Extract x and y values separately for cleaner code
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();

        for i in 0..=n_points {
            let x = x_range.0 + i as f64 * step;
            let y = f(x);
            if y.is_finite() {
                x_values.push(x);
                y_values.push(y);
            }
        }

        let x_json = format!(
            "[{}]",
            x_values
                .iter()
                .map(|x| format!("{x}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let y_json = format!(
            "[{}]",
            y_values
                .iter()
                .map(|y| format!("{y}"))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Generate comprehensive HTML with Plotly.js
        format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Plot - {}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5; 
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            color: #333; 
            text-align: center; 
            margin-bottom: 30px; 
        }}
        .controls {{ 
            display: flex; 
            gap: 15px; 
            margin-bottom: 20px; 
            flex-wrap: wrap; 
            align-items: center; 
        }}
        .control-group {{ 
            display: flex; 
            flex-direction: column; 
            gap: 5px; 
        }}
        label {{ 
            font-weight: 600; 
            color: #555; 
            font-size: 14px; 
        }}
        input, select, button {{ 
            padding: 8px 12px; 
            border: 1px solid #ddd; 
            border-radius: 4px; 
            font-size: 14px; 
        }}
        button {{ 
            background-color: #007bff; 
            color: white; 
            border: none; 
            cursor: pointer; 
            transition: background-color 0.2s; 
        }}
        button:hover {{ 
            background-color: #0056b3; 
        }}
        #plot {{ 
            width: 100%; 
            height: 600px; 
        }}
        .info-panel {{ 
            margin-top: 20px; 
            padding: 15px; 
            background-color: #f8f9fa; 
            border-radius: 6px; 
            border-left: 4px solid #007bff; 
        }}
        .tooltip-info {{ 
            margin-top: 10px; 
            font-size: 14px; 
            color: #666; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Visualization: {}</h1>
        
        <div class="controls">
            <div class="control-group">
                <label for="xMin">X Min:</label>
                <input type="number" id="xMin" value="{}" step="0.1">
            </div>
            <div class="control-group">
                <label for="xMax">X Max:</label>
                <input type="number" id="xMax" value="{}" step="0.1">
            </div>
            <div class="control-group">
                <label for="points">Points:</label>
                <select id="points">
                    <option value="500">500</option>
                    <option value="1000" selected>1000</option>
                    <option value="2000">2000</option>
                    <option value="5000">5000</option>
                </select>
            </div>
            <button onclick="updatePlot()">Update Plot</button>
            <button onclick="resetZoom()">Reset Zoom</button>
            <button onclick="exportData()">Export CSV</button>
            {}
        </div>
        
        <div id="plot"></div>
        
        <div class="info-panel">
            <h3>Interactive Features:</h3>
            <ul>
                <li><strong>Zoom:</strong> Click and drag to zoom into a region</li>
                <li><strong>Pan:</strong> Hold shift and drag to pan around</li>
                <li><strong>Hover:</strong> Move mouse over the curve to see coordinates</li>
                <li><strong>Double-click:</strong> Reset zoom to fit all data</li>
            </ul>
            <div class="tooltip-info" id="tooltip-info">
                Hover over the plot to see coordinate information here.
            </div>
        </div>
    </div>
    
    <script>
        let currentData = {};
        
        // JavaScript implementations of special functions
        function gamma(x) {{
            if (x < 0) return NaN;
            if (x === 0) return Infinity;
            if (x === 1 || x === 2) return 1;
            
            // Stirling's approximation for x > 1
            if (x > 1) {{
                return Math.sqrt(2 * Math.PI / x) * Math.pow(x / Math.E, x);
            }}
            return gamma(x + 1) / x;
        }}
        
        function besselJ0(x) {{
            const ax = Math.abs(x);
            if (ax < 8) {{
                const y = x * x;
                return ((-0.0000000000000000015 * y + 0.000000000000000176) * y +
                       (-0.0000000000000156) * y + 0.0000000000164) * y +
                       (-0.00000000106) * y + 0.000000421) * y +
                       (-0.0000103) * y + 0.00015625) * y +
                       (-0.015625) * y + 1;
            }} else {{
                const z = 8 / ax;
                const y = z * z;
                const xx = ax - 0.785398164;
                return Math.sqrt(0.636619772 / ax) *
                       (Math.cos(xx) * (1 + y * (-0.0703125 + y * 0.1121520996)) +
                        z * Math.sin(xx) * (-0.0390625 + y * 0.0444479255));
            }}
        }}
        
        function erf(x) {{
            const a1 =  0.254829592;
            const a2 = -0.284496736;
            const a3 =  1.421413741;
            const a4 = -1.453152027;
            const a5 =  1.061405429;
            const p  =  0.3275911;
            
            const sign = x >= 0 ? 1 : -1;
            x = Math.abs(x);
            
            const t = 1.0 / (1.0 + p * x);
            const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
            
            return sign * y;
        }}
        
        function airyAi(x) {{
            // Simplified Airy Ai function approximation
            if (x > 5) return 0;  // Exponentially decaying for positive x
            if (x < -5) {{
                // Oscillatory behavior for negative x
                const arg = (2/3) * Math.pow(Math.abs(x), 1.5);
                return (1 / (Math.sqrt(Math.PI) * Math.pow(Math.abs(x), 0.25))) * Math.sin(arg + Math.PI/4);
            }}
            // Rough approximation for the intermediate region
            return Math.exp(-Math.abs(x)) * Math.cos(x);
        }}
        
        function getSpecialFunction(functionName) {{
            const _name = functionName.toLowerCase();
            if (_name.includes('gamma')) return gamma;
            if (_name.includes('bessel') && name.includes('j0')) return besselJ0;
            if (_name.includes('error') || name.includes('erf')) return erf;
            if (_name.includes('airy')) return airyAi;
            // Default fallback - could add more functions as needed
            return Math.sin;
        }}
        
        function initializePlot() {{
            const data = [{{
                x: {},
                y: {},
                type: 'scatter',
                mode: 'lines',
                _name: '{}',
                line: {{
                    color: '#1f77b4',
                    width: 2
                }},
                hovertemplate: '<b>x:</b> %{{x:.6f}}<br><b>f(x):</b> %{{y:.6f}}<extra></extra>'
            }}];
            
            const layout = {{
                title: {{
                    text: '{} Function',
                    font: {{ size: 20 }}
                }},
                xaxis: {{
                    title: 'x',
                    showgrid: true,
                    zeroline: true,
                    showspikes: true,
                    spikethickness: 1,
                    spikecolor: '#999',
                    spikemode: 'across'
                }},
                yaxis: {{
                    title: 'f(x)',
                    showgrid: true,
                    zeroline: true,
                    showspikes: true,
                    spikethickness: 1,
                    spikecolor: '#999',
                    spikemode: 'across'
                }},
                hovermode: 'closest',
                showlegend: true,
                plot_bgcolor: 'white',
                paper_bgcolor: 'white'
            }};
            
            const plotConfig = {{
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: [
                    'pan2d',
                    'zoomin2d',
                    'zoomout2d',
                    'autoScale2d',
                    'hoverClosestCartesian',
                    'hoverCompareCartesian'
                ],
                toImageButtonOptions: {{
                    format: 'png',
                    filename: '{}_plot',
                    height: 600,
                    width: 800,
                    scale: 1
                }}
            }};
            
            Plotly.newPlot('plot', data, layout, plotConfig);
            
            // Add hover event listener for tooltip info
            document.getElementById('plot').on('plotly_hover', function(data) {{
                const point = data.points[0];
                document.getElementById('tooltip-info').innerHTML = 
                    `<strong>Coordinates:</strong> x = ${{point.x.toFixed(6)}}, f(x) = ${{point.y.toFixed(6)}}`;
            }});
            
            currentData = {{ x: {}, y: {} }};
        }}
        
        function updatePlot() {{
            const xMin = parseFloat(document.getElementById('xMin').value);
            const xMax = parseFloat(document.getElementById('xMax').value);
            const nPoints = parseInt(document.getElementById('points').value);
            
            // Generate data points using actual special function implementations
            const step = (xMax - xMin) / nPoints;
            const x = [];
            const y = [];
            
            for (let i = 0; i <= nPoints; i++) {{
                const xVal = xMin + i * step;
                x.push(xVal);
                // Use appropriate special function based on function _name
                const func = getSpecialFunction('{}');
                const yVal = func(xVal);
                y.push(isFinite(yVal) ? yVal : NaN);
            }}
            
            Plotly.restyle('plot', {{'x': [x], 'y': [y]}});
            currentData = {{ x: x, y: y }};
        }}
        
        function resetZoom() {{
            Plotly.relayout('plot', {{
                'xaxis.autorange': true,
                'yaxis.autorange': true
            }});
        }}
        
        function exportData() {{
            let csv = 'x,f(x)\\n';
            for (let i = 0; i < currentData.x.length; i++) {{
                csv += `${{currentData.x[i]}},${{currentData.y[i]}}\\n`;
            }}
            
            const blob = new Blob([csv], {{ type: 'text/csv' }});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = '{}_data.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }}
        
        // Initialize the plot when the page loads
        window.onload = initializePlot;
    </script>
</body>
</html>
        "#,
            function_name,
            function_name,
            x_range.0,
            x_range.1,
            if config.enable_tooltips {
                r#"<button onclick="toggleTooltips()">Toggle Info</button>"#
            } else {
                ""
            },
            x_json,
            y_json,
            function_name,
            function_name,
            function_name,
            function_name, // For the getSpecialFunction call
            x_json,
            y_json,
            function_name,
            function_name // For the CSV download filename
        )
    }

    /// Create interactive plots for common special functions
    pub fn create_gamma_plot() -> String {
        use crate::gamma::gamma;
        let config = InteractivePlotConfig {
            enable_zoom: true,
            enable_pan: true,
            enable_tooltips: true,
            enable_export: true,
        };
        create_interactive_plot(gamma, config, (0.1, 5.0), "Gamma")
    }

    pub fn create_bessel_j0_plot() -> String {
        use crate::bessel::j0;
        let config = InteractivePlotConfig {
            enable_zoom: true,
            enable_pan: true,
            enable_tooltips: true,
            enable_export: true,
        };
        create_interactive_plot(j0, config, (-10.0, 10.0), "Bessel J0")
    }

    pub fn create_erf_plot() -> String {
        use crate::erf::erf;
        let config = InteractivePlotConfig {
            enable_zoom: true,
            enable_pan: true,
            enable_tooltips: true,
            enable_export: true,
        };
        create_interactive_plot(erf, config, (-3.0, 3.0), "Error Function")
    }

    /// Create a comparison plot with multiple special functions
    pub fn create_comparison_plot() -> String {
        // This would create a plot comparing multiple functions
        let template = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Special Functions Comparison</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; }
        #plot { width: 100%; height: 700px; }
        .controls { margin-bottom: 20px; text-align: center; }
        button { margin: 5px; padding: 10px 20px; border: none; border-radius: 4px; background-color: #007bff; color: white; cursor: pointer; }
        button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Special Functions Comparison</h1>
        <div class="controls">
            <button onclick="showGamma()">Gamma Function</button>
            <button onclick="showBessel()">Bessel J0</button>
            <button onclick="showErf()">Error Function</button>
            <button onclick="showAll()">Show All</button>
        </div>
        <div id="plot"></div>
    </div>
    
    <script>
        function generateData(func, range, nPoints, name) {
            const step = (range[1] - range[0]) / nPoints;
            const x = [];
            const y = [];
            
            for (let i = 0; i <= nPoints; i++) {
                const xVal = range[0] + i * step;
                x.push(xVal);
                y.push(func(xVal));
            }
            
            return {
                x: x,
                y: y,
                type: 'scatter',
                mode: 'lines',
                name: name,
                line: { width: 2 }
            };
        }
        
        function gamma(x) {
            // Simplified gamma function approximation for demo
            if (x < 0) return NaN;
            if (x === 0) return Infinity;
            if (x === 1 || x === 2) return 1;
            
            // Stirling's approximation for simplicity
            if (x > 1) {
                return Math.sqrt(2 * Math.PI / x) * Math.pow(x / Math.E, x);
            }
            return gamma(x + 1) / x;
        }
        
        function besselJ0(x) {
            // Simplified Bessel J0 approximation
            const ax = Math.abs(x);
            if (ax < 8) {
                const y = x * x;
                return ((-0.0000000000000000015 * y + 0.000000000000000176) * y +
                       (-0.0000000000000156) * y + 0.0000000000164) * y +
                       (-0.00000000106) * y + 0.000000421) * y +
                       (-0.0000103) * y + 0.00015625) * y +
                       (-0.015625) * y + 1;
            } else {
                const z = 8 / ax;
                const y = z * z;
                const xx = ax - 0.785398164;
                return Math.sqrt(0.636619772 / ax) *
                       (Math.cos(xx) * (1 + y * (-0.0703125 + y * 0.1121520996)) +
                        z * Math.sin(xx) * (-0.0390625 + y * 0.0444479255));
            }
        }
        
        function erf(x) {
            // Simplified error function approximation
            const a1 =  0.254829592;
            const a2 = -0.284496736;
            const a3 =  1.421413741;
            const a4 = -1.453152027;
            const a5 =  1.061405429;
            const p  =  0.3275911;
            
            const sign = x >= 0 ? 1 : -1;
            x = Math.abs(x);
            
            const t = 1.0 / (1.0 + p * x);
            const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
            
            return sign * y;
        }
        
        function showGamma() {
            const data = [generateData(gamma, [0.1, 5], 1000, 'Gamma(x)')];
            Plotly.newPlot('plot', data, {
                title: 'Gamma Function',
                xaxis: { title: 'x' },
                yaxis: { title: 'Γ(x)' }
            });
        }
        
        function showBessel() {
            const data = [generateData(besselJ0, [-15, 15], 1000, 'J₀(x)')];
            Plotly.newPlot('plot', data, {
                title: 'Bessel Function of the First Kind (J₀)',
                xaxis: { title: 'x' },
                yaxis: { title: 'J₀(x)' }
            });
        }
        
        function showErf() {
            const data = [generateData(erf, [-3, 3], 1000, 'erf(x)')];
            Plotly.newPlot('plot', data, {
                title: 'Error Function',
                xaxis: { title: 'x' },
                yaxis: { title: 'erf(x)' }
            });
        }
        
        function showAll() {
            const data = [
                generateData(x => gamma(x) / 10, [0.1, 3], 500, 'Γ(x)/10'),
                generateData(besselJ0, [-10, 10], 500, 'J₀(x)'),
                generateData(erf, [-3, 3], 500, 'erf(x)')
            ];
            Plotly.newPlot('plot', data, {
                title: 'Special Functions Comparison',
                xaxis: { title: 'x' },
                yaxis: { title: 'f(x)' }
            });
        }
        
        // Initialize with gamma function
        window.onload = showGamma;
    </script>
</body>
</html>
        "#;

        template.to_string()
    }
}

/// Export functions for different formats
pub mod export {
    use super::*;

    /// Export formats
    pub enum ExportFormat {
        PNG,
        SVG,
        #[cfg(feature = "pdf")]
        PDF,
        LaTeX,
        CSV,
    }

    /// Export plot data in various formats
    pub fn export_plot_data<F>(
        f: F,
        x_range: (f64, f64),
        n_points: usize,
        format: ExportFormat,
    ) -> Result<Vec<u8>, Box<dyn Error>>
    where
        F: Fn(f64) -> f64,
    {
        match format {
            ExportFormat::CSV => {
                let mut csv_data = String::from("x,y\n");
                let step = (x_range.1 - x_range.0) / n_points as f64;

                for i in 0..=n_points {
                    let x = x_range.0 + i as f64 * step;
                    let y = f(x);
                    csv_data.push_str(&format!("{},{}\n", x, y));
                }

                Ok(csv_data.into_bytes())
            }
            ExportFormat::LaTeX => {
                // Generate LaTeX/TikZ code
                let mut latex = String::from("\\begin{tikzpicture}\n\\begin{axis}[\n");
                latex.push_str("    xlabel=$x$,\n    ylabel=$f(x)$,\n]\n");
                latex.push_str("\\addplot[blue,thick] coordinates {\n");

                let step = (x_range.1 - x_range.0) / n_points as f64;
                for i in 0..=n_points {
                    let x = x_range.0 + i as f64 * step;
                    let y = f(x);
                    if y.is_finite() {
                        latex.push_str(&format!("    ({},{})\n", x, y));
                    }
                }

                latex.push_str("};\n\\end{axis}\n\\end{tikzpicture}\n");
                Ok(latex.into_bytes())
            }
            #[cfg(feature = "pdf")]
            ExportFormat::PDF => pdf_export::render_pdf(&f, x_range, n_points),
            ExportFormat::PNG => {
                // Generate PNG using plotters
                let mut png_data = Vec::new();
                {
                    let backend =
                        plotters::backend::BitMapBackend::with_buffer(&mut png_data, (800, 600))
                            .into_drawing_area();
                    backend
                        .fill(&plotters::style::colors::WHITE)
                        .map_err(|e| format!("Failed to fill background: {}", e))?;

                    let mut chart = plotters::chart::ChartBuilder::on(&backend)
                        .caption("Special Function Plot", ("sans-serif", 30))
                        .margin(10)
                        .x_label_area_size(30)
                        .y_label_area_size(40)
                        .build_cartesian_2d(x_range.0..x_range.1, -2f64..2f64)
                        .map_err(|e| format!("Failed to build chart: {}", e))?;

                    chart
                        .configure_mesh()
                        .x_desc("x")
                        .y_desc("f(x)")
                        .draw()
                        .map_err(|e| format!("Failed to draw mesh: {}", e))?;

                    // Generate data _points
                    let data: Vec<(f64, f64)> = (0..=n_points)
                        .map(|i| {
                            let x =
                                x_range.0 + i as f64 * (x_range.1 - x_range.0) / n_points as f64;
                            let y = f(x);
                            (x, y)
                        })
                        .filter(|(_, y)| y.is_finite())
                        .collect();

                    chart
                        .draw_series(plotters::series::LineSeries::new(
                            data,
                            &plotters::style::colors::BLUE,
                        ))
                        .map_err(|e| format!("Failed to draw series: {}", e))?;

                    backend
                        .present()
                        .map_err(|e| format!("Failed to present plot: {}", e))?;
                }
                // Convert to PNG bytes - this is a simplified approach
                // In a real implementation, you'd need proper PNG encoding
                Ok(png_data)
            }
            ExportFormat::SVG => {
                // Generate SVG using plotters
                let mut svg_data = String::new();
                {
                    let backend =
                        plotters::backend::SVGBackend::with_string(&mut svg_data, (800, 600));
                    let root = backend.into_drawing_area();
                    root.fill(&plotters::style::colors::WHITE)
                        .map_err(|e| format!("Failed to fill background: {}", e))?;

                    let mut chart = plotters::chart::ChartBuilder::on(&root)
                        .caption("Special Function Plot", ("sans-serif", 30))
                        .margin(10)
                        .x_label_area_size(30)
                        .y_label_area_size(40)
                        .build_cartesian_2d(x_range.0..x_range.1, -2f64..2f64)
                        .map_err(|e| format!("Failed to build chart: {}", e))?;

                    chart
                        .configure_mesh()
                        .x_desc("x")
                        .y_desc("f(x)")
                        .draw()
                        .map_err(|e| format!("Failed to draw mesh: {}", e))?;

                    // Generate data _points
                    let data: Vec<(f64, f64)> = (0..=n_points)
                        .map(|i| {
                            let x =
                                x_range.0 + i as f64 * (x_range.1 - x_range.0) / n_points as f64;
                            let y = f(x);
                            (x, y)
                        })
                        .filter(|(_, y)| y.is_finite())
                        .collect();

                    chart
                        .draw_series(plotters::series::LineSeries::new(
                            data,
                            &plotters::style::colors::BLUE,
                        ))
                        .map_err(|e| format!("Failed to draw series: {}", e))?;

                    root.present()
                        .map_err(|e| format!("Failed to present plot: {}", e))?;
                }
                Ok(svg_data.into_bytes())
            }
        }
    }

    /// Pure-Rust PDF rendering for the PDF export branch of [`export_plot_data`].
    ///
    /// Builds a single landscape A4 page using `printpdf` with the following layout:
    ///
    /// * Page: 297 mm × 210 mm (A4 landscape).
    /// * Plot region: an inner rectangle obtained by stripping a 30 mm margin on every side.
    /// * Curve: `n_points + 1` samples of `f` over `x_range`, with non-finite y-values dropped
    ///   *before* the y-range is computed so they neither bias auto-scaling nor produce broken
    ///   line segments.
    /// * Y-range: data extent inflated by a 5 % padding factor; degenerates to `[ymin-1, ymax+1]`
    ///   when the curve is flat (so a horizontal line still renders inside the frame).
    /// * Axes: black 1 pt outlines along the bottom and left edges of the plot region.
    /// * Tick marks: 8 along x and 6 along y, with numeric labels rendered with the standard
    ///   PDF Helvetica built-in font (no font embedding required, keeps the output small and
    ///   the dependency tree free of font files).
    /// * Titles: chart title centred above the plot, x-axis label centred below, y-axis label
    ///   rotated 90° on the left side via `TextMatrix::TranslateRotate`.
    /// * Curve: drawn as a polyline (`Op::DrawLine`) in blue (RGB 0.0 / 0.4 / 0.8) with a 1.2 pt
    ///   stroke. Runs of finite samples that are interrupted by a non-finite value are split
    ///   into separate `DrawLine` operations so discontinuities remain visible rather than
    ///   being bridged by a misleading straight segment.
    #[cfg(feature = "pdf")]
    pub(super) mod pdf_export {
        use super::*;
        use printpdf::{
            BuiltinFont, Color, Line, LinePoint, Mm, Op, PdfDocument, PdfFontHandle, PdfPage,
            PdfSaveOptions, Point, Pt, Rgb, TextItem, TextMatrix,
        };

        // --- Page geometry constants (millimetres) ----------------------------------------

        const PAGE_WIDTH_MM: f32 = 297.0; // A4 landscape long side
        const PAGE_HEIGHT_MM: f32 = 210.0; // A4 landscape short side
        const MARGIN_MM: f32 = 30.0;

        const PLOT_X_MIN_MM: f32 = MARGIN_MM;
        const PLOT_X_MAX_MM: f32 = PAGE_WIDTH_MM - MARGIN_MM;
        const PLOT_Y_MIN_MM: f32 = MARGIN_MM;
        const PLOT_Y_MAX_MM: f32 = PAGE_HEIGHT_MM - MARGIN_MM;
        const PLOT_WIDTH_MM: f32 = PLOT_X_MAX_MM - PLOT_X_MIN_MM;
        const PLOT_HEIGHT_MM: f32 = PLOT_Y_MAX_MM - PLOT_Y_MIN_MM;

        // --- Tick / typography constants -------------------------------------------------

        const X_TICKS: usize = 8;
        const Y_TICKS: usize = 6;
        const TICK_LEN_MM: f32 = 2.0;
        const AXIS_LINE_PT: f32 = 1.0;
        const CURVE_LINE_PT: f32 = 1.2;
        const TITLE_FONT_PT: f32 = 14.0;
        const LABEL_FONT_PT: f32 = 11.0;
        const TICK_FONT_PT: f32 = 9.0;

        // --- Y-range padding -------------------------------------------------------------

        const Y_PADDING_FACTOR: f32 = 0.05;

        /// Black, no ICC profile.
        fn black() -> Color {
            Color::Rgb(Rgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                icc_profile: None,
            })
        }

        /// Curve colour — a moderately saturated blue.
        fn curve_color() -> Color {
            Color::Rgb(Rgb {
                r: 0.0,
                g: 0.4,
                b: 0.8,
                icc_profile: None,
            })
        }

        /// Convert a data-space x-coordinate to a page-space x in millimetres.
        fn x_to_mm(x: f64, x_range: (f64, f64)) -> f32 {
            let span = x_range.1 - x_range.0;
            // Caller guarantees x_range.0 < x_range.1; if it doesn't, we still produce a
            // well-defined value (mid-plot) instead of NaN, which keeps the renderer safe.
            if span.abs() < f64::EPSILON {
                PLOT_X_MIN_MM + PLOT_WIDTH_MM * 0.5
            } else {
                let t = ((x - x_range.0) / span) as f32;
                PLOT_X_MIN_MM + t.clamp(0.0, 1.0) * PLOT_WIDTH_MM
            }
        }

        /// Convert a data-space y-coordinate to a page-space y in millimetres.
        fn y_to_mm(y: f64, y_range: (f64, f64)) -> f32 {
            let span = y_range.1 - y_range.0;
            if span.abs() < f64::EPSILON {
                PLOT_Y_MIN_MM + PLOT_HEIGHT_MM * 0.5
            } else {
                let t = ((y - y_range.0) / span) as f32;
                PLOT_Y_MIN_MM + t.clamp(0.0, 1.0) * PLOT_HEIGHT_MM
            }
        }

        /// Format a tick label compactly: integer if essentially integral, else 4 sig figs.
        fn fmt_tick(value: f64) -> String {
            if !value.is_finite() {
                return "n/a".to_string();
            }
            let rounded = value.round();
            if (value - rounded).abs() < 1e-9 && value.abs() < 1e7 {
                return format!("{}", rounded as i64);
            }
            let abs = value.abs();
            if abs >= 1e4 || (abs > 0.0 && abs < 1e-3) {
                format!("{:.3e}", value)
            } else {
                format!("{:.4}", value)
            }
        }

        /// Sample `f` over `x_range` with `n_points + 1` points; returns the (possibly
        /// non-finite) y-values aligned with their x-positions.
        fn sample_curve<F>(f: &F, x_range: (f64, f64), n_points: usize) -> Vec<(f64, f64)>
        where
            F: Fn(f64) -> f64,
        {
            // We deliberately use n_points + 1 samples (matches PNG/SVG branches and includes
            // both endpoints).
            let n = n_points.max(1);
            let step = (x_range.1 - x_range.0) / n as f64;
            (0..=n)
                .map(|i| {
                    let x = x_range.0 + i as f64 * step;
                    (x, f(x))
                })
                .collect()
        }

        /// Compute the plotted y-range from the sample buffer, ignoring non-finite y-values.
        /// Falls back to a centred unit interval when no finite samples exist.
        fn finite_y_range(samples: &[(f64, f64)]) -> (f64, f64) {
            let mut y_min = f64::INFINITY;
            let mut y_max = f64::NEG_INFINITY;
            for &(_, y) in samples {
                if y.is_finite() {
                    if y < y_min {
                        y_min = y;
                    }
                    if y > y_max {
                        y_max = y;
                    }
                }
            }
            if !y_min.is_finite() || !y_max.is_finite() {
                // No finite sample at all — give the curve a small canvas so the axes still
                // render correctly.
                return (-1.0, 1.0);
            }
            if (y_max - y_min).abs() < f64::EPSILON {
                // Flat curve: synthesise a 1-unit window so the line is interior to the box.
                return (y_min - 1.0, y_max + 1.0);
            }
            let pad = (y_max - y_min) * Y_PADDING_FACTOR as f64;
            (y_min - pad, y_max + pad)
        }

        /// Build a single polyline `Op::DrawLine` for a contiguous run of finite points.
        fn polyline_op(points: &[(f32, f32)]) -> Op {
            Op::DrawLine {
                line: Line {
                    points: points
                        .iter()
                        .map(|&(mx, my)| LinePoint {
                            p: Point {
                                x: Mm(mx).into(),
                                y: Mm(my).into(),
                            },
                            bezier: false,
                        })
                        .collect(),
                    is_closed: false,
                },
            }
        }

        /// Helper: emit a labelled text block at a page-space anchor in millimetres.
        fn text_at(text: &str, anchor_mm: (f32, f32), size_pt: f32) -> Vec<Op> {
            vec![
                Op::StartTextSection,
                Op::SetTextCursor {
                    pos: Point::new(Mm(anchor_mm.0), Mm(anchor_mm.1)),
                },
                Op::SetFont {
                    font: PdfFontHandle::Builtin(BuiltinFont::Helvetica),
                    size: Pt(size_pt),
                },
                Op::SetLineHeight { lh: Pt(size_pt) },
                Op::SetFillColor { col: black() },
                Op::ShowText {
                    items: vec![TextItem::Text(text.to_string())],
                },
                Op::EndTextSection,
            ]
        }

        /// Helper: emit a vertical (90° CCW) text block centred along the y-axis.
        fn text_rotated(text: &str, anchor_mm: (f32, f32), size_pt: f32) -> Vec<Op> {
            // Convert mm -> pt for the text matrix (which expects raw points).
            let x_pt = Mm(anchor_mm.0).into_pt().0;
            let y_pt = Mm(anchor_mm.1).into_pt().0;
            vec![
                Op::StartTextSection,
                Op::SetFont {
                    font: PdfFontHandle::Builtin(BuiltinFont::Helvetica),
                    size: Pt(size_pt),
                },
                Op::SetLineHeight { lh: Pt(size_pt) },
                Op::SetFillColor { col: black() },
                Op::SetTextMatrix {
                    matrix: TextMatrix::TranslateRotate(Pt(x_pt), Pt(y_pt), 90.0),
                },
                Op::ShowText {
                    items: vec![TextItem::Text(text.to_string())],
                },
                Op::EndTextSection,
            ]
        }

        /// A single straight line in page-space (mm), drawn with the current outline colour
        /// and thickness.
        fn line_segment_mm(from: (f32, f32), to: (f32, f32)) -> Op {
            Op::DrawLine {
                line: Line {
                    points: vec![
                        LinePoint {
                            p: Point::new(Mm(from.0), Mm(from.1)),
                            bezier: false,
                        },
                        LinePoint {
                            p: Point::new(Mm(to.0), Mm(to.1)),
                            bezier: false,
                        },
                    ],
                    is_closed: false,
                },
            }
        }

        /// Emit the plot frame (left + bottom axis lines) plus tick marks and tick labels.
        fn axes_and_ticks(x_range: (f64, f64), y_range: (f64, f64)) -> Vec<Op> {
            let mut ops: Vec<Op> = vec![
                Op::SetOutlineColor { col: black() },
                Op::SetOutlineThickness {
                    pt: Pt(AXIS_LINE_PT),
                },
                // Bottom axis (x).
                line_segment_mm(
                    (PLOT_X_MIN_MM, PLOT_Y_MIN_MM),
                    (PLOT_X_MAX_MM, PLOT_Y_MIN_MM),
                ),
                // Left axis (y).
                line_segment_mm(
                    (PLOT_X_MIN_MM, PLOT_Y_MIN_MM),
                    (PLOT_X_MIN_MM, PLOT_Y_MAX_MM),
                ),
            ];

            // X-axis ticks + labels.
            for i in 0..=X_TICKS {
                let t = i as f32 / X_TICKS as f32;
                let x_mm = PLOT_X_MIN_MM + t * PLOT_WIDTH_MM;
                ops.push(line_segment_mm(
                    (x_mm, PLOT_Y_MIN_MM),
                    (x_mm, PLOT_Y_MIN_MM - TICK_LEN_MM),
                ));
                let value = x_range.0 + (x_range.1 - x_range.0) * t as f64;
                let label = fmt_tick(value);
                // Approximate text width centring (Helvetica char ~ 0.55em wide).
                let approx_width_mm =
                    label.chars().count() as f32 * (TICK_FONT_PT * 0.55) / 2.834_646;
                let label_x = x_mm - approx_width_mm * 0.5;
                let label_y = PLOT_Y_MIN_MM - TICK_LEN_MM - 4.0;
                ops.extend(text_at(&label, (label_x, label_y), TICK_FONT_PT));
            }

            // Y-axis ticks + labels.
            for i in 0..=Y_TICKS {
                let t = i as f32 / Y_TICKS as f32;
                let y_mm = PLOT_Y_MIN_MM + t * PLOT_HEIGHT_MM;
                ops.push(line_segment_mm(
                    (PLOT_X_MIN_MM, y_mm),
                    (PLOT_X_MIN_MM - TICK_LEN_MM, y_mm),
                ));
                let value = y_range.0 + (y_range.1 - y_range.0) * t as f64;
                let label = fmt_tick(value);
                let approx_width_mm =
                    label.chars().count() as f32 * (TICK_FONT_PT * 0.55) / 2.834_646;
                // Right-aligned label with a 1 mm gap before the tick mark.
                let label_x = PLOT_X_MIN_MM - TICK_LEN_MM - 1.0 - approx_width_mm;
                let label_y = y_mm - (TICK_FONT_PT * 0.35) / 2.834_646;
                ops.extend(text_at(&label, (label_x, label_y), TICK_FONT_PT));
            }

            ops
        }

        /// Draw the curve, splitting at non-finite gaps so discontinuities aren't bridged.
        fn curve_polylines(
            samples: &[(f64, f64)],
            y_range: (f64, f64),
            x_range: (f64, f64),
        ) -> Vec<Op> {
            let mut ops: Vec<Op> = vec![
                Op::SetOutlineColor { col: curve_color() },
                Op::SetOutlineThickness {
                    pt: Pt(CURVE_LINE_PT),
                },
            ];

            let mut current: Vec<(f32, f32)> = Vec::new();
            for &(x, y) in samples {
                if y.is_finite() {
                    let mx = x_to_mm(x, x_range);
                    let my = y_to_mm(y, y_range);
                    current.push((mx, my));
                } else if current.len() >= 2 {
                    ops.push(polyline_op(&current));
                    current.clear();
                } else {
                    current.clear();
                }
            }
            if current.len() >= 2 {
                ops.push(polyline_op(&current));
            }

            ops
        }

        /// Render the chart and return PDF bytes.
        pub fn render_pdf<F>(
            f: &F,
            x_range: (f64, f64),
            n_points: usize,
        ) -> Result<Vec<u8>, Box<dyn Error>>
        where
            F: Fn(f64) -> f64,
        {
            // Validate the x-range once up-front so the rest of the pipeline can assume a
            // monotone increasing interval and finite endpoints.
            if !(x_range.0.is_finite() && x_range.1.is_finite()) {
                return Err(format!(
                    "PDF generation failed: x_range endpoints must be finite, got ({}, {})",
                    x_range.0, x_range.1
                )
                .into());
            }
            if x_range.0 >= x_range.1 {
                return Err(format!(
                    "PDF generation failed: x_range must be strictly increasing, got ({}, {})",
                    x_range.0, x_range.1
                )
                .into());
            }
            if n_points == 0 {
                return Err("PDF generation failed: n_points must be > 0"
                    .to_string()
                    .into());
            }

            let samples = sample_curve(f, x_range, n_points);
            let y_range = finite_y_range(&samples);

            let mut ops: Vec<Op> = Vec::new();

            // Plot frame interior left blank — only axes, ticks, labels and curve are drawn.
            ops.extend(axes_and_ticks(x_range, y_range));

            // Chart title — centred horizontally above the plot region.
            {
                let title = "Special Function Plot";
                let approx_width_mm =
                    title.chars().count() as f32 * (TITLE_FONT_PT * 0.55) / 2.834_646;
                let title_x = (PAGE_WIDTH_MM - approx_width_mm) * 0.5;
                let title_y = PLOT_Y_MAX_MM + 12.0;
                ops.extend(text_at(title, (title_x, title_y), TITLE_FONT_PT));
            }

            // X-axis label.
            {
                let label = "x";
                let approx_width_mm =
                    label.chars().count() as f32 * (LABEL_FONT_PT * 0.55) / 2.834_646;
                let label_x = (PLOT_X_MIN_MM + PLOT_X_MAX_MM) * 0.5 - approx_width_mm * 0.5;
                let label_y = PLOT_Y_MIN_MM - TICK_LEN_MM - 14.0;
                ops.extend(text_at(label, (label_x, label_y), LABEL_FONT_PT));
            }

            // Y-axis label (rotated 90°, anchored on the left margin centred vertically).
            {
                let label = "f(x)";
                let approx_width_mm =
                    label.chars().count() as f32 * (LABEL_FONT_PT * 0.55) / 2.834_646;
                let label_x = PLOT_X_MIN_MM - 18.0;
                let label_y = (PLOT_Y_MIN_MM + PLOT_Y_MAX_MM) * 0.5 - approx_width_mm * 0.5;
                ops.extend(text_rotated(label, (label_x, label_y), LABEL_FONT_PT));
            }

            // The curve last so it sits on top of the axes.
            ops.extend(curve_polylines(&samples, y_range, x_range));

            let mut doc = PdfDocument::new("Special Function Plot");
            let page = PdfPage::new(Mm(PAGE_WIDTH_MM), Mm(PAGE_HEIGHT_MM), ops);

            let bytes = doc
                .with_pages(vec![page])
                .save(&PdfSaveOptions::default(), &mut Vec::new());

            // Sanity check: a minimal PDF still has a `%PDF-` signature; if the underlying
            // serializer ever returned an empty buffer we'd want to surface that as an error
            // rather than producing an invalid PDF blob.
            if bytes.len() < 5 || !bytes.starts_with(b"%PDF-") {
                return Err(
                    "PDF generation failed: serializer returned a non-PDF byte buffer"
                        .to_string()
                        .into(),
                );
            }

            Ok(bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_config() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert!(config.show_grid);
    }

    #[test]
    fn test_export_csv() {
        let data = export::export_plot_data(|x| x * x, (0.0, 1.0), 10, export::ExportFormat::CSV)
            .expect("Operation failed");

        let csv = String::from_utf8(data).expect("Operation failed");
        assert!(csv.contains("x,y\n"));
        assert!(csv.contains("0,0\n"));
        assert!(csv.contains("1,1\n"));
    }

    /// PDF export must produce a buffer that begins with the standard `%PDF-` magic header.
    #[cfg(feature = "pdf")]
    #[test]
    fn test_export_pdf_magic_bytes() {
        let data = export::export_plot_data(
            |x: f64| x.sin(),
            (-std::f64::consts::PI, std::f64::consts::PI),
            100,
            export::ExportFormat::PDF,
        )
        .expect("PDF export should succeed for sin(x) over (-π, π)");

        assert!(
            data.len() >= 5 && data.starts_with(b"%PDF-"),
            "PDF output must start with %PDF- magic bytes; got first 16 bytes: {:?}",
            &data[..data.len().min(16)]
        );
    }

    /// A real chart with axes, ticks, labels and a curve is well over 1 KB. A trivial empty
    /// PDF (header + xref + trailer + EOF) is ~300 bytes; this test guards against the PDF
    /// arm regressing back to a placeholder document.
    #[cfg(feature = "pdf")]
    #[test]
    fn test_export_pdf_meaningful_size() {
        let data = export::export_plot_data(
            |x: f64| x.cos(),
            (-2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI),
            200,
            export::ExportFormat::PDF,
        )
        .expect("PDF export should succeed for cos(x)");

        assert!(
            data.len() > 1024,
            "Rendered PDF should be > 1 KB to indicate a non-trivial chart; got {} bytes",
            data.len()
        );
    }

    /// Round-trip the PDF bytes through the filesystem to ensure they survive a write/read
    /// cycle unchanged. Uses `std::env::temp_dir()` per the project's test I/O policy.
    #[cfg(feature = "pdf")]
    #[test]
    fn test_export_pdf_roundtrip_via_temp_file() {
        use std::fs;
        use std::io::Write;

        let data = export::export_plot_data(
            |x: f64| 1.0 / (1.0 + x * x),
            (-5.0, 5.0),
            150,
            export::ExportFormat::PDF,
        )
        .expect("PDF export should succeed for the Cauchy/Lorentzian peak");

        let mut path = std::env::temp_dir();
        path.push(format!(
            "scirs2_special_pdf_export_{}.pdf",
            std::process::id()
        ));

        {
            let mut file = fs::File::create(&path).expect("create temp pdf file");
            file.write_all(&data).expect("write pdf bytes");
            file.flush().expect("flush pdf bytes");
        }

        let read_back = fs::read(&path).expect("read pdf bytes back");
        assert_eq!(
            data.len(),
            read_back.len(),
            "Written and read-back PDF must be byte-for-byte identical in length",
        );
        assert_eq!(
            data, read_back,
            "Written and read-back PDF must be byte-for-byte identical",
        );

        let _ = fs::remove_file(&path);
    }

    /// Functions that produce non-finite values (e.g. `1/x` near 0) must be handled by the
    /// PDF renderer: those samples are filtered before y-range computation and split the
    /// curve into multiple polylines instead of bridging the asymptote with a straight line.
    #[cfg(feature = "pdf")]
    #[test]
    fn test_export_pdf_non_finite_filtering() {
        // Helper closure asserts PDF magic + a meaningful (>1KB) chart for one case.
        let assert_pdf_with_filtered_samples =
            |label: &str, f: &dyn Fn(f64) -> f64, range: (f64, f64), n: usize| {
                let data = export::export_plot_data(f, range, n, export::ExportFormat::PDF)
                    .unwrap_or_else(|e| {
                        panic!("PDF export must not panic on non-finite samples ({label}): {e}");
                    });
                assert!(
                    data.starts_with(b"%PDF-"),
                    "[{label}] Non-finite-aware rendering must still emit a valid PDF header"
                );
                assert!(
                    data.len() > 1024,
                    "[{label}] Even after filtering non-finite samples, the chart should still \
                     be substantial; got {} bytes",
                    data.len()
                );
            };

        // `1/x` over the symmetric interval (-1, 1) with `n_points = 100` produces a step
        // of `2/100 = 0.02`, so the sample at `i = 50` lands exactly on `x = 0` and yields
        // +∞ — directly exercising the non-finite branch in `curve_polylines`.
        assert_pdf_with_filtered_samples("1/x straddling 0", &|x| 1.0 / x, (-1.0, 1.0), 100);

        // `tan(x)` near (±π/2): the endpoint samples blow up, splitting the polyline near
        // both extremes.
        assert_pdf_with_filtered_samples(
            "tan(x) near asymptotes",
            &|x: f64| x.tan(),
            (
                -std::f64::consts::FRAC_PI_2 + 1e-9,
                std::f64::consts::FRAC_PI_2 - 1e-9,
            ),
            200,
        );

        // `sqrt(x)` over (-1, 1): every negative-x sample produces NaN, so half of the
        // samples are filtered out — exercises the "drop run when current.len() < 2" path.
        assert_pdf_with_filtered_samples(
            "sqrt(x) with negative-half NaN region",
            &|x: f64| x.sqrt(),
            (-1.0, 1.0),
            100,
        );

        // Adversarial: a function whose every output is non-finite must still produce a
        // valid PDF (just with no curve drawn — only axes / labels / placeholder y-range).
        let data = export::export_plot_data(
            |_x: f64| f64::NAN,
            (-1.0, 1.0),
            50,
            export::ExportFormat::PDF,
        )
        .expect("PDF export must succeed even when every sample is non-finite");
        assert!(
            data.starts_with(b"%PDF-"),
            "Empty-curve PDF must still have valid header"
        );
    }

    /// Edge case: an x-range with non-monotonic endpoints must be rejected with a useful
    /// error message rather than producing a malformed PDF.
    #[cfg(feature = "pdf")]
    #[test]
    fn test_export_pdf_rejects_inverted_range() {
        let result =
            export::export_plot_data(|x: f64| x, (1.0, -1.0), 10, export::ExportFormat::PDF);
        assert!(result.is_err(), "inverted x-range must be rejected");
    }

    /// Edge case: zero samples must be rejected — a chart with no data points has nothing
    /// to render and would silently produce an empty curve in earlier iterations.
    #[cfg(feature = "pdf")]
    #[test]
    fn test_export_pdf_rejects_zero_points() {
        let result = export::export_plot_data(|x: f64| x, (0.0, 1.0), 0, export::ExportFormat::PDF);
        assert!(result.is_err(), "n_points = 0 must be rejected");
    }
}
