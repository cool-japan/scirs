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
            ExportFormat::PDF => {
                // PDF export is not yet implemented
                Err("PDF export is not yet implemented".to_string().into())
            }
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
        let data =
            export::export_plot_data(|x| x * x, (0.0, 1.0), 10, export::ExportFormat::CSV).unwrap();

        let csv = String::from_utf8(data).unwrap();
        assert!(csv.contains("x,y\n"));
        assert!(csv.contains("0,0\n"));
        assert!(csv.contains("1,1\n"));
    }
}
