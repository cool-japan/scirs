//! Real exporters for [`super::Visualization3DBuilder`]'s 3D plot formats.
//!
//! Mirrors the fix applied to the 2D `get_exporter` dispatcher: the 3D
//! counterpart, `get_3d_exporter`, previously mapped every
//! [`super::VisualizationFormat`] other than `PlotlyJson` onto
//! `Plotly3DExporter`, so e.g. requesting `Svg`/`MatplotlibPython` 3D output
//! silently produced Plotly JSON mislabeled as a different format. This
//! module provides genuine, format-distinguishing exporters:
//!
//! - `MatplotlibPython` / `Gnuplot`: both tools have first-class 3D plotting
//!   modes (`mpl_toolkits.mplot3d`, `splot`), so these emit real 3D scripts.
//! - `D3Json`: D3 is an unopinionated data-binding library -- the existing
//!   2D `D3Exporter` is likewise just a data export, not a chart spec -- so
//!   an `{x, y, z}` data export is a faithful "D3 3D export" with no
//!   fabricated chart-type claim.
//! - `Svg` / `Html`: SVG is a bare drawing canvas with no inherent
//!   dimensionality restriction, so these render a real orthographic
//!   look-at projection of the 3D primitives (equivalent to the classic
//!   `gluLookAt` view transform, dropping the depth component).
//! - `BokehJson` / `VegaLite`: both are genuinely 2D-only grammars --
//!   Bokeh's model/glyph system and Vega-Lite's encoding channels have no
//!   z/scene concept at all -- so fabricating a "3D Bokeh document" or a `z`
//!   encoding channel would invent schema neither library actually has.
//!   These return an explicit [`IoError::UnsupportedFormat`] instead of
//!   silently mislabeling Plotly output as one of them.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::error::{IoError, Result};
use crate::metadata::Metadata;

use super::{CameraConfig, DataSeries3D, Plot3DConfig, PlotType3D, Visualization3DExporter};

/// Escapes text for safe inclusion in XML/SVG attribute values and content.
/// (Deliberately duplicated from `super::svg`'s private helper rather than
/// exposed across modules, to keep this fix isolated from the
/// already-verified 2D SVG exporter.)
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ============================================================================
// 3D -> 2D projection geometry
// ============================================================================

type Point3 = [f64; 3];

fn sub(a: Point3, b: Point3) -> Point3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: Point3, b: Point3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: Point3, b: Point3) -> Point3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: Point3) -> f64 {
    dot(a, a).sqrt()
}
fn normalize(a: Point3) -> Point3 {
    let n = norm(a);
    if n > 1e-9 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Computes each axis's overall `(min, max)` across every series' raw `x`/`y`/`z`
/// values, so 3D data can be normalized into a roughly unit-scaled cube before
/// applying the camera transform (matching the scale [`CameraConfig::default`]'s
/// `eye = [1.25, 1.25, 1.25]` is designed for). Degenerate (empty or
/// zero-width) axes fall back to a unit range so normalization never divides
/// by zero or produces NaN.
fn compute_bounds(data: &[DataSeries3D]) -> (Point3, Point3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for series in data {
        for &v in &series.x {
            lo[0] = lo[0].min(v);
            hi[0] = hi[0].max(v);
        }
        for &v in &series.y {
            lo[1] = lo[1].min(v);
            hi[1] = hi[1].max(v);
        }
        for &v in &series.z {
            lo[2] = lo[2].min(v);
            hi[2] = hi[2].max(v);
        }
    }
    for axis in 0..3 {
        if !lo[axis].is_finite() || !hi[axis].is_finite() {
            lo[axis] = -1.0;
            hi[axis] = 1.0;
        } else if hi[axis] - lo[axis] < 1e-12 {
            let mid = lo[axis];
            lo[axis] = mid - 1.0;
            hi[axis] = mid + 1.0;
        }
    }
    (lo, hi)
}

/// Maps a raw data-space point into `[-1, 1]^3` given the bounds from
/// [`compute_bounds`] (each axis normalized independently, matching the
/// "auto" aspect-ratio behaviour most 3D plotting tools default to).
fn normalize_point(p: Point3, lo: Point3, hi: Point3) -> Point3 {
    let mut out = [0.0; 3];
    for axis in 0..3 {
        let mid = (lo[axis] + hi[axis]) / 2.0;
        let half = (hi[axis] - lo[axis]) / 2.0;
        out[axis] = if half > 1e-12 {
            (p[axis] - mid) / half
        } else {
            0.0
        };
    }
    out
}

/// Orthographically projects normalized 3D scene points onto a 2D view
/// plane using the standard look-at view basis (the same `eye`/`side`/`up`
/// construction as the classic `gluLookAt`), then dropping the resulting
/// depth component. This is a real, standard axonometric projection
/// technique, not an approximation specific to any one plotting library.
struct Projector {
    eye: Point3,
    side: Point3,
    up: Point3,
}

impl Projector {
    fn new(camera: &CameraConfig) -> Self {
        let eye = camera.eye;
        let mut forward = normalize(sub(camera.center, eye));
        if norm(forward) < 1e-9 {
            // Degenerate (eye == center): fall back to a default viewing axis.
            forward = [0.0, 0.0, -1.0];
        }
        let mut world_up = normalize(camera.up);
        if norm(world_up) < 1e-9 {
            world_up = [0.0, 0.0, 1.0];
        }
        let mut side = cross(forward, world_up);
        if norm(side) < 1e-9 {
            // `up` parallel to the view direction: pick an arbitrary perpendicular.
            let fallback = if forward[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            side = cross(forward, fallback);
        }
        let side = normalize(side);
        let up = normalize(cross(side, forward));
        Projector { eye, side, up }
    }

    /// Projects a single 3D point (already normalized into the camera's
    /// scene space) to 2D view-plane coordinates.
    fn project(&self, p: Point3) -> (f64, f64) {
        let rel = sub(p, self.eye);
        (dot(rel, self.side), dot(rel, self.up))
    }
}

/// Renders `data` under `config` as a complete, self-contained SVG document:
/// a projected axis triad plus each supported series rendered as its 2D
/// projection (`Scatter3D` as circles, `Line3D` as a polyline, `Surface` as
/// a wireframe mesh). `Mesh3D`/`Isosurface`/`Volume` are skipped, consistent
/// with how every other exporter in this module handles plot (sub)types it
/// doesn't natively render (e.g. `Plotly3DExporter`'s own `_ => continue`).
fn render_svg_3d(data: &[DataSeries3D], config: &Plot3DConfig) -> String {
    let width = config.width.unwrap_or(800) as f64;
    let height = config.height.unwrap_or(600) as f64;
    let margin = 50.0_f64;
    let plot_w = (width - 2.0 * margin).max(1.0);
    let plot_h = (height - 2.0 * margin).max(1.0);

    let (lo, hi) = compute_bounds(data);
    let projector = Projector::new(&config.camera);

    struct Projected<'a> {
        series: &'a DataSeries3D,
        pts: Vec<(f64, f64)>,
        cols: usize,
    }

    let mut projected: Vec<Projected> = Vec::new();
    let mut all_2d: Vec<(f64, f64)> = Vec::new();

    for series in data {
        match series.plot_type {
            PlotType3D::Scatter3D | PlotType3D::Line3D => {
                let n = series.x.len().min(series.y.len()).min(series.z.len());
                let pts: Vec<(f64, f64)> = (0..n)
                    .map(|i| {
                        let p = normalize_point([series.x[i], series.y[i], series.z[i]], lo, hi);
                        projector.project(p)
                    })
                    .collect();
                all_2d.extend(pts.iter().copied());
                projected.push(Projected {
                    series,
                    pts,
                    cols: 0,
                });
            }
            PlotType3D::Surface => {
                let cols = series.x.len().max(1);
                let rows = series.y.len().max(1);
                let mut pts = Vec::with_capacity(rows * cols);
                for r in 0..rows {
                    for c in 0..cols {
                        let xv = series.x.get(c).copied().unwrap_or(0.0);
                        let yv = series.y.get(r).copied().unwrap_or(0.0);
                        let zv = series.z.get(r * cols + c).copied().unwrap_or(0.0);
                        let p = normalize_point([xv, yv, zv], lo, hi);
                        pts.push(projector.project(p));
                    }
                }
                all_2d.extend(pts.iter().copied());
                projected.push(Projected { series, pts, cols });
            }
            _ => {}
        }
    }

    // Project the normalized-space axis triad (from the "min" corner along
    // each axis) so real axis reference lines can be drawn, and so the 2D
    // bounding box used to fit everything into the SVG canvas accounts for
    // them even when there's little or no series data.
    let axis_o = projector.project([-1.0, -1.0, -1.0]);
    let axis_x = projector.project([1.0, -1.0, -1.0]);
    let axis_y = projector.project([-1.0, 1.0, -1.0]);
    let axis_z = projector.project([-1.0, -1.0, 1.0]);

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(px, py) in all_2d.iter().chain([&axis_o, &axis_x, &axis_y, &axis_z]) {
        x_min = x_min.min(px);
        x_max = x_max.max(px);
        y_min = y_min.min(py);
        y_max = y_max.max(py);
    }
    // Written as a positive well-ordered check (rather than negating a `>`
    // comparison) since `f64` is only partially ordered: a NaN bound must
    // also fall back to the default range, not merely "not > ".
    let (x_min, x_max) = if x_min.is_finite() && x_max.is_finite() && x_max > x_min {
        (x_min, x_max)
    } else {
        (-1.0, 1.0)
    };
    let (y_min, y_max) = if y_min.is_finite() && y_max.is_finite() && y_max > y_min {
        (y_min, y_max)
    } else {
        (-1.0, 1.0)
    };

    let x_scale = |x: f64| margin + (x - x_min) / (x_max - x_min) * plot_w;
    // SVG y grows downward, matching the convention the 2D renderer uses.
    let y_scale = |y: f64| margin + plot_h - (y - y_min) / (y_max - y_min) * plot_h;

    let mut body = String::new();

    // Axis triad.
    let (ox, oy) = (x_scale(axis_o.0), y_scale(axis_o.1));
    for (end, label) in [
        (axis_x, config.x_axis.title.as_deref()),
        (axis_y, config.y_axis.title.as_deref()),
        (axis_z, config.z_axis.title.as_deref()),
    ] {
        let (ex, ey) = (x_scale(end.0), y_scale(end.1));
        body.push_str(&format!(
            "<line x1=\"{ox:.2}\" y1=\"{oy:.2}\" x2=\"{ex:.2}\" y2=\"{ey:.2}\" stroke=\"#333\" stroke-width=\"1.5\"/>\n"
        ));
        if let Some(label) = label {
            body.push_str(&format!(
                "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\">{}</text>\n",
                ex,
                ey,
                xml_escape(label)
            ));
        }
    }

    const PALETTE: [&str; 6] = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    ];
    for (i, p) in projected.iter().enumerate() {
        let color = p
            .series
            .style
            .color
            .as_deref()
            .unwrap_or(PALETTE[i % PALETTE.len()]);
        match p.series.plot_type {
            PlotType3D::Scatter3D => {
                for &(px, py) in &p.pts {
                    body.push_str(&format!(
                        "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.1}\" fill=\"{color}\"/>\n",
                        x_scale(px),
                        y_scale(py),
                        p.series.style.size.unwrap_or(3.0).max(0.5),
                    ));
                }
            }
            PlotType3D::Line3D => {
                let points: String = p
                    .pts
                    .iter()
                    .map(|&(px, py)| format!("{:.2},{:.2}", x_scale(px), y_scale(py)))
                    .collect::<Vec<_>>()
                    .join(" ");
                body.push_str(&format!(
                    "<polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>\n"
                ));
            }
            PlotType3D::Surface => {
                // `.max(1)` above already guarantees `cols >= 1`, so this
                // division can never be by zero.
                let cols = p.cols.max(1);
                let rows = p.pts.len() / cols;
                for r in 0..rows {
                    for c in 0..cols {
                        let (px, py) = p.pts[r * cols + c];
                        let (sx, sy) = (x_scale(px), y_scale(py));
                        if c + 1 < cols {
                            let (px2, py2) = p.pts[r * cols + c + 1];
                            body.push_str(&format!(
                                "<line x1=\"{sx:.2}\" y1=\"{sy:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{color}\"/>\n",
                                x_scale(px2), y_scale(py2)
                            ));
                        }
                        if r + 1 < rows {
                            let (px2, py2) = p.pts[(r + 1) * cols + c];
                            body.push_str(&format!(
                                "<line x1=\"{sx:.2}\" y1=\"{sy:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{color}\"/>\n",
                                x_scale(px2), y_scale(py2)
                            ));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" \
         viewBox=\"0 0 {:.0} {:.0}\" font-family=\"sans-serif\">\n",
        width, height, width, height
    );
    svg.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{:.0}\" height=\"{:.0}\" fill=\"white\"/>\n",
        width, height
    ));
    if let Some(title) = &config.title {
        let escaped = xml_escape(title);
        svg.push_str(&format!("<title>{escaped}</title>\n"));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"20\" text-anchor=\"middle\" font-size=\"16\">{escaped}</text>\n",
            width / 2.0
        ));
    }
    svg.push_str(&body);
    svg.push_str("</svg>\n");
    svg
}

// ============================================================================
// Exporters
// ============================================================================

/// Real matplotlib `mpl_toolkits.mplot3d` script exporter.
pub(super) struct Matplotlib3DExporter;

impl Visualization3DExporter for Matplotlib3DExporter {
    fn export_3d(
        &self,
        data: &[DataSeries3D],
        config: &Plot3DConfig,
        _metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let mut script = String::from(
            "import matplotlib.pyplot as plt\n\
             import numpy as np\n\
             from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the '3d' projection)\n\n",
        );
        script.push_str(&format!(
            "fig = plt.figure(figsize=({}, {}))\nax = fig.add_subplot(111, projection='3d')\n\n",
            config.width.unwrap_or(800) as f64 / 100.0,
            config.height.unwrap_or(600) as f64 / 100.0,
        ));

        for series in data {
            match series.plot_type {
                PlotType3D::Scatter3D => {
                    script.push_str(&format!(
                        "ax.scatter({:?}, {:?}, {:?}",
                        series.x, series.y, series.z
                    ));
                    if let Some(name) = &series.name {
                        script.push_str(&format!(", label='{}'", name));
                    }
                    script.push_str(")\n");
                }
                PlotType3D::Line3D => {
                    script.push_str(&format!(
                        "ax.plot({:?}, {:?}, {:?}",
                        series.x, series.y, series.z
                    ));
                    if let Some(name) = &series.name {
                        script.push_str(&format!(", label='{}'", name));
                    }
                    script.push_str(")\n");
                }
                PlotType3D::Surface => {
                    let cols = series.x.len().max(1);
                    let z_rows: Vec<&[f64]> = series.z.chunks(cols).collect();
                    script.push_str(&format!(
                        "X, Y = np.meshgrid({:?}, {:?})\n",
                        series.x, series.y
                    ));
                    script.push_str(&format!("Z = np.array({:?})\n", z_rows));
                    script.push_str("ax.plot_surface(X, Y, Z)\n");
                }
                _ => continue,
            }
        }

        if let Some(title) = &config.title {
            script.push_str(&format!("\nax.set_title('{}')\n", title));
        }
        if let Some(l) = &config.x_axis.title {
            script.push_str(&format!("ax.set_xlabel('{}')\n", l));
        }
        if let Some(l) = &config.y_axis.title {
            script.push_str(&format!("ax.set_ylabel('{}')\n", l));
        }
        if let Some(l) = &config.z_axis.title {
            script.push_str(&format!("ax.set_zlabel('{}')\n", l));
        }
        script.push_str("ax.legend()\n");
        script.push_str("plt.tight_layout()\n");
        script.push_str("plt.show()\n");

        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(script.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }
}

/// Real gnuplot `splot` (3D plotting mode) script exporter.
pub(super) struct Gnuplot3DExporter;

impl Visualization3DExporter for Gnuplot3DExporter {
    fn export_3d(
        &self,
        data: &[DataSeries3D],
        config: &Plot3DConfig,
        _metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let mut script = String::new();
        script.push_str("set terminal png size ");
        script.push_str(&format!(
            "{},{}\n",
            config.width.unwrap_or(800),
            config.height.unwrap_or(600)
        ));
        script.push_str("set output 'plot3d.png'\n\n");
        if let Some(title) = &config.title {
            script.push_str(&format!("set title '{}'\n", title));
        }
        if let Some(l) = &config.x_axis.title {
            script.push_str(&format!("set xlabel '{}'\n", l));
        }
        if let Some(l) = &config.y_axis.title {
            script.push_str(&format!("set ylabel '{}'\n", l));
        }
        if let Some(l) = &config.z_axis.title {
            script.push_str(&format!("set zlabel '{}'\n", l));
        }
        script.push_str("set grid\n\n");

        let mut plottable: Vec<&DataSeries3D> = Vec::new();
        script.push_str("splot ");
        let mut first = true;
        for (i, series) in data.iter().enumerate() {
            let style = match series.plot_type {
                PlotType3D::Scatter3D => Some("with points"),
                PlotType3D::Line3D => Some("with lines"),
                PlotType3D::Surface => Some("with lines"),
                _ => None,
            };
            if let Some(style) = style {
                if !first {
                    script.push_str(", ");
                }
                first = false;
                script.push_str(&format!(
                    "'-' using 1:2:3 {} title '{}'",
                    style,
                    series.name.as_deref().unwrap_or(&format!("Series {}", i))
                ));
                plottable.push(series);
            }
        }
        script.push_str("\n\n");

        for series in plottable {
            match series.plot_type {
                PlotType3D::Surface => {
                    let cols = series.x.len().max(1);
                    for (r, row) in series.z.chunks(cols).enumerate() {
                        let yv = series.y.get(r).copied().unwrap_or(0.0);
                        for (c, &zv) in row.iter().enumerate() {
                            let xv = series.x.get(c).copied().unwrap_or(0.0);
                            script.push_str(&format!("{} {} {}\n", xv, yv, zv));
                        }
                        // Blank line = new scanline: required by gnuplot's
                        // `splot ... with lines` to draw a grid mesh instead
                        // of one continuous zig-zag line across rows.
                        script.push('\n');
                    }
                }
                _ => {
                    let n = series.x.len().min(series.y.len()).min(series.z.len());
                    for i in 0..n {
                        script.push_str(&format!(
                            "{} {} {}\n",
                            series.x[i], series.y[i], series.z[i]
                        ));
                    }
                }
            }
            script.push_str("e\n");
        }

        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(script.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }
}

/// Real D3.js-style 3D data exporter: an array-of-records `{x, y, z}` JSON
/// document (see module docs for why this, unlike Bokeh/Vega-Lite, is a
/// faithful "D3 3D export" rather than a fabrication).
pub(super) struct D3ThreeDExporter;

impl Visualization3DExporter for D3ThreeDExporter {
    fn export_3d(
        &self,
        data: &[DataSeries3D],
        config: &Plot3DConfig,
        _metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        fn plot_type_name(t: &PlotType3D) -> &'static str {
            match t {
                PlotType3D::Scatter3D => "scatter3d",
                PlotType3D::Surface => "surface",
                PlotType3D::Mesh3D => "mesh3d",
                PlotType3D::Line3D => "line3d",
                PlotType3D::Isosurface => "isosurface",
                PlotType3D::Volume => "volume",
            }
        }

        let series: Vec<serde_json::Value> = data
            .iter()
            .map(|s| {
                let n = s.x.len().min(s.y.len()).min(s.z.len());
                let values: Vec<serde_json::Value> = (0..n)
                    .map(|i| serde_json::json!({"x": s.x[i], "y": s.y[i], "z": s.z[i]}))
                    .collect();
                serde_json::json!({
                    "name": s.name,
                    "type": plot_type_name(&s.plot_type),
                    "values": values,
                    "style": {
                        "color": s.style.color,
                        "opacity": s.style.opacity,
                    },
                })
            })
            .collect();

        let doc = serde_json::json!({
            "format": "d3-3d",
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "series": series,
            "scene": {
                "xaxis": {"title": config.x_axis.title},
                "yaxis": {"title": config.y_axis.title},
                "zaxis": {"title": config.z_axis.title},
                "camera": {
                    "eye": config.camera.eye,
                    "center": config.camera.center,
                    "up": config.camera.up,
                },
            },
        });

        let json_str = serde_json::to_string_pretty(&doc)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(json_str.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }
}

/// SVG exporter: renders a real orthographic projection of the 3D
/// primitives (see [`render_svg_3d`]).
pub(super) struct Svg3DExporter;

impl Visualization3DExporter for Svg3DExporter {
    fn export_3d(
        &self,
        data: &[DataSeries3D],
        config: &Plot3DConfig,
        _metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let svg = render_svg_3d(data, config);
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(svg.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }
}

/// Self-contained HTML exporter: embeds the same projected SVG rendering
/// `Svg3DExporter` produces directly in the page body.
pub(super) struct Html3DExporter;

impl Visualization3DExporter for Html3DExporter {
    fn export_3d(
        &self,
        data: &[DataSeries3D],
        config: &Plot3DConfig,
        _metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let svg = render_svg_3d(data, config);
        let title = xml_escape(config.title.as_deref().unwrap_or("3D Visualization"));
        let html = format!(
            "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>{title}</title>\n\
             </head>\n<body>\n{svg}</body>\n</html>\n"
        );
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(html.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }
}

/// Vega-Lite has no 3D/z encoding channel anywhere in its grammar (its
/// channel set is x, y, x2, y2, color, size, shape, theta, radius,
/// latitude, longitude -- never z/depth), so a "3D Vega-Lite spec" cannot be
/// produced without inventing a channel the format doesn't have. Returns an
/// honest [`IoError::UnsupportedFormat`] rather than silently mislabeling
/// Plotly (or any other format's) output as Vega-Lite.
pub(super) struct VegaLite3DUnsupported;

impl Visualization3DExporter for VegaLite3DUnsupported {
    fn export_3d(
        &self,
        _data: &[DataSeries3D],
        _config: &Plot3DConfig,
        _metadata: &Metadata,
        _path: &Path,
    ) -> Result<()> {
        Err(IoError::UnsupportedFormat(
            "Vega-Lite 3D export: Vega-Lite's grammar has no z/depth encoding channel, so 3D \
             data cannot be represented as a Vega-Lite spec. Use PlotlyJson, MatplotlibPython, \
             Gnuplot, D3Json, Svg, or Html for 3D visualization export instead."
                .to_string(),
        ))
    }
}

/// Bokeh's model/glyph system has no native 3D scene, camera, or z-axis
/// concept (there is no `Scatter3D`/`Surface3D` model), so a "3D Bokeh
/// document" cannot be produced without inventing model types the library
/// doesn't have. Returns an honest [`IoError::UnsupportedFormat`] rather
/// than silently mislabeling Plotly (or any other format's) output as Bokeh.
pub(super) struct Bokeh3DUnsupported;

impl Visualization3DExporter for Bokeh3DUnsupported {
    fn export_3d(
        &self,
        _data: &[DataSeries3D],
        _config: &Plot3DConfig,
        _metadata: &Metadata,
        _path: &Path,
    ) -> Result<()> {
        Err(IoError::UnsupportedFormat(
            "Bokeh 3D export: Bokeh's model/glyph system has no native 3D scene or z-axis \
             concept, so 3D data cannot be represented as a genuine Bokeh document. Use \
             PlotlyJson, MatplotlibPython, Gnuplot, D3Json, Svg, or Html for 3D visualization \
             export instead."
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualization::SeriesStyle;

    fn sample_data() -> Vec<DataSeries3D> {
        vec![DataSeries3D {
            name: Some("cloud".to_string()),
            x: vec![1.0, -2.5, 3.5, 0.25],
            y: vec![2.5, 1.0, -1.5, 4.0],
            z: vec![-3.0, 4.5, 0.5, -1.25],
            plot_type: PlotType3D::Scatter3D,
            style: SeriesStyle::default(),
        }]
    }

    fn surface_data() -> Vec<DataSeries3D> {
        vec![DataSeries3D {
            name: Some("terrain".to_string()),
            x: vec![0.0, 1.0, 2.0],
            y: vec![0.0, 1.0],
            // 2 rows x 3 cols, row-major, non-constant.
            z: vec![0.5, 1.5, -0.5, 2.5, -1.0, 3.0],
            plot_type: PlotType3D::Surface,
            style: SeriesStyle::default(),
        }]
    }

    fn camera(eye: [f64; 3]) -> Plot3DConfig {
        Plot3DConfig {
            camera: CameraConfig {
                eye,
                center: [0.0, 0.0, 0.0],
                up: [0.0, 0.0, 1.0],
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_projector_orthonormal_basis() {
        let p = Projector::new(&CameraConfig::default());
        assert!((dot(p.side, p.side) - 1.0).abs() < 1e-9);
        assert!((dot(p.up, p.up) - 1.0).abs() < 1e-9);
        assert!(
            dot(p.side, p.up).abs() < 1e-9,
            "side and up must be perpendicular"
        );
    }

    #[test]
    fn test_different_cameras_yield_different_projections() {
        // Regression test for the original bug's spirit applied to the new
        // renderer: the camera must genuinely participate in the output, not
        // just be accepted and ignored.
        let data = sample_data();
        let svg_a = render_svg_3d(&data, &camera([1.25, 1.25, 1.25]));
        let svg_b = render_svg_3d(&data, &camera([0.0, 0.0, 4.0]));
        assert_ne!(
            svg_a, svg_b,
            "changing the camera eye must change the rendered projection"
        );
    }

    #[test]
    fn test_render_svg_3d_scatter_is_well_formed() {
        let svg = render_svg_3d(&sample_data(), &Plot3DConfig::default());
        assert!(svg.starts_with("<svg"));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert!(svg.contains("<circle"), "scatter3d must render circles");
        // 4 points in sample_data.
        assert_eq!(svg.matches("<circle").count(), 4);
    }

    #[test]
    fn test_render_svg_3d_surface_wireframe() {
        let svg = render_svg_3d(&surface_data(), &Plot3DConfig::default());
        assert!(svg.starts_with("<svg"));
        // 2 rows x 3 cols: 2 horizontal segments/row * 2 rows + 1 vertical *
        // 3 cols = 4 + 3 = 7 mesh line segments, plus the 3 axis lines = 10.
        assert_eq!(svg.matches("<line").count(), 10);
    }

    #[test]
    fn test_matplotlib_3d_script_shape() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("test_3d_mpl_unit_{}.py", std::process::id()));
        Matplotlib3DExporter
            .export_3d(
                &sample_data(),
                &Plot3DConfig::default(),
                &Metadata::new(),
                &path,
            )
            .expect("Operation failed");
        let script = std::fs::read_to_string(&path).expect("Operation failed");
        assert!(script.contains("projection='3d'"));
        assert!(script.contains("ax.scatter"));
        assert!(
            script.contains("-2.5"),
            "actual data values must appear, not fabricated ones"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_matplotlib_3d_surface_script() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("test_3d_mpl_surface_{}.py", std::process::id()));
        Matplotlib3DExporter
            .export_3d(
                &surface_data(),
                &Plot3DConfig::default(),
                &Metadata::new(),
                &path,
            )
            .expect("Operation failed");
        let script = std::fs::read_to_string(&path).expect("Operation failed");
        assert!(script.contains("plot_surface"));
        assert!(script.contains("meshgrid"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_gnuplot_3d_script_shape() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("test_3d_gnuplot_unit_{}.gp", std::process::id()));
        Gnuplot3DExporter
            .export_3d(
                &sample_data(),
                &Plot3DConfig::default(),
                &Metadata::new(),
                &path,
            )
            .expect("Operation failed");
        let script = std::fs::read_to_string(&path).expect("Operation failed");
        assert!(script.contains("splot"));
        assert!(script.contains("using 1:2:3"));
        assert!(script.contains("-2.5"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_d3_3d_json_shape() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("test_3d_d3_unit_{}.json", std::process::id()));
        D3ThreeDExporter
            .export_3d(
                &sample_data(),
                &Plot3DConfig::default(),
                &Metadata::new(),
                &path,
            )
            .expect("Operation failed");
        let json = std::fs::read_to_string(&path).expect("Operation failed");
        let value: serde_json::Value = serde_json::from_str(&json).expect("Operation failed");
        assert_eq!(value["format"], "d3-3d");
        assert!(
            value.get("data").is_none(),
            "must not reuse Plotly's 'data' key"
        );
        let values = value["series"][0]["values"]
            .as_array()
            .expect("Operation failed");
        assert_eq!(values.len(), 4);
        assert_eq!(values[1]["x"], -2.5);
        assert_eq!(values[1]["z"], 4.5);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_vegalite_3d_is_honest_unsupported_error() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("test_3d_vega_unit_{}.json", std::process::id()));
        let result = VegaLite3DUnsupported.export_3d(
            &sample_data(),
            &Plot3DConfig::default(),
            &Metadata::new(),
            &path,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(IoError::UnsupportedFormat(_))));
        assert!(
            !path.exists(),
            "must not silently create a mislabeled file for an unsupported format"
        );
    }

    #[test]
    fn test_bokeh_3d_is_honest_unsupported_error() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("test_3d_bokeh_unit_{}.json", std::process::id()));
        let result = Bokeh3DUnsupported.export_3d(
            &sample_data(),
            &Plot3DConfig::default(),
            &Metadata::new(),
            &path,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(IoError::UnsupportedFormat(_))));
        assert!(!path.exists());
    }
}
