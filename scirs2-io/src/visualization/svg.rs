//! Native SVG rendering and the SVG/self-contained-HTML exporters.
//!
//! Split out of `visualization.rs` to keep that file under the workspace's
//! 2000-line-per-file limit; this module has no dependents outside
//! `super` (the `visualization` module), which wires [`SvgExporter`] and
//! [`HtmlExporter`] into `get_exporter`.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::error::{IoError, Result};
use crate::metadata::Metadata;

use super::{DataSeries, PlotConfig, PlotType, VisualizationExporter};

/// Escapes text for safe inclusion in XML/SVG attribute values and content.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Wraps `body` (already-positioned SVG element markup) in a complete,
/// self-contained `<svg>` document with a white background and an optional
/// title, sized to `width`x`height`.
fn svg_document(width: f64, height: f64, title: Option<&str>, body: &str) -> String {
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" \
         viewBox=\"0 0 {:.0} {:.0}\" font-family=\"sans-serif\">\n",
        width, height, width, height
    );
    svg.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{:.0}\" height=\"{:.0}\" fill=\"white\"/>\n",
        width, height
    ));
    if let Some(t) = title {
        let escaped = xml_escape(t);
        svg.push_str(&format!("<title>{escaped}</title>\n"));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"20\" text-anchor=\"middle\" font-size=\"16\">{escaped}</text>\n",
            width / 2.0
        ));
    }
    svg.push_str(body);
    svg.push_str("</svg>\n");
    svg
}

/// Computes equal-width histogram bins (bin centers and counts) over raw
/// sample `values`, mirroring the client-side auto-binning that Plotly's own
/// histogram trace performs -- needed here because SVG has no client-side
/// logic to do it for us.
fn histogram_bins(values: &[f64], num_bins: usize) -> (Vec<f64>, Vec<f64>) {
    if values.is_empty() || num_bins == 0 {
        return (Vec::new(), Vec::new());
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if max > min { max - min } else { 1.0 };
    let bin_width = range / num_bins as f64;

    let mut counts = vec![0.0_f64; num_bins];
    for &v in values {
        let idx = (((v - min) / bin_width) as usize).min(num_bins - 1);
        counts[idx] += 1.0;
    }
    let centers: Vec<f64> = (0..num_bins)
        .map(|i| min + bin_width * (i as f64 + 0.5))
        .collect();
    (centers, counts)
}

/// A simple blue -> green -> red colour ramp for heatmap cells (`t` in `[0, 1]`).
fn heatmap_color(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let r = (t * 255.0).round() as u8;
    let g = ((1.0 - (t - 0.5).abs() * 2.0).clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = ((1.0 - t) * 255.0).round() as u8;
    (r, g, b)
}

/// Renders a `Heatmap` series as a grid of coloured `<rect>` cells filling the
/// plot area, using the same `(cols, rows, z)` encoding `add_heatmap` stores
/// the grid in.
fn render_svg_heatmap(
    series: &DataSeries,
    config: &PlotConfig,
    width: f64,
    height: f64,
    margin: f64,
    plot_w: f64,
    plot_h: f64,
) -> String {
    let cols = series
        .x
        .as_ref()
        .and_then(|x| x.first())
        .copied()
        .unwrap_or(1.0)
        .max(1.0) as usize;
    let rows = series.y.first().copied().unwrap_or(1.0).max(1.0) as usize;
    let z = series.z.clone().unwrap_or_default();
    let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
    let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z_range = if z_max > z_min { z_max - z_min } else { 1.0 };

    let cell_w = plot_w / cols as f64;
    let cell_h = plot_h / rows as f64;

    let mut body = String::new();
    for r in 0..rows {
        for c in 0..cols {
            let value = z.get(r * cols + c).copied().unwrap_or(z_min);
            let t = (value - z_min) / z_range;
            let (red, green, blue) = heatmap_color(t);
            let x = margin + c as f64 * cell_w;
            let y = margin + r as f64 * cell_h;
            body.push_str(&format!(
                "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"rgb({red},{green},{blue})\"/>\n",
                x,
                y,
                cell_w.max(0.5),
                cell_h.max(0.5)
            ));
        }
    }
    svg_document(width, height, config.title.as_deref(), &body)
}

/// One series' worth of data reduced to plottable `(x, y)` points (or, for
/// histograms, bin centers/counts) in the shared linear axis space.
struct PreparedSeries<'a> {
    series: &'a DataSeries,
    xs: Vec<f64>,
    ys: Vec<f64>,
}

/// Renders `data` under `config` as a complete, self-contained SVG document.
///
/// Supports every plot type producible via `VisualizationBuilder`'s `add_*`
/// methods (line, scatter, histogram, heatmap) plus directly-constructed
/// `Bar` series; any other `PlotType` is skipped for primitive rendering,
/// consistent with how the other exporters in this module handle plot types
/// they don't natively support (e.g. `PlotlyExporter`'s `_ => continue`).
fn render_svg(data: &[DataSeries], config: &PlotConfig) -> String {
    let width = config.width.unwrap_or(800) as f64;
    let height = config.height.unwrap_or(600) as f64;
    let margin = 50.0_f64;
    let plot_w = (width - 2.0 * margin).max(1.0);
    let plot_h = (height - 2.0 * margin).max(1.0);

    // Heatmaps use their own (cols, rows, z-grid) encoding rather than the
    // shared linear x/y axis space the other plot types use, so render the
    // first one found as a standalone grid document.
    if let Some(heatmap) = data
        .iter()
        .find(|s| matches!(s.plot_type, PlotType::Heatmap))
    {
        return render_svg_heatmap(heatmap, config, width, height, margin, plot_w, plot_h);
    }

    let mut prepared: Vec<PreparedSeries> = Vec::new();
    for series in data {
        match series.plot_type {
            PlotType::Line | PlotType::Scatter | PlotType::Bar => {
                let xs = series
                    .x
                    .clone()
                    .unwrap_or_else(|| (0..series.y.len()).map(|i| i as f64).collect());
                prepared.push(PreparedSeries {
                    series,
                    xs,
                    ys: series.y.clone(),
                });
            }
            PlotType::Histogram => {
                let (edges, counts) = histogram_bins(&series.y, 10);
                prepared.push(PreparedSeries {
                    series,
                    xs: edges,
                    ys: counts,
                });
            }
            _ => {}
        }
    }

    if prepared.is_empty() {
        return svg_document(width, height, config.title.as_deref(), "");
    }

    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for p in &prepared {
        for &v in &p.xs {
            x_min = x_min.min(v);
            x_max = x_max.max(v);
        }
        for &v in &p.ys {
            y_min = y_min.min(v);
            y_max = y_max.max(v);
        }
    }
    if !x_max.is_finite() || !x_min.is_finite() || x_max <= x_min {
        x_min = if x_min.is_finite() { x_min } else { 0.0 };
        x_max = x_min + 1.0;
    }
    // Bars/histograms should always show their zero baseline.
    if prepared
        .iter()
        .any(|p| matches!(p.series.plot_type, PlotType::Bar | PlotType::Histogram))
    {
        y_min = y_min.min(0.0);
        y_max = y_max.max(0.0);
    }
    if !y_max.is_finite() || !y_min.is_finite() || y_max <= y_min {
        y_min = if y_min.is_finite() { y_min } else { 0.0 };
        y_max = y_min + 1.0;
    }

    let x_scale = |x: f64| margin + (x - x_min) / (x_max - x_min) * plot_w;
    let y_scale = |y: f64| margin + plot_h - (y - y_min) / (y_max - y_min) * plot_h;

    const PALETTE: [&str; 6] = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    ];
    let mut body = String::new();

    // Axes (simple L-shaped border) plus min/max tick labels.
    body.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#333\"/>\n",
        margin,
        margin + plot_h,
        margin + plot_w,
        margin + plot_h
    ));
    body.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"#333\"/>\n",
        margin,
        margin,
        margin,
        margin + plot_h
    ));
    body.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\" text-anchor=\"middle\">{:.3}</text>\n",
        margin,
        margin + plot_h + 15.0,
        x_min
    ));
    body.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\" text-anchor=\"middle\">{:.3}</text>\n",
        margin + plot_w,
        margin + plot_h + 15.0,
        x_max
    ));
    body.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\" text-anchor=\"end\">{:.3}</text>\n",
        margin - 5.0,
        margin + plot_h,
        y_min
    ));
    body.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\" text-anchor=\"end\">{:.3}</text>\n",
        margin - 5.0,
        margin + 5.0,
        y_max
    ));
    if let Some(title) = &config.x_axis.title {
        body.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"12\" text-anchor=\"middle\">{}</text>\n",
            margin + plot_w / 2.0,
            height - 5.0,
            xml_escape(title)
        ));
    }
    if let Some(title) = &config.y_axis.title {
        body.push_str(&format!(
            "<text x=\"12\" y=\"{:.2}\" font-size=\"12\" text-anchor=\"middle\" \
             transform=\"rotate(-90 12 {:.2})\">{}</text>\n",
            margin + plot_h / 2.0,
            margin + plot_h / 2.0,
            xml_escape(title)
        ));
    }

    for (i, p) in prepared.iter().enumerate() {
        let color = p
            .series
            .style
            .color
            .as_deref()
            .unwrap_or(PALETTE[i % PALETTE.len()]);
        match p.series.plot_type {
            PlotType::Line => {
                let points: String =
                    p.xs.iter()
                        .zip(p.ys.iter())
                        .map(|(&x, &y)| format!("{:.2},{:.2}", x_scale(x), y_scale(y)))
                        .collect::<Vec<_>>()
                        .join(" ");
                body.push_str(&format!(
                    "<polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>\n"
                ));
            }
            PlotType::Scatter => {
                for (&x, &y) in p.xs.iter().zip(p.ys.iter()) {
                    body.push_str(&format!(
                        "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"3\" fill=\"{color}\"/>\n",
                        x_scale(x),
                        y_scale(y)
                    ));
                }
            }
            PlotType::Bar | PlotType::Histogram => {
                let bar_w = (plot_w / p.xs.len().max(1) as f64 * 0.8).max(1.0);
                let y0 = y_scale(0.0);
                for (&x, &y) in p.xs.iter().zip(p.ys.iter()) {
                    let px = x_scale(x) - bar_w / 2.0;
                    let yy = y_scale(y);
                    let (rect_y, rect_h) = if yy <= y0 {
                        (yy, y0 - yy)
                    } else {
                        (y0, yy - y0)
                    };
                    body.push_str(&format!(
                        "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{color}\" opacity=\"0.8\"/>\n",
                        px, rect_y, bar_w, rect_h.max(0.0)
                    ));
                }
            }
            _ => {}
        }
    }

    svg_document(width, height, config.title.as_deref(), &body)
}

/// SVG exporter: renders the plot's primitives (lines, points, bars, binned
/// histograms, heatmap grids) directly as scaled SVG shapes.
pub(super) struct SvgExporter;

impl VisualizationExporter for SvgExporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let svg = self.to_string(data, config, metadata)?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(svg.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        _metadata: &Metadata,
    ) -> Result<String> {
        Ok(render_svg(data, config))
    }
}

/// Self-contained HTML exporter: embeds the same SVG rendering `SvgExporter`
/// produces directly in the page body, so the output needs no external
/// scripts, network access, or CDN to display (unlike a plotting library
/// that requires loading e.g. `plotly.js`/`d3.js` from a CDN).
pub(super) struct HtmlExporter;

impl VisualizationExporter for HtmlExporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let html = self.to_string(data, config, metadata)?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(html.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        _metadata: &Metadata,
    ) -> Result<String> {
        let svg = render_svg(data, config);
        let title = xml_escape(config.title.as_deref().unwrap_or("Visualization"));
        Ok(format!(
            "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>{title}</title>\n\
             </head>\n<body>\n{svg}</body>\n</html>\n"
        ))
    }
}
