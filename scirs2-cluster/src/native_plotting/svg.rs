//! SVG rendering primitives — canvas, elements, animation engine, and interactive controller.

use super::types::{AnimationEngine, InteractiveController, SvgCanvas, SvgElement};
use std::collections::HashMap;

impl SvgCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            elements: Vec::new(),
            styles: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.elements.clear();
    }

    pub fn add_element(&mut self, element: SvgElement) {
        self.elements.push(element);
    }

    pub fn to_svg(&self) -> String {
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">"#,
            self.width, self.height
        );

        // Add styles
        svg.push_str("<defs><style>");
        for (selector, style) in &self.styles {
            svg.push_str(&format!("{} {{ {} }}", selector, style));
        }
        svg.push_str("</style></defs>");

        // Add main group
        svg.push_str(r#"<g class="main-group">"#);

        // Add elements
        for element in &self.elements {
            svg.push_str(&element.to_svg());
        }

        svg.push_str("</g></svg>");
        svg
    }
}

impl SvgElement {
    pub fn to_svg(&self) -> String {
        match self {
            SvgElement::Circle {
                cx,
                cy,
                r,
                fill,
                stroke,
                stroke_width,
                opacity,
            } => {
                format!(
                    r#"<circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="{}" opacity="{}" />"#,
                    cx, cy, r, fill, stroke, stroke_width, opacity
                )
            }
            SvgElement::Line {
                x1,
                y1,
                x2,
                y2,
                stroke,
                stroke_width,
                opacity,
            } => {
                format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" opacity="{}" />"#,
                    x1, y1, x2, y2, stroke, stroke_width, opacity
                )
            }
            SvgElement::Path {
                d,
                fill,
                stroke,
                stroke_width,
                opacity,
            } => {
                format!(
                    r#"<path d="{}" fill="{}" stroke="{}" stroke-width="{}" opacity="{}" />"#,
                    d, fill, stroke, stroke_width, opacity
                )
            }
            SvgElement::Text {
                x,
                y,
                content,
                font_size,
                fill,
                text_anchor,
            } => {
                format!(
                    r#"<text x="{}" y="{}" font-size="{}" fill="{}" text-anchor="{}">{}</text>"#,
                    x, y, font_size, fill, text_anchor, content
                )
            }
            SvgElement::Group {
                id,
                elements,
                transform,
            } => {
                let mut group = format!(r#"<g id="{}" transform="{}">"#, id, transform);
                for element in elements {
                    group.push_str(&element.to_svg());
                }
                group.push_str("</g>");
                group
            }
        }
    }
}

impl AnimationEngine {
    pub fn new(fps: f64) -> Self {
        Self {
            frames: Vec::new(),
            current_frame: 0,
            frame_duration: 1000.0 / fps,
            total_duration: 0.0,
        }
    }
}

impl InteractiveController {
    pub fn new() -> Self {
        Self {
            zoom_level: 1.0,
            pan_offset: (0.0, 0.0),
            selected_elements: Vec::new(),
            hover_element: None,
        }
    }
}
