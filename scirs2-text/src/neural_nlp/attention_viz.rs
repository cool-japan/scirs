//! Attention visualization utilities.
//!
//! Extracts and formats per-head attention weights produced by
//! [`super::TransformerTextEncoder::encode_with_attention`] for inspection.

use scirs2_core::ndarray::{Array2, Array3};

// ─── Types ─────────────────────────────────────────────────────────────────────

/// A single layer×head attention heatmap.
#[derive(Debug, Clone)]
pub struct AttentionHeatmap {
    /// Index of the encoder layer.
    pub layer_idx: usize,
    /// Index of the attention head.
    pub head_idx: usize,
    /// Token strings corresponding to rows/columns.
    pub tokens: Vec<String>,
    /// Attention weight matrix `[seq_len, seq_len]`.
    pub weights: Array2<f32>,
}

/// Full attention visualization across all layers and heads.
#[derive(Debug, Clone)]
pub struct AttentionVisualization {
    /// Number of encoder layers.
    pub n_layers: usize,
    /// Number of attention heads per layer.
    pub n_heads: usize,
    /// Token strings.
    pub token_strs: Vec<String>,
    /// All `n_layers × n_heads` individual heatmaps.
    pub heatmaps: Vec<AttentionHeatmap>,
    /// Attention weights averaged across all layers and all heads `[seq, seq]`.
    pub mean_attention: Array2<f32>,
}

// ─── impl ─────────────────────────────────────────────────────────────────────

impl AttentionVisualization {
    /// Build a visualization from raw attention weight tensors.
    ///
    /// `attention_weights[layer]` has shape `[n_heads, seq, seq]`.
    pub fn from_attention_weights(tokens: Vec<String>, attention_weights: &[Array3<f32>]) -> Self {
        let n_layers = attention_weights.len();
        let seq = tokens.len();

        if n_layers == 0 || seq == 0 {
            return Self {
                n_layers: 0,
                n_heads: 0,
                token_strs: tokens,
                heatmaps: Vec::new(),
                mean_attention: Array2::zeros((0, 0)),
            };
        }

        let n_heads = attention_weights[0].shape()[0];
        let mut heatmaps = Vec::with_capacity(n_layers * n_heads);
        let mut mean_acc = Array2::zeros((seq, seq));
        let total = (n_layers * n_heads) as f32;

        for (layer_idx, layer_attn) in attention_weights.iter().enumerate() {
            for head_idx in 0..n_heads {
                // Slice [seq, seq] for this head
                let weights = layer_attn
                    .index_axis(scirs2_core::ndarray::Axis(0), head_idx)
                    .to_owned(); // Array2<f32>

                mean_acc += &weights;

                heatmaps.push(AttentionHeatmap {
                    layer_idx,
                    head_idx,
                    tokens: tokens.clone(),
                    weights,
                });
            }
        }

        mean_acc.mapv_inplace(|v| v / total);

        Self {
            n_layers,
            n_heads,
            token_strs: tokens,
            heatmaps,
            mean_attention: mean_acc,
        }
    }

    /// Return the top-`k` tokens by mean attention received (column sums).
    ///
    /// Returns `(token_string, mean_weight)` pairs sorted descending by weight.
    pub fn top_attended_tokens(&self, k: usize) -> Vec<(String, f32)> {
        let seq = self.token_strs.len();
        if seq == 0 || k == 0 {
            return Vec::new();
        }

        // Column sum of mean_attention: how much each token is attended to
        let col_sums: Vec<f32> = (0..seq)
            .map(|j| self.mean_attention.column(j).sum())
            .collect();

        let mut indexed: Vec<(usize, f32)> = col_sums.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let take = k.min(seq);
        indexed
            .into_iter()
            .take(take)
            .map(|(i, w)| (self.token_strs[i].clone(), w))
            .collect()
    }

    /// Export the specified head's attention as an ASCII heatmap.
    ///
    /// Uses `░▒▓█` blocks scaled to the [0, 1] range.
    pub fn to_ascii_heatmap(&self, layer: usize, head: usize) -> String {
        let heatmap = self
            .heatmaps
            .iter()
            .find(|h| h.layer_idx == layer && h.head_idx == head);

        let Some(hm) = heatmap else {
            return format!("No heatmap for layer={layer}, head={head}");
        };

        let seq = hm.tokens.len();
        // Token column header
        let max_tok_len = hm.tokens.iter().map(|t| t.len()).max().unwrap_or(4);
        let cell_w = max_tok_len + 1;

        let mut out = String::new();

        // Header row
        out.push_str(&" ".repeat(cell_w));
        for tok in &hm.tokens {
            let padded = format!("{tok:>width$}", width = cell_w);
            out.push_str(&padded);
        }
        out.push('\n');

        for i in 0..seq {
            // Row label
            let label = format!("{:>width$}", hm.tokens[i], width = cell_w);
            out.push_str(&label);
            for j in 0..seq {
                let w = hm.weights[[i, j]].clamp(0.0, 1.0);
                let ch = if w < 0.25 {
                    '░'
                } else if w < 0.5 {
                    '▒'
                } else if w < 0.75 {
                    '▓'
                } else {
                    '█'
                };
                let cell = format!("{:>width$}", ch, width = cell_w);
                out.push_str(&cell);
            }
            out.push('\n');
        }

        out
    }

    /// Flatten all heatmap weights into a single `f32` vector.
    ///
    /// Order: `[layer0_head0_weights..., layer0_head1_weights..., layer1_head0_weights..., ...]`.
    pub fn to_flat_vec(&self) -> Vec<f32> {
        let mut v = Vec::new();
        for hm in &self.heatmaps {
            v.extend(hm.weights.iter().copied());
        }
        v
    }
}
