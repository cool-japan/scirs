// Local Binary Pattern (LBP) feature extraction for images

use crate::error::SignalResult;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

#[allow(unused_imports)]
/// Extract Local Binary Pattern features from an image
#[allow(dead_code)]
pub fn extract_lbp_features(
    image: &Array2<f64>,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    let shape = image.shape();
    let height = shape[0];
    let width = shape[1];

    if height < 3 || width < 3 {
        // Not enough pixels for LBP calculation
        return Ok(());
    }

    // Calculate basic LBP
    let mut lbp_hist = vec![0; 256]; // 8-bit LBP histogram

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            let center = image[[i, j]];
            let mut code = 0u8;

            // Compare each neighbor with the center and set bits accordingly
            if image[[i - 1, j - 1]] >= center {
                code |= 0x01;
            }
            if image[[i - 1, j]] >= center {
                code |= 0x02;
            }
            if image[[i - 1, j + 1]] >= center {
                code |= 0x04;
            }
            if image[[i, j + 1]] >= center {
                code |= 0x08;
            }
            if image[[i + 1, j + 1]] >= center {
                code |= 0x10;
            }
            if image[[i + 1, j]] >= center {
                code |= 0x20;
            }
            if image[[i + 1, j - 1]] >= center {
                code |= 0x40;
            }
            if image[[i, j - 1]] >= center {
                code |= 0x80;
            }

            lbp_hist[code as usize] += 1;
        }
    }

    // Normalize histogram
    let total = ((height - 2) * (width - 2)) as f64;
    let lbp_hist_norm: Vec<f64> = lbp_hist.iter().map(|&count| count as f64 / total).collect();

    // Calculate uniformity (number of 0/1 transitions in binary representation)
    let mut uniformity = vec![0; 256];
    uniformity
        .iter_mut()
        .enumerate()
        .for_each(|(i, transitions)| {
            let mut count = 0;
            let mut prev_bit = i & 0x01;

            // Count transitions by shifting bits and comparing
            for bit_pos in 0..8 {
                let current_bit = (i >> bit_pos) & 0x01;
                if current_bit != prev_bit {
                    count += 1;
                }
                prev_bit = current_bit;
            }

            // Check last transition from bit 7 back to bit 0
            if ((i >> 7) & 0x01) != (i & 0x01) {
                count += 1;
            }

            *transitions = count;
        });

    // Count uniform patterns (0 or 2 transitions)
    let uniform_count = lbp_hist
        .iter()
        .enumerate()
        .filter(|&(i, _)| uniformity[i] <= 2)
        .map(|(_, &count)| count)
        .sum::<usize>();

    features.insert(
        "lbp_uniform_ratio".to_string(),
        uniform_count as f64 / total,
    );

    // Store histogram features
    let mut energy = 0.0;
    let mut entropy = 0.0;

    for &p in &lbp_hist_norm {
        if p > 0.0 {
            energy += p * p;
            entropy -= p * p.ln();
        }
    }

    features.insert("lbp_energy".to_string(), energy);
    features.insert("lbp_entropy".to_string(), entropy);

    // Group patterns into four categories:
    // - Spots (uniform patterns with 0 1-bits)
    // - Flat areas (uniform patterns with 8 1-bits)
    // - Edges (uniform patterns with 2,4,6 1-bits)
    // - Corners (uniform patterns with 1,3,5,7 1-bits)

    let mut spots = 0;
    let mut flat = 0;
    let mut edges = 0;
    let mut corners = 0;

    for i in 0..256 {
        if uniformity[i] <= 2 {
            // Count number of 1 bits
            let mut bit_count = 0;
            let mut val = i;
            while val > 0 {
                bit_count += val & 1;
                val >>= 1;
            }

            match bit_count {
                0 => spots += lbp_hist[i],
                8 => flat += lbp_hist[i],
                2 | 4 | 6 => edges += lbp_hist[i],
                1 | 3 | 5 | 7 => corners += lbp_hist[i],
                _ => {} // Shouldn't happen for 8-bit patterns
            }
        }
    }

    features.insert("lbp_spots".to_string(), spots as f64 / total);
    features.insert("lbp_flat".to_string(), flat as f64 / total);
    features.insert("lbp_edges".to_string(), edges as f64 / total);
    features.insert("lbp_corners".to_string(), corners as f64 / total);

    Ok(())
}
