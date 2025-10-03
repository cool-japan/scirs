// Continuous wavelet transform implementation details

use scirs2_core::numeric::Complex64;

/// Helper function to convolve real signal with complex filter using 'same' mode
///
/// Optimized for the CWT case with real input data
#[allow(unused_imports)]
#[allow(dead_code)]
pub fn convolve_complex_same_real(x: &[Complex64], h: &[Complex64]) -> Vec<Complex64> {
    let nx = x.len();
    let nh = h.len();
    let n_out = nx;

    // Allocate output buffer
    let mut out = vec![Complex64::new(0.0, 0.0); nx + nh - 1];

    // Perform convolution - since h is typically much smaller than x in the CWT case,
    // we optimize by iterating through h in the outer loop for better cache locality
    if nh < nx {
        for j in 0..nh {
            for i in 0..nx {
                out[i + j] += x[i] * h[j];
            }
        }
    } else {
        // Fall back to standard convolution when h is larger
        for i in 0..nx {
            for j in 0..nh {
                out[i + j] += x[i] * h[j];
            }
        }
    }

    // Extract the middle part ('same' mode)
    let start = (nh - 1) / 2;
    out.iter().skip(start).take(n_out).copied().collect()
}

/// Helper function to convolve complex signal with complex filter using 'same' mode
///
/// Handles fully complex CWT computation
#[allow(dead_code)]
pub fn convolve_complex_same_complex(x: &[Complex64], h: &[Complex64]) -> Vec<Complex64> {
    let nx = x.len();
    let nh = h.len();
    let n_out = nx;

    // Allocate output buffer
    let mut out = vec![Complex64::new(0.0, 0.0); nx + nh - 1];

    // Perform convolution - since h is typically much smaller than x in the CWT case,
    // we optimize by iterating through h in the outer loop for better cache locality
    if nh < nx {
        for j in 0..nh {
            for i in 0..nx {
                out[i + j] += x[i] * h[j];
            }
        }
    } else {
        // Fall back to standard convolution when h is larger
        for i in 0..nx {
            for j in 0..nh {
                out[i + j] += x[i] * h[j];
            }
        }
    }

    // Extract the middle part ('same' mode)
    let start = (nh - 1) / 2;
    out.iter().skip(start).take(n_out).copied().collect()
}
