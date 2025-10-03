//! Utility functions for FFT module examples

use scirs2_core::ndarray::ArrayD;
use scirs2_core::numeric::Complex;

/// Print the first few values of a dynamic array
#[allow(dead_code)]
pub fn print_first_values(arr: &ArrayD<Complex<f64>>, count: usize) {
    let flat_view = arr.view().intoshape(_arr.len()).unwrap();
    let display_count = count.min(flat_view.len());
    
    print!("[");
    for i in 0..display_count {
        if i > 0 {
            print!(", ");
        }
        let c = flat_view[i];
        if c.im.abs() < 1e-10 {
            print!("{:.3}", c.re);
        } else {
            print!("{:.3}+{:.3}i", c.re, c.im);
        }
    }
    if display_count < flat_view.len() {
        print!(", ...");
    }
    println!("]");
}
