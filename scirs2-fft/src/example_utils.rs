//! Utility functions for FFT module examples

use ndarray::ArrayD;
use num_complex::Complex;

/// Print the first few values of a dynamic array
pub fn print_first_values(arr: &ArrayD<Complex<f64>>, count: usize) {
    let flat_view = arr.view().into_shape(arr.len()).unwrap();
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