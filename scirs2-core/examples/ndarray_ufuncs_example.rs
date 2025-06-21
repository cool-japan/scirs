// Example demonstrating usage of ndarray extensions and universal functions

use ndarray::array;
use scirs2_core::ndarray_ext::{mask_select, reshape_2d, stack_2d, transpose_2d};
use scirs2_core::ufuncs::{binary2d, math2d, reduction};

fn main() {
    println!("SciRS2-Core ndarray_ext and ufuncs Example");
    println!("==========================================\n");

    // Create some example arrays
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
    let c = array![10.0, 20.0, 30.0];
    let mask = array![[true, false, true], [false, true, false]];

    // Demonstrate ndarray_ext functions
    println!("Original array a:");
    println!("{:?}\n", a);

    // Reshape
    let reshaped = reshape_2d(a.view(), (6, 1)).unwrap();
    println!("Reshaped a (6x1):");
    println!("{:?}\n", reshaped);

    // Stack
    let stacked = stack_2d(&[a.view(), b.view()], 0).unwrap();
    println!("Stack a and b along rows:");
    println!("{:?}\n", stacked);

    // Transpose
    let transposed = transpose_2d(a.view());
    println!("Transpose of a:");
    println!("{:?}\n", transposed);

    // Mask select
    let masked = mask_select(a.view(), mask.view()).unwrap();
    println!("Mask select from a:");
    println!("{:?}\n", masked);

    // Demonstrate ufuncs functions
    println!("\nUniversal Functions (ufuncs) Examples");
    println!("===================================\n");

    // Math operations
    let sin_a = math2d::sin(&a.view());
    println!("sin(a):");
    println!("{:?}\n", sin_a);

    let cos_a = math2d::cos(&a.view());
    println!("cos(a):");
    println!("{:?}\n", cos_a);

    // Binary operations with broadcasting
    let add_broadcast = binary2d::add(&a.view(), &c.view()).unwrap();
    println!("a + c (broadcasting):");
    println!("{:?}\n", add_broadcast);

    let mul_broadcast = binary2d::multiply(&a.view(), &c.view()).unwrap();
    println!("a * c (broadcasting):");
    println!("{:?}\n", mul_broadcast);

    // Reduction operations
    let sum_all = reduction::sum(&a.view(), None);
    println!("sum(a) [all elements]:");
    println!("{:?}\n", sum_all);

    let sum_cols = reduction::sum(&a.view(), Some(0));
    println!("sum(a) [along columns]:");
    println!("{:?}\n", sum_cols);

    let mean_rows = reduction::mean(&a.view(), Some(1));
    println!("mean(a) [along rows]:");
    println!("{:?}\n", mean_rows);
}
