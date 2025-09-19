// Example demonstrating the enhanced ndarray extensions and ufuncs functionality

use ndarray::{array, Array, Array2};
use scirs2_core::ndarray_ext::{
    // Array manipulation operations
    manipulation::{concatenate_2d, flip_2d, pad_2d, repeat_2d, roll_2d, tile_2d, vstack_1d},
    // Matrix operations
    matrix::{_diag as diag, block_diag, eye, kron, toeplitz, trace, tridiagonal},
};
use scirs2_core::ufuncs::{
    ceil,
    cos,
    cube,
    exp,
    floor,
    log,
    log10,
    max,
    // Reduction operations
    mean,
    min,
    rad2deg,
    round,
    // Math operations from math2d
    sin,
    sqrt,
    square,
    std,
};
use std::f64::consts::PI;

#[allow(dead_code)]
fn print_title(title: &str) {
    println!("\n{}", title);
    println!("{}", "=".repeat(title.len()));
}

#[allow(dead_code)]
fn main() {
    println!("SciRS2-Core Advanced ndarray_ext and ufuncs Example");
    println!("===================================================\n");

    // Create some example arrays
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
    let c = array![10.0, 20.0, 30.0];
    // Create a 2D array from a 1D array (wrapped as rows)
    let angles_1d = array![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0];
    let angles = array![[0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0]];

    // Demonstrate matrix creation functions
    println!("\n=== Matrix Creation ===");

    let identity = eye::<f64>(3);
    println!("Identity matrix (3x3):");
    println!("{}", identity);

    let diag_mat = diag(array![1.0, 2.0, 3.0].view());
    println!("\nDiagonal matrix from [1, 2, 3]:");
    println!("{}", diag_mat);

    let tr = trace(array![[1.0, 2.0], [3.0, 4.0]].view()).unwrap();
    println!("\nTrace of [[1, 2], [3, 4]]: {}", tr);

    let k = kron(
        array![[1.0, 2.0], [3.0, 4.0]].view(),
        array![[0.0, 5.0], [6.0, 7.0]].view(),
    );
    println!("\nKronecker product:");
    println!("{}", k);

    let block = block_diag(&[
        array![[1.0, 2.0], [3.0, 4.0]].view(),
        array![[5.0, 6.0], [7.0, 8.0]].view(),
    ]);
    println!("\nBlock diagonal matrix:");
    println!("{}", block);

    let toep = toeplitz(array![1.0, 2.0, 3.0].view(), array![1.0, 4.0, 7.0].view()).unwrap();
    println!("\nToeplitz matrix:");
    println!("{}", toep);

    let tridiag = tridiagonal(
        array![1.0, 2.0, 3.0].view(), // main diagonal
        array![4.0, 5.0].view(),      // lower diagonal
        array![6.0, 7.0].view(),      // upper diagonal
    )
    .unwrap();
    println!("\nTridiagonal matrix:");
    println!("{}", tridiag);

    // Demonstrate array manipulation functions
    println!("\n=== Array Manipulation ===");

    println!("Original array a:");
    println!("{}", a);

    let flipped = flip_2d(a.view(), true, true);
    println!("\nFlipped along both axes:");
    println!("{}", flipped);

    let rolled = roll_2d(a.view(), 1, 1);
    println!("\nRolled by 1 along both axes:");
    println!("{}", rolled);

    let tiled = tile_2d(array![[1.0, 2.0], [3.0, 4.0]].view(), 2, 2);
    println!("\nTiled 2x2:");
    println!("{}", tiled);

    let repeated = repeat_2d(array![[1.0, 2.0], [3.0, 4.0]].view(), 2, 2);
    println!("\nRepeated each element 2x2:");
    println!("{}", repeated);

    let padded = pad_2d(a.view(), ((1, 1), (2, 2)), 0.0);
    println!("\nPadded with zeros:");
    println!("{}", padded);

    let concat_h = concatenate_2d(&[a.view(), b.view()], 1).unwrap();
    println!("\nConcatenated horizontally:");
    println!("{}", concat_h);

    let vstacked = vstack_1d(&[c.view(), array![40.0, 50.0, 60.0].view()]).unwrap();
    println!("\nVertically stacked 1D arrays:");
    println!("{}", vstacked);

    // Demonstrate mathematical operations
    println!("\n=== Mathematical Functions ===");

    println!("Original angles (in radians):");
    println!("{}", angles_1d);

    let sin_values = sin(&angles.view());
    println!("\nSine of angles:");
    println!("{}", sin_values);

    let cos_values = cos(&angles.view());
    println!("\nCosine of angles:");
    println!("{}", cos_values);

    let exp_values = exp(&array![[0.0, 1.0, 2.0]].view());
    println!("\nExponential of [0, 1, 2]:");
    println!("{}", exp_values);

    let log_values = log(&array![[1.0, 2.0, 10.0]].view());
    println!("\nNatural logarithm of [1, 2, 10]:");
    println!("{}", log_values);

    let log10_values = log10(&array![[1.0, 10.0, 100.0]].view());
    println!("\nBase-10 logarithm of [1, 10, 100]:");
    println!("{}", log10_values);

    let sqrt_values = sqrt(&array![[1.0, 4.0, 9.0]].view());
    println!("\nSquare root of [1, 4, 9]:");
    println!("{}", sqrt_values);

    let rounded = round(&array![[1.1, 1.5, 1.9]].view());
    println!("\nRounded values of [1.1, 1.5, 1.9]:");
    println!("{}", rounded);

    let floored = floor(&array![[1.1, 1.5, 1.9]].view());
    println!("\nFloored values of [1.1, 1.5, 1.9]:");
    println!("{}", floored);

    let ceiled = ceil(&array![[1.1, 1.5, 1.9]].view());
    println!("\nCeiled values of [1.1, 1.5, 1.9]:");
    println!("{}", ceiled);

    let squared = square(&array![[1.0, 2.0, 3.0]].view());
    println!("\nSquared values of [1, 2, 3]:");
    println!("{}", squared);

    let cubed = cube(&array![[1.0, 2.0, 3.0]].view());
    println!("\nCubed values of [1, 2, 3]:");
    println!("{}", cubed);

    let degrees = rad2deg(&angles.view());
    println!("\nAngles converted to degrees:");
    println!("{}", degrees);

    // Demonstrate statistical operations
    println!("\n=== Statistical Functions ===");

    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    println!("Data matrix:");
    println!("{}", data);

    let mean_all = mean(&data.view(), None);
    println!("\nMean of all elements: {}", mean_all[0]);

    let mean_rows = mean(&data.view(), Some(0));
    println!("\nMean of each column:");
    println!("{}", mean_rows);

    let mean_cols = mean(&data.view(), Some(1));
    println!("\nMean of each row:");
    println!("{}", mean_cols);

    let std_all = std(&data.view(), None);
    println!("\nStandard deviation of all elements: {}", std_all[0]);

    let min_all = min(&data.view(), None);
    println!("\nMinimum value: {}", min_all[0]);

    let max_all = max(&data.view(), None);
    println!("\nMaximum value: {}", max_all[0]);

    // Demonstrate a complete data analysis workflow
    println!("\n=== Complete Data Analysis Workflow ===");

    // Create a dataset with some noise
    let measurements = array![
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.3],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2]
    ];

    println!("Dataset (shape {:?}):", measurements.shape());

    // Calculate descriptive statistics
    let column_means = mean(&measurements.view(), Some(0));
    let column_std = std(&measurements.view(), Some(0));

    println!("\nColumn means:");
    println!("{}", column_means);

    println!("\nColumn standard deviations:");
    println!("{}", column_std);

    // Z-score standardization (subtract mean, divide by std)
    let mut standardized = Array2::<f64>::zeros(measurements.raw_dim());
    for i in 0..measurements.shape()[0] {
        for j in 0..measurements.shape()[1] {
            standardized[[i, j]] = (measurements[[i, j]] - column_means[j]) / column_std[j];
        }
    }

    println!("\nStandardized data (first 3 rows):");
    for i in 0..3 {
        println!("{:?}", standardized.row(i));
    }

    // Calculate correlation matrix (simplified example)
    let n_cols = measurements.shape()[1];
    let mut corr = Array2::<f64>::zeros((n_cols, n_cols));

    for i in 0..n_cols {
        for j in 0..n_cols {
            let std_i = standardized.column(i);
            let std_j = standardized.column(j);

            let mut sum_product = 0.0;
            for k in 0..standardized.shape()[0] {
                sum_product += std_i[k] * std_j[k];
            }

            // Normalize by n-1 to get correlation
            corr[[i, j]] = sum_product / (standardized.shape()[0] as f64 - 1.0);
        }
    }

    println!("\nCorrelation matrix:");
    println!("{}", corr);

    println!("\nExample complete!");
}
