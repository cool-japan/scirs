//! Matrix statistical functions example
//!
//! This example demonstrates statistical functions for matrices,
//! such as covariance and correlation computation.

use scirs2_core::ndarray::array;
use scirs2_linalg::error::LinalgResult;
use scirs2_linalg::stats::covariance::mahalanobis_distance;
use scirs2_linalg::stats::{correlationmatrix, covariancematrix};

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("Matrix Statistical Functions Example");
    println!("==================================\n");

    // Create a sample data matrix
    // Each row is a sample, each column is a variable
    let data = array![
        [1.2, 2.3, 3.1],
        [2.1, 1.5, 2.2],
        [1.8, 1.9, 3.3],
        [2.5, 2.2, 4.1],
        [1.5, 2.0, 3.5]
    ];

    println!("Data matrix:");
    for i in 0..data.nrows() {
        println!("{:?}", data.row(i));
    }
    println!();

    // Compute covariance matrix
    let cov = covariancematrix(&data.view(), None)?;

    println!("Covariance matrix:");
    for i in 0..cov.nrows() {
        for j in 0..cov.ncols() {
            print!("{:.6} ", cov[[i, j]]);
        }
        println!();
    }
    println!();

    // Compute correlation matrix
    let corr = correlationmatrix(&data.view(), None)?;

    println!("Correlation matrix:");
    for i in 0..corr.nrows() {
        for j in 0..corr.ncols() {
            print!("{:.6} ", corr[[i, j]]);
        }
        println!();
    }
    println!();

    // Compute mean of the data
    let mut mean = array![0.0, 0.0, 0.0];
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            mean[j] += data[[i, j]];
        }
    }
    for j in 0..mean.len() {
        mean[j] /= data.nrows() as f64;
    }

    println!("Mean vector: {:?}", mean);

    // Compute Mahalanobis distance for a new sample
    let new_sample = array![2.0, 2.0, 3.0];
    let dist = mahalanobis_distance(&new_sample.view(), &mean.view(), &cov.view())?;

    println!("New sample: {:?}", new_sample);
    println!("Mahalanobis distance from mean: {:.6}", dist);

    Ok(())
}
