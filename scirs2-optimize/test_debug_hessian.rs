use ndarray::{array, ArrayView1};
use scirs2_optimize::sparse_numdiff::{sparse_hessian, SparseFiniteDiffOptions};
use scirs2_sparse::sparray::SparseArray;

fn sphere(x: &ArrayView1<f64>) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

fn main() {
    let x = array![1.0, 2.0];

    // Test the default method
    println!("Testing default method:");
    let hess = sparse_hessian(sphere, &x.view(), None, None, None).unwrap();

    println!("Hessian computed:");
    let hess_dense = hess.to_array();
    println!("{:?}", hess_dense);

    // Debug: check the raw sparse data
    let data = hess.get_data();
    let indices = hess.get_indices();
    let indptr = hess.get_indptr();

    println!("\nSparse representation:");
    println!("Data: {:?}", data);
    println!("Indices: {:?}", indices);
    println!("Indptr: {:?}", indptr);

    println!("\nShape: {:?}", hess.shape());

    // Test with explicit 3-point method
    println!("\n\nTesting 3-point method explicitly:");
    let mut options = SparseFiniteDiffOptions::default();
    options.method = "3-point".to_string();

    let hess2 = sparse_hessian(sphere, &x.view(), None, None, Some(options)).unwrap();
    let hess2_dense = hess2.to_array();
    println!("{:?}", hess2_dense);
}
