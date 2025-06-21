use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

fn main() {
    println!("Testing SVD S component extraction...");

    ag::run::<f64, _, _>(|g| {
        // Create a 2x2 matrix for simplicity
        let matrix = convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0]], g);
        println!(
            "Original matrix shape: {:?}",
            matrix.eval(g).unwrap().shape()
        );

        // Compute SVD
        let (u, s, v) = svd(matrix);

        // Check shapes and values
        println!("\nSVD components:");

        println!("U (should be 2x2):");
        match u.eval(g) {
            Ok(arr) => {
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);
            }
            Err(e) => println!("Error evaluating U: {:?}", e),
        }

        println!("\nS (should be a 1D vector with 2 values):");
        match s.eval(g) {
            Ok(arr) => {
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);

                // Print detailed information about the S component
                println!("S ndim: {}", arr.ndim());
                println!("S size: {}", arr.len());
            }
            Err(e) => println!("Error evaluating S: {:?}", e),
        }

        println!("\nV (should be 2x2):");
        match v.eval(g) {
            Ok(arr) => {
                println!("Shape: {:?}", arr.shape());
                println!("Values: {:?}", arr);
            }
            Err(e) => println!("Error evaluating V: {:?}", e),
        }
    });
}
