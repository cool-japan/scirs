//! Simple test for specialized matrices

use scirs2_core::ndarray::array;
use scirs2_linalg::specialized::{BandedMatrix, SymmetricMatrix, TridiagonalMatrix};
use scirs2_linalg::SpecializedMatrix;

#[allow(dead_code)]
fn main() {
    println!("Testing specialized matrices...");

    // Test tridiagonal matrix
    let diag = array![2.0, 2.0, 2.0, 2.0, 2.0]; // Main diagonal
    let superdiag = array![-1.0, -1.0, -1.0, -1.0]; // Superdiagonal
    let subdiag = array![-1.0, -1.0, -1.0, -1.0]; // Subdiagonal

    let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view())
        .expect("Failed to create tridiagonal matrix");

    let _dense = tri.to_dense().expect("Failed to convert to dense");
    println!("Tridiagonal matrix created successfully!");
    println!("Size: {}x{}", tri.nrows(), tri.ncols());

    // Test banded matrix
    let mut band_data = scirs2_core::ndarray::Array2::zeros((3, 4)); // 3 diagonals (1+1+1), 4 columns

    // Lower diagonal
    band_data[[0, 0]] = 1.0;
    band_data[[0, 1]] = 2.0;
    band_data[[0, 2]] = 3.0;

    // Main diagonal
    band_data[[1, 0]] = 4.0;
    band_data[[1, 1]] = 5.0;
    band_data[[1, 2]] = 6.0;
    band_data[[1, 3]] = 7.0;

    // Upper diagonal
    band_data[[2, 0]] = 8.0;
    band_data[[2, 1]] = 9.0;
    band_data[[2, 2]] = 10.0;

    let band =
        BandedMatrix::new(band_data.view(), 1, 1, 4, 4).expect("Failed to create banded matrix");

    println!("Banded matrix created successfully!");
    println!("Size: {}x{}", band.nrows(), band.ncols());

    // Test symmetric matrix
    let sym_data = array![
        [2.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [0.0, 0.0, -1.0, 2.0]
    ];

    let sym =
        SymmetricMatrix::frommatrix(&sym_data.view()).expect("Failed to create symmetric matrix");

    println!("Symmetric matrix created successfully!");
    println!("Size: {}x{}", sym.nrows(), sym.ncols());

    println!("All tests passed!");
}
