#[allow(dead_code)]
fn main() {
    println!("Testing ndarray shapes");

    // Create scalar (0D)
    let scalar = scirs2_core::ndarray::arr0(3.5);
    println!("Scalar shape: {:?}", scalar.shape());
    println!("Scalar value: {}", scalar[[]]);
    println!("Scalar ndim: {}", scalar.ndim());

    // Create 1D array
    let vector = scirs2_core::ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]);
    println!("Vector shape: {:?}", vector.shape());
    println!("Vector[0]: {}", vector[0]);
    println!("Vector ndim: {}", vector.ndim());

    // Create 2D array
    let matrix = scirs2_core::ndarray::array![[1.0, 2.0], [3.0, 4.0]];
    println!("Matrix shape: {:?}", matrix.shape());
    println!("Matrix[0,0]: {}", matrix[[0, 0]]);
    println!("Matrix ndim: {}", matrix.ndim());

    // Convert to dynamic arrays
    let scalar_dyn = scalar.into_dyn();
    let vector_dyn = vector.into_dyn();
    let matrix_dyn = matrix.into_dyn();

    println!("Scalar (dyn) shape: {:?}", scalar_dyn.shape());
    println!("Vector (dyn) shape: {:?}", vector_dyn.shape());
    println!("Matrix (dyn) shape: {:?}", matrix_dyn.shape());

    println!("Scalar (dyn) value: {}", scalar_dyn[[]]);
    println!("Vector (dyn)[0]: {}", vector_dyn[[0]]);
    println!("Matrix (dyn)[0,0]: {}", matrix_dyn[[0, 0]]);

    // Create an array directly
    let direct = scirs2_core::ndarray::Array::<f64, scirs2_core::ndarray::Ix2>::zeros((2, 2));
    println!("Direct shape: {:?}", direct.shape());

    // Clone the arrays
    let scalar_clone = scalar_dyn.clone();
    let matrix_clone = matrix_dyn.clone();

    println!("Scalar clone shape: {:?}", scalar_clone.shape());
    println!("Matrix clone shape: {:?}", matrix_clone.shape());
}
