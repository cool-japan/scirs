use scirs2_core::ndarray::Array2;

#[allow(dead_code)]
fn main() {
    let array =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    println!("Original array:");
    for i in 0..3 {
        for j in 0..3 {
            print!("{} ", array[[i, j]]);
        }
        println!();
    }
}
