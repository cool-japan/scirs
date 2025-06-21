use ag::tensor_ops::*;
use ndarray::array;
use scirs2_autograd as ag;

#[test]
fn debug_svd_reconstruction() {
    ag::run(|g| {
        let a = convert_to_tensor(array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]], g);
        println!("Original matrix A:");
        println!("{:?}", a.eval(g).unwrap());

        let (u, s, v) = svd(a);

        let u_val = u.eval(g).unwrap();
        let s_val = s.eval(g).unwrap();
        let v_val = v.eval(g).unwrap();

        println!("\nU shape: {:?}", u_val.shape());
        println!("U:\n{:?}", u_val);

        println!("\nS shape: {:?}", s_val.shape());
        println!("S: {:?}", s_val);

        println!("\nV shape: {:?}", v_val.shape());
        println!("V:\n{:?}", v_val);

        // Try reconstruction
        let s_diag = diag(s);
        println!("\nS_diag shape: {:?}", s_diag.shape());
        match s_diag.eval(g) {
            Ok(val) => println!("S_diag:\n{:?}", val),
            Err(e) => println!("S_diag eval error: {:?}", e),
        }

        let us = matmul(u, &s_diag);
        match us.eval(g) {
            Ok(val) => {
                println!("\nU * S shape: {:?}", val.shape());
                println!("U * S:\n{:?}", val);
            }
            Err(e) => println!("U * S eval error: {:?}", e),
        }

        let reconstructed = matmul(us, &transpose(v, &[1, 0]));
        match reconstructed.eval(g) {
            Ok(val) => {
                println!("\nReconstructed shape: {:?}", val.shape());
                println!("Reconstructed:\n{:?}", val);
            }
            Err(e) => println!("Reconstruction eval error: {:?}", e),
        }
    });
}
