// Minimal test of linear algebra operations
use ag::tensor_ops::*;
use scirs2_autograd as ag;
use scirs2_core::ndarray::array;

#[allow(dead_code)]
fn main() {
    ag::run(|ctx| {
        // Test basic operations
        let identity = eye(3, ctx);
        println!("Identity matrix created");

        let a = convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0]], ctx);
        let tr = trace(a);
        println!("Trace computed");

        let values = convert_to_tensor(array![1.0, 2.0, 3.0], ctx);
        let diagonal = diag(values);
        println!("Diagonal matrix created");

        // Try to evaluate
        match identity.eval(ctx) {
            Ok(val) => println!("Identity: {:?}", val),
            Err(e) => println!("Error: {:?}", e),
        }

        match tr.eval(ctx) {
            Ok(val) => println!("Trace: {:?}", val),
            Err(e) => println!("Error: {:?}", e),
        }

        match diagonal.eval(ctx) {
            Ok(val) => println!("Diagonal: {:?}", val),
            Err(e) => println!("Error: {:?}", e),
        }
    });
}
