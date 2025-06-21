use scirs2_autograd as ag;

fn main() {
    println!("Testing tensor shape handling (DEBUGGING):");

    ag::run::<f64, _, _>(|g| {
        // 1. Create a tensor with simple constructor
        println!("\n=== 1. Testing zeros tensor creation ===");
        let z = ag::tensor_ops::zeros(&[2, 2], g);
        let z_eval = z.eval(g).unwrap();
        println!("Zeros tensor shape: {:?}", z_eval.shape());
        println!("Zeros tensor data:\n{:?}", z_eval);

        // 2. Create a variable from array
        println!("\n=== 2. Testing variable creation ===");
        let test_array = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        println!("Original array shape: {:?}", test_array.shape());
        let v = ag::tensor_ops::variable(test_array.clone(), g);
        let v_eval = v.eval(g).unwrap();
        println!("Variable tensor shape: {:?}", v_eval.shape());
        println!("Variable tensor data:\n{:?}", v_eval);

        // 3. Try accessing elements
        println!("\n=== 3. Testing element access ===");
        if v_eval.shape().len() == 2 {
            println!("v_eval[0,0] = {}", v_eval[[0, 0]]);
            println!("v_eval[0,1] = {}", v_eval[[0, 1]]);
            println!("v_eval[1,0] = {}", v_eval[[1, 0]]);
            println!("v_eval[1,1] = {}", v_eval[[1, 1]]);
        } else {
            println!(
                "ERROR: Cannot access elements due to wrong shape: {:?}",
                v_eval.shape()
            );
        }

        // 4. Try more advanced operations
        println!("\n=== 4. Testing matrix operations ===");
        let det = ag::tensor_ops::determinant(v);
        let det_eval = det.eval(g).unwrap();
        println!("Determinant shape: {:?}", det_eval.shape());
        if det_eval.shape().is_empty() {
            println!("Determinant value: {}", det_eval[[]]);
        } else {
            println!("ERROR: Determinant has wrong shape: {:?}", det_eval.shape());
        }
    });
}
