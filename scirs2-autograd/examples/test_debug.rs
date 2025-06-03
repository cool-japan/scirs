use ag::ndarray::array;
use ag::tensor_ops as T;
use scirs2_autograd as ag;

fn main() {
    ag::run::<f32, _, _>(|ctx| {
        println!("=== Debug Test ===");

        // Create simple data
        let a_data = array![[1.0, 2.0], [3.0, 4.0]];
        let b_data = array![[5.0, 6.0], [7.0, 8.0]];
        println!("Input data shape: {:?}", a_data.shape());

        // Test 1: Direct tensor creation (works)
        println!("\n1. Direct tensor creation:");
        let a = T::convert_to_tensor(a_data.clone(), ctx);
        let b = T::convert_to_tensor(b_data.clone(), ctx);
        let c = T::matmul(&a, &b);
        let result = ctx.evaluator().push(&c).run();
        if let Ok(r) = &result[0] {
            println!("Direct matmul result: {:?}", r);
        }

        // Test 2: Using placeholders (problematic)
        println!("\n2. Using placeholders:");
        let p_a = ctx.placeholder("a", &[2, 2]);
        let p_b = ctx.placeholder("b", &[2, 2]);

        // Debug: evaluate placeholder directly
        let feeder = ag::Feeder::new()
            .push(p_a, a_data.view().into_dyn())
            .push(p_b, b_data.view().into_dyn());

        let result = ctx.evaluator().push(&p_a).set_feeder(feeder).run();

        if let Ok(r) = &result[0] {
            println!("Placeholder evaluation result shape: {:?}", r.shape());
            println!("Placeholder evaluation result: {:?}", r);
        }

        // Now try matmul with placeholders
        println!("\n3. Matmul with placeholders:");
        let p_c = T::matmul(&p_a, &p_b);
        let feeder2 = ag::Feeder::new()
            .push(p_a, a_data.view().into_dyn())
            .push(p_b, b_data.view().into_dyn());

        let result2 = ctx.evaluator().push(&p_c).set_feeder(feeder2).run();

        if let Ok(r) = &result2[0] {
            println!("Placeholder matmul result: {:?}", r);
        } else if let Err(e) = &result2[0] {
            println!("Placeholder matmul error: {:?}", e);
        }
    });
}
