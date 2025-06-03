use ag::tensor_ops as T;
use ndarray;
use scirs2_autograd as ag;

fn main() {
    println!("Testing gradient computation for z = 2x^2 + 3y + 1");

    ag::run(|ctx| {
        let x = ctx.placeholder("x", &[]);
        let y = ctx.placeholder("y", &[]);

        // Build the expression z = 2x^2 + 3y + 1
        let x_squared = x * x;
        let two_x_squared = 2. * x_squared;
        let three_y = 3. * y;
        let z = two_x_squared + three_y + 1.;

        // Compute gradients
        let gy = T::grad(&[z], &[y])[0];
        let gx = T::grad(&[z], &[x])[0];

        // Evaluate dz/dy (should be 3)
        let gy_result = gy.eval(ctx).unwrap();
        println!(
            "dz/dy = {:?} (expected: 3.0)",
            gy_result[ndarray::IxDyn(&[])]
        );

        // Evaluate dz/dx at x=2 (should be 2*2*x = 8)
        let gx_result = ctx
            .evaluator()
            .push(&gx)
            .feed(x, ndarray::arr0(2.).view().into_dyn())
            .run();
        println!(
            "dz/dx at x=2 = {:?} (expected: 8.0)",
            gx_result[0].as_ref().unwrap()[ndarray::IxDyn(&[])]
        );

        // Let's also trace the intermediate values
        println!("\nTracing intermediate values:");

        // Test individual gradient computations
        let g_three_y = T::grad(&[three_y], &[y])[0];
        println!(
            "d(3y)/dy = {:?} (expected: 3.0)",
            g_three_y.eval(ctx).unwrap()[ndarray::IxDyn(&[])]
        );

        // Test gradient of x^2
        let g_x_squared = T::grad(&[x_squared], &[x])[0];
        let g_x_squared_at_2 = ctx
            .evaluator()
            .push(&g_x_squared)
            .feed(x, ndarray::arr0(2.).view().into_dyn())
            .run();
        println!(
            "d(x^2)/dx at x=2 = {:?} (expected: 4.0)",
            g_x_squared_at_2[0].as_ref().unwrap()[ndarray::IxDyn(&[])]
        );

        // Test gradient of 2*x^2
        let g_two_x_squared = T::grad(&[two_x_squared], &[x])[0];
        let g_two_x_squared_at_2 = ctx
            .evaluator()
            .push(&g_two_x_squared)
            .feed(x, ndarray::arr0(2.).view().into_dyn())
            .run();
        println!(
            "d(2x^2)/dx at x=2 = {:?} (expected: 8.0)",
            g_two_x_squared_at_2[0].as_ref().unwrap()[ndarray::IxDyn(&[])]
        );
    });
}
