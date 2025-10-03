#[cfg(feature = "autograd")]
#[allow(unused_imports)]
mod example {
    // use scirs2_core::ndarray::array;  // Temporarily disabled
    use scirs2_autograd::error::Result as AutogradResult;
    // use scirs2_autograd::variable::Variable;  // Temporarily disabled
    // Import the implemented autograd transformations
    // use scirs2_linalg::autograd::transformations::variable::{
    //     project, reflectionmatrix, rotationmatrix_2d, scalingmatrix, shearmatrix,
    // };  // Temporarily disabled

    pub fn run() -> AutogradResult<()> {
        println!("Matrix Transformations with Autograd Example");
        println!("--------------------------------------------\n");

        // Note: This example is temporarily disabled due to API mismatch.
        // The Variable type in scirs2_autograd is RefCell<NdArray<F>>, not a struct with tensor field.
        // The transformation functions in autograd::transformations::variable expect a different Variable type.
        println!("Autograd transformations are temporarily disabled due to API mismatch.");

        /* Disabled code - needs API fix
        // Example 1: 2D Rotation Matrix
        println!("Example 1: 2D Rotation Matrix");

        // Create scalar tensor for angle (45 degrees)
        let angle_data = array![std::f64::consts::PI / 4.0].into_dyn();
        let angle = Variable::fromarray(angle_data, true);

        let rotation = rotationmatrix_2d(&angle)?;

        println!("Rotation matrix for 45 degrees:");
        println!("{:?}\n", rotation.tensor.data());

        // Apply the rotation to a vector and calculate gradient
        // Create unit vector along x-axis
        let v_data = array![1.0, 0.0].into_dyn();
        let v = Variable::fromarray(v_data, false);

        // Apply rotation to vector (using matmul)
        let rotated_v = rotation.matmul(&v)?;
        println!("Rotated vector [1, 0]:");
        println!("{:?}\n", rotated_v.tensor.data());

        // Compute the gradient of the y-coordinate with respect to the angle
        let y_axis_data = array![0.0, 1.0].into_dyn();
        let y_axis = Variable::fromarray(y_axis_data, false);

        let y_coord = rotated_v.dot(&y_axis)?;

        // Create gradient for backward pass (typically filled with ones)
        let grad_data = array![1.0].into_dyn();
        y_coord.backward(Some(grad_data))?;

        println!("Gradient of y-coordinate with respect to angle:");
        println!("{:?}\n", angle.tensor.grad());

        // Example 2: Scaling Matrix
        println!("Example 2: Scaling Matrix");
        let scales_data = array![2.0, 0.5].into_dyn(); // Scale x by 2, y by 0.5
        let scales = Variable::fromarray(scales_data, true);
        let scaling = scalingmatrix(&scales)?;

        println!("Scaling matrix for [2.0, 0.5]:");
        println!("{:?}\n", scaling.tensor.data());

        // Apply scaling to a vector and calculate gradient
        let v_data = array![1.0, 1.0].into_dyn(); // Vector [1, 1]
        let v = Variable::fromarray(v_data, false);
        let scaled_v = scaling.matmul(&v)?;
        println!("Scaled vector [1, 1]:");
        println!("{:?}\n", scaled_v.tensor.data());

        // Example 3: Reflection Matrix
        println!("Example 3: Reflection Matrix");
        let normal_data = array![1.0, 1.0].into_dyn(); // Normal vector for reflection plane
        let normal = Variable::fromarray(normal_data, true);
        let reflection = reflectionmatrix(&normal)?;

        println!("Reflection matrix for normal [1, 1]:");
        println!("{:?}\n", reflection.tensor.data());

        // Apply reflection to a vector
        let v_data = array![1.0, 0.0].into_dyn(); // Vector [1, 0]
        let v = Variable::fromarray(v_data, false);
        let reflected_v = reflection.matmul(&v)?;
        println!("Reflected vector [1, 0]:");
        println!("{:?}\n", reflected_v.tensor.data());

        // Example 4: Shear Matrix
        println!("Example 4: Shear Matrix");
        let shear_factor_data = array![0.5].into_dyn();
        let shear_factor = Variable::fromarray(shear_factor_data, true);
        let shear = shearmatrix(&shear_factor, 0, 1, 2)?; // Shear in x direction

        println!("Shear matrix for factor 0.5:");
        println!("{:?}\n", shear.tensor.data());

        // Apply shear to a vector
        let v_data = array![1.0, 1.0].into_dyn(); // Vector [1, 1]
        let v = Variable::fromarray(v_data, false);
        let sheared_v = shear.matmul(&v)?;
        println!("Sheared vector [1, 1]:");
        println!("{:?}\n", sheared_v.tensor.data());

        // Example 5: Orthogonal Projection
        println!("Example 5: Orthogonal Projection");
        // Create a matrix whose columns span a subspace
        let a_data = array![[1.0, 0.0], [1.0, 1.0]].into_dyn();
        let a = Variable::fromarray(a_data, true);

        // Create a vector to project
        let x_data = array![2.0, 3.0].into_dyn();
        let x = Variable::fromarray(x_data, false);

        // Project the vector onto the column space of A
        let projected = project(&a, &x)?;

        println!("Matrix A (columns span the subspace):");
        println!("{:?}", a.tensor.data());
        println!("\nVector x to project:");
        println!("{:?}", x.tensor.data());
        println!("\nProjection of x onto the column space of A:");
        println!("{:?}\n", projected.tensor.data());

        // Instead of the norm, let's compute the sum of the projection
        // to avoid the shape issue with the norm function
        let sum_val = projected.sum()?;

        // Create gradient for backward pass (typically filled with ones)
        let grad_data = array![1.0].into_dyn();
        sum_val.backward(Some(grad_data))?;

        println!("Gradient of projection sum with respect to A:");
        println!("{:?}", a.tensor.grad());
        */
        // End of disabled code

        Ok(())
    }
}

#[allow(dead_code)]
fn main() {
    #[cfg(feature = "autograd")]
    {
        example::run().unwrap();
    }

    #[cfg(not(feature = "autograd"))]
    {
        println!("This example requires the 'autograd' feature.");
        println!(
            "Run with: cargo run --example matrix_transformations_example --features autograd"
        );
    }
}
