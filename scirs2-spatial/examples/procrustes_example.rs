use ndarray::{array, Array2, ArrayView2};
use scirs2_spatial::{error::SpatialResult, procrustes, procrustes_extended};

fn main() -> SpatialResult<()> {
    println!("Procrustes Analysis Example");
    println!("==========================\n");

    // 2D Example
    println!("2D Procrustes Analysis Example");
    println!("--------------------------");

    // Create two datasets where one is a rotated, scaled, and reflected version of the other
    let a = array![[1.0, 3.0], [1.0, 2.0], [1.0, 1.0], [2.0, 1.0]];

    let b = array![[4.0, -2.0], [4.0, -4.0], [4.0, -6.0], [2.0, -6.0]];

    println!("Original datasets:");
    println!("Dataset A:");
    print_matrix(&a.view());

    println!("\nDataset B:");
    print_matrix(&b.view());

    // Perform Procrustes analysis
    let (mtx1, mtx2, disparity) = procrustes(&a.view(), &b.view())?;

    println!("\nAfter Procrustes transformation:");
    println!(
        "Disparity (squared error between transformed matrices): {:.10}",
        disparity
    );

    println!("\nStandardized dataset A:");
    print_matrix(&mtx1.view());

    println!("\nTransformed dataset B:");
    print_matrix(&mtx2.view());

    // 3D Example
    println!("\n\n3D Procrustes Analysis Example");
    println!("--------------------------");

    // Create two 3D point sets with known transformation
    let points3d_a = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ];

    // Apply known transformation:
    // - Scale by 2
    // - Rotate around z-axis by 45 degrees
    // - Translate by [5, 5, 5]
    let points3d_b = apply_transform(&points3d_a.view(), 2.0, 45.0, &array![5.0, 5.0, 5.0]);

    println!("Original 3D datasets:");
    println!("Dataset A:");
    print_matrix(&points3d_a.view());

    println!("\nDataset B (After scaling, rotation, and translation):");
    print_matrix(&points3d_b.view());

    // Perform Extended Procrustes analysis
    let (transformed, params, disparity) =
        procrustes_extended(&points3d_a.view(), &points3d_b.view(), true, true, true)?;

    println!("\nAfter Extended Procrustes transformation:");
    println!("Disparity: {:.10}", disparity);
    println!("Scale factor recovered: {:.6}", params.scale);
    println!("Rotation matrix recovered:\n{:.6}", params.rotation);
    println!(
        "Translation vector recovered: [{:.6}, {:.6}, {:.6}]",
        params.translation[0], params.translation[1], params.translation[2]
    );

    println!("\nTransformed dataset B (should match A):");
    print_matrix(&transformed.view());

    // Perform Procrustes with constraints
    println!("\n\nProcrustes with Constraints Example");
    println!("----------------------------------");

    // No scaling allowed
    let (_transformed_no_scale, params_no_scale, disparity_no_scale) =
        procrustes_extended(&points3d_a.view(), &points3d_b.view(), false, true, true)?;

    println!("Without scaling:");
    println!("Disparity: {:.10}", disparity_no_scale);
    println!("Scale factor: {:.6} (should be 1.0)", params_no_scale.scale);

    // No reflection allowed
    let (_transformed_no_reflection, _params_no_reflection, disparity_no_reflection) =
        procrustes_extended(&points3d_a.view(), &points3d_b.view(), true, false, true)?;

    println!("\nWithout reflection:");
    println!("Disparity: {:.10}", disparity_no_reflection);

    // No translation allowed
    let (_transformed_no_translation, params_no_translation, disparity_no_translation) =
        procrustes_extended(&points3d_a.view(), &points3d_b.view(), true, true, false)?;

    println!("\nWithout translation:");
    println!("Disparity: {:.10}", disparity_no_translation);
    println!(
        "Translation vector: [{:.6}, {:.6}, {:.6}] (should be zeros)",
        params_no_translation.translation[0],
        params_no_translation.translation[1],
        params_no_translation.translation[2]
    );

    // Apply the transformation to new data
    println!("\n\nApplying Transformation to New Data");
    println!("---------------------------------");

    let new_points = array![[2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 2.0]];

    println!("New points in A coordinate system:");
    print_matrix(&new_points.view());

    let transformed_new = params.transform(&new_points.view());

    println!("\nNew points transformed to B coordinate system:");
    print_matrix(&transformed_new.view());

    Ok(())
}

/// Utility function to print a matrix
fn print_matrix(mat: &ArrayView2<f64>) {
    for row in mat.rows() {
        print!("  [");
        for (j, &val) in row.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{:.4}", val);
        }
        println!("]");
    }
}

/// Apply a transform to 3D points
///
/// This applies:
/// 1. Scale by scale_factor
/// 2. Rotate around z-axis by angle_degrees
/// 3. Translate by translation_vector
fn apply_transform(
    points: &ArrayView2<f64>,
    scale_factor: f64,
    angle_degrees: f64,
    translation_vector: &ndarray::Array1<f64>,
) -> Array2<f64> {
    let angle_radians = angle_degrees * std::f64::consts::PI / 180.0;
    let cos_angle = angle_radians.cos();
    let sin_angle = angle_radians.sin();

    // Create rotation matrix for z-axis rotation
    let rotation = array![
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle, 0.0],
        [0.0, 0.0, 1.0]
    ];

    // Apply scale, rotation, and translation
    let mut result = points.to_owned() * scale_factor;
    result = result.dot(&rotation.t());

    // Apply translation
    for mut row in result.rows_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            *val += translation_vector[i];
        }
    }

    result
}
