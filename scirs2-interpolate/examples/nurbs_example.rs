use scirs2_core::ndarray::array;
use scirs2_interpolate::bspline::ExtrapolateMode;
use scirs2_interpolate::nurbs::{make_nurbs_circle, make_nurbs_sphere, NurbsCurve, NurbsSurface};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NURBS (Non-Uniform Rational B-Splines) Examples");
    println!("===============================================");

    // Example 1: Simple quadratic NURBS curve (a parabola)
    println!("\nExample 1: Simple quadratic NURBS curve (parabola)");
    let control_points = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let weights = array![1.0, 1.0, 1.0]; // Equal weights (equivalent to a B-spline)
    let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let degree = 2;

    let nurbs_curve = NurbsCurve::new(
        &control_points.view(),
        &weights.view(),
        &knots.view(),
        degree,
        ExtrapolateMode::Extrapolate,
    )?;

    // Evaluate at various points
    println!("  Point at t=0.0: {:?}", nurbs_curve.evaluate(0.0)?);
    println!("  Point at t=0.5: {:?}", nurbs_curve.evaluate(0.5)?);
    println!("  Point at t=1.0: {:?}", nurbs_curve.evaluate(1.0)?);

    // Example 2: NURBS curve with varying weights
    println!("\nExample 2: NURBS curve with varying weights");
    let control_points = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let weights = array![1.0, 2.0, 1.0]; // Higher weight for middle point
    let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let degree = 2;

    let weighted_nurbs = NurbsCurve::new(
        &control_points.view(),
        &weights.view(),
        &knots.view(),
        degree,
        ExtrapolateMode::Extrapolate,
    )?;

    // Compare points with uniform weights vs. varying weights
    println!(
        "  Uniform weights at t=0.5: {:?}",
        nurbs_curve.evaluate(0.5)?
    );
    println!(
        "  Varying weights at t=0.5: {:?}",
        weighted_nurbs.evaluate(0.5)?
    );
    // Notice how the varying weights pull the curve closer to the middle control point

    // Example 3: Creating a NURBS circle
    println!("\nExample 3: NURBS circle");
    let center = array![0.0, 0.0];
    let radius = 1.0;
    let circle = make_nurbs_circle([center[0], center[1]], radius, None, None)?;

    println!(
        "  Circle point at t=0.0   (0°):   {:?}",
        circle.evaluate(0.0)?
    );
    println!(
        "  Circle point at t=0.25 (90°):   {:?}",
        circle.evaluate(0.25)?
    );
    println!(
        "  Circle point at t=0.5  (180°):  {:?}",
        circle.evaluate(0.5)?
    );
    println!(
        "  Circle point at t=0.75 (270°):  {:?}",
        circle.evaluate(0.75)?
    );
    println!(
        "  Circle point at t=1.0  (360°):  {:?}",
        circle.evaluate(1.0)?
    );

    // Verify that points are on the circle (distance from center should be radius)
    let p = circle.evaluate(0.3)?;
    let dist = (f64::powi(p[0] - center[0], 2) + f64::powi(p[1] - center[1], 2)).sqrt();
    println!("  Distance from center for point at t=0.3: {}", dist);
    println!("  (Should be equal to radius = {})", radius);

    // Example 4: NURBS arc (partial circle)
    println!("\nExample 4: NURBS arc (quarter circle)");
    let start_angle = 0.0;
    let end_angle = std::f64::consts::PI / 2.0; // 90 degrees
    let arc = make_nurbs_circle(
        [center[0], center[1]],
        radius,
        Some(start_angle),
        Some(end_angle),
    )?;

    println!("  Arc point at t=0.0  (0°):  {:?}", arc.evaluate(0.0)?);
    println!("  Arc point at t=0.5 (45°):  {:?}", arc.evaluate(0.5)?);
    println!("  Arc point at t=1.0 (90°):  {:?}", arc.evaluate(1.0)?);

    // Example 5: Simple NURBS surface (a plane)
    println!("\nExample 5: Simple NURBS surface (plane)");
    let control_points = array![
        [0.0, 0.0, 0.0], // (0,0)
        [1.0, 0.0, 0.0], // (0,1)
        [0.0, 1.0, 0.0], // (1,0)
        [1.0, 1.0, 0.0]  // (1,1)
    ];
    let weights = array![1.0, 1.0, 1.0, 1.0]; // Equal weights
    let knots_u = array![0.0, 0.0, 1.0, 1.0];
    let knots_v = array![0.0, 0.0, 1.0, 1.0];
    let degree_u = 1;
    let degree_v = 1;

    let nurbs_surface = NurbsSurface::new(
        &control_points.view(),
        &weights.view(),
        2,
        2, // 2x2 grid of control points
        &knots_u.view(),
        &knots_v.view(),
        degree_u,
        degree_v,
        ExtrapolateMode::Extrapolate,
    )?;

    println!(
        "  Surface at (u,v)=(0.0,0.0): {:?}",
        nurbs_surface.evaluate(0.0, 0.0)?
    );
    println!(
        "  Surface at (u,v)=(0.5,0.5): {:?}",
        nurbs_surface.evaluate(0.5, 0.5)?
    );
    println!(
        "  Surface at (u,v)=(1.0,1.0): {:?}",
        nurbs_surface.evaluate(1.0, 1.0)?
    );

    // Example 6: NURBS surface with varying weights
    println!("\nExample 6: NURBS surface with varying weights");
    let control_points = array![
        [0.0, 0.0, 0.0], // (0,0)
        [1.0, 0.0, 0.0], // (0,1)
        [0.0, 1.0, 0.0], // (1,0)
        [1.0, 1.0, 0.0]  // (1,1)
    ];
    let weights = array![1.0, 1.0, 1.0, 2.0]; // Higher weight for (1,1) corner
    let knots_u = array![0.0, 0.0, 1.0, 1.0];
    let knots_v = array![0.0, 0.0, 1.0, 1.0];
    let degree_u = 1;
    let degree_v = 1;

    let weighted_surface = NurbsSurface::new(
        &control_points.view(),
        &weights.view(),
        2,
        2,
        &knots_u.view(),
        &knots_v.view(),
        degree_u,
        degree_v,
        ExtrapolateMode::Extrapolate,
    )?;

    // Compare middle point for uniform vs weighted surface
    println!(
        "  Uniform weights at (u,v)=(0.5,0.5): {:?}",
        nurbs_surface.evaluate(0.5, 0.5)?
    );
    println!(
        "  Varying weights at (u,v)=(0.5,0.5): {:?}",
        weighted_surface.evaluate(0.5, 0.5)?
    );
    // Notice the middle point is pulled toward the (1,1) corner with the higher weight

    // Example 7: NURBS sphere
    println!("\nExample 7: NURBS sphere");
    let center = array![0.0, 0.0, 0.0];
    let radius = 1.0;
    let sphere = make_nurbs_sphere([center[0], center[1], center[2]], radius)?;

    // Evaluate at various points on the sphere
    println!(
        "  Sphere at (u,v)=(0.0,0.0): {:?}",
        sphere.evaluate(0.0, 0.0)?
    );
    println!(
        "  Sphere at (u,v)=(0.5,0.5): {:?}",
        sphere.evaluate(0.5, 0.5)?
    );
    println!(
        "  Sphere at (u,v)=(1.0,0.0): {:?}",
        sphere.evaluate(1.0, 0.0)?
    );

    // Verify points are on the sphere (distance from center should be radius)
    let p = sphere.evaluate(0.3, 0.7)?;
    let dist = (f64::powi(p[0] - center[0], 2)
        + f64::powi(p[1] - center[1], 2)
        + f64::powi(p[2] - center[2], 2))
    .sqrt();
    println!(
        "  Distance from center for point at (u,v)=(0.3,0.7): {}",
        dist
    );
    println!("  (Should be approximately equal to radius = {})", radius);

    // Example 8: Derivatives of NURBS curves
    println!("\nExample 8: Derivatives of NURBS curves");
    let control_points = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let weights = array![1.0, 1.0, 1.0];
    let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let degree = 2;

    let nurbs_curve = NurbsCurve::new(
        &control_points.view(),
        &weights.view(),
        &knots.view(),
        degree,
        ExtrapolateMode::Extrapolate,
    )?;

    println!(
        "  Derivative at t=0.0: {:?}",
        nurbs_curve.derivative(0.0, 1)?
    );
    println!(
        "  Derivative at t=0.5: {:?}",
        nurbs_curve.derivative(0.5, 1)?
    );
    println!(
        "  Derivative at t=1.0: {:?}",
        nurbs_curve.derivative(1.0, 1)?
    );

    // Example 9: Sampling a NURBS curve at multiple parameter values
    println!("\nExample 9: Sampling a NURBS curve");
    let t_values = scirs2_core::ndarray::Array1::<f64>::linspace(0.0, 1.0, 5);
    let curve_points = circle.evaluate_array(&t_values.view())?;

    println!("  Circle points at t = [0.0, 0.25, 0.5, 0.75, 1.0]:");
    for i in 0..t_values.len() {
        println!(
            "    t = {:.2}: ({:.6}, {:.6})",
            t_values[i],
            curve_points[[i, 0]],
            curve_points[[i, 1]]
        );
    }

    // Example 10: Sampling a NURBS surface at multiple parameter values
    println!("\nExample 10: Sampling a NURBS surface on a grid");
    let u_values = scirs2_core::ndarray::Array1::<f64>::linspace(0.0, 1.0, 3); // [0.0, 0.5, 1.0]
    let v_values = scirs2_core::ndarray::Array1::<f64>::linspace(0.0, 1.0, 3);
    let surface_points = nurbs_surface.evaluate_array(&u_values.view(), &v_values.view(), true)?;

    println!("  Surface points at (u,v) grid [0.0, 0.5, 1.0] x [0.0, 0.5, 1.0]:");
    for i in 0..u_values.len() {
        for j in 0..v_values.len() {
            let idx = i * v_values.len() + j;
            println!(
                "    (u,v) = ({:.1},{:.1}): ({:.6}, {:.6}, {:.6})",
                u_values[i],
                v_values[j],
                surface_points[[idx, 0]],
                surface_points[[idx, 1]],
                surface_points[[idx, 2]]
            );
        }
    }

    Ok(())
}
