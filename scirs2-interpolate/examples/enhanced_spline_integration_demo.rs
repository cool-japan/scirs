//! Enhanced Spline Integration with Extrapolation Support Demo
//!
//! This example demonstrates the new `integrate_with_extrapolation` method
//! that provides SciPy-compatible integration beyond spline domain boundaries.

use scirs2_core::ndarray::Array1;
use scirs2_interpolate::error::InterpolateResult;
use scirs2_interpolate::interp1d::ExtrapolateMode;
use scirs2_interpolate::spline::CubicSpline;

#[allow(dead_code)]
fn main() -> InterpolateResult<()> {
    println!("=== Enhanced Spline Integration with Extrapolation Demo ===\n");

    // Create sample data: f(x) = x^2 from x=0 to x=3
    let x = Array1::from(vec![0.0, 1.0, 2.0, 3.0]);
    let y = Array1::from(vec![0.0, 1.0, 4.0, 9.0]);

    // Create a natural cubic spline
    let spline = CubicSpline::new(&x.view(), &y.view())?;

    println!("Created cubic spline for f(x) = x^2 over domain [0, 3]");
    println!("Data points: {:?}", x);
    println!("Values: {:?}\n", y);

    // 1. Standard integration within domain
    println!("1. Integration within spline domain:");
    let integral_within = spline.integrate(0.5, 2.5)?;
    println!("   ∫[0.5, 2.5] spline(x) dx = {:.6}", integral_within);

    // For comparison, analytical integral of x^2 from 0.5 to 2.5 is:
    // [x³/3] from 0.5 to 2.5 = (2.5³ - 0.5³)/3 = (15.625 - 0.125)/3 = 5.167
    println!(
        "   Expected (x² analytical): {:.6}\n",
        (2.5_f64.powi(3) - 0.5_f64.powi(3)) / 3.0
    );

    // 2. Integration with extrapolation beyond domain
    println!("2. Integration with extrapolation beyond domain:");

    // Try to integrate from -1 to 4 (extends beyond [0, 3] domain)
    // First, show the error behavior
    match spline.integrate_with_extrapolation(-1.0, 4.0, Some(ExtrapolateMode::Error)) {
        Ok(_) => println!("   Unexpected success with Error mode"),
        Err(e) => println!("   Error mode (expected): {}", e),
    }

    // Now with linear extrapolation
    let integral_linear_extrap =
        spline.integrate_with_extrapolation(-1.0, 4.0, Some(ExtrapolateMode::Extrapolate))?;
    println!(
        "   ∫[-1, 4] with linear extrapolation = {:.6}",
        integral_linear_extrap
    );

    // With constant extrapolation
    let integral_const_extrap =
        spline.integrate_with_extrapolation(-1.0, 4.0, Some(ExtrapolateMode::Nearest))?;
    println!(
        "   ∫[-1, 4] with constant extrapolation = {:.6}\n",
        integral_const_extrap
    );

    // 3. Demonstrate different extrapolation regions
    println!("3. Integration in different extrapolation regions:");

    // Left extrapolation only
    let left_extrap =
        spline.integrate_with_extrapolation(-2.0, -1.0, Some(ExtrapolateMode::Extrapolate))?;
    println!("   ∫[-2, -1] (left extrapolation) = {:.6}", left_extrap);

    // Right extrapolation only
    let right_extrap =
        spline.integrate_with_extrapolation(4.0, 5.0, Some(ExtrapolateMode::Extrapolate))?;
    println!("   ∫[4, 5] (right extrapolation) = {:.6}", right_extrap);

    // Mixed: left extrapolation + interior
    let mixed_left =
        spline.integrate_with_extrapolation(-1.0, 2.0, Some(ExtrapolateMode::Extrapolate))?;
    println!("   ∫[-1, 2] (left extrap + interior) = {:.6}", mixed_left);

    // Mixed: interior + right extrapolation
    let mixed_right =
        spline.integrate_with_extrapolation(1.0, 4.0, Some(ExtrapolateMode::Extrapolate))?;
    println!(
        "   ∫[1, 4] (interior + right extrap) = {:.6}\n",
        mixed_right
    );

    // 4. Compare constant vs linear extrapolation
    println!("4. Extrapolation method comparison:");

    let bounds = [(-1.0, 0.0), (3.0, 4.0)];
    let labels = ["Left extrapolation", "Right extrapolation"];

    for (i, &(a, b)) in bounds.iter().enumerate() {
        println!("   {} ∫[{}, {}]:", labels[i], a, b);

        let const_result =
            spline.integrate_with_extrapolation(a, b, Some(ExtrapolateMode::Nearest))?;
        let linear_result =
            spline.integrate_with_extrapolation(a, b, Some(ExtrapolateMode::Extrapolate))?;

        println!("     Constant extrapolation: {:.6}", const_result);
        println!("     Linear extrapolation:   {:.6}", linear_result);
        println!(
            "     Difference:             {:.6}",
            f64::abs(linear_result - const_result)
        );
    }

    println!("\n=== Demo Complete ===");
    println!("The enhanced integration method provides SciPy-compatible");
    println!("extrapolation capabilities while maintaining numerical stability.");

    Ok(())
}
