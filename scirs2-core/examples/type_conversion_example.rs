use num_complex::Complex64;
use scirs2_core::types::{ComplexExt, ComplexOps, NumericConversion};

fn main() {
    println!("Type Conversion Example");

    // Only run the example if the types feature is enabled
    #[cfg(feature = "types")]
    {
        println!("\n--- Numeric Conversion Example ---");
        numeric_conversion_example();

        println!("\n--- Numeric Conversion with Error Handling ---");
        numeric_conversion_error_example();

        println!("\n--- Clamping and Rounding Example ---");
        clamping_rounding_example();

        println!("\n--- Complex Number Operations ---");
        complex_operations_example();

        println!("\n--- Complex Number Conversion ---");
        complex_conversion_example();

        println!("\n--- Batch Conversion Example ---");
        batch_conversion_example();
    }

    #[cfg(not(feature = "types"))]
    println!("Types feature not enabled. Run with --features=\"types\" to see the example.");
}

#[cfg(feature = "types")]
fn numeric_conversion_example() {
    // Convert between numeric types
    let float_value: f64 = 42.5;

    // Convert to integer (truncates)
    let int_result: i32 = float_value.to_numeric().unwrap();
    println!("f64 {} -> i32 {}", float_value, int_result);

    // Convert integer to float (exact)
    let int_value: i32 = 42;
    let float_result: f64 = int_value.to_numeric().unwrap();
    println!("i32 {} -> f64 {}", int_value, float_result);

    // Convert between integer types
    let large_int: i64 = 1_000_000;
    let medium_int: i32 = large_int.to_numeric().unwrap();
    println!("i64 {} -> i32 {}", large_int, medium_int);
}

#[cfg(feature = "types")]
fn numeric_conversion_error_example() {
    // Try to convert a value that's too large for the target type
    let large_value: f64 = 1e20;
    let result: Result<i32, _> = large_value.to_numeric();

    match result {
        Ok(value) => println!("Conversion succeeded: {}", value),
        Err(err) => println!("Conversion failed: {}", err),
    }

    // Try to convert a negative value to an unsigned type
    let negative_value: i32 = -5;
    let result: Result<u32, _> = negative_value.to_numeric();

    match result {
        Ok(value) => println!("Conversion succeeded: {}", value),
        Err(err) => println!("Conversion failed: {}", err),
    }

    // Try to convert a float with fractional part to an integer
    let fraction_value: f64 = 42.75;
    let result: Result<i32, _> = fraction_value.to_numeric();

    match result {
        Ok(value) => println!("Conversion succeeded: {}", value),
        Err(err) => println!(
            "Conversion failed (expected precision loss warning): {}",
            err
        ),
    }
}

#[cfg(feature = "types")]
fn clamping_rounding_example() {
    // Demonstrate clamping for out-of-range values
    let too_large: f64 = 1e10;
    let clamped: i32 = too_large.to_numeric_clamped();
    println!("Clamping {} to i32: {}", too_large, clamped);

    let too_small: f64 = -1e10;
    let clamped: i32 = too_small.to_numeric_clamped();
    println!("Clamping {} to i32: {}", too_small, clamped);

    // Demonstrate rounding for fractional values
    let fractional: f64 = 42.3;
    let rounded: i32 = fractional.to_numeric_rounded();
    println!("Rounding {} to i32: {}", fractional, rounded);

    let fractional: f64 = 42.7;
    let rounded: i32 = fractional.to_numeric_rounded();
    println!("Rounding {} to i32: {}", fractional, rounded);
}

#[cfg(feature = "types")]
fn complex_operations_example() {
    // Create complex numbers
    let z1 = Complex64::new(3.0, 4.0);
    let z2 = Complex64::new(1.0, 2.0);

    // Basic operations
    println!("z1 = {}", z1.to_algebraic_string());
    println!("z2 = {}", z2.to_algebraic_string());
    println!("z1 + z2 = {}", (z1 + z2).to_algebraic_string());
    println!("z1 * z2 = {}", (z1 * z2).to_algebraic_string());

    // Additional operations from ComplexOps trait
    println!("Magnitude of z1: {:.4}", z1.magnitude());
    println!("Phase of z1: {:.4} rad", z1.phase());
    println!("Distance between z1 and z2: {:.4}", z1.distance(z2));

    // Convert to polar form
    let (mag, phase) = z1.to_polar();
    println!("z1 in polar form: {:.4}∠{:.4}rad", mag, phase);
    println!("z1 in polar string: {}", z1.to_polar_string());

    // Normalize z1 (make its magnitude 1)
    let z_normalized = z1.normalize();
    println!("Normalized z1: {}", z_normalized.to_algebraic_string());
    println!(
        "Magnitude of normalized z1: {:.4}",
        z_normalized.magnitude()
    );

    // Rotate z1 by 90 degrees (π/2 radians)
    let z_rotated = z1.rotate(std::f64::consts::PI / 2.0);
    println!("z1 rotated by 90°: {}", z_rotated.to_algebraic_string());
}

#[cfg(feature = "types")]
fn complex_conversion_example() {
    // Create a complex number with f64 components
    let z64 = Complex64::new(123.456, 789.012);
    println!(
        "Original complex number (Complex64): {}",
        z64.to_algebraic_string()
    );

    // Convert to Complex32
    let z32 = z64.convert_complex::<f32>().unwrap();
    println!("Converted to Complex32: {}", z32);

    // Convert back to Complex64
    let z64_back = z32.convert_complex::<f64>().unwrap();
    println!(
        "Converted back to Complex64: {}",
        z64_back.to_algebraic_string()
    );

    // Try to convert a complex number with very large components
    let large_z = Complex64::new(1e40, 1e40);
    let result = large_z.convert_complex::<f32>();

    match result {
        Ok(z) => println!("Conversion succeeded: {}", z),
        Err(err) => println!("Conversion failed (expected): {}", err),
    }
}

#[cfg(feature = "types")]
fn batch_conversion_example() {
    use scirs2_core::types::convert;

    // Convert a slice of values
    let float_values = vec![1.1, 2.2, 3.3, 4.4, 5.5];

    // Convert to integers with error checking
    let int_result: Result<Vec<i32>, _> = convert::slice_to_numeric(&float_values);
    match int_result {
        Ok(ints) => println!("Slice conversion with error checking: {:?}", ints),
        Err(err) => println!("Conversion failed: {}", err),
    }

    // Convert with clamping (no errors)
    let clamped_ints = convert::slice_to_numeric_clamped::<_, i32>(&float_values);
    println!("Slice conversion with clamping: {:?}", clamped_ints);

    // Convert a slice of complex numbers
    let complex_values = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
    ];

    // Convert complex values to a different complex type
    let converted_result = convert::complex_slice_to_complex::<f64, f32>(&complex_values);
    match converted_result {
        Ok(converted) => println!("Complex slice conversion: {:?}", converted),
        Err(err) => println!("Complex conversion failed: {}", err),
    }

    // Convert real slice to complex slice
    let real_values = vec![1.0, 2.0, 3.0, 4.0];
    let complex_result = convert::real_to_complex::<f64, f64>(&real_values);
    match complex_result {
        Ok(complex_vals) => println!("Real to complex conversion: {:?}", complex_vals),
        Err(err) => println!("Real to complex conversion failed: {}", err),
    }

    // Convenience functions for common conversions
    let z32 = num_complex::Complex32::new(1.0, 2.0);
    let z64 = convert::complex32_to_complex64(z32);
    println!("Quick conversion from Complex32 to Complex64: {}", z64);

    let back_to_32 = convert::complex64_to_complex32(z64);
    println!(
        "Quick conversion from Complex64 to Complex32: {}",
        back_to_32
    );
}
