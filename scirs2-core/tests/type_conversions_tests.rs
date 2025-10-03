#[cfg(feature = "types")]
mod type_conversion_tests {
    use approx::assert_relative_eq;
    use scirs2_core::numeric::Complex64;
    use scirs2_core::types::{ComplexExt, ComplexOps, NumericConversion};

    #[test]
    fn test_valid_numeric_conversions() {
        // Convert between numeric types
        assert_eq!(42.0f64.to_numeric::<i32>().unwrap(), 42);
        assert_eq!((-42.0f64).to_numeric::<i32>().unwrap(), -42);
        assert_eq!(42i32.to_numeric::<f64>().unwrap(), 42.0);
        assert_eq!(42u8.to_numeric::<u16>().unwrap(), 42u16);
    }

    #[test]
    fn test_numeric_conversionerrors() {
        // Test overflow
        assert!(1e20f64.to_numeric::<i32>().is_err());

        // Test underflow
        assert!((-1e20f64).to_numeric::<i32>().is_err());

        // Test precision loss
        assert!(42.5f64.to_numeric::<i32>().is_err());

        // Test negative to unsigned conversion
        assert!((-1i32).to_numeric::<u32>().is_err());
    }

    #[test]
    fn test_numeric_clamping() {
        // Test clamping for out-of-range values
        assert_eq!(1e20f64.to_numeric_clamped::<i32>(), i32::MAX);
        assert_eq!((-1e20f64).to_numeric_clamped::<i32>(), i32::MIN);

        // Test valid conversions with clamping
        assert_eq!(42.0f64.to_numeric_clamped::<i32>(), 42);
    }

    #[test]
    fn test_numeric_rounding() {
        // Test rounding for fractional values
        assert_eq!(42.3f64.to_numeric_rounded::<i32>(), 42);
        assert_eq!(42.7f64.to_numeric_rounded::<i32>(), 43);
        assert_eq!((-42.7f64).to_numeric_rounded::<i32>(), -43);
    }

    #[test]
    fn test_complex_operations() {
        // Test complex number operations
        let z1 = Complex64::new(3.0, 4.0);
        let z2 = Complex64::new(1.0, 2.0);

        // Basic properties
        assert_relative_eq!(z1.magnitude(), 5.0, epsilon = 1e-10);
        assert_relative_eq!(z1.phase(), 0.9272952180016122, epsilon = 1e-10);

        // Distance
        assert_relative_eq!(z1.distance(z2), 2.82842712474619, epsilon = 1e-10);

        // Normalization
        let z_norm = z1.normalize();
        assert_relative_eq!(z_norm.magnitude(), 1.0, epsilon = 1e-10);

        // Rotation
        let z_rot = z1.rotate(std::f64::consts::PI / 2.0);
        assert_relative_eq!(z_rot.re, -4.0, epsilon = 1e-10);
        assert_relative_eq!(z_rot.im, 3.0, epsilon = 1e-10);

        // Polar form
        let (mag, phase) = z1.to_polar();
        let z_back = Complex64::from_polar(mag, phase);
        assert_relative_eq!(z_back.re, z1.re, epsilon = 1e-10);
        assert_relative_eq!(z_back.im, z1.im, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_conversion() {
        // Test complex number conversion
        let z1 = Complex64::new(3.0, 4.0);

        // Convert to Complex32
        let z2 = z1.convert_complex::<f32>().unwrap();
        assert_relative_eq!(z2.re, 3.0f32, epsilon = 1e-6);
        assert_relative_eq!(z2.im, 4.0f32, epsilon = 1e-6);

        // Convert back to Complex64
        let z3 = z2.convert_complex::<f64>().unwrap();
        assert_relative_eq!(z3.re, 3.0f64, epsilon = 1e-6);
        assert_relative_eq!(z3.im, 4.0f64, epsilon = 1e-6);
    }

    #[test]
    fn test_complex_conversionerrors() {
        // Test complex number conversion errors
        let large_z = Complex64::new(1e40, 1e40);

        // This should error on conversion to f32
        assert!(large_z.convert_complex::<f32>().is_err());
    }

    #[test]
    fn test_batch_conversions() {
        use scirs2_core::types::convert;

        // Test slice conversion
        let float_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let int_values = convert::slice_to_numeric::<_, i32>(&float_values).unwrap();

        assert_eq!(int_values, vec![1, 2, 3, 4, 5]);

        // Test slice conversion with clamping
        let large_values = vec![1e10, 2e10, 3e10];
        let clamped = convert::slice_to_numeric_clamped::<_, i32>(&large_values);

        assert_eq!(clamped, vec![i32::MAX, i32::MAX, i32::MAX]);

        // Test complex slice conversion
        let complex_values = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];

        let converted = convert::complex_slice_to_complex::<_, f32>(&complex_values).unwrap();

        assert_relative_eq!(converted[0].re, 1.0f32, epsilon = 1e-6);
        assert_relative_eq!(converted[0].im, 2.0f32, epsilon = 1e-6);
        assert_relative_eq!(converted[1].re, 3.0f32, epsilon = 1e-6);
        assert_relative_eq!(converted[1].im, 4.0f32, epsilon = 1e-6);

        // Test real to complex conversion
        let real_values = vec![1.0, 2.0, 3.0];
        let complex = convert::real_to_complex::<_, f64>(&real_values).unwrap();

        assert_relative_eq!(complex[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(complex[0].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(complex[1].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(complex[1].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(complex[2].re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(complex[2].im, 0.0, epsilon = 1e-10);
    }
}
