//! Utility functions for HFFT operations
//!
//! This module contains helper functions for the Hermitian Fast Fourier Transform operations.

use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use std::fmt::Debug;

/// Try to convert a value to Complex64
///
/// This function attempts to convert different types to Complex64:
/// - Complex64 values are passed through
/// - Complex32 values are converted to Complex64
/// - Other complex types are parsed from their debug representation
/// - Primitive numeric types are converted to Complex64 with zero imaginary part
///
/// # Arguments
///
/// * `val` - The value to convert
///
/// # Returns
///
/// * `Some(Complex64)` if the conversion was successful
/// * `None` if the conversion failed
pub(crate) fn try_as_complex<T: Copy + Debug + 'static + NumCast>(val: T) -> Option<Complex64> {
    // Check if the value is a Complex64 directly
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        unsafe {
            let ptr = &val as *const T as *const Complex64;
            return Some(*ptr);
        }
    }

    // Check for complex32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<scirs2_core::numeric::Complex32>() {
        unsafe {
            let ptr = &val as *const T as *const scirs2_core::numeric::Complex32;
            let complex32 = *ptr;
            return Some(Complex64::new(complex32.re as f64, complex32.im as f64));
        }
    }

    // Handle other common complex number types by name-based detection
    // This is safer than trying to convert directly, as it avoids potential memory issues
    let type_name = std::any::type_name::<T>();
    if type_name.contains("Complex") {
        // For complex types, try to get the representation and parse it
        let debug_str = format!("{val:?}");

        // Try to extract re and im values using split and parse
        let re_im: Vec<f64> = debug_str
            .split(&[',', '(', ')', '{', '}', ':', ' '][..])
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();

        // If we found exactly two numbers, assume they're re and im
        if re_im.len() == 2 {
            return Some(Complex64::new(re_im[0], re_im[1]));
        }
    }

    // Handle primitive number types directly for better performance
    // For numeric primitives, we convert to Complex64 with zero imaginary part
    macro_rules! handle_primitive {
        ($type:ty) => {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$type>() {
                unsafe {
                    let ptr = &val as *const T as *const $type;
                    return Some(Complex64::new(*ptr as f64, 0.0));
                }
            }
        };
    }

    // Handle common numeric types
    handle_primitive!(f64);
    handle_primitive!(f32);
    handle_primitive!(i32);
    handle_primitive!(i64);
    handle_primitive!(u32);
    handle_primitive!(u64);
    handle_primitive!(i16);
    handle_primitive!(u16);
    handle_primitive!(i8);
    handle_primitive!(u8);

    // For other potential complex types, try to parse from Debug representation
    // This is a more robust approach for complex types from other libraries
    let debug_str = format!("{val:?}");
    if debug_str.contains("Complex") || (debug_str.contains("re") && debug_str.contains("im")) {
        // Extract numbers from the debug string
        let re_im: Vec<f64> = debug_str
            .split(&[',', '(', ')', '{', '}', ':', ' '][..])
            .filter_map(|s| {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    trimmed.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .collect();

        // Try different approaches to extract values
        if re_im.len() == 2 {
            // If we found exactly two numbers, assume they're re and im
            return Some(Complex64::new(re_im[0], re_im[1]));
        } else if debug_str.contains("re:") && debug_str.contains("im:") {
            // For more complex representations like { re: 1.0, im: 2.0 }
            let re_str = debug_str
                .split("re:")
                .nth(1)
                .and_then(|s| s.split(',').next());
            let im_str = debug_str
                .split("im:")
                .nth(1)
                .and_then(|s| s.split('}').next());

            if let (Some(re_s), Some(im_s)) = (re_str, im_str) {
                if let (Ok(re), Ok(im)) = (re_s.trim().parse::<f64>(), im_s.trim().parse::<f64>()) {
                    return Some(Complex64::new(re, im));
                }
            }
        }
    }

    // As a last resort, try generic NumCast conversion
    NumCast::from(val).map(|v| Complex64::new(v, 0.0))
}
