use scirs2_core::ndarray::array;
use scirs2_linalg::{inv, solve};

#[test]
#[allow(dead_code)]
fn test_enhanced_error_singularmatrix() {
    // Test enhanced error messages for singular matrix
    let singular = array![[1.0, 2.0], [2.0, 4.0]];

    // Test with inverse
    let result = inv(&singular.view(), None);
    assert!(result.is_err());
    let error = result.unwrap_err();

    // Check that the error contains diagnostic information
    let error_str = format!("{}", error);
    println!("Actual error message for inv: {}", error_str);
    assert!(error_str.contains("Singular matrix"));
    assert!(error_str.contains("Matrix shape"));
    assert!(error_str.contains("Consider the following"));
    println!("Enhanced error message:\n{}", error_str);
}

#[test]
#[allow(dead_code)]
fn test_enhanced_error_solve() {
    // Test enhanced error messages for solve
    let singular = array![[1.0, 2.0], [2.0, 4.0]];
    let b = array![1.0, 2.0];

    let result = solve(&singular.view(), &b.view(), None);
    assert!(result.is_err());
    let error = result.unwrap_err();

    // Check that the error contains diagnostic information
    let error_str = format!("{}", error);
    println!("Actual error message for solve: {}", error_str);
    assert!(error_str.contains("Singular matrix"));
    assert!(error_str.contains("Matrix shape"));
    // For small matrices, solve() calls inv() internally, so we get inv's error message
    assert!(error_str.contains("Consider the following"));
    println!("Solve error message:\n{}", error_str);
}

#[test]
#[allow(dead_code)]
fn test_error_no_enhancement_for_dimension_error() {
    // Test that dimension errors don't get enhanced
    let non_square = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let result = inv(&non_square.view(), None);
    assert!(result.is_err());
    let error = result.unwrap_err();

    // Check that the error does not contain enhanced diagnostic information
    let error_str = format!("{}", error);
    assert!(error_str.contains("must be square"));
    // The dimension error may contain "Matrix shape" in the error message but not as diagnostic info
    // It should not contain hints or other diagnostic markers
    assert!(!error_str.contains("Hint:"));
}
