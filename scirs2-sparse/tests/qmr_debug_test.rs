use scirs2_sparse::error::SparseError;
use scirs2_sparse::linalg::{qmr, QMROptions};

#[test]
fn test_qmr_not_implemented() {
    // Test that QMR is not implemented yet
    let identity = scirs2_sparse::linalg::IdentityOperator::<f64>::new(3);
    let b = vec![1.0, 2.0, 3.0];
    let options = QMROptions::default();

    match qmr(&identity, &b, options) {
        Err(SparseError::NotImplemented(msg)) => {
            assert!(msg.contains("QMR solver is not yet implemented"));
        }
        _ => panic!("Expected NotImplemented error"),
    }
}
