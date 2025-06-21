#[cfg(feature = "memory_efficient")]
mod tests {
    use scirs2_core::error::{CoreError, ErrorContext};
    use scirs2_core::memory_efficient::{register_fusion, FusedOp, OpFusion};
    use std::any::{Any, TypeId};
    use std::sync::Arc;

    // A simple operation for testing
    #[derive(Clone)]
    struct SquareOp;

    impl FusedOp for SquareOp {
        fn name(&self) -> &str {
            "SquareOp"
        }

        fn input_type(&self) -> TypeId {
            TypeId::of::<f64>()
        }

        fn output_type(&self) -> TypeId {
            TypeId::of::<f64>()
        }

        fn can_fuse_with(&self, other: &dyn FusedOp) -> bool {
            other.name() == "SqrtOp"
        }

        fn fuse_with(&self, other: &dyn FusedOp) -> Arc<dyn FusedOp> {
            if other.name() == "SqrtOp" {
                // Square followed by sqrt is identity
                Arc::new(IdentityOp)
            } else {
                self.clone_op()
            }
        }

        fn apply(&self, input: &dyn Any) -> Result<Box<dyn Any>, CoreError> {
            let x = input
                .downcast_ref::<f64>()
                .ok_or_else(|| CoreError::InvalidArgument(ErrorContext::new("Expected f64")))?;
            Ok(Box::new(x * x))
        }

        fn clone_op(&self) -> Arc<dyn FusedOp> {
            Arc::new(self.clone())
        }
    }

    // Another simple operation for testing
    #[derive(Clone)]
    struct SqrtOp;

    impl FusedOp for SqrtOp {
        fn name(&self) -> &str {
            "SqrtOp"
        }

        fn input_type(&self) -> TypeId {
            TypeId::of::<f64>()
        }

        fn output_type(&self) -> TypeId {
            TypeId::of::<f64>()
        }

        fn can_fuse_with(&self, other: &dyn FusedOp) -> bool {
            other.name() == "SquareOp"
        }

        fn fuse_with(&self, other: &dyn FusedOp) -> Arc<dyn FusedOp> {
            if other.name() == "SquareOp" {
                // Sqrt followed by square is identity
                Arc::new(IdentityOp)
            } else {
                self.clone_op()
            }
        }

        fn apply(&self, input: &dyn Any) -> Result<Box<dyn Any>, CoreError> {
            let x = input
                .downcast_ref::<f64>()
                .ok_or_else(|| CoreError::InvalidArgument(ErrorContext::new("Expected f64")))?;

            if *x < 0.0 {
                return Err(CoreError::InvalidArgument(ErrorContext::new(
                    "Cannot take sqrt of negative number",
                )));
            }

            Ok(Box::new(x.sqrt()))
        }

        fn clone_op(&self) -> Arc<dyn FusedOp> {
            Arc::new(self.clone())
        }
    }

    // Identity operation for testing fusion
    #[derive(Clone)]
    struct IdentityOp;

    impl FusedOp for IdentityOp {
        fn name(&self) -> &str {
            "IdentityOp"
        }

        fn input_type(&self) -> TypeId {
            TypeId::of::<f64>()
        }

        fn output_type(&self) -> TypeId {
            TypeId::of::<f64>()
        }

        fn can_fuse_with(&self, _other: &dyn FusedOp) -> bool {
            true
        }

        fn fuse_with(&self, other: &dyn FusedOp) -> Arc<dyn FusedOp> {
            other.clone_op()
        }

        fn apply(&self, input: &dyn Any) -> Result<Box<dyn Any>, CoreError> {
            let x = input
                .downcast_ref::<f64>()
                .ok_or_else(|| CoreError::InvalidArgument(ErrorContext::new("Expected f64")))?;
            Ok(Box::new(*x))
        }

        fn clone_op(&self) -> Arc<dyn FusedOp> {
            Arc::new(self.clone())
        }
    }

    #[test]
    fn test_op_fusion_creation() {
        let fusion = OpFusion::new();

        // New fusion should be empty
        assert!(fusion.is_empty());
        assert_eq!(fusion.num_ops(), 0);
    }

    #[test]
    fn test_op_fusion_add_op() {
        let mut fusion = OpFusion::new();

        // Add a square operation
        let square_op = Arc::new(SquareOp);
        fusion.add_op(square_op).unwrap();

        // Should now have one operation
        assert!(!fusion.is_empty());
        assert_eq!(fusion.num_ops(), 1);
    }

    #[test]
    fn test_op_fusion_type_mismatch() {
        let mut fusion = OpFusion::new();

        // Add a square operation
        let square_op = Arc::new(SquareOp);
        fusion.add_op(square_op).unwrap();

        // Try to add an operation with mismatched types
        struct MismatchOp;

        impl FusedOp for MismatchOp {
            fn name(&self) -> &str {
                "MismatchOp"
            }
            fn input_type(&self) -> TypeId {
                TypeId::of::<i32>()
            }
            fn output_type(&self) -> TypeId {
                TypeId::of::<i32>()
            }
            fn can_fuse_with(&self, _: &dyn FusedOp) -> bool {
                false
            }
            fn fuse_with(&self, _: &dyn FusedOp) -> Arc<dyn FusedOp> {
                Arc::new(MismatchOp)
            }
            fn apply(&self, _: &dyn Any) -> Result<Box<dyn Any>, CoreError> {
                Ok(Box::new(0))
            }
            fn clone_op(&self) -> Arc<dyn FusedOp> {
                Arc::new(MismatchOp)
            }
        }

        let mismatch_op = Arc::new(MismatchOp);
        let result = fusion.add_op(mismatch_op);

        // Should fail due to type mismatch
        assert!(result.is_err());
    }

    #[test]
    fn test_op_fusion_optimize() {
        let mut fusion = OpFusion::new();

        // Add square followed by sqrt, which should optimize to identity
        let square_op = Arc::new(SquareOp);
        let sqrt_op = Arc::new(SqrtOp);

        fusion.add_op(square_op).unwrap();
        fusion.add_op(sqrt_op).unwrap();

        // Before optimization
        assert_eq!(fusion.num_ops(), 2);

        // After optimization
        fusion.optimize().unwrap();

        // Should have combined into a single operation
        assert_eq!(fusion.num_ops(), 1);
    }

    #[test]
    fn test_op_fusion_apply() {
        let mut fusion = OpFusion::new();

        // Add square operation
        let square_op = Arc::new(SquareOp);
        fusion.add_op(square_op).unwrap();

        // Apply to input
        let input = 3.0;
        let result = fusion.apply(input).unwrap();

        // Downcast and check
        let output = result.downcast_ref::<f64>().unwrap();
        assert_eq!(*output, 9.0); // 3 squared = 9
    }

    #[test]
    fn test_op_fusion_register() {
        // Register a square operation for f64
        let square_op = Arc::new(SquareOp);
        register_fusion::<f64>(square_op).unwrap();

        // This is hard to test fully because the registry is global and we can't easily
        // query it directly, but at least we can verify registration doesn't error
    }

    #[test]
    fn test_empty_op_fusion_optimize() {
        let mut fusion = OpFusion::new();

        // Optimizing an empty fusion should succeed but do nothing
        fusion.optimize().unwrap();
        assert!(fusion.is_empty());
    }

    #[test]
    fn test_single_op_fusion_optimize() {
        let mut fusion = OpFusion::new();

        // Add a single operation
        let square_op = Arc::new(SquareOp);
        fusion.add_op(square_op).unwrap();

        // Optimizing a fusion with one op should succeed but do nothing
        fusion.optimize().unwrap();
        assert_eq!(fusion.num_ops(), 1);
    }

    #[test]
    fn test_op_fusion_apply_type_mismatch() {
        let mut fusion = OpFusion::new();

        // Add square operation for f64
        let square_op = Arc::new(SquareOp);
        fusion.add_op(square_op).unwrap();

        // Try to apply with wrong type
        let input = 3i32;
        let result = fusion.apply(input);

        // Should fail due to type mismatch
        assert!(result.is_err());
    }
}
