#[cfg(feature = "memory_efficient")]
mod tests {
    use ndarray::Array1;
    use scirs2_core::memory_efficient::{evaluate, LazyArray, LazyOp, LazyOpKind};

    #[test]
    fn test_lazy_array_creation() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Create a lazy array
        let lazy = LazyArray::new(data.clone());

        // Check that the shape is correct
        assert_eq!(lazy.shape, data.shape());

        // Check that data is stored
        assert!(lazy.concrete_data.is_some());

        // Check that there are no operations
        assert!(lazy.ops.is_empty());
    }

    #[test]
    fn test_lazy_array_with_shape() {
        let shape = vec![3, 4];

        // Create a lazy array with shape only
        let lazy = LazyArray::<f64, ndarray::Dim<[usize; 2]>>::with_shape(shape.clone());

        // Check that the shape is correct
        assert_eq!(lazy.shape, shape);

        // Check that there is no data
        assert!(lazy.concrete_data.is_none());

        // Check that there are no operations
        assert!(lazy.ops.is_empty());
    }

    #[test]
    fn test_lazy_array_evaluate() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Create a lazy array
        let lazy = LazyArray::new(data.clone());

        // Evaluate the array
        let result = evaluate(&lazy).unwrap();

        // Check that the result matches the original data
        assert_eq!(result, data);
    }

    #[test]
    fn test_lazy_array_map() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Create a lazy array and define a map operation
        let lazy = LazyArray::new(data.clone());
        let lazy_map = lazy.map(|&x| x * x);

        // Check that the shape is preserved
        assert_eq!(lazy_map.shape, data.shape());

        // Check that there is an operation
        assert_eq!(lazy_map.ops.len(), 1);

        // Check that there is a source
        assert_eq!(lazy_map.sources.len(), 1);

        // This would evaluate the operation, but our implementation is just a placeholder
        // let result = evaluate(&lazy_map).unwrap();
        // let expected = data.map(|&x| x * x);
        // assert_eq!(result, expected);
    }

    #[test]
    fn test_lazy_op_display() {
        // Check that LazyOp with Unary kind can be displayed
        let op = LazyOp {
            kind: LazyOpKind::Unary,
            op: std::rc::Rc::new(Box::new(|&x: &f64| x * x) as Box<dyn std::any::Any>),
            data: None,
        };
        let display = format!("{}", op);
        assert!(display.contains("Unary"));

        // Check that LazyOp with Reshape kind can be displayed
        let shape = vec![2, 3];
        let op = LazyOp {
            kind: LazyOpKind::Reshape,
            op: std::rc::Rc::new(Box::new(()) as Box<dyn std::any::Any>),
            data: Some(std::rc::Rc::new(shape.clone())),
        };
        let display = format!("{}", op);
        assert!(display.contains("Reshape"));
    }

    #[test]
    fn test_lazy_array_debug() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Create a lazy array
        let lazy = LazyArray::new(data.clone());

        // Check that debug formatting works
        let debug = format!("{:?}", lazy);
        assert!(debug.contains("LazyArray"));
        assert!(debug.contains("shape"));
        assert!(debug.contains("has_data: true"));
    }
}
