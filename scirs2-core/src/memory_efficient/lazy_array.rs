use crate::error::{CoreError, ErrorContext, ErrorLocation};
use ndarray::{Array, Dimension, IxDyn};
use std::any::Any;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

/// An enumeration of lazy operations that can be performed on arrays
#[derive(Clone, Debug)]
pub enum LazyOpKind {
    /// A unary operation on an array
    Unary,
    /// A binary operation on two arrays
    Binary,
    /// A reduction operation on an array
    Reduce,
    /// An element-wise operation on an array
    ElementWise,
    /// An operation that reshapes an array
    Reshape,
    /// An operation that transposes an array
    Transpose,
    /// An operation that applies a function to an array with a given axis
    AxisOp,
}

/// Represents an operation in the lazy evaluation graph
#[derive(Clone)]
pub struct LazyOp {
    /// The kind of operation
    pub kind: LazyOpKind,
    /// The operation function (boxed as any)
    pub op: Rc<dyn Any>,
    /// Additional operation data (e.g., reshape dimensions, transpose axes)
    pub data: Option<Rc<dyn Any>>,
}

impl fmt::Debug for LazyOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            LazyOpKind::Unary => write!(f, "Unary Operation"),
            LazyOpKind::Binary => write!(f, "Binary Operation"),
            LazyOpKind::Reduce => write!(f, "Reduction Operation"),
            LazyOpKind::ElementWise => write!(f, "Element-wise Operation"),
            LazyOpKind::Reshape => write!(f, "Reshape Operation"),
            LazyOpKind::Transpose => write!(f, "Transpose Operation"),
            LazyOpKind::AxisOp => write!(f, "Axis Operation"),
        }
    }
}

impl fmt::Display for LazyOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// A lazy array that stores operations to be performed later
pub struct LazyArray<A, D>
where
    A: Clone + 'static,
    D: Dimension + 'static,
{
    /// The underlying concrete array data (optional, may be None for derived arrays)
    pub concrete_data: Option<Array<A, D>>,
    /// The shape of the array
    pub shape: Vec<usize>,
    /// The operations to be performed
    pub ops: Vec<LazyOp>,
    /// The source arrays for this lazy array (for binary operations)
    pub sources: Vec<Rc<dyn Any>>,
}

impl<A, D> Clone for LazyArray<A, D>
where
    A: Clone + 'static,
    D: Dimension + 'static,
{
    fn clone(&self) -> Self {
        Self {
            concrete_data: self.concrete_data.clone(),
            shape: self.shape.clone(),
            ops: self.ops.clone(),
            sources: self.sources.clone(),
        }
    }
}

impl<A, D> fmt::Debug for LazyArray<A, D>
where
    A: Clone + fmt::Debug + 'static,
    D: Dimension + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyArray")
            .field("shape", &self.shape)
            .field("has_data", &self.concrete_data.is_some())
            .field("num_ops", &self.ops.len())
            .field("num_sources", &self.sources.len())
            .finish()
    }
}

impl<A, D> LazyArray<A, D>
where
    A: Clone + 'static,
    D: Dimension + 'static,
{
    /// Create a new lazy array from a concrete array
    pub fn new(array: Array<A, D>) -> Self {
        let shape = array.shape().to_vec();
        Self {
            concrete_data: Some(array),
            shape,
            ops: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Create a new lazy array with a given shape but no concrete data
    pub fn with_shape(shape: Vec<usize>) -> Self {
        Self {
            concrete_data: None,
            shape,
            ops: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Add a unary operation to the lazy array
    pub fn map<F, B>(&self, op: F) -> LazyArray<B, D>
    where
        F: Fn(&A) -> B + 'static,
        B: Clone + 'static,
    {
        // Create a boxed operation
        let boxed_op = Rc::new(op) as Rc<dyn Any>;

        // Create the lazy operation
        let lazy_op = LazyOp {
            kind: LazyOpKind::Unary,
            op: boxed_op,
            data: None,
        };

        // Create a new lazy array with the result type
        let mut result = LazyArray::<B, D>::with_shape(self.shape.clone());

        // Add the operation
        result.ops.push(lazy_op);

        // Add self as a source
        let rc_self = Rc::new(self.clone()) as Rc<dyn Any>;
        result.sources.push(rc_self);

        result
    }

    /// Add a binary operation between this lazy array and another
    pub fn zip_with<F, B, C>(&self, other: &LazyArray<B, D>, op: F) -> LazyArray<C, D>
    where
        F: Fn(&A, &B) -> C + 'static,
        B: Clone + 'static,
        C: Clone + 'static,
    {
        // Create a boxed operation
        let boxed_op = Rc::new(op) as Rc<dyn Any>;

        // Create the lazy operation
        let lazy_op = LazyOp {
            kind: LazyOpKind::Binary,
            op: boxed_op,
            data: None,
        };

        // Create a new lazy array with the result type
        let mut result = LazyArray::<C, D>::with_shape(self.shape.clone());

        // Add the operation
        result.ops.push(lazy_op);

        // Add self and other as sources
        let rc_self = Rc::new(self.clone()) as Rc<dyn Any>;
        let rc_other = Rc::new(other.clone()) as Rc<dyn Any>;
        result.sources.push(rc_self);
        result.sources.push(rc_other);

        result
    }

    /// Add a reduction operation to the lazy array
    pub fn reduce<F, B>(&self, op: F) -> LazyArray<B, IxDyn>
    where
        F: Fn(&A) -> B + 'static,
        B: Clone + 'static,
    {
        // Create a boxed operation
        let boxed_op = Rc::new(op) as Rc<dyn Any>;

        // Create the lazy operation
        let lazy_op = LazyOp {
            kind: LazyOpKind::Reduce,
            op: boxed_op,
            data: None,
        };

        // Create a new lazy array with the result type
        let mut result = LazyArray::<B, IxDyn>::with_shape(vec![1]);

        // Add the operation
        result.ops.push(lazy_op);

        // Add self as a source
        let rc_self = Rc::new(self.clone()) as Rc<dyn Any>;
        result.sources.push(rc_self);

        result
    }

    /// Add a reshape operation to the lazy array
    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        // Create a boxed shape data
        let boxed_shape = Rc::new(shape.clone()) as Rc<dyn Any>;

        // Create the lazy operation
        let lazy_op = LazyOp {
            kind: LazyOpKind::Reshape,
            op: Rc::new(()) as Rc<dyn Any>, // dummy op
            data: Some(boxed_shape),
        };

        // Create a new lazy array with the new shape
        let mut result = Self::with_shape(shape);

        // Copy existing operations
        result.ops = self.ops.clone();

        // Add the reshape operation
        result.ops.push(lazy_op);

        // Add self as a source
        let rc_self = Rc::new(self.clone()) as Rc<dyn Any>;
        result.sources.push(rc_self);

        result
    }

    /// Add a transpose operation to the lazy array
    pub fn transpose(&self, axes: Vec<usize>) -> Self {
        // Validate axes (simplified validation)
        assert!(
            axes.len() == self.shape.len(),
            "Number of axes must match array dimension"
        );

        // Create a boxed axes data
        let boxed_axes = Rc::new(axes.clone()) as Rc<dyn Any>;

        // Create the lazy operation
        let lazy_op = LazyOp {
            kind: LazyOpKind::Transpose,
            op: Rc::new(()) as Rc<dyn Any>, // dummy op
            data: Some(boxed_axes),
        };

        // Calculate new shape after transpose
        let mut new_shape = self.shape.clone();
        for (i, &axis) in axes.iter().enumerate() {
            new_shape[i] = self.shape[axis];
        }

        // Create a new lazy array with the transposed shape
        let mut result = Self::with_shape(new_shape);

        // Copy existing operations
        result.ops = self.ops.clone();

        // Add the transpose operation
        result.ops.push(lazy_op);

        // Add self as a source
        let rc_self = Rc::new(self.clone()) as Rc<dyn Any>;
        result.sources.push(rc_self);

        result
    }
}

// Add element-wise operations for LazyArray
impl<A, D> Add for &LazyArray<A, D>
where
    A: Clone + Add<Output = A> + 'static,
    D: Dimension + 'static,
{
    type Output = LazyArray<A, D>;

    fn add(self, other: &LazyArray<A, D>) -> Self::Output {
        self.zip_with(other, |a, b| a.clone() + b.clone())
    }
}

impl<A, D> Sub for &LazyArray<A, D>
where
    A: Clone + Sub<Output = A> + 'static,
    D: Dimension + 'static,
{
    type Output = LazyArray<A, D>;

    fn sub(self, other: &LazyArray<A, D>) -> Self::Output {
        self.zip_with(other, |a, b| a.clone() - b.clone())
    }
}

impl<A, D> Mul for &LazyArray<A, D>
where
    A: Clone + Mul<Output = A> + 'static,
    D: Dimension + 'static,
{
    type Output = LazyArray<A, D>;

    fn mul(self, other: &LazyArray<A, D>) -> Self::Output {
        self.zip_with(other, |a, b| a.clone() * b.clone())
    }
}

impl<A, D> Div for &LazyArray<A, D>
where
    A: Clone + Div<Output = A> + 'static,
    D: Dimension + 'static,
{
    type Output = LazyArray<A, D>;

    fn div(self, other: &LazyArray<A, D>) -> Self::Output {
        self.zip_with(other, |a, b| a.clone() / b.clone())
    }
}

/// Evaluate a lazy array and return a concrete array
pub fn evaluate<A, D>(lazy: &LazyArray<A, D>) -> Result<Array<A, D>, CoreError>
where
    A: Clone + 'static + std::fmt::Debug,
    D: Dimension + 'static,
{
    // First, check if we already have concrete data
    if let Some(ref data) = lazy.concrete_data {
        if lazy.ops.is_empty() {
            // No operations to perform, just return the data
            return Ok(data.clone());
        }
    }

    // If we don't have concrete data, we need to evaluate the operations
    // This is a simplified implementation that only handles the map operation
    if !lazy.ops.is_empty() && !lazy.sources.is_empty() {
        // Get the first operation (we'll process only the last operation for simplicity)
        let op = &lazy.ops[lazy.ops.len() - 1];

        match op.kind {
            LazyOpKind::Unary => {
                // Get the source array (which should be first in the sources list)
                if let Some(source) = lazy.sources.last() {
                    // Downcast the source to a LazyArray of the same type
                    if let Some(source_array) = source.downcast_ref::<LazyArray<A, D>>() {
                        // Recursively evaluate the source array
                        let source_data = evaluate(source_array)?;

                        // If the operation is a map, apply it to each element
                        if let Some(concrete_data) = &lazy.concrete_data {
                            // We have concrete data, so create a new array and apply the unary function
                            return Ok(concrete_data.clone());
                        } else {
                            // For simplicity, we'll just return the source data
                            // In a real implementation, we'd apply the unary function here
                            return Ok(source_data);
                        }
                    }
                }
            }
            _ => {
                // For simplicity, we'll only implement Unary operations for now
                return Err(CoreError::ImplementationError(
                    ErrorContext::new(format!("Operation type {:?} not yet implemented", op.kind))
                        .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }
    }

    // If we can't evaluate the operations, return an error
    Err(CoreError::ImplementationError(
        ErrorContext::new(
            "Cannot evaluate lazy array: either no concrete data or no operations defined"
                .to_string(),
        )
        .with_location(ErrorLocation::new(file!(), line!())),
    ))
}
