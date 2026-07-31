//! A small extension of [ndarray](https://github.com/rust-ndarray/ndarray)
//!
//! Mainly provides `array_gen`, which is a collection of array generator functions.
use crate::error::OpResult;
use crate::error_helpers::try_from_numeric;
use crate::ndarray;

use crate::Float;

/// alias for `scirs2_core::ndarray::Array<T, IxDyn>`
pub type NdArray<T> = scirs2_core::ndarray::Array<T, scirs2_core::ndarray::IxDyn>;

/// alias for `scirs2_core::ndarray::ArrayView<T, IxDyn>`
pub type NdArrayView<'a, T> = scirs2_core::ndarray::ArrayView<'a, T, scirs2_core::ndarray::IxDyn>;

/// alias for `scirs2_core::ndarray::RawArrayView<T, IxDyn>`
pub type RawNdArrayView<T> = scirs2_core::ndarray::RawArrayView<T, scirs2_core::ndarray::IxDyn>;

/// alias for `scirs2_core::ndarray::RawArrayViewMut<T, IxDyn>`
pub type RawNdArrayViewMut<T> =
    scirs2_core::ndarray::RawArrayViewMut<T, scirs2_core::ndarray::IxDyn>;

/// alias for `scirs2_core::ndarray::ArrayViewMut<T, IxDyn>`
pub type NdArrayViewMut<'a, T> =
    scirs2_core::ndarray::ArrayViewMut<'a, T, scirs2_core::ndarray::IxDyn>;

#[inline]
/// This works well only for small arrays
pub(crate) fn asshape<T: Float>(x: &NdArrayView<T>) -> Vec<usize> {
    x.iter().map(|a| a.to_usize().unwrap_or(0)).collect()
}

#[inline]
pub(crate) fn expand_dims<T: Float>(x: NdArray<T>, axis: usize) -> NdArray<T> {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape_with_order(shape)
        .expect("Shape conversion failed - this is a bug")
}

#[inline]
pub(crate) fn roll_axis<T: Float>(
    arg: &mut NdArray<T>,
    to: scirs2_core::ndarray::Axis,
    from: scirs2_core::ndarray::Axis,
) {
    let i = to.index();
    let mut j = from.index();
    if j > i {
        while i != j {
            arg.swap_axes(i, j);
            j -= 1;
        }
    } else {
        while i != j {
            arg.swap_axes(i, j);
            j += 1;
        }
    }
}

#[inline]
pub(crate) fn normalize_negative_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

#[inline]
pub(crate) fn normalize_negative_axes<T: Float>(axes: &NdArrayView<T>, ndim: usize) -> Vec<usize> {
    let mut axes_ret: Vec<usize> = Vec::with_capacity(axes.len());
    for &axis in axes.iter() {
        let axis = if axis < T::zero() {
            (T::from(ndim).unwrap_or_else(|| T::zero()) + axis)
                .to_usize()
                .unwrap_or(0)
        } else {
            axis.to_usize().unwrap_or(0)
        };
        axes_ret.push(axis);
    }
    axes_ret
}

#[inline]
pub(crate) fn sparse_to_dense<T: Float>(arr: &NdArrayView<T>) -> Vec<usize> {
    let mut axes: Vec<usize> = vec![];
    for (i, &a) in arr.iter().enumerate() {
        if a == T::one() {
            axes.push(i);
        }
    }
    axes
}

#[allow(unused)]
#[inline]
pub(crate) fn is_fully_transposed(strides: &[scirs2_core::ndarray::Ixs]) -> bool {
    let mut ret = true;
    for w in strides.windows(2) {
        if w[0] > w[1] {
            ret = false;
            break;
        }
    }
    ret
}

/// Creates a zero array in the specified shape.
#[inline]
#[allow(dead_code)]
pub fn zeros<T: Float>(shape: &[usize]) -> NdArray<T> {
    NdArray::<T>::zeros(shape)
}

/// Creates a one array in the specified shape.
#[inline]
#[allow(dead_code)]
pub fn ones<T: Float>(shape: &[usize]) -> NdArray<T> {
    NdArray::<T>::ones(shape)
}

/// Creates a constant array in the specified shape.
#[inline]
#[allow(dead_code)]
pub fn constant<T: Float>(value: T, shape: &[usize]) -> NdArray<T> {
    NdArray::<T>::from_elem(shape, value)
}

use scirs2_core::random::{ChaCha8Rng, Rng, RngExt, SeedableRng, TryRng};

/// Random number generator for ndarray
///
/// Uses ChaCha8Rng internally because StdRng does not implement Clone in rand 0.10.
#[derive(Clone)]
pub struct ArrayRng<A> {
    rng: ChaCha8Rng,
    _phantom: std::marker::PhantomData<A>,
}

// Implement TryRng for ArrayRng by delegating to the internal ChaCha8Rng.
// In rand_core 0.10, TryRng<Error=Infallible> auto-provides Rng and (deprecated) RngCore.
impl<A> TryRng for ArrayRng<A> {
    type Error = std::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.rng.next_u32())
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.rng.next_u64())
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Self::Error> {
        self.rng.fill_bytes(dest);
        Ok(())
    }
}

impl<A: Float> ArrayRng<A> {
    /// Creates a new random number generator with the default seed.
    pub fn new() -> Self {
        Self::from_seed(0)
    }

    /// Creates a new random number generator with the specified seed.
    pub fn from_seed(seed: u64) -> Self {
        let rng = ChaCha8Rng::seed_from_u64(seed);
        Self {
            rng,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a reference to the internal RNG
    pub fn as_rng(&self) -> &ChaCha8Rng {
        &self.rng
    }

    /// Returns a mutable reference to the internal RNG
    pub fn as_rng_mut(&mut self) -> &mut ChaCha8Rng {
        &mut self.rng
    }

    /// Creates a uniform random array in the specified shape.
    /// Values are in the range [0, 1)
    pub fn random(&mut self, shape: &[usize]) -> NdArray<A> {
        let len = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(
                A::from(self.rng.random::<f64>()).expect("Shape conversion failed - this is a bug"),
            );
        }
        NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), data)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"))
    }

    /// Creates a normal random array in the specified shape.
    /// Values are drawn from a normal distribution with the specified mean and standard deviation.
    pub fn normal(&mut self, shape: &[usize], mean: f64, std: f64) -> NdArray<A> {
        use scirs2_core::random::{Distribution, Normal};
        let normal = Normal::new(mean, std)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"));
        let len = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(
                A::from(normal.sample(&mut self.rng))
                    .expect("Shape conversion failed - this is a bug"),
            );
        }
        NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), data)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"))
    }

    /// Creates a uniform random array in the specified shape.
    /// Values are in the range [low, high).
    pub fn uniform(&mut self, shape: &[usize], low: f64, high: f64) -> NdArray<A> {
        use scirs2_core::random::{Distribution, Uniform};
        let uniform = Uniform::new(low, high)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"));
        let len = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(
                A::from(uniform.sample(&mut self.rng))
                    .expect("Shape conversion failed - this is a bug"),
            );
        }
        NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), data)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"))
    }

    /// Creates a random array with Glorot/Xavier uniform initialization.
    /// For a tensor with shape (in_features, out_features),
    /// samples are drawn from Uniform(-sqrt(6/(in_features+out_features)), sqrt(6/(in_features+out_features))).
    pub fn glorot_uniform(&mut self, shape: &[usize]) -> NdArray<A> {
        assert!(shape.len() >= 2, "shape must have at least 2 dimensions");
        let fan_in = shape[shape.len() - 2];
        let fan_out = shape[shape.len() - 1];
        let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();
        self.uniform(shape, -scale, scale)
    }

    /// Creates a random array with Glorot/Xavier normal initialization.
    /// For a tensor with shape (in_features, out_features),
    /// samples are drawn from Normal(0, sqrt(2/(in_features+out_features))).
    pub fn glorot_normal(&mut self, shape: &[usize]) -> NdArray<A> {
        assert!(shape.len() >= 2, "shape must have at least 2 dimensions");
        let fan_in = shape[shape.len() - 2];
        let fan_out = shape[shape.len() - 1];
        let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
        self.normal(shape, 0.0, scale)
    }

    /// Creates a random array with He/Kaiming uniform initialization.
    /// For a tensor with shape (in_features, out_features),
    /// samples are drawn from Uniform(-sqrt(6/in_features), sqrt(6/in_features)).
    pub fn he_uniform(&mut self, shape: &[usize]) -> NdArray<A> {
        assert!(shape.len() >= 2, "shape must have at least 2 dimensions");
        let fan_in = shape[shape.len() - 2];
        let scale = (6.0 / fan_in as f64).sqrt();
        self.uniform(shape, -scale, scale)
    }

    /// Creates a random array with He/Kaiming normal initialization.
    /// For a tensor with shape (in_features, out_features),
    /// samples are drawn from Normal(0, sqrt(2/in_features)).
    pub fn he_normal(&mut self, shape: &[usize]) -> NdArray<A> {
        assert!(shape.len() >= 2, "shape must have at least 2 dimensions");
        let fan_in = shape[shape.len() - 2];
        let scale = (2.0 / fan_in as f64).sqrt();
        self.normal(shape, 0.0, scale)
    }

    /// Creates a random array from the standard normal distribution.
    pub fn standard_normal(&mut self, shape: &[usize]) -> NdArray<A> {
        self.normal(shape, 0.0, 1.0)
    }

    /// Creates a random array from the standard uniform distribution.
    pub fn standard_uniform(&mut self, shape: &[usize]) -> NdArray<A> {
        self.uniform(shape, 0.0, 1.0)
    }

    /// Creates a random array from the bernoulli distribution.
    pub fn bernoulli(&mut self, shape: &[usize], p: f64) -> NdArray<A> {
        use scirs2_core::random::{Bernoulli, Distribution};
        let bernoulli =
            Bernoulli::new(p).unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"));
        let len = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let val = if bernoulli.sample(&mut self.rng) {
                A::one()
            } else {
                A::zero()
            };
            data.push(val);
        }
        NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), data)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"))
    }

    /// Creates a random array from the exponential distribution.
    pub fn exponential(&mut self, shape: &[usize], lambda: f64) -> NdArray<A> {
        use scirs2_core::random::{Distribution, Exp};
        let exp =
            Exp::new(lambda).unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"));
        let len = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(
                A::from(exp.sample(&mut self.rng))
                    .expect("Shape conversion failed - this is a bug"),
            );
        }
        NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), data)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"))
    }

    /// Creates a random array from the log-normal distribution.
    pub fn log_normal(&mut self, shape: &[usize], mean: f64, stddev: f64) -> NdArray<A> {
        use scirs2_core::random::{Distribution, LogNormal};
        let log_normal = LogNormal::new(mean, stddev)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"));
        let len = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(
                A::from(log_normal.sample(&mut self.rng))
                    .expect("Shape conversion failed - this is a bug"),
            );
        }
        NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), data)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"))
    }

    /// Creates a random array from the gamma distribution.
    pub fn gamma(&mut self, shape: &[usize], shape_param: f64, scale: f64) -> NdArray<A> {
        use scirs2_core::random::{Distribution, Gamma};
        let gamma = Gamma::new(shape_param, scale)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"));
        let len = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(
                A::from(gamma.sample(&mut self.rng))
                    .expect("Shape conversion failed - this is a bug"),
            );
        }
        NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(shape), data)
            .unwrap_or_else(|_| panic!("Shape conversion failed - this is a bug"))
    }
}

impl<A: Float> Default for ArrayRng<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Check whether `shape` is the empty shape `[]`, i.e. the shape of a
/// genuinely 0-dimensional (rank-0) array.
///
/// A rank-0 array is the *only* shape that can be read with the
/// zero-length index `scirs2_core::ndarray::IxDyn(&[])`; every call site in
/// this crate that branches on `is_scalarshape` before doing exactly that
/// read relies on this.
///
/// This deliberately does **not** also match `[1]` (a rank-1 array holding
/// a single element), even though a previous version of this predicate did
/// (and its doc comment said so). `[1]` is *not* index-compatible with
/// `IxDyn(&[])`: indexing a rank-1 array with a zero-length index is a
/// dimensionality mismatch and panics. Treating `[1]` as scalar caused
/// exactly that panic in the comparison ops (`Equal`, `NotEqual`,
/// `Greater`, `Lesser`, ... in `tensor_ops::math_ops`) and in `DivOp`
/// whenever an operand happened to have shape `[1]`:
/// `is_scalarshape(&[1])` returned `true`, so the caller took the
/// "extract the scalar via `x[IxDyn(&[])]`" branch and crashed instead of
/// falling through to ordinary (correct) broadcasting.
///
/// Callers that instead want "does this shape have exactly one element,
/// whatever its rank" (some broadcast/reduction shortcuts do) should
/// compare `shape.iter().product::<usize>() == 1` directly rather than
/// reusing this predicate -- ndarray's own `broadcast()` already
/// right-aligns and pads missing leading dimensions with an implicit size
/// of 1, so a shape-`[1]` operand still broadcasts correctly even when it
/// is no longer routed through the 0-d fast path here.
#[inline]
#[allow(dead_code)]
pub fn is_scalarshape(shape: &[usize]) -> bool {
    shape.is_empty()
}

/// Create a scalar shape (empty shape)
#[inline]
#[allow(dead_code)]
pub fn scalarshape() -> Vec<usize> {
    vec![]
}

/// Create an array from a scalar value
#[inline]
#[allow(dead_code)]
pub fn from_scalar<T: Float>(value: T) -> NdArray<T> {
    NdArray::<T>::from_elem(scirs2_core::ndarray::IxDyn(&[1]), value)
}

/// Get shape of an ndarray view
#[inline]
#[allow(dead_code)]
pub fn shape_of_view<T>(view: &NdArrayView<'_, T>) -> Vec<usize> {
    view.shape().to_vec()
}

/// Get shape of an ndarray
#[inline]
#[allow(dead_code)]
pub fn shape_of<T>(array: &NdArray<T>) -> Vec<usize> {
    array.shape().to_vec()
}

/// Get default random number generator
#[inline]
#[allow(dead_code)]
pub fn get_default_rng<A: Float>() -> ArrayRng<A> {
    ArrayRng::<A>::default()
}

/// Create a deep copy of an ndarray
#[inline]
#[allow(dead_code)]
pub fn deep_copy<T: Float + Clone>(array: &NdArrayView<'_, T>) -> NdArray<T> {
    array.to_owned()
}

/// Select elements from an array along an axis
#[inline]
#[allow(dead_code)]
pub fn select<T: Float + Clone>(
    array: &NdArrayView<'_, T>,
    axis: scirs2_core::ndarray::Axis,
    indices: &[usize],
) -> NdArray<T> {
    let mut shape = array.shape().to_vec();
    shape[axis.index()] = indices.len();

    let mut result = NdArray::<T>::zeros(scirs2_core::ndarray::IxDyn(&shape));

    for (i, &idx) in indices.iter().enumerate() {
        let slice = array.index_axis(axis, idx);
        result.index_axis_mut(axis, i).assign(&slice);
    }

    result
}

/// Check if two shapes are compatible for broadcasting
#[inline]
#[allow(dead_code)]
pub fn are_broadcast_compatible(shape1: &[usize], shape2: &[usize]) -> bool {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let min_len = std::cmp::min(len1, len2);

    for i in 0..min_len {
        let dim1 = shape1[len1 - 1 - i];
        let dim2 = shape2[len2 - 1 - i];
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }
    true
}

/// Compute the shape resulting from broadcasting two shapes together
#[inline]
#[allow(dead_code)]
pub fn broadcastshape(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
    if !are_broadcast_compatible(shape1, shape2) {
        return None;
    }

    let len1 = shape1.len();
    let len2 = shape2.len();
    let result_len = std::cmp::max(len1, len2);
    let mut result = Vec::with_capacity(result_len);

    for i in 0..result_len {
        let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };
        let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };
        result.push(std::cmp::max(dim1, dim2));
    }

    result.reverse();
    Some(result)
}

/// Array generation functions
pub mod array_gen {
    use super::*;

    /// Creates a zero array in the specified shape.
    #[inline]
    pub fn zeros<T: Float>(shape: &[usize]) -> NdArray<T> {
        NdArray::<T>::zeros(shape)
    }

    /// Creates a one array in the specified shape.
    #[inline]
    pub fn ones<T: Float>(shape: &[usize]) -> NdArray<T> {
        NdArray::<T>::ones(shape)
    }

    /// Creates a 2D identity matrix of the specified size.
    #[inline]
    pub fn eye<T: Float>(n: usize) -> NdArray<T> {
        let mut result = NdArray::<T>::zeros(scirs2_core::ndarray::IxDyn(&[n, n]));
        for i in 0..n {
            result[[i, i]] = T::one();
        }
        result
    }

    /// Creates a constant array in the specified shape.
    #[inline]
    pub fn constant<T: Float>(value: T, shape: &[usize]) -> NdArray<T> {
        NdArray::<T>::from_elem(shape, value)
    }

    /// Generates a random array in the specified shape with values between 0 and 1.
    pub fn random<T: Float>(shape: &[usize]) -> NdArray<T> {
        let mut rng = ArrayRng::<T>::default();
        rng.random(shape)
    }

    /// Generates a random normal array in the specified shape.
    pub fn randn<T: Float>(shape: &[usize]) -> NdArray<T> {
        let mut rng = ArrayRng::<T>::default();
        rng.normal(shape, 0.0, 1.0)
    }

    /// Creates a Glorot/Xavier uniform initialized array in the specified shape.
    pub fn glorot_uniform<T: Float>(shape: &[usize]) -> NdArray<T> {
        let mut rng = ArrayRng::<T>::default();
        rng.glorot_uniform(shape)
    }

    /// Creates a Glorot/Xavier normal initialized array in the specified shape.
    pub fn glorot_normal<T: Float>(shape: &[usize]) -> NdArray<T> {
        let mut rng = ArrayRng::<T>::default();
        rng.glorot_normal(shape)
    }

    /// Creates a He/Kaiming uniform initialized array in the specified shape.
    pub fn he_uniform<T: Float>(shape: &[usize]) -> NdArray<T> {
        let mut rng = ArrayRng::<T>::default();
        rng.he_uniform(shape)
    }

    /// Creates a He/Kaiming normal initialized array in the specified shape.
    pub fn he_normal<T: Float>(shape: &[usize]) -> NdArray<T> {
        let mut rng = ArrayRng::<T>::default();
        rng.he_normal(shape)
    }

    /// Creates an array with a linearly spaced sequence from start to end.
    pub fn linspace<T: Float>(start: T, end: T, num: usize) -> NdArray<T> {
        if num <= 1 {
            return if num == 0 {
                NdArray::<T>::zeros(scirs2_core::ndarray::IxDyn(&[0]))
            } else {
                NdArray::<T>::from_elem(scirs2_core::ndarray::IxDyn(&[1]), start)
            };
        }

        let step = (end - start) / T::from(num - 1).unwrap_or_else(|| T::one());
        let mut data = Vec::with_capacity(num);

        for i in 0..num {
            data.push(start + step * T::from(i).unwrap_or_else(|| T::zero()));
        }

        NdArray::<T>::from_shape_vec(scirs2_core::ndarray::IxDyn(&[num]), data)
            .expect("Shape conversion failed - this is a bug")
    }

    /// Creates an array of evenly spaced values within a given interval.
    pub fn arange<T: Float>(start: T, end: T, step: T) -> NdArray<T> {
        let size = ((end - start) / step).to_f64().unwrap_or(0.0).ceil() as usize;
        let mut data = Vec::with_capacity(size);

        let mut current = start;
        while current < end {
            data.push(current);
            current += step;
        }

        NdArray::<T>::from_shape_vec(scirs2_core::ndarray::IxDyn(&[data.len()]), data)
            .expect("Shape conversion failed - this is a bug")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_scalarshape_matches_rank_zero_only() {
        // The bug: `[1]` used to be misclassified as scalar, which then made
        // callers index it with the rank-0 index `IxDyn(&[])` and panic.
        assert!(is_scalarshape(&[]));
        assert!(!is_scalarshape(&[1]));
        assert!(!is_scalarshape(&[1, 1]));
        assert!(!is_scalarshape(&[3]));
        assert!(!is_scalarshape(&[0]));
    }

    /// Regression test for the `Equal`/`NotEqual`/... comparison ops
    /// (`tensor_ops::math_ops::impl_cmp_op!`): before the `is_scalarshape`
    /// fix, a shape-`[1]` operand was misidentified as scalar, so the op
    /// indexed it with the rank-0 index `IxDyn(&[])` and panicked. Uses
    /// non-constant data so a fabricated/garbage result would be caught,
    /// not just a crash.
    #[test]
    fn equal_on_shape_one_operands_does_not_panic_and_is_correct() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let a = crate::tensor_ops::convert_to_tensor(
                NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[1]), vec![5.0f64])
                    .expect("Operation failed"),
                ctx,
            );
            let b = crate::tensor_ops::convert_to_tensor(
                NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[1]), vec![7.0f64])
                    .expect("Operation failed"),
                ctx,
            );

            let eq_self = crate::tensor_ops::equal(a, a);
            let result_self = eq_self.eval(ctx).expect("Operation failed");
            assert_eq!(result_self.shape(), &[1]);
            assert_eq!(
                result_self.iter().copied().collect::<Vec<_>>(),
                vec![1.0f64]
            );

            let eq_diff = crate::tensor_ops::equal(a, b);
            let result_diff = eq_diff.eval(ctx).expect("Operation failed");
            assert_eq!(
                result_diff.iter().copied().collect::<Vec<_>>(),
                vec![0.0f64]
            );
        });
    }

    /// Regression test for `DivOp` on shape-`[1]` operands (same root cause
    /// as above). Uses distinct non-constant values so a wrong (e.g.
    /// fabricated-zero or mismatched-shape) result would be caught.
    #[test]
    fn div_on_shape_one_operands_does_not_panic_and_is_correct() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let a = crate::tensor_ops::convert_to_tensor(
                NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[1]), vec![10.0f64])
                    .expect("Operation failed"),
                ctx,
            );
            let b = crate::tensor_ops::convert_to_tensor(
                NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[1]), vec![4.0f64])
                    .expect("Operation failed"),
                ctx,
            );

            let result = crate::tensor_ops::div(a, b)
                .eval(ctx)
                .expect("Operation failed");
            assert_eq!(result.shape(), &[1]);
            assert_eq!(result.iter().copied().collect::<Vec<_>>(), vec![2.5f64]);
        });
    }

    /// A shape-`[1]` operand against a larger tensor must still broadcast
    /// correctly (not just avoid panicking) now that it is no longer routed
    /// through the 0-d fast path.
    #[test]
    fn div_broadcasts_shape_one_against_larger_tensor() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let a = crate::tensor_ops::convert_to_tensor(
                NdArray::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[3]),
                    vec![10.0f64, 20.0, 30.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let b = crate::tensor_ops::convert_to_tensor(
                NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[1]), vec![2.0f64])
                    .expect("Operation failed"),
                ctx,
            );

            let result = crate::tensor_ops::div(a, b)
                .eval(ctx)
                .expect("Operation failed");
            assert_eq!(
                result.iter().copied().collect::<Vec<_>>(),
                vec![5.0f64, 10.0, 15.0]
            );
        });
    }
}
