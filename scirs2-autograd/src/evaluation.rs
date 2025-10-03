use crate::ndarray_ext::{NdArray, NdArrayView, RawNdArrayView};

use crate::tensor::Tensor;
use crate::{Context, Graph};
use crate::{EvalError, Float};

use std::collections::HashMap;

/// Unique id for a placeholder tensor
#[derive(Clone, Copy)]
pub enum PlaceholderKey {
    Name(&'static str),
    ID(usize),
}

/// placeholder name or `Tensor` itself.
pub trait Placeholder {
    fn key(&self) -> PlaceholderKey;
}

#[allow(clippy::needless_lifetimes)]
impl<'g, F: Float> Placeholder for Tensor<'g, F> {
    fn key(&self) -> PlaceholderKey {
        PlaceholderKey::ID(self.id)
    }
}

impl Placeholder for &'static str {
    fn key(&self) -> PlaceholderKey {
        PlaceholderKey::Name(self)
    }
}

/// Helper structure for tensor evaluations.
///
/// `Evaluator` can buffer evaluation targets with useful `push` and `extend` functions
/// and runs batched evaluation.
/// You can also use `feed` method to feed NdArrays to placeholders.
///
///    ```
/// use scirs2_autograd as ag;
///
/// ag::run(|ctx| {
///    let a = ctx.placeholder("a", &[]);
///    let x = a + a;
///    let y = a * a;
///    let z = a / a;
///
///    let result = ctx.evaluator()
///        .extend(&[x, y, z])
///        .feed(a, scirs2_core::ndarray::arr0(2.).view().into_dyn())
///        .run();
///    println!("{:?}", result);
/// });
///    ```
pub struct Evaluator<'c, 'g, F: Float> {
    targets: Vec<&'g Tensor<'g, F>>,
    ctx: &'c Context<'g, F>,
    feeder: Option<Feeder<'g, F>>,
}

// public APIs
impl<'c, 'g, F: Float> Evaluator<'c, 'g, F> {
    /// Creates new Evaluator with Context.
    pub fn new(ctx: &'c Context<'g, F>) -> Self {
        Evaluator {
            targets: vec![],
            ctx,
            feeder: None,
        }
    }

    /// Makes a copy of self with different targets.
    pub fn fork(&self) -> Self {
        Self {
            targets: vec![],
            ctx: self.ctx,
            feeder: self.feeder.clone(),
        }
    }

    /// Registers `tensor` as an evaluation target.
    pub fn push<'t, T>(mut self, tensor: &'t T) -> Self
    where
        T: AsRef<Tensor<'g, F>> + 'g,
        't: 'g,
    {
        // Get the reference with the right lifetime
        let tensor_ref = tensor.as_ref();
        // Store the reference
        self.targets.push(tensor_ref);
        self
    }

    /// Registers `tensors` as evaluation targets.
    pub fn extend<'t, T>(mut self, tensors: &'t [T]) -> Self
    where
        T: AsRef<Tensor<'g, F>> + 'g,
        't: 'g,
    {
        for t in tensors {
            let tensor_ref = t.as_ref();
            self.targets.push(tensor_ref);
        }
        self
    }

    /// Sets `feeder`.
    pub fn set_feeder(mut self, feeder: Feeder<'g, F>) -> Self {
        self.feeder = Some(feeder);
        self
    }

    /// Feeds a placeholder tensor with a value.
    pub fn feed<P: Placeholder, V: Into<NdArrayView<'g, F>>>(
        mut self,
        placeholder: P,
        value: V,
    ) -> Self {
        if let Some(ref mut f) = self.feeder {
            let cloned_f = f.clone();
            *f = cloned_f.push(placeholder, value);
        } else {
            let f = Feeder::new();
            self.feeder = Some(f.push(placeholder, value));
        }
        self
    }

    /// Simple wrapper for `self.push(tensor).run()`.
    pub fn eval<T: AsRef<Tensor<'g, F>> + 'g>(self, tensor: T) -> Result<NdArray<F>, EvalError> {
        // Use clone here to avoid lifetime issues
        let tensor_ref = tensor.as_ref();
        let ret = self.push(tensor_ref).run();
        match ret.first() {
            Some(v) => match v {
                Ok(array) => Ok(array.to_owned()),
                Err(e) => Err(e.clone()),
            },
            None => Err(EvalError::Other("No tensor evaluated".to_string())),
        }
    }

    /// Consumes input feeds (placeholders) and tensors, and runs tensor operations.
    pub fn run(self) -> Vec<Result<NdArray<F>, EvalError>> {
        let mut ret = vec![];
        let mut placeholders = HashMap::new();

        // Prepare input feeds
        if let Some(feeder) = &self.feeder {
            for (key, array_, phantom) in &feeder.feeds {
                match key {
                    PlaceholderKey::Name(name) => {
                        if let Some(tid) = self.ctx.get_tensor_by_name(name) {
                            placeholders.insert(tid, array_);
                        } else {
                            ret.push(Err(EvalError::VariableError(format!(
                                "Placeholder not found: {name:?}"
                            ))));
                            return ret;
                        }
                    }
                    PlaceholderKey::ID(id) => {
                        placeholders.insert(*id, array_);
                    }
                }
            }
        }

        // Evaluate each tensor
        if self.targets.is_empty() {
            // If no target is specified, just return empty array
            return ret;
        }

        let results = Graph::eval_tensors(self.targets.as_slice(), &placeholders, self.ctx);
        for r in results {
            match r {
                Ok(array) => {
                    ret.push(Ok(array));
                }
                Err(e) => {
                    ret.push(Err(EvalError::OpError(e)));
                }
            }
        }
        ret
    }
}

/// `Feeder` contains placeholder-array pairs.
/// `Context::eval` consumes this.
///
/// You can add your placeholder-array pairs with `push` method.
///
///    ```
/// use scirs2_autograd as ag;
///
/// ag::run(|ctx| {
///     let a = ctx.placeholder("a", &[]);
///     let b = ctx.placeholder("b", &[]);
///     let expr = a * b;
///
///     let mut feeder = ag::Feeder::new();
///     let result = ctx.evaluator()
///         .push(&expr)
///         .set_feeder(feeder.push(a, scirs2_core::ndarray::arr0(10.).view().into_dyn()).push(b, scirs2_core::ndarray::arr0(20.).view().into_dyn()))
///         .run();
///     println!("{:?}", result[0]);  // => Ok(arr0(200.0))
/// });
///    ```
#[derive(Clone)]
pub struct Feeder<'g, F: Float> {
    feeds: Vec<(
        PlaceholderKey,
        RawNdArrayView<F>,
        std::marker::PhantomData<&'g ()>,
    )>,
}

#[allow(clippy::needless_lifetimes)]
impl<'g, F: Float> Default for Feeder<'g, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'g, F: Float> Feeder<'g, F> {
    /// Creates an empty Feeder.
    pub fn new() -> Self {
        Feeder { feeds: vec![] }
    }

    /// Adds a placeholder-value pair.
    pub fn push<P: Placeholder, V: Into<NdArrayView<'g, F>>>(
        mut self,
        placeholder: P,
        value: V,
    ) -> Self {
        let value = value.into();
        let key = placeholder.key();
        unsafe {
            let raw_view: RawNdArrayView<F> = std::mem::transmute(value);
            self.feeds.push((key, raw_view, std::marker::PhantomData));
        }
        self
    }
}
