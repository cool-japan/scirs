use crate::tensor::{Tensor, TensorInternal};

use crate::error::OpError;
use crate::ndarray_ext::{NdArrayView, RawNdArrayView};
use crate::op;
use crate::variable::{VariableID, VariableNamespace};
use crate::{tensor_ops as T, Evaluator};
use crate::{Float, NdArray, VariableEnvironment};
use std::collections::{HashMap, HashSet};

use std::cell::{Ref, RefCell, RefMut};
use std::fmt;
use std::ops::Deref;

pub type TensorID = usize;

/// Graph represents a computation graph holding tensors inside.
///
/// NOTE:
/// You won't be using this struct directly because this is generally accessed via `Context::deref()`.
pub struct Graph<F: Float> {
    pub(crate) node_set: RefCell<Vec<TensorInternal<F>>>,
    pub(crate) variable2node: RefCell<HashMap<VariableID, TensorID>>,
}

pub const NUM_NODES_WARN: usize = 50_000;
pub const NUM_NODES_CRITICAL: usize = 500_000;

impl<'graph, F: Float> Graph<F> {
    #[inline]
    pub fn eval_tensors(
        tensors: &[&Tensor<F>],
        feeds: &HashMap<TensorID, &RawNdArrayView<F>>,
        ctx: &Context<F>,
    ) -> Vec<Result<NdArray<F>, OpError>> {
        // The original tensors we want to compute values for
        let mut results = Vec::with_capacity(tensors.len());

        // Early return if there are no tensors to evaluate
        if tensors.is_empty() {
            return results;
        }

        // Collect all nodes needed for evaluation in topological order
        let mut eval_nodes = Vec::new();
        let mut visited = HashSet::new();

        // Helper function to collect nodes in topological order
        fn collect_nodes_topo<F: Float>(
            node_id: TensorID,
            graph: &Graph<F>,
            eval_nodes: &mut Vec<TensorID>,
            visited: &mut HashSet<TensorID>,
        ) {
            if visited.contains(&node_id) {
                return;
            }

            // Mark as visited to avoid cycles
            visited.insert(node_id);

            // Get the node's dependencies (incoming nodes)
            let incoming = graph.access_inner(node_id).incoming_nodes.clone();

            // Process dependencies first (depth-first)
            for incoming_node in &incoming {
                collect_nodes_topo(incoming_node.id, graph, eval_nodes, visited);
            }

            // Add this node after its dependencies
            eval_nodes.push(node_id);
        }

        // Collect nodes for all target tensors
        for tensor in tensors {
            collect_nodes_topo(tensor.id, ctx.as_graph(), &mut eval_nodes, &mut visited);
        }

        // Map to store computed values for each node
        let mut computed_values: HashMap<TensorID, NdArray<F>> = HashMap::new();

        // Add feed values to the computed values
        for (&id, &feed_view) in feeds.iter() {
            // Convert the RawNdArrayView back to a regular NdArrayView and then to owned array
            unsafe {
                let view: NdArrayView<F> = std::mem::transmute(feed_view.clone());
                let owned_array = view.to_owned();
                computed_values.insert(id, owned_array);
            }
        }

        // Evaluate nodes in topological order
        for node_id in eval_nodes {
            // Skip if already computed (e.g., from feeds)
            if computed_values.contains_key(&node_id) {
                continue;
            }

            let node = ctx.as_graph().access_inner(node_id);

            // If this is a variable node, fetch its data from the VariableEnvironment
            if let Some(variable_id) = node.variable_id {
                // Get the variable data from the environment
                if let Some(var_array) = ctx.var_env_ref.get_array_by_id(variable_id) {
                    let borrowed_array = var_array.borrow();
                    let cloned_array = borrowed_array.clone();
                    computed_values.insert(node_id, cloned_array);
                    continue;
                } else {
                    let err = OpError::RuntimeError(format!(
                        "Variable with ID {} not found in VariableEnvironment",
                        variable_id
                    ));

                    // If this is one of our target tensors, add an error to the result
                    for tensor in tensors {
                        if tensor.id == node_id {
                            results.push(Err(err.clone()));
                        }
                    }
                    continue;
                }
            }

            // If this is a placeholder but no feed was provided, return an error
            if node.placeholder_name.is_some() && !computed_values.contains_key(&node_id) {
                let placeholder_name = node.placeholder_name.unwrap_or("<unnamed>");
                let err = OpError::RuntimeError(format!(
                    "No feed value provided for placeholder '{}'",
                    placeholder_name
                ));

                // If this is one of our target tensors, add an error to the result
                for tensor in tensors {
                    if tensor.id == node_id {
                        results.push(Err(err.clone()));
                    }
                }

                // Skip this node since we can't compute it
                continue;
            }

            // Get inputs for this operation
            let mut input_arrays = Vec::with_capacity(node.incoming_nodes.len());

            // Collect input arrays from computed values
            for input_node in &node.incoming_nodes {
                if let Some(input_array) = computed_values.get(&input_node.id) {
                    input_arrays.push(input_array.clone());
                } else {
                    // If an input wasn't computed, there's a bug in our topological sort
                    let err = OpError::RuntimeError(format!(
                        "Input node {} for node {} was not computed - possible cycle in graph",
                        input_node.id, node_id
                    ));

                    // If this is one of our target tensors, add an error to the result
                    for tensor in tensors {
                        if tensor.id == node_id {
                            results.push(Err(err.clone()));
                        }
                    }

                    // Skip this node
                    continue;
                }
            }

            // We no longer need a separate output_arrays variable

            // Create compute context with cloned input arrays
            let cloned_inputs = input_arrays.clone();
            let mut compute_ctx = op::ComputeContext::with_inputs(cloned_inputs);

            // Execute the operation
            match node.get_op().compute(&mut compute_ctx) {
                Ok(()) => {
                    // Operation succeeded, store the output
                    let outputs = compute_ctx.get_outputs();
                    if !outputs.is_empty() {
                        computed_values.insert(node_id, outputs[0].clone());
                    } else {
                        // Operation produced no output
                        let err = OpError::RuntimeError(format!(
                            "Operation {} did not produce any output",
                            node.get_op().name()
                        ));

                        // If this is one of our target tensors, add an error to the result
                        for tensor in tensors {
                            if tensor.id == node_id {
                                results.push(Err(err.clone()));
                            }
                        }
                    }
                }
                Err(err) => {
                    // Operation failed
                    // If this is one of our target tensors, add an error to the result
                    for tensor in tensors {
                        if tensor.id == node_id {
                            results.push(Err(err.clone()));
                        }
                    }
                }
            }
        }

        // Collect results for the requested tensors
        results.clear(); // Clear any error results added during evaluation
        for tensor in tensors {
            if let Some(value) = computed_values.get(&tensor.id) {
                results.push(Ok(value.clone()));
            } else {
                results.push(Err(OpError::RuntimeError(format!(
                    "Failed to compute tensor {}",
                    tensor.id
                ))));
            }
        }

        results
    }

    #[inline]
    pub fn get_tensor_by_name(&self, name: &'static str) -> Option<TensorID> {
        // Search through all tensors to find one with matching placeholder name
        let nodes = self.node_set.borrow();
        for (id, node) in nodes.iter().enumerate() {
            if let Some(placeholder_name) = node.placeholder_name {
                if placeholder_name == name {
                    return Some(id);
                }
            }
        }
        None
    }

    #[inline]
    pub(crate) fn install(&'graph self, mut node: TensorInternal<F>) -> TensorID {
        let mut inner = self.node_set.borrow_mut();
        let id = inner.len();
        if id == NUM_NODES_WARN {
            eprintln!(
                "Too many tensors in this graph: {}. \
            Use Graph::clear, or move the training loop out of the `run` block",
                NUM_NODES_WARN
            )
        }
        if id > NUM_NODES_CRITICAL {
            panic!(
                "Maximum graph size exceeded: {}. \
            Use Graph::clear, or move the training loop out of the `run` block",
                NUM_NODES_CRITICAL
            )
        }
        node.id = id;
        inner.push(node);
        id
    }

    #[inline(always)]
    pub(crate) fn access_inner(&self, id: TensorID) -> Ref<TensorInternal<F>> {
        let borrow = self.node_set.borrow();
        Ref::map(borrow, |t| &t[id])
    }

    #[inline(always)]
    pub(crate) fn access_inner_mut(&self, id: TensorID) -> RefMut<TensorInternal<F>> {
        let borrow = self.node_set.borrow_mut();
        RefMut::map(borrow, |t| &mut t[id])
    }

    #[inline(always)]
    pub(crate) fn tensor(&'graph self, id: TensorID) -> Tensor<'graph, F> {
        Tensor { id, graph: self }
    }

    #[inline]
    pub(crate) fn topo_rank(&self, id: TensorID) -> usize {
        self.node_set.borrow()[id].topo_rank
    }

    #[inline]
    pub fn variable_by_id(&self, vid: VariableID) -> Tensor<F> {
        let tid = {
            let temp = self.variable2node.borrow();
            temp.get(&vid).cloned()
        };
        if let Some(tid) = tid {
            // use existing tensor
            self.tensor(tid)
        } else {
            // allocate a new tensor
            let allocated = Tensor::builder(self)
                .set_variable(vid)
                .build(crate::tensor_ops::basic_source_ops::Variable);
            // register vid -> tid map
            self.variable2node.borrow_mut().insert(vid, allocated.id);
            allocated
        }
    }
}

impl<T: Float> fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let set = &*self.node_set.borrow();
        let mut buf = format!("graph size: {}\n", set.len());
        for node in set {
            buf += format!("{}\n", node).as_str();
        }
        write!(f, "{}", buf)
    }
}

/// Creates and runs a computation graph.
///
/// See [Context].
pub fn run<F, FN, R>(f: FN) -> R
where
    F: Float,
    FN: FnOnce(&mut Context<F>) -> R,
{
    let graph_internal = Graph {
        node_set: RefCell::new(Vec::with_capacity(512)),
        variable2node: RefCell::new(HashMap::new()),
    };
    let mut ctx = Context {
        var_env_ref: &mut VariableEnvironment::new(),
        graph: graph_internal,
    };
    f(&mut ctx)
}

/// Generates and runs a computation graph
///
/// Each time [run] is invoked, a new `Context` allocating a [Graph] is passed to the closure, in which tensors are generated and evaluated.
/// It's faster to understand if you see [Tensor]'s documentation.
///
/// In order to bind `Tensor`s to pre-defined variable arrays, use [VariableEnvironment::run] instead.
/// See [crate::variable]
pub struct Context<'env, F: Float> {
    pub(crate) graph: Graph<F>,
    pub(crate) var_env_ref: &'env VariableEnvironment<F>,
}

impl<'graph, 'env, F: Float> Context<'env, F> {
    /// Get or create a variable namespace with the specified name.
    ///
    /// Use `namespace_mut` for mutable operations such as variables registrations.
    #[inline]
    pub fn namespace(&'env self, namespace_id: &'static str) -> VariableNamespace<'env, F> {
        self.var_env_ref.namespace(namespace_id)
    }

    /// Get or create the *default* variable namespace.
    ///
    /// Use `namespace_mut` for mutable operations such as variables registrations.
    #[inline]
    pub fn default_namespace(&'env self) -> VariableNamespace<'env, F> {
        self.var_env_ref.default_namespace()
    }

    /// Returns a reference to the current VariableEnvironment
    #[inline]
    pub fn env(&'graph self) -> &'env VariableEnvironment<F> {
        self.var_env_ref
    }

    /// Creates an evaluator for the graph.
    ///
    /// This method is used to evaluate tensors in the graph.
    #[inline]
    pub fn evaluator(&'graph self) -> Evaluator<'graph, 'graph, F> {
        Evaluator::new(self)
    }

    /// Evaluates tensors in the graph.
    ///
    /// This is an internal method used by tensor.eval()
    #[inline]
    pub fn eval(
        &'graph self,
        tensors: &[&Tensor<'graph, F>],
        feeds: &HashMap<TensorID, RawNdArrayView<F>>,
        _var_env: &'env VariableEnvironment<F>,
    ) -> Vec<Result<NdArray<F>, OpError>> {
        // Create a temporary HashMap to store references
        let temp_feeds: HashMap<TensorID, &RawNdArrayView<F>> =
            feeds.iter().map(|(k, v)| (*k, v)).collect();
        Graph::eval_tensors(tensors, &temp_feeds, self)
    }

    /// Removes all tensors in this graph.
    ///
    /// Note that any tensors allocated prior to this method call are invalid.
    #[inline]
    pub fn clear(&mut self) {
        self.graph.node_set.borrow_mut().clear();
        self.graph.variable2node.borrow_mut().clear();
    }

    /// Creates a placeholder tensor in a [Graph].
    ///
    /// placeholder is a named tensor whose value can be specified when evaluating a computation graph.
    /// You can designate the `shape` of the placeholder and `shape[i]` can be a positive
    /// value or -1 which means an dim of arbitrary size.
    ///
    /// Use [Evaluator::feed] and [Feeder::push] in order to assign ArrayViews to placeholders.
    ///    ```
    /// use scirs2_autograd as ag;
    /// use ag::ndarray::array;
    ///
    /// ag::run(|ctx| {
    ///     // be aware that x1 and x3 represent the same value
    ///     let x1 = ctx.placeholder("x", &[-1, 2]);
    ///     let x2 = ctx.placeholder("y", &[-1, 2]);
    ///     let x3 = ctx.placeholder("x", &[-1, 2]);
    ///     let sum = x1 + x2 + x3;
    ///
    ///     let arr = &array![[1., 1.]].into_dyn();
    ///
    ///     let result = ctx.evaluator()
    ///         .push(&sum)
    ///         .feed("x", arr.view()) // feed for x1 and x3
    ///         .feed("y", arr.view()) // feed for x2
    ///         .feed(x2, arr.view()) // same as .feed("y", ...)
    ///         .run();
    ///     assert_eq!(result[0], Ok(arr + arr + arr));
    /// });
    ///    ```
    ///
    /// See also [tensor_ops::convert_to_tensor].
    #[inline]
    pub fn placeholder(&'graph self, name: &'static str, shape: &[isize]) -> Tensor<'graph, F> {
        // Check if a placeholder with this name already exists
        if let Some(existing_id) = self.get_tensor_by_name(name) {
            // Return the existing placeholder
            return self.tensor(existing_id);
        }

        // Create a new placeholder tensor with the given name and shape
        Tensor::builder(self)
            .set_placeholder_name(name)
            .set_known_shape(shape)
            .build(T::basic_source_ops::Placeholder)
    }
}

#[allow(clippy::needless_lifetimes)]
impl<'env, F: Float> Deref for Context<'env, F> {
    type Target = Graph<F>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

pub trait AsGraph<F: Float> {
    fn as_graph(&self) -> &Graph<F>;

    // Get a reference to the variable environment
    fn env_ref(&self) -> &VariableEnvironment<F>;

    // Get a reference to the context (if available)
    fn context_ref(&self) -> Option<&Context<F>> {
        None
    }

    // Get or create a variable tensor by ID
    fn variable_by_id(&self, vid: VariableID) -> Tensor<F> {
        self.as_graph().variable_by_id(vid)
    }
}

impl<F: Float> AsGraph<F> for Graph<F> {
    #[inline]
    fn as_graph(&self) -> &Graph<F> {
        self
    }

    // Return a reference to the current variable environment
    // This is a simple placeholder implementation for AsGraph trait
    #[inline]
    fn env_ref(&self) -> &VariableEnvironment<F> {
        // This should never be called in practice since we simplified the variable function
        panic!("env_ref called on Graph, but Graph has no associated environment")
    }
}

impl<F: Float> AsGraph<F> for Context<'_, F> {
    #[inline]
    fn as_graph(&self) -> &Graph<F> {
        &self.graph
    }

    #[inline]
    fn env_ref(&self) -> &VariableEnvironment<F> {
        self.var_env_ref
    }

    #[inline]
    fn context_ref(&self) -> Option<&Context<F>> {
        Some(self)
    }
}

impl<F: Float> Default for Graph<F> {
    fn default() -> Self {
        Self {
            node_set: RefCell::new(Vec::new()),
            variable2node: RefCell::new(HashMap::new()),
        }
    }
}

#[inline]
pub(crate) fn assert_same_graph<F: Float>(a: &impl AsGraph<F>, b: &impl AsGraph<F>) {
    assert_eq!(
        a.as_graph() as *const _,
        b.as_graph() as *const _,
        "Detected tensors belonging to different graphs"
    );
}

#[test]
#[should_panic]
fn test_mixed_graph() {
    VariableEnvironment::<f32>::new().run(|g| {
        let a = T::zeros(&[1], g);
        VariableEnvironment::<f32>::new().run(|g2| {
            let b = T::zeros(&[1], g2);
            let _ = a + b;
        });
    });
}
