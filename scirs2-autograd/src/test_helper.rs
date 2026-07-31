//! Provides helper functions for testing.
use crate::evaluation::Feeder;
use crate::tensor::Tensor;
use crate::tensor_ops::*;
use crate::{ndarray_ext, Context, Float};

/// Checks the validity of `gradients` with finite difference trick.
/// For this test only, `variables` must be *shared* variables.
#[allow(dead_code)]
pub fn check_theoretical_grads<'g, 't, 'v, F: Float, A>(
    objective: A,
    gradients: &'t [A],
    variables: &'t [A],
    feeder: Feeder<'v, F>,
    eps: F,
    tol: F,
    g: &'g Context<F>,
) where
    A: AsRef<Tensor<'g, F>> + Copy + 'g,
    't: 'g,
    'v: 'g,
{
    let objective = sum_all(objective);
    // backprop
    let theoretical_grads = g
        .evaluator()
        .extend(gradients)
        .set_feeder(feeder.clone())
        .run();

    // for each variable nodes
    for (var_node, th_grad) in variables.iter().zip(theoretical_grads) {
        // Copy gradient array if needed
        let th_copied = if th_grad
            .as_ref()
            .expect("Operation failed")
            .is_standard_layout()
        {
            None
        } else {
            Some(ndarray_ext::deep_copy(
                &th_grad.as_ref().expect("Operation failed").view(),
            ))
        };
        let th_ptr = if let Some(ref inner) = th_copied {
            inner.as_ptr()
        } else {
            th_grad.as_ref().expect("Operation failed").as_ptr()
        };

        // for each values
        let v_len = g
            .env()
            .get_array_by_id(
                var_node
                    .as_ref()
                    .get_variable_id()
                    .expect("This is not a variable"),
            )
            .expect("variable array not found")
            .borrow()
            .len();

        for i in 0..v_len as isize {
            let evacuated;
            // +
            // SAFETY PROOF:
            // Preconditions:
            //   1. Index i is within bounds: 0 <= i < v_len (verified by loop bounds)
            //   2. guard_mut holds valid exclusive reference to array
            //   3. Array length matches v_len (verified earlier)
            // Guarantees:
            //   - No out-of-bounds access (i < v_len ensured by loop)
            //   - No data races (exclusive &mut via borrow_mut)
            //   - Pointer arithmetic is valid (offset within allocated array)
            // Verification:
            //   - Loop bound: i in 0..v_len ensures valid index
            //   - Array length verified by earlier borrow().len() == v_len
            debug_assert!(
                i >= 0 && i < v_len as isize,
                "Index {} out of bounds (len: {})",
                i,
                v_len
            );
            unsafe {
                let mut guard_mut = g
                    .env()
                    .get_array_by_id(
                        var_node
                            .as_ref()
                            .get_variable_id()
                            .expect("This is not a variable"),
                    )
                    .expect("variable array not found")
                    .borrow_mut();
                let head = guard_mut.as_mut_ptr();
                // SAFETY: i < v_len verified by loop and assertion above
                evacuated = *head.offset(i);
                *head.offset(i) = evacuated + eps;
            }

            // eval
            let obj_pos_orig = g
                .evaluator()
                .push(&objective)
                .set_feeder(feeder.clone())
                .run()
                .remove(0)
                .expect("Operation failed");
            let obj_pos = if obj_pos_orig.is_standard_layout() {
                obj_pos_orig
            } else {
                ndarray_ext::deep_copy(&obj_pos_orig.view())
            };

            // SAFETY: i < v_len verified by loop bounds and assertion above
            unsafe {
                let mut guard_mut = g
                    .env()
                    .get_array_by_id(
                        var_node
                            .as_ref()
                            .get_variable_id()
                            .expect("This is not a variable"),
                    )
                    .expect("variable array not found")
                    .borrow_mut();

                let head = guard_mut.as_mut_ptr();
                // SAFETY: i < v_len verified by loop bounds
                *head.offset(i) = evacuated - eps;
            }

            // eval
            let obj_neg_orig = g
                .evaluator()
                .push(&objective)
                .set_feeder(feeder.clone())
                .run()
                .remove(0)
                .expect("Operation failed");
            let obj_neg = if obj_neg_orig.is_standard_layout() {
                obj_neg_orig
            } else {
                ndarray_ext::deep_copy(&obj_neg_orig.view())
            };

            // restore
            // SAFETY: i < v_len verified by loop bounds
            unsafe {
                let mut guard_mut = g
                    .env()
                    .get_array_by_id(
                        var_node
                            .as_ref()
                            .get_variable_id()
                            .expect("This is not a variable"),
                    )
                    .expect("variable array not found")
                    .borrow_mut();
                let head = guard_mut.as_mut_ptr();
                // SAFETY: i < v_len verified by loop bounds
                *head.offset(i) = evacuated;
            }

            let two = F::one() + F::one();
            let g_num = (obj_pos - obj_neg).sum() / (two * eps);
            // SAFETY: i < theoretical_grad.len() verified by loop (v_len == theoretical_grad.len())
            let g_th = unsafe { *th_ptr.offset(i) };

            // compare
            let diff = (g_num - g_th).abs();
            if diff > tol {
                panic!(
                    "Gradient checking failed with too large error: numerical={g_num}, theoretical={g_th}"
                );
            }
        }
    }
}

/// Numerically verifies the analytical gradients of `objective` w.r.t. `params`.
///
/// Every entry of every parameter is perturbed by `±epsilon` and the resulting central
/// difference of `sum_all(objective)` is compared against the reverse-mode gradient.
/// Returns `true` only if every element agrees to within `tolerance` (relative to the
/// magnitude of the numerical value).
///
/// `params` must be **variables** created through a [`crate::VariableEnvironment`]
/// (`ctx.variable(...)` / `VariableNamespace::slot`), because the check works by writing
/// perturbed values back into the environment. A `params` entry that is not a variable,
/// or an objective that fails to evaluate, makes the function return `false` — it never
/// reports success it did not verify.
///
/// This used to ignore all of its arguments and unconditionally return `true`, which made
/// it incapable of detecting a broken gradient — the exact thing it exists to detect.
#[allow(dead_code)]
pub fn gradient_check<'g, F: Float>(
    ctx: &'g Context<'g, F>,
    objective: &Tensor<'g, F>,
    params: &[Tensor<'g, F>],
    epsilon: F,
    tolerance: F,
) -> bool {
    if epsilon <= F::zero() || tolerance < F::zero() || params.is_empty() {
        return false;
    }

    let loss = sum_all(objective);

    // Analytical gradients.
    let analytical = grad(&[loss], params);
    let mut analytical_values = Vec::with_capacity(params.len());
    for gt in &analytical {
        match gt.eval(ctx) {
            Ok(arr) => analytical_values.push(arr.iter().copied().collect::<Vec<F>>()),
            Err(_) => return false,
        }
    }

    let two = F::one() + F::one();

    for (p_index, param) in params.iter().enumerate() {
        let Some(vid) = param.get_variable_id() else {
            // Not a variable: its value cannot be perturbed, so nothing can be verified.
            return false;
        };
        let Some(cell) = ctx.env().get_array_by_id(vid) else {
            return false;
        };

        let len = { cell.borrow().len() };
        if analytical_values[p_index].len() != len {
            return false;
        }

        for i in 0..len {
            let original = {
                let view = cell.borrow();
                match view.iter().nth(i) {
                    Some(v) => *v,
                    None => return false,
                }
            };

            let mut sample = |value: F| -> Option<F> {
                {
                    let mut view = cell.borrow_mut();
                    match view.iter_mut().nth(i) {
                        Some(slot) => *slot = value,
                        None => return None,
                    }
                }
                let evaluated = loss.eval(ctx).ok()?;
                Some(evaluated.iter().copied().fold(F::zero(), |a, b| a + b))
            };

            let plus = sample(original + epsilon);
            let minus = sample(original - epsilon);

            // Always restore the original value, whatever happened above.
            {
                let mut view = cell.borrow_mut();
                if let Some(slot) = view.iter_mut().nth(i) {
                    *slot = original;
                }
            }

            let (Some(plus), Some(minus)) = (plus, minus) else {
                return false;
            };

            let numerical = (plus - minus) / (two * epsilon);
            let analytic = analytical_values[p_index][i];
            let diff = (numerical - analytic).abs();
            if diff > tolerance * (F::one() + numerical.abs()) {
                return false;
            }
        }
    }

    true
}

/// Structural summary of a model's graph node, as produced by
/// [`print_model_summary`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct ModelSummary {
    pub tensor_id: usize,
    pub num_inputs: usize,
    /// Statically known shape, if any. `Tensor::shape()` returns an empty
    /// `Vec` both for a genuine 0-d scalar and for a shape it could not
    /// resolve, so an empty value here is not proof the model is scalar.
    pub knownshape: Vec<usize>,
    pub requires_grad: bool,
    pub is_source: bool,
}

/// Prints a summary of the model's graph-node structure to help with
/// debugging, and returns it for programmatic use.
///
/// A [`Tensor`] is a lazy graph handle with no evaluation [`Context`]
/// available here, so this reports structural metadata (id, input arity,
/// statically known shape, differentiability) rather than runtime values;
/// evaluate `model` separately (e.g. via [`Tensor::eval`]) to inspect actual
/// data. This used to ignore `model` entirely and print a hard-coded
/// `"[placeholder]"` string regardless of what was passed in.
#[allow(dead_code)]
pub fn print_model_summary<F: Float>(model: &Tensor<F>) -> ModelSummary {
    let summary = ModelSummary {
        tensor_id: model.id(),
        num_inputs: model.num_inputs(),
        knownshape: model.shape(),
        requires_grad: model.requires_grad(),
        is_source: model.is_source(),
    };
    println!(
        "Model summary: id={}, inputs={}, shape={:?}, requires_grad={}, is_source={}",
        summary.tensor_id,
        summary.num_inputs,
        summary.knownshape,
        summary.requires_grad,
        summary.is_source,
    );
    summary
}

/// Memory usage estimate produced by [`profile_memory_usage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct MemoryUsageReport {
    /// Total element count across `model`'s output plus every tensor in
    /// `inputs`.
    pub total_elements: usize,
    /// `total_elements * size_of::<F>()`.
    pub total_bytes: usize,
    /// Whether `model` could actually be evaluated (no unresolved
    /// variable/placeholder stood in the way) to obtain its *real* element
    /// count, as opposed to falling back to its statically known shape.
    pub model_evaluated: bool,
}

/// Estimates the memory footprint of `model` and `inputs` to help with
/// debugging, and returns the estimate for programmatic use.
///
/// Evaluates `model` (and each of `inputs`) through its own bare
/// [`crate::Graph`] to obtain its *real* element count whenever that graph
/// contains no unresolved variable/placeholder; when it does (evaluation
/// needs an explicit [`Context`] this function does not receive), falls
/// back to the tensor's statically known shape. This used to ignore both
/// arguments entirely and print a hard-coded `"[placeholder]"` string.
#[allow(dead_code)]
pub fn profile_memory_usage<F: Float>(
    model: &Tensor<F>,
    inputs: &[Tensor<F>],
) -> MemoryUsageReport {
    let elem_size = std::mem::size_of::<F>();

    fn element_count<F: Float>(t: &Tensor<F>) -> (usize, bool) {
        match t.eval(t.graph()) {
            Ok(arr) => (arr.len(), true),
            Err(_) => (t.shape().iter().product(), false),
        }
    }

    let (model_elements, model_evaluated) = element_count(model);
    let mut total_elements = model_elements;
    for input in inputs {
        let (elements, _) = element_count(input);
        total_elements += elements;
    }

    let report = MemoryUsageReport {
        total_elements,
        total_bytes: total_elements * elem_size,
        model_evaluated,
    };
    println!(
        "Memory usage: ~{} bytes across {} elements (model + {} inputs, {} bytes/element, model evaluated: {})",
        report.total_bytes,
        report.total_elements,
        inputs.len(),
        elem_size,
        report.model_evaluated,
    );
    report
}

/// Actually runs `forward_fn`, evaluates its output to force real
/// computation, times the whole thing, prints a summary, and returns the
/// measured [`std::time::Duration`].
///
/// `Tensor`s are lazy, so building `forward_fn`'s output alone would not run
/// any computation; this evaluates the result (through its own bare
/// [`crate::Graph`]) so the measured time reflects the forward pass, not
/// just graph construction. This used to discard `forward_fn` unused via
/// `let _ = (model, inputs, forward_fn)` -- never calling it at all -- and
/// print a hard-coded `"[placeholder]"` string.
#[allow(dead_code)]
pub fn measure_computation_time<'a, F: Float, G>(
    model: &'a Tensor<'a, F>,
    inputs: &'a [Tensor<'a, F>],
    forward_fn: G,
) -> std::time::Duration
where
    G: FnOnce(&'a Tensor<'a, F>, &'a [Tensor<'a, F>]) -> Tensor<'a, F>,
{
    let start = std::time::Instant::now();
    let output = forward_fn(model, inputs);
    let eval_result = output.eval(output.graph());
    let elapsed = start.elapsed();

    match eval_result {
        Ok(array) => println!(
            "Computation time: {elapsed:?} (output shape {:?}, {} elements)",
            array.shape(),
            array.len()
        ),
        Err(e) => println!(
            "Computation time: {elapsed:?} (forward pass built the graph, but could not \
             be evaluated without an explicit Context -- {e:?}; timing reflects graph \
             construction only)"
        ),
    }

    elapsed
}

#[cfg(test)]
mod helper_fn_tests {
    use super::*;

    #[test]
    fn print_model_summary_reports_real_structure_not_a_placeholder() {
        crate::run(|ctx: &mut Context<f64>| {
            let a = crate::tensor_ops::convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[4]),
                    vec![1.0f64, 2.0, 3.0, 4.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let b = crate::tensor_ops::convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[4]),
                    vec![5.0f64, 6.0, 7.0, 8.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let sum = a + b;

            let summary = print_model_summary(&sum);
            // `sum` is a binary op: exactly 2 inputs, not the arbitrary
            // constant a placeholder implementation would report.
            assert_eq!(summary.num_inputs, 2);
            assert_eq!(summary.tensor_id, sum.id());
            assert!(!summary.is_source);
        });
    }

    #[test]
    fn profile_memory_usage_counts_real_elements_for_non_constant_shapes() {
        crate::run(|ctx: &mut Context<f64>| {
            let model = crate::tensor_ops::convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[4]),
                    vec![1.0f64, 2.0, 3.0, 4.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let input0 = crate::tensor_ops::convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[3]),
                    vec![10.0f64, 20.0, 30.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let input1 = crate::tensor_ops::convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[2, 2]),
                    vec![1.0f64, 2.0, 3.0, 4.0],
                )
                .expect("Operation failed"),
                ctx,
            );

            let report = profile_memory_usage(&model, &[input0, input1]);
            // 4 (model) + 3 (input0) + 4 (input1) = 11 -- not a fixed/fabricated count.
            assert_eq!(report.total_elements, 11);
            assert_eq!(report.total_bytes, 11 * std::mem::size_of::<f64>());
            assert!(report.model_evaluated);
        });
    }

    #[test]
    fn measure_computation_time_actually_invokes_forward_fn_and_times_it() {
        // The old stub discarded `forward_fn` via `let _ = (...)` and never called it
        // (so `called` would stay `false`), nor did it evaluate the result (so a
        // fabricated/near-zero duration could not be distinguished from a real one --
        // sleeping inside `forward_fn` and requiring the measured duration to cover
        // that sleep rules both out).
        crate::run(|ctx: &mut Context<f64>| {
            let model = crate::tensor_ops::convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[3]),
                    vec![2.0f64, 4.0, 6.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let inputs: Vec<Tensor<f64>> = vec![];
            let called = std::cell::Cell::new(false);
            let sleep_for = std::time::Duration::from_millis(20);

            let duration = measure_computation_time(&model, &inputs, |m, _inputs| {
                called.set(true);
                std::thread::sleep(sleep_for);
                crate::tensor_ops::arithmetic::mul(*m, *m)
            });

            assert!(
                called.get(),
                "measure_computation_time must actually call forward_fn"
            );
            assert!(
                duration >= sleep_for,
                "measured duration ({duration:?}) must cover the real work done inside forward_fn"
            );
        });
    }
}
