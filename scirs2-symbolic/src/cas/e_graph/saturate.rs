//! Equality saturation: repeatedly apply rewrite rules until fixpoint or budget.
//!
//! # Strategy
//!
//! Each saturation iteration:
//! 1. Extract the current cheapest representative for every class (snapshot).
//! 2. For every class, for every enode in that class:
//!    - Reconstruct the enode's representative `LoweredOp` from the snapshot.
//!    - Try to match each rule's LHS against that `LoweredOp`.
//!    - On match, instantiate the RHS, add it to the e-graph, and union.
//! 3. Call `rebuild()` to repair the hashcons.
//! 4. Repeat until no union occurred, or the budget is exhausted.
//!
//! The "extract snapshot first" approach avoids repeated extraction per rule
//! per enode and keeps each iteration bounded by O(rules × classes × nodes).
//!
//! # Budget
//!
//! [`SaturationBudget`] limits both iteration count and total node count.
//! When the budget is exhausted the e-graph is in a valid (partial saturation)
//! state; the caller should still extract and return a valid form.

use std::collections::HashMap;

use super::build::EGraph;
use super::extract::extract_class;
use super::node::ClassId;
use crate::cas::pattern::{instantiate, match_pattern, Bindings, Pattern};
use crate::eml::LoweredOp;

/// A single rewrite rule for use within the e-graph saturation engine.
///
/// Rules are structurally identical to [`crate::cas::identity_db::Identity`]
/// but without the metadata fields.
pub(crate) struct EGraphRule {
    pub lhs: Pattern,
    pub rhs: Pattern,
}

/// Budget controlling how long saturation runs.
///
/// When `max_iterations` outer loops complete or `max_nodes` enodes are in
/// the graph, saturation stops. The result is still extracted and returned;
/// it may not be fully saturated but is always a valid `LoweredOp`.
#[derive(Debug, Clone)]
pub struct SaturationBudget {
    /// Maximum saturation loop iterations (default: 30).
    pub max_iterations: u32,
    /// Maximum total enodes in the graph (default: 10,000).
    pub max_nodes: u32,
}

impl Default for SaturationBudget {
    fn default() -> Self {
        SaturationBudget {
            max_iterations: 30,
            max_nodes: 10_000,
        }
    }
}

/// Run equality saturation on `egraph` using the given `rules`.
///
/// Applies rules iteratively until no new unions are produced or the budget
/// is exhausted. After each iteration, `rebuild()` is called to re-canonicalize
/// the hashcons.
///
/// Returns `true` if any union was performed during saturation.
pub(crate) fn saturate(
    egraph: &mut EGraph,
    rules: &[EGraphRule],
    budget: &SaturationBudget,
) -> bool {
    let mut any_union = false;

    for _iter in 0..budget.max_iterations {
        if egraph.total_nodes() >= budget.max_nodes {
            break;
        }

        // Snapshot: extract cheapest representative for each class.
        // Collect and sort class ids so that saturation order is deterministic.
        // std::collections::HashMap uses a randomised hash seed, making `.keys()`
        // iteration order non-deterministic across runs and threads. Sorting by
        // the monotonically-assigned `ClassId` (u32) gives a stable traversal
        // order that does not depend on the OS thread scheduler or the hash seed,
        // eliminating the race condition in `test_saturation_with_identity_db`.
        let mut class_ids: Vec<ClassId> = egraph.classes.keys().copied().collect();
        class_ids.sort_unstable();

        // For each class that is a root, extract cheapest representative.
        let mut snapshot: HashMap<ClassId, LoweredOp> = HashMap::new();
        for &cls in &class_ids {
            let root = egraph.find(cls);
            if let std::collections::hash_map::Entry::Vacant(entry) = snapshot.entry(root) {
                let rep = extract_class(egraph, root);
                entry.insert(rep);
            }
        }

        // Apply rules.
        let mut union_this_iter = false;
        // We collect (lhs_class, rhs_class) pairs to union after iterating,
        // to avoid holding references into egraph while it is mutated.
        let mut to_union: Vec<(ClassId, ClassId)> = Vec::new();

        for &cls in &class_ids {
            let root = egraph.find(cls);
            // Get the representative LoweredOp from snapshot.
            let rep = match snapshot.get(&root) {
                Some(r) => r.clone(),
                None => continue,
            };

            for rule in rules {
                let mut bindings: Bindings = Bindings::default();
                if match_pattern(&rule.lhs, &rep, &mut bindings) {
                    // Instantiate RHS.
                    match instantiate(&rule.rhs, &bindings) {
                        Ok(rhs_op) => {
                            // Add rhs_op to get rhs_class, then record the union.
                            let rhs_class = egraph.add(&rhs_op);
                            let lhs_root = egraph.find(root);
                            let rhs_root = egraph.find(rhs_class);
                            if lhs_root != rhs_root {
                                to_union.push((lhs_root, rhs_root));
                            }
                        }
                        Err(_) => continue,
                    }
                }
            }
        }

        // Apply collected unions.
        for (a, b) in to_union {
            if egraph.union(a, b) {
                union_this_iter = true;
                any_union = true;
            }
        }

        if union_this_iter {
            egraph.rebuild();
        } else {
            // No new unions this iteration — saturation is complete.
            break;
        }
    }

    any_union
}

/// Match and apply rules against a single `LoweredOp` representative.
///
/// Exposed for testing individual rule applications.
pub(crate) fn match_and_apply_rules(
    egraph: &mut EGraph,
    rules: &[EGraphRule],
    budget: &mut SaturationBudget,
) -> bool {
    saturate(egraph, rules, budget)
}

/// Build the default rule set for saturation.
///
/// Includes:
/// - Algebraic simplification rules: x+0=x, 0+x=x, x*1=x, 1*x=x,
///   x-0=x, x/1=x, x*0=0, 0*x=0.
/// - Log/exp cancellation rules: ln(exp(x))=x, exp(ln(x))=x.
/// - Log/exp product rules: ln(x*y)=ln(x)+ln(y), exp(x)*exp(y)=exp(x+y).
/// - Power simplification: x^1=x, x^0=1.
/// - Commutativity of addition and multiplication.
pub(crate) fn default_rules() -> Vec<EGraphRule> {
    use crate::cas::pattern::{BinaryKind, UnaryKind};

    let p = |v: u32| Pattern::PatVar(v);
    let c0 = Pattern::PatConst(0.0);
    let c1 = Pattern::PatConst(1.0);
    let ci0 = || Pattern::PatConstInt(0);
    let ci1 = || Pattern::PatConstInt(1);

    macro_rules! op2 {
        ($k:expr, $l:expr, $r:expr) => {
            Pattern::PatOp2($k, Box::new($l), Box::new($r))
        };
    }
    macro_rules! op1 {
        ($k:expr, $c:expr) => {
            Pattern::PatOp1($k, Box::new($c))
        };
    }

    vec![
        // x + 0 = x
        EGraphRule {
            lhs: op2!(BinaryKind::Add, p(0), c0.clone()),
            rhs: p(0),
        },
        // 0 + x = x
        EGraphRule {
            lhs: op2!(BinaryKind::Add, c0.clone(), p(0)),
            rhs: p(0),
        },
        // x - 0 = x
        EGraphRule {
            lhs: op2!(BinaryKind::Sub, p(0), c0.clone()),
            rhs: p(0),
        },
        // x * 1 = x
        EGraphRule {
            lhs: op2!(BinaryKind::Mul, p(0), c1.clone()),
            rhs: p(0),
        },
        // 1 * x = x
        EGraphRule {
            lhs: op2!(BinaryKind::Mul, c1.clone(), p(0)),
            rhs: p(0),
        },
        // x / 1 = x
        EGraphRule {
            lhs: op2!(BinaryKind::Div, p(0), c1.clone()),
            rhs: p(0),
        },
        // x * 0 = 0
        EGraphRule {
            lhs: op2!(BinaryKind::Mul, p(0), ci0()),
            rhs: Pattern::PatConstInt(0),
        },
        // 0 * x = 0
        EGraphRule {
            lhs: op2!(BinaryKind::Mul, ci0(), p(0)),
            rhs: Pattern::PatConstInt(0),
        },
        // x ^ 1 = x  (PatConstInt 1)
        EGraphRule {
            lhs: op2!(BinaryKind::Pow, p(0), ci1()),
            rhs: p(0),
        },
        // x ^ 0 = 1  (PatConstInt 0)
        EGraphRule {
            lhs: op2!(BinaryKind::Pow, p(0), ci0()),
            rhs: Pattern::PatConstInt(1),
        },
        // ln(exp(x)) = x
        EGraphRule {
            lhs: op1!(UnaryKind::Ln, op1!(UnaryKind::Exp, p(0))),
            rhs: p(0),
        },
        // exp(ln(x)) = x
        EGraphRule {
            lhs: op1!(UnaryKind::Exp, op1!(UnaryKind::Ln, p(0))),
            rhs: p(0),
        },
        // ln(x*y) = ln(x) + ln(y)
        EGraphRule {
            lhs: op1!(UnaryKind::Ln, op2!(BinaryKind::Mul, p(0), p(1))),
            rhs: op2!(
                BinaryKind::Add,
                op1!(UnaryKind::Ln, p(0)),
                op1!(UnaryKind::Ln, p(1))
            ),
        },
        // exp(x) * exp(y) = exp(x + y)
        EGraphRule {
            lhs: op2!(
                BinaryKind::Mul,
                op1!(UnaryKind::Exp, p(0)),
                op1!(UnaryKind::Exp, p(1))
            ),
            rhs: op1!(UnaryKind::Exp, op2!(BinaryKind::Add, p(0), p(1))),
        },
        // x + y = y + x  (commutativity of addition)
        EGraphRule {
            lhs: op2!(BinaryKind::Add, p(0), p(1)),
            rhs: op2!(BinaryKind::Add, p(1), p(0)),
        },
        // x * y = y * x  (commutativity of multiplication)
        EGraphRule {
            lhs: op2!(BinaryKind::Mul, p(0), p(1)),
            rhs: op2!(BinaryKind::Mul, p(1), p(0)),
        },
        // Distribution: (p + q) * r = p*r + q*r
        EGraphRule {
            lhs: op2!(BinaryKind::Mul, op2!(BinaryKind::Add, p(0), p(1)), p(2)),
            rhs: op2!(
                BinaryKind::Add,
                op2!(BinaryKind::Mul, p(0), p(2)),
                op2!(BinaryKind::Mul, p(1), p(2))
            ),
        },
        // Distribution: r * (p + q) = r*p + r*q
        EGraphRule {
            lhs: op2!(BinaryKind::Mul, p(2), op2!(BinaryKind::Add, p(0), p(1))),
            rhs: op2!(
                BinaryKind::Add,
                op2!(BinaryKind::Mul, p(2), p(0)),
                op2!(BinaryKind::Mul, p(2), p(1))
            ),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cas::e_graph::node::ClassId;
    use crate::eml::LoweredOp;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    #[test]
    fn test_saturation_finds_x_plus_0() {
        let mut eg = EGraph::new();
        let op = LoweredOp::Add(Box::new(var(0)), Box::new(c(0.0)));
        let root = eg.add(&op);
        let rules = default_rules();
        let budget = SaturationBudget::default();
        saturate(&mut eg, &rules, &budget);
        let root_canon = ClassId(eg.union_find.find_root_immutable(root.0));
        let extracted = extract_class(&eg, root_canon);
        // After saturation, x+0 should be equivalent to x.
        let id_x = eg.add(&var(0));
        // Both should be in the same class.
        assert_eq!(
            eg.find(root),
            eg.find(id_x),
            "x+0 and x should be in same class after saturation"
        );
        // Extracted form should not be a nested +0 anymore (it should simplify to x).
        // The extracted form is var(0) since cost(x) = 1 < cost(x+0) = 3.
        assert_eq!(extracted, var(0), "extracted form should be x");
    }

    #[test]
    fn test_budget_exhaustion_valid_form() {
        let mut eg = EGraph::new();
        let op = LoweredOp::Add(Box::new(var(0)), Box::new(c(0.0)));
        let root = eg.add(&op);
        let rules = default_rules();
        let budget = SaturationBudget {
            max_iterations: 1,
            max_nodes: 10,
        };
        saturate(&mut eg, &rules, &budget);
        // Even with tiny budget, extract should return a valid LoweredOp.
        let root_canon = ClassId(eg.union_find.find_root_immutable(root.0));
        let extracted = extract_class(&eg, root_canon);
        // Not a panic; any valid LoweredOp is acceptable.
        let _ = extracted; // just check no panic
    }

    #[test]
    fn test_saturation_foil_distribution() {
        // (x + y) * (a + b) — test that distribution rule fires and creates the
        // distributed form as an equivalent class.
        let mut eg = EGraph::new();
        let x = var(0);
        let y = var(1);
        let a = var(2);
        let b = var(3);
        let x_plus_y = LoweredOp::Add(Box::new(x.clone()), Box::new(y.clone()));
        let a_plus_b = LoweredOp::Add(Box::new(a.clone()), Box::new(b.clone()));
        let foil = LoweredOp::Mul(Box::new(x_plus_y), Box::new(a_plus_b));
        let root = eg.add(&foil);

        // Build the distributed form: x*a + x*b + y*a + y*b
        // Actually rule fires as (x+y)*r = x*r + y*r, so (x+y)*(a+b):
        // with r=(a+b): x*(a+b) + y*(a+b)
        let distributed_partial = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(var(0)),
                Box::new(LoweredOp::Add(Box::new(var(2)), Box::new(var(3)))),
            )),
            Box::new(LoweredOp::Mul(
                Box::new(var(1)),
                Box::new(LoweredOp::Add(Box::new(var(2)), Box::new(var(3)))),
            )),
        );
        let distributed_id = eg.add(&distributed_partial);

        let rules = default_rules();
        let budget = SaturationBudget {
            max_iterations: 5,
            max_nodes: 2000,
        };
        saturate(&mut eg, &rules, &budget);

        // After saturation, the root (foil) and the distributed form should be equivalent.
        let root_canon = eg.find(root);
        let dist_canon = eg.find(distributed_id);
        assert_eq!(
            root_canon, dist_canon,
            "foil and its distribution should be in same e-class after saturation"
        );
    }
}
