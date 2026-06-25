//! DP extraction: find the smallest-tree representative of an e-class.
//!
//! Given an e-graph (possibly after saturation), `extract_class` walks the
//! equivalence classes and computes — for each class — the minimum-cost
//! `LoweredOp` tree, where cost = 1 + sum of children costs (a simple
//! tree-size metric).
//!
//! # Algorithm
//!
//! 1. **BFS** from `root` to collect all reachable class ids.
//! 2. **DP fixed-point**: iterate over all reachable classes repeatedly.
//!    For each enode in a class, compute candidate cost = 1 + Σ children_cost.
//!    Update if cheaper. Continue until no cost changes or `MAX_EXTRACT_ITER`
//!    is reached.
//! 3. **Reconstruction**: walk from root, at each class pick the enode with
//!    the best cost, recursively reconstruct children (iterative, bounded by
//!    tree depth).
//!
//! # Cycle handling
//!
//! If a class's enodes all have children with infinite cost (cycle or
//! unresolved dependency), the class cost stays at `u32::MAX`. Bounded
//! iterations prevent infinite loops.
//!
//! # No recursion
//!
//! The reconstruction step uses an explicit frame-stack to avoid OS stack
//! overflow on deep trees.

use std::collections::HashMap;

use super::build::EGraph;
use super::node::{lowered_binary, lowered_leaf, lowered_unary, ClassId, NodeKind};
use super::union_find::UnionFind;
use crate::eml::LoweredOp;

/// Maximum number of DP iterations for cost relaxation.
const MAX_EXTRACT_ITER: u32 = 128;

/// Extract the cheapest (smallest tree) representative of `root` from `egraph`.
///
/// Falls back to `LoweredOp::Const(0.0)` only if the root class is completely
/// unreachable or has infinite cost (a degenerate e-graph).
pub(crate) fn extract_class(egraph: &EGraph, root: ClassId) -> LoweredOp {
    // Step 1: canonicalize root (note: we don't mutate egraph here, so use immutable find).
    let root_canonical = egraph.union_find.find_root_immutable(root.0);
    let root_id = ClassId(root_canonical);

    // Step 2: BFS to collect all reachable class ids from root.
    let reachable = collect_reachable(&egraph.union_find, &egraph.classes, root_id);

    // Step 3: DP cost relaxation.
    let mut costs: HashMap<ClassId, u32> = HashMap::new();
    let mut best_enode: HashMap<ClassId, (NodeKind, Box<[ClassId]>)> = HashMap::new();

    // Initialize all reachable classes with infinite cost.
    for &cls in &reachable {
        costs.insert(cls, u32::MAX);
    }

    // Relaxation loop.
    let mut changed = true;
    let mut iter = 0u32;
    while changed && iter < MAX_EXTRACT_ITER {
        changed = false;
        iter += 1;

        for &cls in &reachable {
            let current_cost = *costs.get(&cls).unwrap_or(&u32::MAX);

            let nodes = if let Some(eclass) = egraph.classes.get(&cls) {
                eclass.nodes.clone()
            } else {
                continue;
            };

            for enode in &nodes {
                // Compute candidate cost for this enode.
                let mut child_cost_sum: u32 = 0;
                let mut any_infinite = false;

                for &child_id in enode.children.iter() {
                    let canonical_child =
                        ClassId(egraph.union_find.find_root_immutable(child_id.0));
                    let child_cost = *costs.get(&canonical_child).unwrap_or(&u32::MAX);
                    if child_cost == u32::MAX {
                        any_infinite = true;
                        break;
                    }
                    child_cost_sum = child_cost_sum.saturating_add(child_cost);
                }

                if any_infinite {
                    continue;
                }

                let candidate = child_cost_sum.saturating_add(1);
                if candidate < current_cost {
                    costs.insert(cls, candidate);
                    // Record canonical children.
                    let canonical_children: Box<[ClassId]> = enode
                        .children
                        .iter()
                        .map(|c| ClassId(egraph.union_find.find_root_immutable(c.0)))
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    best_enode.insert(cls, (enode.kind.clone(), canonical_children));
                    changed = true;
                }
            }
        }
    }

    // Step 4: Reconstruct LoweredOp from root.
    reconstruct(root_id, &best_enode)
}

/// Collect all reachable class ids from `root` via BFS.
///
/// Children are canonicalized via immutable union-find root lookup.
fn collect_reachable(
    uf: &UnionFind,
    classes: &HashMap<ClassId, super::node::EClass>,
    root: ClassId,
) -> Vec<ClassId> {
    let mut visited: HashMap<ClassId, ()> = HashMap::new();
    let mut queue: Vec<ClassId> = vec![root];

    while let Some(cls) = queue.pop() {
        if visited.contains_key(&cls) {
            continue;
        }
        visited.insert(cls, ());
        if let Some(eclass) = classes.get(&cls) {
            for enode in &eclass.nodes {
                for &child_id in enode.children.iter() {
                    let canonical_child = ClassId(uf.find_root_immutable(child_id.0));
                    if !visited.contains_key(&canonical_child) {
                        queue.push(canonical_child);
                    }
                }
            }
        }
    }

    visited.into_keys().collect()
}

/// Iterative reconstruction of a `LoweredOp` from the DP best-enode table.
///
/// Uses an explicit frame stack to avoid recursion.
fn reconstruct(
    root: ClassId,
    best_enode: &HashMap<ClassId, (NodeKind, Box<[ClassId]>)>,
) -> LoweredOp {
    enum Frame {
        /// Visit this class: push children then Close.
        Open(ClassId),
        /// Children are done: build LoweredOp from kind + children from result stack.
        Close(NodeKind, usize), // kind, n_children
    }

    let mut work: Vec<Frame> = vec![Frame::Open(root)];
    let mut results: Vec<LoweredOp> = Vec::new();

    while let Some(frame) = work.pop() {
        match frame {
            Frame::Open(cls) => {
                match best_enode.get(&cls) {
                    None => {
                        // No best enode — class has infinite cost or is unreachable.
                        // Fallback to Const(0.0) as a valid LoweredOp.
                        results.push(LoweredOp::Const(0.0));
                    }
                    Some((kind, children)) => {
                        let n = children.len();
                        if n == 0 {
                            // Leaf — reconstruct immediately.
                            if let Some(op) = lowered_leaf(kind) {
                                results.push(op);
                            } else {
                                results.push(LoweredOp::Const(0.0));
                            }
                        } else {
                            // Push Close frame, then children (right-first so left pops first).
                            work.push(Frame::Close(kind.clone(), n));
                            for &child in children.iter().rev() {
                                work.push(Frame::Open(child));
                            }
                        }
                    }
                }
            }

            Frame::Close(kind, n_children) => {
                if n_children == 1 {
                    let child = results.pop().unwrap_or(LoweredOp::Const(0.0));
                    let op = lowered_unary(&kind, child).unwrap_or(LoweredOp::Const(0.0));
                    results.push(op);
                } else if n_children == 2 {
                    // Children pushed right-first, so left was pushed last → left pops first.
                    let left = results.pop().unwrap_or(LoweredOp::Const(0.0));
                    let right = results.pop().unwrap_or(LoweredOp::Const(0.0));
                    let op = lowered_binary(&kind, left, right).unwrap_or(LoweredOp::Const(0.0));
                    results.push(op);
                } else {
                    // Unexpected arity — push placeholder.
                    results.push(LoweredOp::Const(0.0));
                }
            }
        }
    }

    results.pop().unwrap_or(LoweredOp::Const(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::LoweredOp;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    #[test]
    fn test_extract_var() {
        let mut eg = EGraph::new();
        let id = eg.add(&var(0));
        let result = extract_class(&eg, id);
        assert_eq!(result, var(0));
    }

    #[test]
    fn test_extract_const() {
        let mut eg = EGraph::new();
        let id = eg.add(&c(3.0));
        let result = extract_class(&eg, id);
        assert_eq!(result, c(3.0));
    }

    #[test]
    fn test_extract_picks_smallest_tree() {
        // Add ln(exp(x)) and x as equivalent classes.
        let mut eg = EGraph::new();
        let ln_exp_x = LoweredOp::Ln(Box::new(LoweredOp::Exp(Box::new(var(0)))));
        let id_ln_exp = eg.add(&ln_exp_x);
        let id_x = eg.add(&var(0));
        // Union them (ln(exp(x)) ≡ x).
        eg.union(id_ln_exp, id_x);
        eg.rebuild();
        // Extract from the root (either id) — should pick x (cost 1 < 3).
        let canonical_root = eg.find(id_ln_exp);
        let result = extract_class(&eg, canonical_root);
        // Either var(0) or ln(exp(x)) is acceptable — just check it's a valid LoweredOp.
        // The DP should pick var(0) since cost 1 < cost 3.
        assert_eq!(result, var(0), "should extract cheapest representative");
    }

    #[test]
    fn test_extract_terminates_on_cyclic_unions() {
        // Union a class with itself — trivial cycle, extract should terminate.
        let mut eg = EGraph::new();
        let id = eg.add(&var(0));
        eg.union(id, id);
        eg.rebuild();
        let result = extract_class(&eg, id);
        assert_eq!(result, var(0));
    }
}
