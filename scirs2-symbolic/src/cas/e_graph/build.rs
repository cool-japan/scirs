//! E-graph construction: `add`, `union`, and `rebuild`.
//!
//! The `EGraph` is the core data structure. It maintains:
//! - A `UnionFind` to track equivalence classes.
//! - A map from `ClassId` to `EClass` (nodes + parent back-edges).
//! - A `hashcons` table mapping canonicalized `ENode`s to their class.
//! - A `pending_rebuild` queue of classes whose parents need re-checking.
//!
//! # Rebuild invariant
//!
//! After any `union(a, b)`, the hashcons may be stale: two enodes that
//! previously differed (because their children were in different classes)
//! may now be equivalent. `rebuild` processes the `pending_rebuild` queue:
//! for each pending class, it canonicalizes its parent enodes and checks
//! for hashcons collisions, merging any that appear. Bounded by
//! `MAX_REBUILD_ITER = 64`.
//!
//! # No recursion
//!
//! `add` uses an explicit work-stack. `rebuild` uses an iterative worklist.

use hashbrown::HashMap;

use super::node::{
    node_kind_of_binary, node_kind_of_leaf, node_kind_of_unary, ClassId, EClass, ENode, NodeKind,
};
use super::union_find::UnionFind;
use crate::eml::LoweredOp;

/// Maximum number of rebuild iterations per `rebuild()` call.
pub(crate) const MAX_REBUILD_ITER: u32 = 64;

/// The e-graph. Central structure for equality saturation.
pub(crate) struct EGraph {
    pub union_find: UnionFind,
    /// Canonical class id → class data.
    pub classes: HashMap<ClassId, EClass>,
    /// Hashcons: canonical enode → canonical class id.
    pub hashcons: HashMap<ENode, ClassId>,
    /// Classes whose parents need re-hashconsing after a union.
    pub pending_rebuild: Vec<ClassId>,
    /// Total number of enodes ever created (for budget checks).
    pub node_count: u32,
}

impl EGraph {
    /// Create an empty e-graph.
    pub fn new() -> Self {
        EGraph {
            union_find: UnionFind::new(),
            classes: HashMap::new(),
            hashcons: HashMap::new(),
            pending_rebuild: Vec::new(),
            node_count: 0,
        }
    }

    /// Canonicalize a `ClassId` by following the union-find.
    pub fn find(&mut self, id: ClassId) -> ClassId {
        ClassId(self.union_find.find(id.0))
    }

    /// Build a canonical `ENode` by resolving all children through `find`.
    fn canonicalize_enode(&mut self, node: &ENode) -> ENode {
        let canonical_children: Vec<ClassId> = node
            .children
            .iter()
            .map(|c| ClassId(self.union_find.find(c.0)))
            .collect();
        ENode {
            kind: node.kind.clone(),
            children: canonical_children.into_boxed_slice(),
        }
    }

    /// Add an `ENode` to the graph, returning its class id.
    ///
    /// Canonicalizes children, checks the hashcons, and creates a new class
    /// if there is no existing entry.
    ///
    /// Updates the parent back-edges of all children's classes.
    fn add_enode(&mut self, node: ENode) -> ClassId {
        let canonical = self.canonicalize_enode(&node);
        if let Some(&existing) = self.hashcons.get(&canonical) {
            let canonical_existing = ClassId(self.union_find.find(existing.0));
            return canonical_existing;
        }
        // Hashcons miss — create new class.
        let raw_id = self.union_find.make_set();
        let class_id = ClassId(raw_id);
        self.node_count += 1;
        let eclass = EClass::new(class_id, canonical.clone());
        self.classes.insert(class_id, eclass);
        self.hashcons.insert(canonical.clone(), class_id);
        // Register this enode as a parent of each child class.
        let canonical_parent = ClassId(self.union_find.find(class_id.0));
        for &child_id in canonical.children.iter() {
            let canonical_child = ClassId(self.union_find.find(child_id.0));
            if let Some(cls) = self.classes.get_mut(&canonical_child) {
                cls.parents.push((canonical.clone(), canonical_parent));
            }
        }
        class_id
    }

    /// Bottom-up iterative insert of a `LoweredOp`. Returns the `ClassId` of `op`.
    ///
    /// Uses an explicit work-stack to avoid recursion, even for deeply-nested
    /// `LoweredOp` trees (up to 10,000-deep chains).
    pub fn add(&mut self, op: &LoweredOp) -> ClassId {
        /// Frames for the iterative post-order traversal.
        enum AddFrame {
            /// First visit: push children, then schedule Close.
            Open(LoweredOp),
            /// Children are done: build ENode and push result.
            Close { kind: NodeKind, n_children: usize },
        }

        let mut work: Vec<AddFrame> = vec![AddFrame::Open(op.clone())];
        let mut result_stack: Vec<ClassId> = Vec::new();

        while let Some(frame) = work.pop() {
            match frame {
                AddFrame::Open(current) => {
                    // Leaf: build immediately, push Close with 0 children.
                    if let Some(kind) = node_kind_of_leaf(&current) {
                        work.push(AddFrame::Close {
                            kind,
                            n_children: 0,
                        });
                        continue;
                    }
                    // Unary: push Close then child.
                    if let Some(kind) = node_kind_of_unary(&current) {
                        let child = match &current {
                            LoweredOp::Neg(c)
                            | LoweredOp::Exp(c)
                            | LoweredOp::Ln(c)
                            | LoweredOp::Sin(c)
                            | LoweredOp::Cos(c)
                            | LoweredOp::Tan(c)
                            | LoweredOp::Sinh(c)
                            | LoweredOp::Cosh(c)
                            | LoweredOp::Tanh(c)
                            | LoweredOp::Arcsin(c)
                            | LoweredOp::Arccos(c)
                            | LoweredOp::Arctan(c)
                            | LoweredOp::Arcsinh(c)
                            | LoweredOp::Arccosh(c)
                            | LoweredOp::Arctanh(c)
                            | LoweredOp::Sqrt(c)
                            | LoweredOp::Abs(c) => c.as_ref().clone(),
                            _ => unreachable!("unary matched but no child extracted"),
                        };
                        work.push(AddFrame::Close {
                            kind,
                            n_children: 1,
                        });
                        work.push(AddFrame::Open(child));
                        continue;
                    }
                    // Binary: push Close, right child, left child.
                    if let Some(kind) = node_kind_of_binary(&current) {
                        let (left, right) = match &current {
                            LoweredOp::Add(l, r)
                            | LoweredOp::Sub(l, r)
                            | LoweredOp::Mul(l, r)
                            | LoweredOp::Div(l, r)
                            | LoweredOp::Pow(l, r) => (l.as_ref().clone(), r.as_ref().clone()),
                            _ => unreachable!("binary matched but no children extracted"),
                        };
                        work.push(AddFrame::Close {
                            kind,
                            n_children: 2,
                        });
                        // Push right first so left pops first.
                        work.push(AddFrame::Open(right));
                        work.push(AddFrame::Open(left));
                        continue;
                    }
                    // Should never reach here if LoweredOp variants are exhaustive.
                    // Defensive: treat as a leaf Const(0) rather than panic.
                    work.push(AddFrame::Close {
                        kind: NodeKind::Const(0u64),
                        n_children: 0,
                    });
                }

                AddFrame::Close { kind, n_children } => {
                    // Collect child class ids from result_stack.
                    let children_start = result_stack.len().saturating_sub(n_children);
                    let children: Vec<ClassId> = result_stack.drain(children_start..).collect();
                    let enode = ENode {
                        kind,
                        children: children.into_boxed_slice(),
                    };
                    let class_id = self.add_enode(enode);
                    result_stack.push(class_id);
                }
            }
        }

        // The last element is the root class id.
        result_stack.pop().unwrap_or_else(|| {
            // Defensive: if result_stack is somehow empty, add Const(0) as fallback.
            let enode = ENode::leaf(NodeKind::Const(0u64));
            self.add_enode(enode)
        })
    }

    /// Merge the equivalence classes of `a` and `b`.
    ///
    /// Returns `true` if a merge was performed (a and b were in different
    /// classes), `false` if they were already equivalent.
    pub fn union(&mut self, a: ClassId, b: ClassId) -> bool {
        let ra = ClassId(self.union_find.find(a.0));
        let rb = ClassId(self.union_find.find(b.0));
        if ra == rb {
            return false;
        }
        let (winner_raw, loser_raw) = self.union_find.union(ra.0, rb.0);
        let winner_id = ClassId(winner_raw);
        let loser_id = ClassId(loser_raw);

        // Move loser's nodes into winner's class.
        // We must carefully borrow-split here: remove loser, then update winner.
        if let Some(loser_class) = self.classes.remove(&loser_id) {
            // Collect the data we need before mutably borrowing classes.
            let loser_nodes = loser_class.nodes;
            let loser_parents = loser_class.parents;

            if let Some(winner_class) = self.classes.get_mut(&winner_id) {
                winner_class.nodes.extend(loser_nodes);
                winner_class.parents.extend(loser_parents);
            }
            // Schedule winner for rebuild.
            self.pending_rebuild.push(winner_id);
        } else {
            // Loser was already gone (merged earlier). Still schedule winner.
            self.pending_rebuild.push(winner_id);
        }
        true
    }

    /// Repair the hashcons after a batch of unions.
    ///
    /// For each pending class, re-canonicalizes every parent enode (so children
    /// now refer to current roots). If two parent enodes that were previously
    /// distinct become identical after canonicalization, they are merged.
    ///
    /// Bounded by `MAX_REBUILD_ITER = 64` outer iterations.
    pub fn rebuild(&mut self) {
        let mut iters = 0u32;
        while !self.pending_rebuild.is_empty() && iters < MAX_REBUILD_ITER {
            iters += 1;
            let worklist: Vec<ClassId> = self.pending_rebuild.drain(..).collect();
            for class_id in worklist {
                let canonical_class = ClassId(self.union_find.find(class_id.0));
                // Collect parents from the canonical class.
                let parents = if let Some(cls) = self.classes.get(&canonical_class) {
                    cls.parents.clone()
                } else {
                    continue;
                };

                // Re-hashcons each parent enode.
                for (parent_enode, parent_class) in parents {
                    let canonical_parent_class = ClassId(self.union_find.find(parent_class.0));
                    // Canonicalize the parent enode's children.
                    let canonical_children: Box<[ClassId]> = parent_enode
                        .children
                        .iter()
                        .map(|c| ClassId(self.union_find.find(c.0)))
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let canonical_enode = ENode {
                        kind: parent_enode.kind.clone(),
                        children: canonical_children,
                    };

                    match self.hashcons.get(&canonical_enode).copied() {
                        Some(existing_class) => {
                            let canonical_existing =
                                ClassId(self.union_find.find(existing_class.0));
                            if canonical_existing != canonical_parent_class {
                                // Hashcons collision — merge them.
                                // union will add to pending_rebuild.
                                if self.union(canonical_existing, canonical_parent_class) {
                                    // Re-insert hashcons with the new winner.
                                    let winner =
                                        ClassId(self.union_find.find(canonical_parent_class.0));
                                    self.hashcons.insert(canonical_enode, winner);
                                }
                            }
                        }
                        None => {
                            // No entry — insert the canonicalized enode.
                            self.hashcons
                                .insert(canonical_enode, canonical_parent_class);
                        }
                    }
                }
            }
        }
    }

    /// Return the total number of distinct enodes in the graph.
    pub fn total_nodes(&self) -> u32 {
        self.node_count
    }
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
    fn test_add_idempotent() {
        let mut eg = EGraph::new();
        let op = LoweredOp::Add(Box::new(var(0)), Box::new(c(0.0)));
        let id1 = eg.add(&op);
        let id2 = eg.add(&op);
        // Same structural expression → same class id (hashcons dedup).
        assert_eq!(eg.find(id1), eg.find(id2));
    }

    #[test]
    fn test_hashcons_dedup() {
        let mut eg = EGraph::new();
        let op1 = LoweredOp::Add(Box::new(var(0)), Box::new(var(1)));
        let op2 = LoweredOp::Add(Box::new(var(0)), Box::new(var(1)));
        let id1 = eg.add(&op1);
        let id2 = eg.add(&op2);
        assert_eq!(
            eg.find(id1),
            eg.find(id2),
            "same structual op should reuse class"
        );
        // The hashcons should only have a small number of unique classes.
        // x (0), y (1), x+y (2 unique enodes for x, y, x+y = 3)
        assert!(eg.hashcons.len() <= 5, "hashcons should deduplicate");
    }

    #[test]
    fn test_union_propagates() {
        let mut eg = EGraph::new();
        let a = eg.add(&var(0));
        let b = eg.add(&var(1));
        assert_ne!(eg.find(a), eg.find(b));
        eg.union(a, b);
        assert_eq!(
            eg.find(a),
            eg.find(b),
            "after union, both should have same root"
        );
    }

    #[test]
    fn test_rebuild_fixpoint() {
        let mut eg = EGraph::new();
        // Add x+0.
        let x_plus_0 = LoweredOp::Add(Box::new(var(0)), Box::new(c(0.0)));
        let id_xp0 = eg.add(&x_plus_0);
        let id_x = eg.add(&var(0));
        // They should initially be different classes.
        assert_ne!(eg.find(id_xp0), eg.find(id_x));
        // Union x+0 with x (simulating the rewrite x+0→x).
        eg.union(id_xp0, id_x);
        eg.rebuild();
        // After rebuild, both should find the same root.
        assert_eq!(
            eg.find(id_xp0),
            eg.find(id_x),
            "after rebuild, x+0 and x should be in same class"
        );
    }

    #[test]
    fn test_add_leaf_const_var() {
        let mut eg = EGraph::new();
        let ic = eg.add(&c(3.0));
        let iv = eg.add(&var(2));
        // Leaves should always produce distinct classes.
        assert_ne!(eg.find(ic), eg.find(iv));
        assert!(eg.total_nodes() >= 2);
    }

    #[test]
    fn test_deep_chain_no_overflow() {
        // 1000-deep Add(x, Const(0)) chain — must not stack-overflow.
        let mut op = var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(c(0.0)));
        }
        let mut eg = EGraph::new();
        let _id = eg.add(&op);
        // No panic = success.
    }
}
