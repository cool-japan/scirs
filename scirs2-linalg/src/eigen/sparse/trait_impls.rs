//! # CsrMatrix - Trait Implementations
//!
//! This module contains trait implementations for `CsrMatrix`.
//!
//! ## Implemented Traits
//!
//! - `SparseMatrix`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::prelude::*;
use std::iter::Sum;

use super::functions::SparseMatrix;
use super::types::CsrMatrix;

impl<F> SparseMatrix<F> for CsrMatrix<F>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    fn nrows(&self) -> usize {
        self.nrows
    }
    fn ncols(&self) -> usize {
        self.ncols
    }
    fn matvec(&self, x: &ArrayView1<F>, y: &mut Array1<F>) -> LinalgResult<()> {
        if x.len() != self.ncols {
            return Err(LinalgError::DimensionError(format!(
                "CsrMatrix::matvec: x has {} elements but matrix has {} columns",
                x.len(),
                self.ncols
            )));
        }
        if y.len() != self.nrows {
            return Err(LinalgError::DimensionError(format!(
                "CsrMatrix::matvec: y has {} elements but matrix has {} rows",
                y.len(),
                self.nrows
            )));
        }
        for i in 0..self.nrows {
            let row_start = if i < self.indptr.len() {
                self.indptr[i]
            } else {
                self.data.len()
            };
            let row_end = if i + 1 < self.indptr.len() {
                self.indptr[i + 1]
            } else {
                self.data.len()
            };
            let mut acc = F::zero();
            for k in row_start..row_end {
                if k < self.data.len() && k < self.indices.len() {
                    let col = self.indices[k];
                    if col < x.len() {
                        acc += self.data[k] * x[col];
                    }
                }
            }
            y[i] = acc;
        }
        Ok(())
    }
    fn is_symmetric(&self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        let n = self.nrows;
        for i in 0..n {
            let row_start = if i < self.indptr.len() {
                self.indptr[i]
            } else {
                return false;
            };
            let row_end = if i + 1 < self.indptr.len() {
                self.indptr[i + 1]
            } else {
                self.data.len()
            };
            for k in row_start..row_end {
                if k >= self.data.len() || k >= self.indices.len() {
                    continue;
                }
                let j = self.indices[k];
                if j >= n {
                    return false;
                }
                let val = self.data[k];
                let ji_start = if j < self.indptr.len() {
                    self.indptr[j]
                } else {
                    return false;
                };
                let ji_end = if j + 1 < self.indptr.len() {
                    self.indptr[j + 1]
                } else {
                    self.data.len()
                };
                let mut found = false;
                for kk in ji_start..ji_end {
                    if kk < self.indices.len()
                        && self.indices[kk] == i
                        && kk < self.data.len()
                        && (self.data[kk] - val).abs() < F::from(1e-14).unwrap_or(F::epsilon())
                    {
                        found = true;
                        break;
                    }
                }
                if !found {
                    return false;
                }
            }
        }
        true
    }
    fn sparsity(&self) -> f64 {
        let total = (self.nrows as f64) * (self.ncols as f64);
        if total == 0.0 {
            return 0.0;
        }
        self.data.len() as f64 / total
    }
}
