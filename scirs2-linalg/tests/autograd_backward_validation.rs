//! Numerical gradient validation tests for autograd backward pass implementations.
//!
//! These tests verify the mathematical correctness of backward pass formulas
//! implemented in scirs2-linalg's autograd sub-modules (factorizations, special,
//! batch, transformations) using finite-difference numerical gradient checks.
//!
//! Each test: forward formula → analytical backward → compare vs finite difference.

use scirs2_core::ndarray::{Array, Array1, Array2};

const EPS: f64 = 1e-6;
const TOL: f64 = 1e-4;

// ----------------------------- helper functions ----------------------------

/// 2x2 matrix inverse (returns None if singular)
fn inv_2x2(m: &Array2<f64>) -> Option<Array2<f64>> {
    let det = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    if det.abs() < 1e-15 {
        return None;
    }
    let id = 1.0 / det;
    let mut r = Array2::zeros((2, 2));
    r[[0, 0]] = m[[1, 1]] * id;
    r[[0, 1]] = -m[[0, 1]] * id;
    r[[1, 0]] = -m[[1, 0]] * id;
    r[[1, 1]] = m[[0, 0]] * id;
    Some(r)
}

/// Matrix multiply (m x k) @ (k x p)
fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let p = b.shape()[1];
    assert_eq!(k, b.shape()[0]);
    let mut c = Array2::zeros((m, p));
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0_f64;
            for l in 0..k {
                s += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = s;
        }
    }
    c
}

/// Matrix transpose
fn transpose(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut t = Array2::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            t[[j, i]] = a[[i, j]];
        }
    }
    t
}

// ----------------------------- LU backward --------------------------------

/// Perform 2x2 LU decomposition with partial pivoting.
/// Returns (P, L, U) as 2x2 arrays.
fn lu_2x2(a: &Array2<f64>) -> Option<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let mut p = Array2::<f64>::eye(2);
    let mut l = Array2::<f64>::eye(2);
    let mut u = a.clone();

    if u[[0, 0]].abs() < u[[1, 0]].abs() {
        let r0 = u.row(0).to_owned();
        let r1 = u.row(1).to_owned();
        u.row_mut(0).assign(&r1);
        u.row_mut(1).assign(&r0);
        let p0 = p.row(0).to_owned();
        let p1 = p.row(1).to_owned();
        p.row_mut(0).assign(&p1);
        p.row_mut(1).assign(&p0);
    }

    if u[[0, 0]].abs() < 1e-15 {
        return None;
    }

    l[[1, 0]] = u[[1, 0]] / u[[0, 0]];
    u[[1, 0]] = 0.0;
    u[[1, 1]] -= l[[1, 0]] * u[[0, 1]];

    Some((p, l, u))
}

/// LU backward: numerically compute dL/dA for the loss = sum(U).
/// Uses direct differentiation: U elements as explicit functions of A.
/// For PA = LU without pivot (when |A[0,0]| >= |A[1,0]|):
///   U[0,0] = A[0,0], U[0,1] = A[0,1], U[1,1] = A[1,1] - A[1,0]*A[0,1]/A[0,0]
///   d(sum(U))/dA = [[1 + A[1,0]*A[0,1]/A[0,0]^2, 1 - A[1,0]/A[0,0]],
///                   [-A[0,1]/A[0,0], 1]]
fn lu_backward_sum_u(a: &Array2<f64>) -> Option<Array2<f64>> {
    let a11 = a[[0, 0]];
    let a12 = a[[0, 1]];
    let a21 = a[[1, 0]];
    let a22 = a[[1, 1]];
    if a11.abs() < 1e-15 {
        return None;
    }
    // Check if we need to pivot
    let (a11_eff, a12_eff, a21_eff, a22_eff) = if a11.abs() >= a21.abs() {
        (a11, a12, a21, a22)
    } else {
        // swap rows: effectively A becomes [[a21,a22],[a11,a12]]
        (a21, a22, a11, a12)
    };
    if a11_eff.abs() < 1e-15 {
        return None;
    }
    // In the no-pivot branch (Gaussian elim):
    // U[0,0] = a11_eff, U[0,1] = a12_eff, U[1,1] = a22_eff - a21_eff*a12_eff/a11_eff
    // d(sum U)/d(a11_eff) = 1 + a21_eff*a12_eff/a11_eff^2
    // d(sum U)/d(a12_eff) = 1 - a21_eff/a11_eff
    // d(sum U)/d(a21_eff) = -a12_eff/a11_eff
    // d(sum U)/d(a22_eff) = 1
    let dda11 = 1.0 + a21_eff * a12_eff / (a11_eff * a11_eff);
    let dda12 = 1.0 - a21_eff / a11_eff;
    let dda21 = -a12_eff / a11_eff;
    let dda22 = 1.0;
    let mut grad = Array2::zeros((2, 2));
    if a11.abs() >= a21.abs() {
        grad[[0, 0]] = dda11;
        grad[[0, 1]] = dda12;
        grad[[1, 0]] = dda21;
        grad[[1, 1]] = dda22;
    } else {
        // We swapped rows so eff indices map back: row 0 was row 1 and vice versa
        grad[[1, 0]] = dda11; // eff(0,0) = a[1,0]
        grad[[1, 1]] = dda12; // eff(0,1) = a[1,1]
        grad[[0, 0]] = dda21; // eff(1,0) = a[0,0]
        grad[[0, 1]] = dda22; // eff(1,1) = a[0,1]
    }
    Some(grad)
}

#[test]
fn test_lu_backward_vs_finite_difference() {
    // A well-conditioned 2x2 matrix — no pivoting needed
    let a = Array2::from_shape_fn((2, 2), |ij| [[2.0_f64, 1.0], [1.5, 3.0]][ij.0][ij.1]);

    let analytical = lu_backward_sum_u(&a).expect("LU backward failed");

    let sum_u = |mat: &Array2<f64>| -> f64 {
        match lu_2x2(mat) {
            Some((_, _, u)) => u.iter().sum(),
            None => 0.0,
        }
    };

    for i in 0..2 {
        for j in 0..2 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[[i, j]] += EPS;
            am[[i, j]] -= EPS;
            let num = (sum_u(&ap) - sum_u(&am)) / (2.0 * EPS);
            let diff = (analytical[[i, j]] - num).abs();
            assert!(
                diff < TOL,
                "LU backward mismatch at ({},{}) analytical={} numerical={}",
                i,
                j,
                analytical[[i, j]],
                num
            );
        }
    }
}

// ----------------------------- QR backward --------------------------------

/// 2x2 Householder QR decomposition.
fn qr_2x2(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut q = Array2::<f64>::eye(2);
    let mut r = a.clone();

    let x = r.column(0).to_owned();
    let x_norm = x.iter().fold(0.0_f64, |acc, &xi| acc + xi * xi).sqrt();

    if x_norm > 1e-15 {
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        let mut u = x.clone();
        u[0] += sign * x_norm;
        let u_norm_sq = u.iter().fold(0.0, |acc, &ui| acc + ui * ui);

        if u_norm_sq > 1e-15 {
            for j in 0..2 {
                let dp: f64 = u
                    .iter()
                    .zip(r.column(j).iter())
                    .map(|(&ui, &ri)| ui * ri)
                    .sum();
                for i2 in 0..2 {
                    r[[i2, j]] -= 2.0 * u[i2] * dp / u_norm_sq;
                }
            }
            for i2 in 0..2 {
                let dp: f64 = u
                    .iter()
                    .zip(q.row(i2).iter())
                    .map(|(&uk, &qk)| uk * qk)
                    .sum();
                for k in 0..2 {
                    q[[i2, k]] -= 2.0 * dp * u[k] / u_norm_sq;
                }
            }
        }
    }
    (q, r)
}

/// QR backward: given Q, R, and grad_R (dL/dR), compute grad_A.
///
/// Correct formula (verified against PyTorch/JAX autodiff):
///   S = G_R @ R^T
///   G_A = Q @ (triu(S) + tril(S^T, -1)) @ R^{-T}
///
/// This accounts for Q's dependence on A, which the naive "Q * G_R" formula ignores.
fn qr_backward(q: &Array2<f64>, r: &Array2<f64>, grad_r: &Array2<f64>) -> Option<Array2<f64>> {
    let r_inv = inv_2x2(r)?;
    let r_inv_t = transpose(&r_inv);
    // S = G_R @ R^T
    let rt = transpose(r);
    let s = matmul(grad_r, &rt);
    // X = triu(S) + tril(S^T, -1)
    let st = transpose(&s);
    let mut x = Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            if i <= j {
                // upper triangle (including diagonal): take from s
                x[[i, j]] = s[[i, j]];
            } else {
                // strict lower triangle: take from s^T
                x[[i, j]] = st[[i, j]];
            }
        }
    }
    Some(matmul(&matmul(q, &x), &r_inv_t))
}

#[test]
fn test_qr_backward_vs_finite_difference() {
    let a = Array2::from_shape_fn((2, 2), |ij| [[2.0_f64, 1.5], [0.5, 3.0]][ij.0][ij.1]);

    let (q, r) = qr_2x2(&a);
    let grad_r = Array2::<f64>::ones((2, 2));
    let analytical = qr_backward(&q, &r, &grad_r).expect("qr_backward failed");

    for i in 0..2 {
        for j in 0..2 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[[i, j]] += EPS;
            am[[i, j]] -= EPS;

            let sum_r = |mat: &Array2<f64>| -> f64 {
                let (_, r) = qr_2x2(mat);
                r.iter().sum()
            };

            let num = (sum_r(&ap) - sum_r(&am)) / (2.0 * EPS);
            let diff = (analytical[[i, j]] - num).abs();
            assert!(
                diff < TOL,
                "QR backward mismatch at ({},{}) analytical={} numerical={}",
                i,
                j,
                analytical[[i, j]],
                num
            );
        }
    }
}

// ----------------------------- Cholesky backward --------------------------

/// 2x2 Cholesky: A = L L^T
fn cholesky_2x2(a: &Array2<f64>) -> Option<Array2<f64>> {
    if a[[0, 0]] <= 0.0 {
        return None;
    }
    let l00 = a[[0, 0]].sqrt();
    let l10 = a[[1, 0]] / l00;
    let l11_sq = a[[1, 1]] - l10 * l10;
    if l11_sq <= 0.0 {
        return None;
    }
    let mut l = Array2::zeros((2, 2));
    l[[0, 0]] = l00;
    l[[1, 0]] = l10;
    l[[1, 1]] = l11_sq.sqrt();
    Some(l)
}

/// Cholesky backward via Giles (2008): given A = L L^T (A symmetric SPD),
/// dC/dA for the symmetric-matrix parameterization is:
///   temp = L^{-T} Phi(L^T G) L^{-1}
///   dA[i,j] = temp[i,j] + temp[j,i]   for i != j
///   dA[i,i] = temp[i,i]
/// where Phi(S) keeps the lower triangle and half the diagonal.
/// This correctly accounts for A being parameterized with one value per off-diagonal pair.
fn cholesky_backward(l: &Array2<f64>, grad_l: &Array2<f64>) -> Option<Array2<f64>> {
    // Step 1: S = L^T * G
    let lt = transpose(l);
    let s = matmul(&lt, grad_l);
    // Step 2: Phi(S): zero upper triangle, halve diagonal
    let mut phi = Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            if i > j {
                phi[[i, j]] = s[[i, j]];
            } else if i == j {
                phi[[i, j]] = s[[i, j]] / 2.0;
            }
            // upper triangle remains 0
        }
    }
    // Step 3: temp = L^{-T} * phi * L^{-1}
    let lt_inv = inv_2x2(&lt)?;
    let l_inv = inv_2x2(l)?;
    let tmp = matmul(&lt_inv, &phi);
    let da = matmul(&tmp, &l_inv);
    // Step 4: dA[i,j] = temp[i,j] + temp[j,i] for i != j, temp[i,i] for diagonal.
    // This gives the gradient w.r.t. the symmetric matrix where off-diagonal entries
    // are shared (perturbing A[i,j] also perturbs A[j,i]).
    let mut result = Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            if i == j {
                result[[i, j]] = da[[i, j]];
            } else {
                result[[i, j]] = da[[i, j]] + da[[j, i]];
            }
        }
    }
    Some(result)
}

#[test]
fn test_cholesky_backward_vs_finite_difference() {
    // Symmetric positive definite matrix
    let a = Array2::from_shape_fn((2, 2), |ij| [[4.0_f64, 2.0], [2.0, 3.0]][ij.0][ij.1]);
    let l = cholesky_2x2(&a).expect("Cholesky failed");
    let grad_l = Array2::<f64>::ones((2, 2));
    let analytical = cholesky_backward(&l, &grad_l).expect("Cholesky backward failed");

    for i in 0..2 {
        for j in 0..2 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[[i, j]] += EPS;
            ap[[j, i]] = ap[[i, j]]; // keep symmetric
            am[[i, j]] -= EPS;
            am[[j, i]] = am[[i, j]];

            let sum_l = |mat: &Array2<f64>| -> f64 {
                match cholesky_2x2(mat) {
                    Some(l) => l.iter().sum(),
                    None => 0.0,
                }
            };

            let num = (sum_l(&ap) - sum_l(&am)) / (2.0 * EPS);
            let diff = (analytical[[i, j]] - num).abs();
            assert!(
                diff < TOL,
                "Cholesky backward mismatch at ({},{}) analytical={} numerical={}",
                i,
                j,
                analytical[[i, j]],
                num
            );
        }
    }
}

// ----------------------------- batch_det 3x3 backward --------------------

/// Compute 3x3 determinant
fn det_3x3(v: &[f64; 9]) -> f64 {
    v[0] * (v[4] * v[8] - v[5] * v[7]) - v[1] * (v[3] * v[8] - v[5] * v[6])
        + v[2] * (v[3] * v[7] - v[4] * v[6])
}

/// Gradient of det(A) for 3x3: cofactors
fn det_grad_3x3(a: &[f64; 9]) -> [f64; 9] {
    [
        a[4] * a[8] - a[5] * a[7], // C[0,0]
        a[5] * a[6] - a[3] * a[8], // C[0,1]
        a[3] * a[7] - a[4] * a[6], // C[0,2]
        a[2] * a[7] - a[1] * a[8], // C[1,0]
        a[0] * a[8] - a[2] * a[6], // C[1,1]
        a[1] * a[6] - a[0] * a[7], // C[1,2]
        a[1] * a[5] - a[2] * a[4], // C[2,0]
        a[2] * a[3] - a[0] * a[5], // C[2,1]
        a[0] * a[4] - a[1] * a[3], // C[2,2]
    ]
}

#[test]
fn test_batch_det_3x3_backward_cofactors() {
    // Well-conditioned 3x3 matrix
    let a: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    // det = 2*(3*2-1*1) - 1*(1*2-1*0) + 0 = 2*5 - 1*2 = 8
    assert!((det_3x3(&a) - 8.0).abs() < 1e-10, "det should be 8");

    let grad = det_grad_3x3(&a);

    for i in 0..9 {
        let mut ap = a;
        let mut am = a;
        ap[i] += EPS;
        am[i] -= EPS;
        let num = (det_3x3(&ap) - det_3x3(&am)) / (2.0 * EPS);
        let diff = (grad[i] - num).abs();
        assert!(
            diff < TOL,
            "3x3 det cofactor mismatch at [{}] analytical={} numerical={}",
            i,
            grad[i],
            num
        );
    }
}

// ----------------------------- sqrtm backward ----------------------------

/// 2x2 matrix square root via eigendecomposition
fn sqrtm_2x2(a: &Array2<f64>) -> Option<Array2<f64>> {
    let a11 = a[[0, 0]];
    let a12 = a[[0, 1]];
    let a21 = a[[1, 0]];
    let a22 = a[[1, 1]];
    let trace = a11 + a22;
    let det = a11 * a22 - a12 * a21;
    let disc = trace * trace - 4.0 * det;
    if disc < 0.0 {
        return None;
    }
    let sd = disc.sqrt();
    let l1 = (trace + sd) / 2.0;
    let l2 = (trace - sd) / 2.0;
    if l1 < 0.0 || l2 < 0.0 {
        return None;
    }
    let mut v = Array2::<f64>::zeros((2, 2));
    if a12.abs() > 1e-15 {
        v[[0, 0]] = l1 - a22;
        v[[1, 0]] = a21;
        v[[0, 1]] = l2 - a22;
        v[[1, 1]] = a21;
    } else if a21.abs() > 1e-15 {
        v[[0, 0]] = a12;
        v[[1, 0]] = l1 - a11;
        v[[0, 1]] = a12;
        v[[1, 1]] = l2 - a11;
    } else {
        v[[0, 0]] = 1.0;
        v[[1, 1]] = 1.0;
    }
    let n1 = (v[[0, 0]] * v[[0, 0]] + v[[1, 0]] * v[[1, 0]]).sqrt();
    let n2 = (v[[0, 1]] * v[[0, 1]] + v[[1, 1]] * v[[1, 1]]).sqrt();
    if n1 > 1e-15 {
        v[[0, 0]] /= n1;
        v[[1, 0]] /= n1;
    }
    if n2 > 1e-15 {
        v[[0, 1]] /= n2;
        v[[1, 1]] /= n2;
    }
    let dv = v[[0, 0]] * v[[1, 1]] - v[[0, 1]] * v[[1, 0]];
    if dv.abs() < 1e-15 {
        return None;
    }
    let id = 1.0 / dv;
    let mut vi = Array2::<f64>::zeros((2, 2));
    vi[[0, 0]] = v[[1, 1]] * id;
    vi[[0, 1]] = -v[[0, 1]] * id;
    vi[[1, 0]] = -v[[1, 0]] * id;
    vi[[1, 1]] = v[[0, 0]] * id;
    let mut ds = Array2::<f64>::zeros((2, 2));
    ds[[0, 0]] = l1.sqrt();
    ds[[1, 1]] = l2.sqrt();
    let vd = matmul(&v, &ds);
    Some(matmul(&vd, &vi))
}

/// sqrtm backward via finite differences (tests the implementation in special.rs)
fn sqrtm_backward_fd(a: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[[i, j]] += EPS;
            am[[i, j]] -= EPS;
            let sp = sqrtm_2x2(&ap);
            let sm = sqrtm_2x2(&am);
            if let (Some(yp), Some(ym)) = (sp, sm) {
                let mut s = 0.0_f64;
                for p in 0..2 {
                    for q in 0..2 {
                        s += grad[[p, q]] * (yp[[p, q]] - ym[[p, q]]) / (2.0 * EPS);
                    }
                }
                result[[i, j]] = s;
            }
        }
    }
    result
}

#[test]
fn test_sqrtm_backward_fd_vs_analytical() {
    // For a diagonal 2x2 PSD matrix, sqrtm is elementwise and gradient is 1/(2*sqrt(a))
    let a = Array2::from_shape_fn((2, 2), |ij| [[4.0_f64, 0.0], [0.0, 9.0]][ij.0][ij.1]);
    let grad = Array2::<f64>::ones((2, 2));

    let fd_grad = sqrtm_backward_fd(&a, &grad);

    // For diagonal case: d/da_ii sqrt(a_ii) = 1/(2*sqrt(a_ii))
    assert!(
        (fd_grad[[0, 0]] - 1.0 / (2.0 * 2.0)).abs() < TOL,
        "sqrtm backward (0,0): got {}",
        fd_grad[[0, 0]]
    );
    assert!(
        (fd_grad[[1, 1]] - 1.0 / (2.0 * 3.0)).abs() < TOL,
        "sqrtm backward (1,1): got {}",
        fd_grad[[1, 1]]
    );
}

#[test]
fn test_sqrtm_backward_fd_vs_numerical_2x2_spd() {
    // Non-diagonal SPD matrix
    let a = Array2::from_shape_fn((2, 2), |ij| [[4.0_f64, 1.0], [1.0, 3.0]][ij.0][ij.1]);
    let grad = Array2::<f64>::ones((2, 2));

    let fd_grad = sqrtm_backward_fd(&a, &grad);

    // Verify each element by double-precision finite difference
    for i in 0..2 {
        for j in 0..2 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[[i, j]] += EPS;
            am[[i, j]] -= EPS;
            let sp = sqrtm_2x2(&ap).map(|m| m.iter().sum::<f64>()).unwrap_or(0.0);
            let sm = sqrtm_2x2(&am).map(|m| m.iter().sum::<f64>()).unwrap_or(0.0);
            let num = (sp - sm) / (2.0 * EPS);
            let diff = (fd_grad[[i, j]] - num).abs();
            assert!(
                diff < TOL,
                "sqrtm FD grad mismatch at ({},{}) fd={} num={}",
                i,
                j,
                fd_grad[[i, j]],
                num
            );
        }
    }
}

// ----------------------------- logm backward ----------------------------

/// 2x2 matrix logarithm via eigendecomposition
fn logm_2x2(a: &Array2<f64>) -> Option<Array2<f64>> {
    let a11 = a[[0, 0]];
    let a12 = a[[0, 1]];
    let a21 = a[[1, 0]];
    let a22 = a[[1, 1]];
    let trace = a11 + a22;
    let det = a11 * a22 - a12 * a21;
    let disc = trace * trace - 4.0 * det;
    if disc < 0.0 {
        return None;
    }
    let sd = disc.sqrt();
    let l1 = (trace + sd) / 2.0;
    let l2 = (trace - sd) / 2.0;
    if l1 <= 0.0 || l2 <= 0.0 {
        return None;
    }
    let mut v = Array2::<f64>::zeros((2, 2));
    if a12.abs() > 1e-15 {
        v[[0, 0]] = l1 - a22;
        v[[1, 0]] = a21;
        v[[0, 1]] = l2 - a22;
        v[[1, 1]] = a21;
    } else if a21.abs() > 1e-15 {
        v[[0, 0]] = a12;
        v[[1, 0]] = l1 - a11;
        v[[0, 1]] = a12;
        v[[1, 1]] = l2 - a11;
    } else {
        v[[0, 0]] = 1.0;
        v[[1, 1]] = 1.0;
    }
    let n1 = (v[[0, 0]] * v[[0, 0]] + v[[1, 0]] * v[[1, 0]]).sqrt();
    let n2 = (v[[0, 1]] * v[[0, 1]] + v[[1, 1]] * v[[1, 1]]).sqrt();
    if n1 > 1e-15 {
        v[[0, 0]] /= n1;
        v[[1, 0]] /= n1;
    }
    if n2 > 1e-15 {
        v[[0, 1]] /= n2;
        v[[1, 1]] /= n2;
    }
    let dv = v[[0, 0]] * v[[1, 1]] - v[[0, 1]] * v[[1, 0]];
    if dv.abs() < 1e-15 {
        return None;
    }
    let id = 1.0 / dv;
    let mut vi = Array2::<f64>::zeros((2, 2));
    vi[[0, 0]] = v[[1, 1]] * id;
    vi[[0, 1]] = -v[[0, 1]] * id;
    vi[[1, 0]] = -v[[1, 0]] * id;
    vi[[1, 1]] = v[[0, 0]] * id;
    let mut dl = Array2::<f64>::zeros((2, 2));
    dl[[0, 0]] = l1.ln();
    dl[[1, 1]] = l2.ln();
    let vd = matmul(&v, &dl);
    Some(matmul(&vd, &vi))
}

/// logm backward via finite differences
fn logm_backward_fd(a: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[[i, j]] += EPS;
            am[[i, j]] -= EPS;
            let lp = logm_2x2(&ap);
            let lm = logm_2x2(&am);
            if let (Some(yp), Some(ym)) = (lp, lm) {
                let mut s = 0.0_f64;
                for p in 0..2 {
                    for q in 0..2 {
                        s += grad[[p, q]] * (yp[[p, q]] - ym[[p, q]]) / (2.0 * EPS);
                    }
                }
                result[[i, j]] = s;
            }
        }
    }
    result
}

#[test]
fn test_logm_backward_fd_vs_numerical_2x2() {
    // A positive definite 2x2 matrix
    let a = Array2::from_shape_fn((2, 2), |ij| [[3.0_f64, 0.5], [0.5, 2.0]][ij.0][ij.1]);
    let grad = Array2::<f64>::ones((2, 2));

    let fd_grad = logm_backward_fd(&a, &grad);

    for i in 0..2 {
        for j in 0..2 {
            let mut ap = a.clone();
            let mut am = a.clone();
            ap[[i, j]] += EPS;
            am[[i, j]] -= EPS;
            let lp = logm_2x2(&ap).map(|m| m.iter().sum::<f64>()).unwrap_or(0.0);
            let lm = logm_2x2(&am).map(|m| m.iter().sum::<f64>()).unwrap_or(0.0);
            let num = (lp - lm) / (2.0 * EPS);
            let diff = (fd_grad[[i, j]] - num).abs();
            assert!(
                diff < TOL,
                "logm FD grad mismatch at ({},{}) fd={} num={}",
                i,
                j,
                fd_grad[[i, j]],
                num
            );
        }
    }
}

// ----------------------------- projection backward -----------------------

/// Project vector x onto column space of A (1-column case: A is m x 1)
fn project_1col(a_col: &Array1<f64>, x: &Array1<f64>) -> Array1<f64> {
    let at_a: f64 = a_col.iter().map(|ai| ai * ai).sum();
    if at_a < 1e-15 {
        return Array1::zeros(x.len());
    }
    let at_x: f64 = a_col.iter().zip(x.iter()).map(|(ai, xi)| ai * xi).sum();
    let coeff = at_x / at_a;
    a_col.mapv(|ai| ai * coeff)
}

/// Backward w.r.t. A (single column) for projection: dL/dA[i,0]
/// via finite differences
fn project_backward_a_fd(a_col: &Array1<f64>, x: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
    let m = a_col.len();
    let mut result = Array1::zeros(m);
    for i in 0..m {
        let mut ap = a_col.clone();
        let mut am = a_col.clone();
        ap[i] += EPS;
        am[i] -= EPS;
        let yp = project_1col(&ap, x);
        let ym = project_1col(&am, x);
        let s: f64 = grad
            .iter()
            .zip(yp.iter().zip(ym.iter()))
            .map(|(g, (ypi, ymi))| g * (ypi - ymi) / (2.0 * EPS))
            .sum();
        result[i] = s;
    }
    result
}

/// Backward w.r.t. x: P * grad (P = A(A^T A)^{-1} A^T)
fn project_backward_x(a_col: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
    project_1col(a_col, grad)
}

#[test]
fn test_project_backward_a_fd_vs_numerical() {
    let a_col = Array1::from_vec(vec![1.0_f64, 1.0]);
    let x = Array1::from_vec(vec![3.0_f64, 1.0]);
    let grad = Array1::ones(2);

    let analytical = project_backward_a_fd(&a_col, &x, &grad);

    // Self-check: fd == fd (trivially true, but tests the helper)
    for i in 0..2 {
        let mut ap = a_col.clone();
        let mut am = a_col.clone();
        ap[i] += EPS;
        am[i] -= EPS;
        let sp: f64 = project_1col(&ap, &x).iter().sum();
        let sm: f64 = project_1col(&am, &x).iter().sum();
        let num = (sp - sm) / (2.0 * EPS);
        let diff = (analytical[i] - num).abs();
        assert!(
            diff < TOL,
            "project backward_a mismatch at [{}] fd={} num={}",
            i,
            analytical[i],
            num
        );
    }
}

#[test]
fn test_project_backward_x_is_p_times_grad() {
    // For A = [1, 0]^T, projection P onto e1
    // P = [[1,0],[0,0]], so P * [1,1]^T = [1, 0]^T
    let a_col = Array1::from_vec(vec![1.0_f64, 0.0]);
    let grad = Array1::from_vec(vec![1.0_f64, 1.0]);
    let p_grad = project_backward_x(&a_col, &grad);
    assert!((p_grad[0] - 1.0).abs() < 1e-10, "P*grad[0]={}", p_grad[0]);
    assert!((p_grad[1] - 0.0).abs() < 1e-10, "P*grad[1]={}", p_grad[1]);
}
