//! Integration tests for the extended Gauss-Legendre quadrature tables (orders 1-10).

use scirs2_integrate::geometric::volume_preserving::gauss_legendre_quadrature;

#[test]
fn test_gauss_legendre_orders_1_to_10_exactness() {
    // GL-n is exact for polynomials of degree 2n-1.
    // Test: integral of x^k over [-1,1] = 2/(k+1) for even k, 0 for odd k.
    for n in 1usize..=10 {
        let (nodes, weights) = gauss_legendre_quadrature(n).expect("should succeed");
        assert_eq!(nodes.len(), n);
        assert_eq!(weights.len(), n);
        for k in 0..(2 * n) {
            let computed: f64 = nodes
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            let exact = if k % 2 == 0 {
                2.0 / (k as f64 + 1.0)
            } else {
                0.0
            };
            assert!(
                (computed - exact).abs() < 1e-12,
                "n={n}, k={k}: got {computed}, expected {exact}, diff={}",
                (computed - exact).abs()
            );
        }
    }
}

#[test]
fn test_gauss_legendre_node_weight_symmetry() {
    for n in 1usize..=10 {
        let (nodes, weights) = gauss_legendre_quadrature(n).expect("should succeed");
        for i in 0..n {
            let j = n - 1 - i;
            assert!(
                (nodes[i] + nodes[j]).abs() < 1e-14,
                "n={n}: nodes[{i}] + nodes[{j}] = {} (should be 0)",
                nodes[i] + nodes[j]
            );
            assert!(
                (weights[i] - weights[j]).abs() < 1e-14,
                "n={n}: weights[{i}] != weights[{j}]"
            );
        }
    }
}

#[test]
fn test_gauss_legendre_weights_sum_to_2() {
    for n in 1usize..=10 {
        let (_, weights) = gauss_legendre_quadrature(n).expect("should succeed");
        let sum: f64 = weights.iter().sum();
        assert!((sum - 2.0).abs() < 1e-13, "n={n}: weight sum = {sum}");
    }
}

#[test]
fn test_gauss_legendre_invalid_orders() {
    assert!(gauss_legendre_quadrature(0).is_err(), "n=0 should fail");
    assert!(gauss_legendre_quadrature(11).is_err(), "n=11 should fail");
}

#[test]
fn test_gauss_legendre_order6_pendulum_hamiltonian_conservation() {
    // Verify key node counts for representative orders.
    let (nodes, weights) = gauss_legendre_quadrature(3).expect("n=3 should succeed");
    assert_eq!(nodes.len(), 3);
    let weight_sum: f64 = weights.iter().sum();
    assert!((weight_sum - 2.0).abs() < 1e-13);

    // Verify n=5 (order 10) also works
    let (nodes5, _) = gauss_legendre_quadrature(5).expect("n=5 should succeed");
    assert_eq!(nodes5.len(), 5);

    // Verify n=10 (order 20) also works
    let (nodes10, _) = gauss_legendre_quadrature(10).expect("n=10 should succeed");
    assert_eq!(nodes10.len(), 10);

    // Hamiltonian conservation check for a simple pendulum using GL-6 quadrature.
    // H(p, q) = p^2/2 - cos(q). We verify that a midpoint-rule estimate of the
    // integral of the Hamiltonian's time derivative over a step is near-zero,
    // which is the property exploited by GL-based variational integrators.
    // Use the n=6 nodes to form a simple collocation estimate.
    let (nodes6, weights6) = gauss_legendre_quadrature(6).expect("n=6 should succeed");
    let dt = 0.01_f64;
    let q0 = 0.5_f64;
    let p0 = 1.0_f64;
    let h0 = p0 * p0 / 2.0 - q0.cos();

    // Interpolate q(t) ≈ q0 + p0*tau*dt (first-order, linear) over one step.
    // The GL quadrature should integrate the Hamiltonian density accurately.
    let h_integrated: f64 = nodes6
        .iter()
        .zip(weights6.iter())
        .map(|(&xi, &wi)| {
            let tau = (xi + 1.0) * 0.5; // map to [0,1]
            let q_tau = q0 + p0 * tau * dt;
            let p_tau = p0 - q0.sin() * tau * dt; // linearised pendulum
            let h_tau = p_tau * p_tau / 2.0 - q_tau.cos();
            wi * h_tau / 2.0 // factor of 1/2 from the interval transformation
        })
        .sum();
    // The average Hamiltonian over the step should be close to H(0) for small dt
    assert!(
        (h_integrated - h0).abs() < 1e-4,
        "Hamiltonian should be nearly conserved over small step: h_integrated={h_integrated}, h0={h0}"
    );
}
