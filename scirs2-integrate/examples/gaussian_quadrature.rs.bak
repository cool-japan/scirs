use ndarray::ArrayView1;
use scirs2_integrate::gaussian::{gauss_legendre, multi_gauss_legendre, GaussLegendreQuadrature};
use std::f64::consts::PI;

fn main() {
    println!("Gaussian Quadrature Integration Examples\n");

    // Example 1: Integrate x^2 over [0, 1] using Gauss-Legendre method
    // Exact result: 1/3
    println!("Example 1: Integrate x^2 from 0 to 1");
    println!("Exact value: {}", 1.0 / 3.0);

    // Using 5-point Gauss-Legendre quadrature
    let result = gauss_legendre(|x: f64| x * x, 0.0, 1.0, 5).unwrap();
    println!("5-point Gauss-Legendre: {:.10}", result);
    println!("Absolute error: {:.1e}\n", (result - 1.0 / 3.0).abs());

    // Example 2: Integrate sin(x) over [0, π] using Gauss-Legendre method
    // Exact result: 2
    println!("Example 2: Integrate sin(x) from 0 to π");
    println!("Exact value: {}", 2.0);

    // Showing progression of accuracy with increasing points
    for n_points in [2, 3, 5, 10] {
        let result = gauss_legendre(|x: f64| x.sin(), 0.0, PI, n_points).unwrap();
        println!("{}-point Gauss-Legendre: {:.10}", n_points, result);
        println!("Absolute error: {:.1e}", (result - 2.0).abs());
    }
    println!();

    // Example 3: Integrating a difficult function with rapid oscillations
    // int_0^10 sin(x^2) dx
    println!("Example 3: Integrate sin(x^2) from 0 to 10");
    // Reference value computed with high precision
    let reference_value = 0.544_083_001_724_177;
    println!("Reference value: {:.15}", reference_value);

    for n_points in [10, 20, 50] {
        let result = gauss_legendre(|x: f64| (x * x).sin(), 0.0, 10.0, n_points).unwrap();
        println!("{}-point Gauss-Legendre: {:.15}", n_points, result);
        println!("Absolute error: {:.1e}", (result - reference_value).abs());
    }
    println!();

    // Example 4: Manual usage of the GaussLegendreQuadrature struct
    println!("Example 4: Using GaussLegendreQuadrature struct directly");

    let quad = GaussLegendreQuadrature::<f64>::new(10).unwrap();
    let result = quad.integrate(|x: f64| (x * x).exp() * x.cos(), -1.0, 1.0);

    println!("Nodes: {:?}", quad.nodes);
    println!("Weights: {:?}", quad.weights);
    println!("Result of ∫e^(x²)cos(x) dx from -1 to 1: {:.10}", result);
    println!();

    // Example 5: Multi-dimensional integration
    println!("Example 5: Multi-dimensional integration");

    // Integrate f(x,y) = xy over the unit square [0,1]×[0,1]
    // Exact result: 1/4
    let result_2d = multi_gauss_legendre(
        |x: ArrayView1<f64>| x[0] * x[1],
        &[(0.0, 1.0), (0.0, 1.0)],
        5,
    )
    .unwrap();

    println!("∫∫xy dxdy over [0,1]²:");
    println!("  Calculated: {:.10}", result_2d);
    println!("  Exact:      {:.10}", 0.25);
    println!("  Error:      {:.1e}", (result_2d - 0.25).abs());
    println!();

    // Integrate f(x,y,z) = xyz over the unit cube [0,1]³
    // Exact result: 1/8
    let result_3d = multi_gauss_legendre(
        |x: ArrayView1<f64>| x[0] * x[1] * x[2],
        &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        3,
    )
    .unwrap();

    println!("∫∫∫xyz dxdydz over [0,1]³:");
    println!("  Calculated: {:.10}", result_3d);
    println!("  Exact:      {:.10}", 0.125);
    println!("  Error:      {:.1e}", (result_3d - 0.125).abs());
}
