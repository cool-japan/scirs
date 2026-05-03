# scirs2-special

[![crates.io](https://img.shields.io/crates/v/scirs2-special.svg)](https://crates.io/crates/scirs2-special)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-special)](https://docs.rs/scirs2-special)

**Special mathematical functions for the SciRS2 scientific computing library (v0.4.3).**

`scirs2-special` provides a comprehensive collection of special mathematical functions modeled after SciPy's `special` module. Version 0.4.3 extends the classical function set with advanced additions: Mathieu functions, Coulomb wave functions, spherical harmonics with Wigner/Gaunt symbols, Jacobi theta functions, Weierstrass elliptic functions, parabolic cylinder functions, Fox H-functions, Appell hypergeometric functions, Meixner-Pollaczek polynomials, Heun functions, polylogarithm, q-analogs, number-theoretic functions, and information-theoretic functions — all as pure Rust.

## Features (v0.4.3)

### Classical Special Functions

#### Gamma and Related
- Gamma function `Γ(z)`, log-gamma `ln Γ(z)`
- Digamma `ψ(z)`, trigamma `ψ'(z)`, polygamma `ψ^(n)(z)`
- Beta function `B(a,b)`, log-beta
- Incomplete gamma: lower `γ(a,x)` and upper `Γ(a,x)`
- Regularized incomplete gamma `P(a,x)`, `Q(a,x)`
- Incomplete beta `I_x(a,b)` and its inverse
- Factorial `n!`, log-factorial, binomial coefficient `C(n,k)`
- Pochhammer symbol (rising factorial)

#### Error Functions
- Error function `erf(x)`, complementary `erfc(x)`
- Scaled complementary `erfcx(x)` (stable for large x)
- Imaginary error function `erfi(x)`
- Dawson integral `F(x)`
- Inverse error function `erfinv(x)`, inverse complementary `erfcinv(x)`

#### Bessel Functions
- First kind `J_n(x)`: orders 0, 1, n (integer and real)
- Second kind `Y_n(x)`: orders 0, 1, n
- Modified first kind `I_n(x)`: exponentially scaled variant
- Modified second kind `K_n(x)`: exponentially scaled variant
- Spherical Bessel: `j_n(x)`, `y_n(x)`
- Hankel functions `H_n^(1)(x)`, `H_n^(2)(x)` (complex)
- Bessel function zeros

#### Elliptic Integrals and Functions
- Complete integrals: `K(k)`, `E(k)`, `Pi(n,k)`
- Incomplete integrals: `F(phi,k)`, `E(phi,k)`, `Pi(phi,n,k)`
- Jacobi elliptic functions: `sn`, `cn`, `dn` and 12 reciprocal/quotient variants
- Carlson symmetric forms: `R_F`, `R_D`, `R_J`, `R_C`

#### Orthogonal Polynomials
- Legendre: `P_n(x)`, associated `P_n^m(x)`
- Chebyshev first kind `T_n(x)` and second kind `U_n(x)`
- Hermite: physicist's `H_n(x)` and probabilist's `He_n(x)`
- Laguerre: `L_n(x)` and associated `L_n^alpha(x)`
- Gegenbauer (ultraspherical): `C_n^lambda(x)`
- Jacobi: `P_n^(alpha,beta)(x)`
- Zernike radial polynomials

#### Airy Functions
- `Ai(x)`, `Bi(x)` and their derivatives `Ai'(x)`, `Bi'(x)`
- Exponentially scaled variants for large |x|
- Complex argument support

#### Hypergeometric Functions
- Confluent hypergeometric limit `_0F_1(b; z)`
- Kummer's confluent `_1F_1(a; b; z)` and Tricomi's `U(a,b,z)`
- Gauss hypergeometric `_2F_1(a,b; c; z)`: analytic continuation for |z| >= 1
- Generalized hypergeometric `_pF_q`

#### Zeta and Related
- Riemann zeta `ζ(s)`, Hurwitz zeta `ζ(s,a)`, Dirichlet eta `η(s)`
- Lerch transcendent `Φ(z,s,a)`
- Lambert W function: real branches W_0 and W_{-1}

#### Other Classical Functions
- Struve functions `H_n(x)` and `L_n(x)` with asymptotic expansions
- Kelvin functions: `ber`, `bei`, `ker`, `kei` and derivatives
- Fresnel integrals `S(x)` and `C(x)` with modulus and phase
- Parabolic cylinder functions: Weber `D_n(x)`, `U(a,x)`, `V(a,x)`
- Spheroidal wave functions: prolate and oblate, angular and radial
- Wright functions: Wright Omega `omega(z)`, Wright Bessel generalizations
- Coulomb wave functions: regular `F_l(eta,rho)`, irregular `G_l(eta,rho)`, `H_l^+/-`
- Logarithmic integral `li(x)`, offset logarithmic integral `Li(x)`
- Exponential integrals `Ei(x)`, `E_n(x)`, `E_1(x)`

### Advanced Special Functions

#### Mathieu Functions
- Characteristic values of even functions `a_r(q)` and odd functions `b_r(q)`
- Even (cosine-elliptic) functions `ce_r(q, x)` with Fourier coefficients
- Odd (sine-elliptic) functions `se_r(q, x)` with Fourier coefficients
- Radial Mathieu functions of the first kind `Mc_r(q, x)` and second kind `Ms_r(q, x)`
- Asymptotic expansions for large |q|

#### Spherical Harmonics and Angular Momentum
- Real spherical harmonics `Y_l^m(theta, phi)` for arbitrary l, m
- Complex spherical harmonics with Condon-Shortley phase
- Gaunt coefficients: integrals of products of three spherical harmonics
- Wigner 3-j symbols by Racah formula
- Wigner 6-j symbols (Racah W-coefficients)
- Wigner 9-j symbols for compound angular momentum coupling
- Clebsch-Gordan coefficients

#### Jacobi Theta Functions
- `theta_1(z, q)`, `theta_2(z, q)`, `theta_3(z, q)`, `theta_4(z, q)`
- Logarithmic derivatives (theta functions of the second kind)
- Quasi-periodicity relations and heat equation connections
- Modular transformations

#### Weierstrass Elliptic Functions
- Weierstrass P-function `p(z; g2, g3)`
- Weierstrass zeta function `zeta(z; g2, g3)`
- Weierstrass sigma function `sigma(z; g2, g3)`
- Elliptic invariants `g2`, `g3` from half-periods `omega1`, `omega2`
- Discriminant `Delta` and Klein j-invariant

#### Parabolic Cylinder Functions (Extended)
- `D_n(x)` for non-integer n via connection to Whittaker functions
- `U(a, x)` and `V(a, x)`: standard parabolic cylinder pair
- Asymptotic expansions for large |x| and large |a|

#### Fox H-Function
- Generalized Fox H-function `H_{p,q}^{m,n}[z | (a,alpha)_p; (b,beta)_q]`
- Series and integral representations
- Special cases: Meijer G-function, Wright function, stable distributions

#### Appell Hypergeometric Functions
- `F_1(a; b1, b2; c; x, y)`: Appell first hypergeometric function
- `F_2(a; b1, b2; c1, c2; x, y)`: Appell second
- `F_3(a1, a2; b1, b2; c; x, y)`: Appell third
- `F_4(a; b; c1, c2; x, y)`: Appell fourth

#### Meixner-Pollaczek Polynomials
- `P_n^lambda(x; phi)`: Meixner-Pollaczek polynomials
- Recurrence relations and generating functions
- Orthogonality weight and inner product

#### Heun Functions
- General Heun function `Hl(a, q; alpha, beta, gamma, delta; z)`
- Confluent Heun function `HeunC`
- Double-confluent Heun function `HeunD`
- Biconfluent Heun function `HeunB`
- Triconfluent Heun function `HeunT`

#### Polylogarithm
- `Li_s(z)` for complex `s` and `z`
- Fermi-Dirac integrals `F_j(x)`
- Bose-Einstein integrals `G_j(x)`
- Clausen function `Cl_2(theta)`

#### Q-Analogs
- Q-Gamma function `Gamma_q(x)`
- Q-Pochhammer symbol `(a; q)_n` and infinite product `(a; q)_infty`
- Q-Binomial coefficient (Gaussian binomial)
- Q-Exponential functions `e_q(z)` and `E_q(z)`
- Q-Bessel functions of the first and second kind
- Q-orthogonal polynomials: big and little q-Jacobi, q-Laguerre, q-Hermite

#### Number Theory Functions
- Ramanujan tau function `tau(n)`
- Euler's totient function `phi(n)` and Jordan's totient
- Liouville function `lambda(n)`, von Mangoldt function `Lambda(n)`
- Mobius function `mu(n)` and Mertens function `M(n)`
- Number of divisors `d(n)`, sum of divisors `sigma_k(n)`
- Partition function `p(n)` via pentagonal recurrence
- Bell polynomials (complete and partial)
- Bernoulli numbers and polynomials; Euler numbers and polynomials
- Stirling numbers of the first and second kind; Lah numbers

#### Information-Theoretic Functions
- KL divergence `D_KL(P || Q)` for discrete and Gaussian distributions
- Jensen-Shannon divergence
- Shannon entropy `H(p)`, Renyi entropy `H_alpha(p)`
- Mutual information and conditional entropy
- Cross-entropy and binary cross-entropy
- Logistic function, softmax, logsumexp (numerically stable)
- Sinc function

#### Combinatorics Extensions
- Bell polynomials `Y_n(x_1,...,x_n)` complete and partial
- Falling factorial, rising factorial (Pochhammer)
- Catalan numbers, Narayana numbers, Motzkin numbers
- Derangement count, subfactorial
- Restricted growth strings and set partition enumeration

#### Orthogonal Polynomial Extensions
- Wilson polynomials and Racah polynomials
- Askey-Wilson polynomials (q-extensions)
- Dual Hahn polynomials, continuous dual Hahn
- Krawtchouk, Meixner, Charlier discrete orthogonal polynomials

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-special = "0.4.3"
```

With parallel processing:

```toml
[dependencies]
scirs2-special = { version = "0.4.3", features = ["parallel"] }
```

### Classical functions

```rust
use scirs2_special::{gamma, bessel, erf, elliptic, orthogonal};

// Gamma(5) = 4! = 24
let g = gamma(5.0f64);
assert!((g - 24.0).abs() < 1e-10);

// Bessel J_0(2.0)
let j0 = bessel::j0(2.0f64);
println!("J_0(2) = {}", j0);

// erf(1.0)
let e = erf::erf(1.0f64);
println!("erf(1) = {}", e);

// Complete elliptic integral K(0.5)
let k = elliptic::ellipk(0.5f64)?;
println!("K(0.5) = {}", k);

// Legendre polynomial P_3(0.5)
let p3 = orthogonal::legendre(3, 0.5f64)?;
println!("P_3(0.5) = {}", p3);
```

### Mathieu functions

```rust
use scirs2_special::mathieu::{mathieu_a, mathieu_ce};

// Characteristic value a_0(q=1.0)
let a0 = mathieu_a(0, 1.0f64)?;
println!("a_0(1.0) = {}", a0);

// Even Mathieu function ce_0(q=1.0, x=0.5)
let ce0 = mathieu_ce(0, 1.0f64, 0.5f64)?;
println!("ce_0(1.0, 0.5) = {}", ce0);
```

### Spherical harmonics and Wigner symbols

```rust
use scirs2_special::spherical_harmonics::{sph_harm, wigner_3j, gaunt};

// Y_2^1(theta=0.5, phi=1.0)
let y21 = sph_harm(2, 1, 0.5f64, 1.0f64)?;
println!("Y_2^1 = {:?}", y21); // complex value

// Wigner 3-j symbol (j1=1, j2=1, j3=2, m1=-1, m2=0, m3=1)
let w3j = wigner_3j(1, 1, 2, -1, 0, 1)?;
println!("Wigner 3-j = {}", w3j);

// Gaunt coefficient integral of Y_2^1 * Y_2^0 * Y_2^(-1)
let g = gaunt(2, 1, 2, 0, 2, -1)?;
println!("Gaunt = {}", g);
```

### Jacobi theta functions

```rust
use scirs2_special::theta_functions::{theta_1, theta_3};

// theta_1(z=0.5, q=0.3)
let t1 = theta_1(0.5f64, 0.3f64)?;
println!("theta_1(0.5, 0.3) = {}", t1);

// theta_3(z=0.0, q=0.5) -- the Jacobi nome series
let t3 = theta_3(0.0f64, 0.5f64)?;
println!("theta_3(0, 0.5) = {}", t3);
```

### Polylogarithm

```rust
use scirs2_special::polylogarithm::polylog;
use num_complex::Complex64;

// Li_2(0.5) = pi^2/12 - (ln 0.5)^2 / 2
let li2 = polylog(2.0f64, Complex64::new(0.5, 0.0))?;
println!("Li_2(0.5) = {}", li2.re);
```

### Q-analogs

```rust
use scirs2_special::q_analogs::{q_gamma, q_pochhammer};

// Q-Gamma(4, q=0.5) -> should approach 3! = 6 as q -> 1
let qg = q_gamma(4.0f64, 0.5f64)?;
println!("Gamma_q(4; q=0.5) = {}", qg);

// Q-Pochhammer (0.3; 0.5)_5
let qp = q_pochhammer(0.3f64, 0.5f64, 5)?;
println!("(0.3; 0.5)_5 = {}", qp);
```

### Information-theoretic functions

```rust
use scirs2_special::information_theoretic::{kl_divergence, shannon_entropy, mutual_information};

let p = vec![0.25, 0.25, 0.25, 0.25f64];
let q = vec![0.5, 0.25, 0.125, 0.125f64];

let kl = kl_divergence(&p, &q)?;
let h = shannon_entropy(&p)?;
println!("KL(P||Q) = {}, H(P) = {}", kl, h);
```

## API Overview

| Module | Description |
|--------|-------------|
| `gamma` | Gamma, log-gamma, digamma, beta, incomplete variants |
| `erf` | Error, complementary error, scaled, imaginary, Dawson |
| `bessel` | J, Y, I, K Bessel; spherical Bessel; Hankel; zeros |
| `elliptic` / `elliptic_ext` | Complete/incomplete integrals; Jacobi functions; Carlson |
| `orthogonal` | Legendre, Chebyshev, Hermite, Laguerre, Gegenbauer, Jacobi, Zernike |
| `airy` / `airy_ext` | Ai, Bi and derivatives; complex argument |
| `hypergeometric` / `hypergeometric_ext` | _0F_1, _1F_1, _2F_1, U, Tricomi |
| `mathieu` | Characteristic values, ce/se functions, radial Mc/Ms |
| `spherical_harmonics` | Real/complex Y_l^m; Gaunt; Wigner 3j/6j/9j; CG coefficients |
| `theta_functions` | Jacobi theta theta_1 through theta_4 |
| `weierstrass` | Weierstrass P, zeta, sigma; elliptic invariants |
| `parabolic_cylinder` | Weber D_n, U, V; asymptotic expansions |
| `fox_h` | Generalized Fox H-function |
| `appell` | Appell F_1, F_2, F_3, F_4 hypergeometric |
| `meixner_pollaczek` | Meixner-Pollaczek polynomials P_n^lambda |
| `heun` | General, confluent, double-confluent, biconfluent, triconfluent Heun |
| `polylogarithm` | Li_s(z), Fermi-Dirac, Bose-Einstein, Clausen |
| `q_analogs` | Q-Gamma, q-Pochhammer, q-exponential, q-binomial |
| `q_bessel` | Q-Bessel functions of first and second kind |
| `q_orthogonal` | Q-Jacobi, q-Laguerre, q-Hermite polynomials |
| `number_theory` / `number_theory_ext` | Ramanujan tau, totient, Mobius, divisors, partitions |
| `information_theoretic` | KL divergence, entropy, mutual information |
| `combinatorial_ext` | Bell polynomials, Catalan, Narayana, Motzkin, Stirling |
| `orthogonal_polynomials` | Wilson, Askey-Wilson, Racah, Krawtchouk |
| `zeta` / `zeta_ext` | Riemann zeta, Hurwitz zeta, Lerch transcendent |
| `struve` / `struve_ext` | Struve H and L with asymptotic expansions |
| `combinatorial` | Factorials, binomial, multinomial, Stirling numbers |
| `lattice` | Lattice functions and modular forms |
| `statistical` | Logistic, softmax, logsumexp, sinc |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | All core and advanced special functions |
| `parallel` | Rayon-based parallel array evaluation |
| `simd` | SIMD-accelerated batch computation via scirs2-core |
| `pdf` | PDF export support via `printpdf` (default: off) |

## Documentation

Full API documentation is available at [docs.rs/scirs2-special](https://docs.rs/scirs2-special).

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
