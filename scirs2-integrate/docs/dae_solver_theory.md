# DAE Solver Theory and Implementation

This document explains the mathematical theory behind the Differential Algebraic Equation (DAE) solvers implemented in the `scirs2-integrate` library, with a focus on the specialized Backward Differentiation Formula (BDF) methods.

## 1. Introduction to DAEs

Differential Algebraic Equations (DAEs) are a combination of differential equations and algebraic constraints. They are more general than Ordinary Differential Equations (ODEs) and appear naturally in many physical systems where constraints are present.

### 1.1 Forms of DAEs

DAEs can be classified into several forms:

#### Semi-explicit form

The semi-explicit form separates the differential and algebraic parts:

```
x'(t) = f(x(t), y(t), t)
0 = g(x(t), y(t), t)
```

where `x` are the differential variables, `y` are the algebraic variables, and `g` represents the constraints.

#### Fully implicit form

The fully implicit form combines all equations:

```
F(t, y(t), y'(t)) = 0
```

where `y` includes both differential and algebraic variables.

### 1.2 DAE Index

The index of a DAE is a measure of how far it is from being an ODE. It roughly corresponds to the number of times you need to differentiate the constraints to obtain a system of ODEs.

- **Index-1 DAEs**: Can be converted to ODEs with one differentiation
- **Index-2 DAEs**: Require two differentiations
- **Index-3 and higher**: Require three or more differentiations

Higher-index DAEs are more challenging to solve numerically.

## 2. Backward Differentiation Formula (BDF) Methods

BDF methods are implicit linear multistep methods particularly effective for stiff ODEs and DAEs. 

### 2.1 BDF Formula

The k-th order BDF formula approximates the derivative as:

```
y'_{n+1} ≈ (1/h) * Σ(j=0 to k) α_j * y_{n+1-j}
```

where `h` is the step size, and `α_j` are the BDF coefficients.

For example, the coefficients for orders 1-5 are:

- Order 1 (Backward Euler): α = [1, -1]
- Order 2: α = [1, -4/3, 1/3]
- Order 3: α = [1, -18/11, 9/11, -2/11]
- Order 4: α = [1, -48/25, 36/25, -16/25, 3/25]
- Order 5: α = [1, -300/137, 300/137, -200/137, 75/137, -12/137]

### 2.2 BDF for ODEs

For an ODE system `y' = f(t, y)`, the BDF method requires solving:

```
Σ(j=0 to k) α_j * y_{n+1-j} = h * f(t_{n+1}, y_{n+1})
```

This is a nonlinear equation that can be solved using Newton's method.

## 3. BDF Methods for DAEs

Applying BDF to DAEs requires special considerations.

### 3.1 BDF for Semi-explicit DAEs

For semi-explicit DAEs:

```
x'(t) = f(x(t), y(t), t)
0 = g(x(t), y(t), t)
```

The BDF discretization gives:

```
Σ(j=0 to k) α_j * x_{n+1-j} = h * f(t_{n+1}, x_{n+1}, y_{n+1})
0 = g(t_{n+1}, x_{n+1}, y_{n+1})
```

This is a coupled system of nonlinear algebraic equations for `(x_{n+1}, y_{n+1})`. 

### 3.2 BDF for Fully Implicit DAEs

For fully implicit DAEs:

```
F(t, y(t), y'(t)) = 0
```

We approximate `y'` using the BDF formula and solve:

```
F(t_{n+1}, y_{n+1}, (1/h) * Σ(j=0 to k) α_j * y_{n+1-j}) = 0
```

This is a nonlinear equation for `y_{n+1}`.

## 4. Implementation Details

### 4.1 Newton's Method for Nonlinear Equations

At each step, we need to solve a nonlinear system using Newton's method:

1. Start with an initial guess (often from extrapolation)
2. Compute the Jacobian matrix (or an approximation of it)
3. Solve a linear system to get the Newton step
4. Update the solution and check for convergence
5. Repeat until convergence or maximum iterations

### 4.2 Jacobian Approximation

The Jacobian can be:
- Computed analytically if the derivatives are available
- Approximated using finite differences
- Approximated using automatic differentiation

### 4.3 Adaptivity

Our implementation uses several adaptive strategies:

**Step Size Control**:
- Estimate local error at each step
- Adjust step size based on error estimates
- Use safety factors to prevent too large changes

**Order Control**:
- Start with a low order (typically 1 or 2)
- Gradually increase order as more past points become available
- Maximum order is limited to 5 for numerical stability

### 4.4 Constraint Handling

For DAEs, constraint satisfaction is crucial:

- Solve constraints simultaneously with differential equations
- Use projection methods to correct constraint violations
- Implement stabilization techniques for higher-index DAEs

## 5. Index Reduction Techniques

For higher-index DAEs, we implement index reduction techniques:

### 5.1 Method of Dummy Derivatives

Introduce additional variables to represent higher derivatives and additional constraints from differentiating the original constraints.

### 5.2 Projection Methods

Project the numerical solution onto the constraint manifold after each step:

1. Solve the system without enforcing constraints
2. Project the solution onto the constraint manifold
3. Continue with the projected solution

### 5.3 Pantelides Algorithm

A systematic approach to reduce the index of a DAE:

1. Create a graph representation of equations and variables
2. Identify structural singularities
3. Add differentiated equations to remove singularities

## 6. Performance Considerations

### 6.1 Linear Solver Selection

The choice of linear solver affects performance:

- Direct solvers: LU decomposition (good for small to medium problems)
- Iterative solvers: GMRES, BiCGSTAB (better for large sparse systems)
- Preconditioners: Important for accelerating iterative solvers

### 6.2 Jacobian Reuse

Computing the Jacobian is expensive. Strategies include:

- Reuse the Jacobian for multiple steps
- Update the Jacobian only when convergence is slow
- Use approximate Jacobians when suitable

### 6.3 Specialized Techniques for Large Systems

For large systems:

- Exploit sparsity patterns
- Use parallel Jacobian computation
- Implement Krylov subspace methods for linear systems

## 7. Future Enhancements

Planned enhancements to our DAE solvers include:

- Krylov-enhanced implicit solvers for large DAEs
- Block-structured preconditioners
- Adjoint methods for sensitivity analysis
- Quasi-Newton methods for reduced computational cost
- Parallelized BDF methods for high-performance computing environments

## 8. References

1. K. E. Brenan, S. L. Campbell, and L. R. Petzold, "Numerical Solution of Initial-Value Problems in Differential-Algebraic Equations"
2. E. Hairer and G. Wanner, "Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems"
3. E. Hairer, S. P. Nørsett, and G. Wanner, "Solving Ordinary Differential Equations I: Nonstiff Problems"
4. U. M. Ascher and L. R. Petzold, "Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations"
5. P. N. Brown, A. C. Hindmarsh, and L. R. Petzold, "Consistent Initial Condition Calculation for Differential-Algebraic Systems"