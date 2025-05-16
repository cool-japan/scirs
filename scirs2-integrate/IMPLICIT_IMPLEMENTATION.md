# Implicit Methods for PDEs - Implementation Summary

This document summarizes the implementation of implicit methods for solving partial differential equations (PDEs) in the scirs2-integrate module.

## Implemented Methods

We have implemented the following implicit methods:

1. **Crank-Nicolson Method**
   - Second-order accurate method (A-stable)
   - Implemented for 1D parabolic PDEs
   - Supports variable diffusion coefficients
   - Handles advection and reaction terms
   - Supports various boundary conditions (Dirichlet, Neumann, Robin, Periodic)

2. **Backward Euler Method**
   - First-order L-stable method
   - Excellent stability for stiff problems
   - Implemented for 1D parabolic PDEs
   - Supports the same features as Crank-Nicolson

3. **Alternating Direction Implicit (ADI) Method**
   - Efficient approach for 2D problems
   - Peaceman-Rachford splitting scheme
   - Handles diffusion, advection, and reaction terms
   - Supports all boundary condition types

## Key Features

- **Linear System Solvers**
  - Specialized tridiagonal solver (Thomas algorithm) for 1D methods
  - General linear system solver with partial pivoting for complex cases
  - Efficient algorithm for matrix setup in ADI methods

- **Boundary Condition Handling**
  - Complete support for Dirichlet, Neumann, Robin, and Periodic conditions
  - Specialized discretization stencils for each boundary type
  - Second-order accurate boundary approximations

- **Flexible PDE Terms**
  - Variable coefficients (space, time, and solution dependent)
  - Full advection-diffusion-reaction equation support
  - Builder pattern for adding optional terms

## Example Applications

Three example programs demonstrate the use of these methods:

1. **implicit_heat_equation.rs**
   - Solves the classic heat equation using both Crank-Nicolson and Backward Euler methods
   - Compares solutions with the analytical result
   - Demonstrates stability and accuracy characteristics

2. **implicit_advection_diffusion_reaction.rs**
   - Demonstrates solving stiff advection-diffusion-reaction equations
   - Shows how implicit methods handle problems where explicit methods would require tiny time steps
   - Compares the different methods in terms of stability

3. **implicit_nonlinear_reaction_diffusion.rs**
   - Shows how to handle nonlinear PDEs like the Fisher-KPP equation
   - Demonstrates linearization techniques for nonlinear terms
   - Analyzes the resulting traveling wave solutions

4. **implicit_adi_2d_heat.rs**
   - Solves the 2D heat equation using the ADI method
   - Demonstrates the efficiency of the operator splitting approach
   - Compares with the analytical solution

5. **implicit_adi_advection_diffusion.rs**
   - Shows how to solve 2D advection-diffusion problems using ADI
   - Demonstrates handling advection in multiple directions
   - Analyzes solution properties like mass conservation

## Documentation

A comprehensive guide has been created:

- **implicit_pde_guide.md**
  - Explains all available methods and their properties
  - Provides examples for different PDE types
  - Includes performance considerations and best practices

## Integration

Currently, the implementation exists in the codebase but is commented out in lib.rs for exports due to compilation issues with other parts of the project. Once these issues are resolved, the new functionality can be made available by uncommented the relevant export statements.

## Future Work

Potential enhancements to consider:

1. **Higher Dimension Support**
   - Extend ADI methods to 3D problems using Douglas-Rachford or Brian extension
   - Implement specialized sparse matrix solvers for multi-dimensional problems

2. **Nonlinear Solver Integration**
   - Add support for fully nonlinear reaction terms using Newton iterations
   - Implement adaptive approaches for stiff nonlinear PDEs

3. **Variable Time-Stepping**
   - Add support for adaptive time-stepping based on error estimates
   - Implement predictor-corrector schemes for improved accuracy

4. **Performance Optimization**
   - Optimize matrix assembly for special cases like constant coefficients
   - Implement specialized LU factorization for repeated solves with the same matrix structure