# LSODA Method Research Notes

## Overview
LSODA (Livermore Solver for Ordinary Differential Equations with Automatic method switching) is an adaptive solver that automatically switches between non-stiff and stiff methods. It uses Adams methods (explicit) for non-stiff regions and BDF methods (implicit) for stiff regions.

## Key Features
- Automatic detection of stiffness during the integration process
- Seamless switching between appropriate methods
- Efficient handling of systems that change character during integration

## Method Components
1. **Adams Methods** (for non-stiff parts):
   - Explicit multistep method
   - Variable order (typically 1-12)
   - Efficient for non-stiff regions
   - Similar to Adams-Bashforth-Moulton methods

2. **BDF Methods** (for stiff parts):
   - Backward Differentiation Formula
   - Implicit multistep method
   - Variable order (typically 1-5)
   - A-stable at lower orders (essential for stiff problems)

3. **Stiffness Detection**:
   - Monitors eigenvalues of the Jacobian matrix
   - Uses heuristics based on integration efficiency
   - Checks if the problem would be more efficiently solved with a different method

## Automatic Method Switching Criteria
- **From Adams to BDF**: When stiffness is detected:
  - Step size becomes very small relative to the scale of the problem
  - Newton iterations converge too slowly for Adams method
  - Heuristic indicates Adams method is inefficient for current state

- **From BDF to Adams**: When stiffness disappears:
  - Step size increases significantly
  - BDF method is determined to be less efficient than Adams would be
  - Jacobian eigenvalues indicate non-stiff behavior

## Implementation Strategy
1. **Base Components**:
   - Reuse existing Adams-Moulton predictor-corrector from RK methods
   - Reuse existing BDF implementation from stiff solver
   - Implement eigenvalue estimation for the Jacobian matrix
   - Create switching logic between methods

2. **Key Functions Required**:
   - Stiffness detection algorithm
   - Method switching with state transfer
   - Order and step size selection for both methods
   - Efficient Jacobian handling and updates

3. **Error Estimation and Step Control**:
   - Similar to existing adaptive methods
   - Need to handle method transitions carefully
   - Order selection strategies for both methods

## Implementation Phases
1. **Phase 1**: Core method switching infrastructure
   - Basic stiffness detection
   - Simple switching logic
   - Essential error control

2. **Phase 2**: Advanced features
   - Optimized stiffness detection
   - Improved Jacobian handling
   - Advanced error estimation and step control
   - Order adaptation strategies

3. **Phase 3**: Optimization and robustness
   - Performance tuning
   - Edge case handling
   - Comprehensive testing

## SciPy LSODA Details
- Based on the ODEPACK Fortran library
- Dynamic switching between Adams (non-stiff) and BDF (stiff)
- API very similar to other scipy.integrate solvers
- Main parameters:
  - Relative and absolute tolerances
  - Method selection (lsoda automatically switches, but can start with either method)
  - Jacobian handling options (analytical, numerical, sparsity)

## Key Challenges
1. **Stiffness Detection**: Accurate and efficient detection without excessive computation
2. **Method Switching**: Smooth transition with minimal disruption to accuracy and stability
3. **Jacobian Handling**: Efficient computation and reuse of Jacobian information
4. **Order Selection**: Optimal order selection for both methods independently

## References
1. Petzold, L. (1983). Automatic selection of methods for solving stiff and nonstiff systems of ordinary differential equations. SIAM Journal on Scientific and Statistical Computing, 4(1), 136-148.
2. Hindmarsh, A. C. (1983). ODEPACK, a systematized collection of ODE solvers. Scientific Computing, 55-64.
3. SciPy implementation: https://github.com/scipy/scipy/blob/main/scipy/integrate/_odepack.py

## Next Steps
1. Study ODEPACK's LSODA implementation in detail
2. Review SciPy's Python wrapper for LSODA
3. Implement core stiffness detection algorithm
4. Design method switching infrastructure
5. Integrate with existing ODE solver framework