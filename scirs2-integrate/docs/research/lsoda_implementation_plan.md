# LSODA Implementation Plan

## Overview
This document outlines the implementation plan for the LSODA (Livermore Solver for Ordinary Differential Equations with Automatic method switching) method in the scirs2-integrate crate.

## Phase 1: Core Infrastructure (Initial Implementation)

### Data Structures
1. **LSodaState**
   - Current ODE state (time, solution, derivatives)
   - Method state (current method in use: Adams or BDF)
   - Integration history (needed for multistep methods)
   - Jacobian information and cache
   - Method switching statistics

2. **StiffnessDetector**
   - Algorithm to detect stiffness based on step size and stability
   - Heuristics for efficiency comparison between methods
   - Optional eigenvalue estimation

### Core Components
1. **LSODA Method Entry Point** (lsoda_method function)
   - Initialize integration
   - Handle method switching
   - Process results and statistics

2. **Method-Specific Implementations**
   - Adams method for non-stiff phases
     - Predictor-corrector approach
     - Variable order implementation
   - BDF method for stiff phases
     - Reuse existing BDF implementation
     - Adapt for method switching

3. **Method Switching Logic**
   - Criteria for switching from Adams to BDF
   - Criteria for switching from BDF to Adams
   - State transfer between methods

## Phase 2: Advanced Features

1. **Performance Optimizations**
   - Efficient Jacobian handling
   - Jacobian reuse strategy
   - Heuristic fine-tuning

2. **Robustness Improvements**
   - Better error estimation
   - Advanced step size control
   - Order selection strategies

3. **User-Friendly Features**
   - Option to influence initial method choice
   - Detailed statistics on method switching
   - Dense output for continuous solutions

## Implementation Roadmap

### Task 1: Adams Method Implementation
- Implement variable-order Adams predictor-corrector
- Support for order 1-12
- Error estimation
- Step size control

### Task 2: Stiffness Detection
- Implement basic stability-based detection
- Add heuristics for method efficiency
- Test on standard stiff/non-stiff problems

### Task 3: Method Switching
- Implement mechanism to transition between methods
- Ensure consistent state during transitions
- Maintain accuracy during method switches

### Task 4: Integration with Existing Framework
- Connect with the main solve_ivp interface
- Reuse compatible components from BDF implementation
- Implement proper error handling

### Task 5: Testing and Validation
- Test on standard benchmark problems
- Verify against SciPy's LSODA implementation
- Profile performance and optimize

## Detailed Algorithm Outline

### Main LSODA Loop
```rust
fn lsoda_method<F, Func>(f: Func, t_span: [F; 2], y0: Array1<F>, opts: ODEOptions<F>) -> IntegrateResult<ODEResult<F>>
{
    // Initialize with Adams method (non-stiff)
    let mut method = Method::Adams;
    let mut lsoda_state = LsodaState::new(y0, &opts);
    
    // Main integration loop
    while t < t_end {
        // Attempt step with current method
        let step_result = match method {
            Method::Adams => adams_step(&mut lsoda_state, &f, &opts),
            Method::BDF => bdf_step(&mut lsoda_state, &f, &opts),
        };
        
        // Handle result
        match step_result {
            Ok(_) => {
                // Step successful - check for method switching
                if method == Method::Adams && stiffness_detector.is_stiff(&lsoda_state) {
                    // Switch from Adams to BDF
                    method = Method::BDF;
                    lsoda_state.prepare_for_method_switch();
                } else if method == Method::BDF && !stiffness_detector.is_stiff(&lsoda_state) {
                    // Switch from BDF to Adams
                    method = Method::Adams;
                    lsoda_state.prepare_for_method_switch();
                }
                
                // Update state for next step
                lsoda_state.advance();
            },
            Err(StepError::TooStiff) => {
                // Force switch to BDF regardless of current method
                method = Method::BDF;
                lsoda_state.prepare_for_method_switch();
            },
            Err(StepError::NotStiff) => {
                // Force switch to Adams regardless of current method
                method = Method::Adams;
                lsoda_state.prepare_for_method_switch();
            },
            Err(e) => return Err(e.into()),
        }
    }
    
    // Prepare and return final result
    Ok(lsoda_state.into_result(method))
}
```

### Stiffness Detection
```rust
impl StiffnessDetector {
    fn is_stiff(&self, state: &LsodaState) -> bool {
        // Criteria for stiffness:
        
        // 1. Step size heuristic
        let h_heuristic = self.step_size_ratio_suggests_stiffness(state);
        
        // 2. Iteration efficiency
        let iter_heuristic = self.iteration_efficiency_suggests_stiffness(state);
        
        // 3. Optional eigenvalue estimation (more expensive but accurate)
        let eigenvalue_heuristic = if self.use_eigenvalue_estimation {
            self.eigenvalues_suggest_stiffness(state)
        } else {
            false
        };
        
        // Combine heuristics with weights
        h_heuristic || iter_heuristic || eigenvalue_heuristic
    }
}
```

## Testing Strategy

1. **Unit Tests**
   - Individual components (Adams, BDF, stiffness detection)
   - Method switching logic
   - Error handling

2. **Integration Tests**
   - Standard ODE problems
   - Stiff problems
   - Non-stiff problems
   - Problems that change character (start non-stiff, become stiff)

3. **Benchmarks**
   - Performance comparison with separate Adams and BDF methods
   - Overhead measurement for stiffness detection
   - Comparison with other libraries on select problems

## Completion Criteria

The LSODA implementation will be considered complete when:

1. It correctly solves a standard set of test problems with accuracy comparable to SciPy's implementation
2. It automatically detects stiffness and switches methods appropriately
3. It provides clear statistics about method switching
4. It is robust and handles edge cases appropriately
5. The implementation is well-documented and easy to maintain