# Bayesian Optimization Design Summary from scikit-optimize

## Core Components

### 1. Space Definition
- `Space` class that manages different parameter types
- Three main parameter types:
  - `Real`: Continuous parameters (with uniform or log-uniform prior)
  - `Integer`: Integer parameters (with uniform or log-uniform prior)
  - `Categorical`: Categorical parameters (with one-hot encoding by default)
- Each parameter type handles its own transformation and inverse transformation
- Parameter bounds and constraints are clearly defined

### 2. Surrogate Models
- Gaussian Process (GP) is the default model
- Also supports:
  - Random Forest
  - Gradient Boosted Regression Trees (GBRT)
  - Extra Trees
- Models must implement the scikit-learn estimator interface with `predict` that supports `return_std=True`

### 3. Acquisition Functions
- Expected Improvement (EI): Default, balances exploration and exploitation
- Lower Confidence Bound (LCB): Configurable exploration-exploitation tradeoff
- Probability of Improvement (PI): More aggressive exploitation
- GP-Hedge: Probabilistically selects from multiple acquisition functions
- Variants for time-aware optimization (EIps, PIps)

### 4. Acquisition Optimization
- Two strategies:
  - `sampling`: Randomly sample points and pick the best (used for categorical spaces)
  - `lbfgs`: Use L-BFGS-B optimizer for continuous spaces
- Allows for multiple restarts to avoid local minima

### 5. Optimizer Loop
- Base optimizer implements the core Bayesian optimization loop
- Separates ask/tell interface from the optimization loop
- Supports both synchronous and asynchronous evaluation
- Handles initialization strategies
- Manages model queuing and history

## Key Design Patterns

### 1. Transformer Pipeline
- Parameters undergo transformations for proper modeling
- Each dimension type has its own transformer
- Transformers implement `transform` and `inverse_transform`
- Pipeline of transformers allows composition

### 2. Ask/Tell Interface
- `ask()`: Get the next point to evaluate
- `tell(x, y)`: Update the model with new observation
- Enables custom optimization loops
- Supports batch evaluation with constant liar strategy

### 3. Initial Point Generators
- Different strategies for initial sampling:
  - Random (default)
  - Sobol sequences
  - Halton sequences
  - Latin Hypercube Sampling (LHS)
  - Grid sampling

### 4. Result Object
- Standard result object with:
  - Best parameter values
  - Function value at minimum
  - All evaluated points and values
  - Surrogate models
  - Search space information

### 5. Callback System
- Callbacks evaluate after each iteration
- Enable monitoring, logging, and early stopping
- Verbose callbacks for progress tracking

## Key Algorithms

### Bayesian Optimization Loop
1. Initialize with random or specified points
2. For each iteration:
   - Fit surrogate model to all observations
   - Optimize acquisition function to find next sampling point
   - Evaluate objective function at the new point
   - Update observations and model

### Acquisition Function Optimization
1. For sampling strategy:
   - Generate random points in the search space
   - Evaluate acquisition function at these points
   - Select point with minimum acquisition value

2. For L-BFGS-B strategy:
   - Start multiple L-BFGS-B runs from different initial points
   - Run local optimization for each
   - Select best minimum across all runs

### GP-Hedge Strategy
1. Maintain a set of acquisition functions
2. For each iteration:
   - Optimize each acquisition function to get candidate points
   - Calculate gain for each acquisition function
   - Probabilistically select one candidate using softmax on gains
   - Update gains based on performance

## Implementation Considerations for Rust

1. **Surrogate Models**:
   - Need a Gaussian Process implementation
   - Consider NDArray for matrix operations
   - Handle numerical stability issues

2. **Parameter Space**:
   - Create ergonomic API for parameter space definition
   - Support proper transformations for different parameter types
   - Efficient sampling from different distributions

3. **Acquisition Functions**:
   - Implement gradient computation for efficient optimization
   - Handle numerical edge cases properly

4. **Optimization**:
   - Implement proper L-BFGS-B optimizer or equivalent
   - Support parallelism for acquisition function optimization

5. **Async Support**:
   - Design for both synchronous and asynchronous evaluation
   - Support batch evaluation efficiently