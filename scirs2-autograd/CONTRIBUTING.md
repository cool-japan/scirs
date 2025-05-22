# Contributing to scirs2-autograd

This document provides guidelines for contributing to the scirs2-autograd module, focusing on automatic differentiation functionality for scientific computing and machine learning in Rust.

## Getting Started

### Development Environment Setup

1. Ensure you have Rust 1.70+ installed
2. Clone the repository and navigate to the scirs2-autograd directory
3. Run tests to ensure everything is working: `cargo test`

### Project Structure

- `src/` - Main source code
  - `tensor_ops/` - Tensor operations with gradient support
  - `optimizers/` - Optimization algorithms (SGD, Adam, etc.)
- `tests/` - Integration and unit tests
- `examples/` - Usage examples and demonstrations
- `docs/` - Additional documentation
- `scripts/` - Development utilities

## Current Development Priorities

### High Priority Issues

1. **Matrix Norm Gradients (Issue #42)**
   - Location: `src/tensor_ops/norm_ops.rs`
   - Tests: `tests/norm_ops_tests.rs` (currently ignored)
   - Documentation: See `MATRIX_NORM_GRADIENTS.md` for implementation guide
   - Priority: High - needed for many ML applications

2. **Gradient System Robustness**
   - Fix placeholder/feeder system issues
   - Improve error handling in gradient computation
   - Add gradient verification utilities

### Medium Priority

- Enhanced linear algebra operations
- More activation functions
- Advanced optimizers
- Memory optimization

## Contributing Guidelines

### Code Style

- Follow the project's existing code style
- Run `cargo fmt` before committing
- Address all `cargo clippy` warnings
- Use meaningful variable and function names
- Add documentation for public APIs

### Testing Requirements

- All new features must include comprehensive tests
- Fix any existing ignored tests when implementing related functionality
- Use the test template in `tests/test_templates/` for matrix operations
- Include both unit tests and integration tests
- Add property-based tests for mathematical operations where appropriate

### Documentation

- Document all public APIs with examples
- Update TODO.md when completing tasks
- Add implementation notes for complex algorithms
- Reference academic papers for mathematical operations

## Specific Contribution Areas

### Matrix Norm Gradients

If you're working on issue #42 (matrix norm gradients):

1. **Read the implementation guide**: `MATRIX_NORM_GRADIENTS.md`
2. **Use the test template**: `tests/test_templates/matrix_norm_test_template.rs`
3. **Start with Frobenius norm**: It's the simplest case
4. **Test thoroughly**: Use both analytical and numerical gradient verification
5. **Handle edge cases**: Zero matrices, ill-conditioned matrices, etc.

Example workflow:
```bash
# Run existing tests to see current failures
./scripts/test_norms.sh frobenius

# Implement fixes in src/tensor_ops/norm_ops.rs
# Add tests in tests/norm_ops_tests.rs

# Verify your implementation
cargo test --package scirs2-autograd test_frobenius_norm -- --nocapture --ignored
```

### Adding New Operations

When adding new tensor operations:

1. **Follow existing patterns**: Look at similar operations in `src/tensor_ops/`
2. **Implement both forward and backward passes**
3. **Add comprehensive tests**
4. **Document the operation**
5. **Add to the appropriate module exports**

### Performance Optimization

For performance improvements:

1. **Benchmark before and after changes**
2. **Profile memory usage**
3. **Consider SIMD opportunities for element-wise operations**
4. **Optimize hot paths identified through profiling**

## Development Workflow

### Standard Development Process

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Implement your changes**
3. **Run the full test suite**: `cargo test`
4. **Fix any warnings**: `cargo clippy`
5. **Format code**: `cargo fmt`
6. **Update documentation** as needed
7. **Commit with descriptive messages**
8. **Create a pull request**

### For Matrix Norm Gradients Specifically

1. **Start with understanding the math**: Read `MATRIX_NORM_GRADIENTS.md`
2. **Implement one norm at a time**: Frobenius → Spectral → Nuclear
3. **Test each implementation thoroughly**
4. **Remove `#[ignore]` from tests as you fix them**
5. **Verify gradients using finite differences**

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test categories
cargo test --package scirs2-autograd norm_ops_tests
cargo test --package scirs2-autograd linalg_tests

# Run ignored tests (currently failing)
cargo test -- --ignored

# Run tests for specific norm (using our utility script)
./scripts/test_norms.sh frobenius
```

## Debugging Tips

### Gradient Issues

- Use finite difference verification to check analytical gradients
- Print intermediate values during computation
- Check for NaN or infinity values in gradients
- Verify tensor shapes at each step

### Numerical Stability

- Use appropriate epsilon values for division operations
- Handle zero or near-zero cases explicitly
- Consider the condition number of matrices in linear algebra operations

### Performance Issues

- Profile with `cargo build --release` and appropriate tools
- Check for unnecessary allocations
- Look for opportunities to use in-place operations

## Questions and Support

- Check existing issues in the repository
- Look at the TODO.md for planned work
- Review the enhancement proposals in `docs/enhancement-proposals/`
- Ask questions in pull request discussions

## Recognition

Contributors who significantly improve the matrix norm gradient implementation will be recognized in the project changelog and documentation.