# Error Handling Guide for scirs2-core

This guide establishes standard patterns for error handling across all scirs2-core modules.

## Core Principles

1. **Use `CoreError` for public APIs** - All public functions should return `CoreResult<T>`
2. **Create module-specific error types** for internal implementation details
3. **Always include error context** with location information
4. **Preserve semantic information** when converting between error types
5. **Use error macros** for consistent context creation

## Standard Module Error Pattern

Every module that needs custom error types should follow this pattern:

```rust
use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

/// Module-specific error type
#[derive(Debug, thiserror::Error)]
pub enum ModuleError {
    #[error("Specific error description: {0}")]
    SpecificError(String),
    
    #[error("Another error with details: {details}")]
    AnotherError { details: String },
    
    // Add more variants as needed
}

/// Convert module errors to core errors with semantic preservation
impl From<ModuleError> for CoreError {
    fn from(err: ModuleError) -> Self {
        match err {
            ModuleError::SpecificError(msg) => {
                CoreError::InvalidArgument(
                    ErrorContext::new(format!("Module specific error: {}", msg))
                        .with_location(ErrorLocation::new(file!(), line!()))
                )
            }
            ModuleError::AnotherError { details } => {
                CoreError::ComputationError(
                    ErrorContext::new(details)
                        .with_location(ErrorLocation::new(file!(), line!()))
                )
            }
        }
    }
}
```

## Error Creation Patterns

### Use Error Macros

Always use the provided error macros for consistency:

```rust
// For domain errors
return Err(domain_error!("Value {} is outside valid domain", value));

// For dimension errors  
return Err(dimension_error!("Expected shape {:?}, got {:?}", expected, actual));

// For value errors
return Err(value_error!("Invalid parameter: {}", param));

// For computation errors
return Err(computation_error!("Operation failed: {}", reason));

// For generic error context
return Err(CoreError::InvalidArgument(
    error_context!("Invalid configuration: {}", config)
));
```

### Avoid Manual Error Construction

❌ **Don't do this:**
```rust
return Err(CoreError::InvalidArgument(
    ErrorContext::new("Bad input".to_string())
));
```

✅ **Do this instead:**
```rust
return Err(CoreError::InvalidArgument(
    error_context!("Bad input: expected positive value, got {}", value)
));
```

## Error Propagation

### Use `?` Operator with Context

When propagating errors, add context:

```rust
fn process_data(data: &[f64]) -> CoreResult<Vec<f64>> {
    validate_data(data)
        .map_err(|e| CoreError::InvalidArgument(
            error_context!("Data validation failed")
                .with_cause(Box::new(e))
        ))?;
    
    // Process data...
    Ok(processed)
}
```

### Chain Error Causes

When wrapping errors, preserve the cause chain:

```rust
match some_operation() {
    Ok(result) => Ok(result),
    Err(e) => Err(CoreError::ComputationError(
        error_context!("High-level operation failed")
            .with_cause(Box::new(e.into()))
    ))
}
```

## Module-Specific Guidelines

### GPU Module
- Use `CoreError::GpuError` for GPU-specific failures
- Include device information in error context
- Map backend-specific errors to semantic CoreError variants

### Memory Module
- Use `CoreError::MemoryError` for allocation failures
- Include size and alignment information
- Add memory statistics when relevant

### Validation Module
- Use `CoreError::ValidationError` for validation failures
- Include field paths and constraint information
- Preserve validation severity levels

### Array Protocol
- Use `CoreError::ShapeError` for shape mismatches
- Use `CoreError::DtypeError` for type mismatches
- Include operation name and array information

## Error Recovery

### Use Circuit Breakers

For operations that might fail repeatedly:

```rust
use crate::error::recovery::{CircuitBreaker, CircuitBreakerConfig};

let breaker = CircuitBreaker::new(CircuitBreakerConfig {
    failure_threshold: 5,
    success_threshold: 2,
    timeout: Duration::from_secs(30),
});

let result = breaker.call(|| {
    // Potentially failing operation
    risky_operation()
})?;
```

### Implement Retry Logic

For transient failures:

```rust
use crate::error::recovery::{RetryPolicy, RetryConfig};

let policy = RetryPolicy::new(RetryConfig {
    max_attempts: 3,
    initial_delay: Duration::from_millis(100),
    max_delay: Duration::from_secs(5),
    multiplier: 2.0,
    jitter: true,
});

let result = policy.retry(|| {
    // Operation that might fail transiently
    network_operation()
})?;
```

## Testing Error Handling

### Test Error Cases

Always test error paths:

```rust
#[test]
fn test_error_handling() {
    let result = function_that_might_fail(-1);
    assert!(result.is_err());
    
    let err = result.unwrap_err();
    match err {
        CoreError::InvalidArgument(ctx) => {
            assert!(ctx.message().contains("negative value"));
        }
        _ => panic!("Unexpected error type"),
    }
}
```

### Test Error Context

Verify error context is properly set:

```rust
#[test]
fn test_error_context() {
    let result = operation_with_context();
    if let Err(err) = result {
        let ctx = err.context();
        assert!(ctx.location().is_some());
        assert!(ctx.cause().is_some());
    }
}
```

## Best Practices

1. **Be specific** - Use the most specific CoreError variant available
2. **Be descriptive** - Include relevant values and context in error messages
3. **Be consistent** - Follow module patterns established in this guide
4. **Be helpful** - Suggest fixes or alternatives in error messages when possible
5. **Be traceable** - Always include location information for debugging

## Migration Checklist

When updating existing code to follow these standards:

- [ ] Replace manual ErrorContext creation with error macros
- [ ] Add location information to all errors
- [ ] Implement proper From conversions for module errors
- [ ] Use semantic CoreError variants instead of generic ones
- [ ] Add error recovery where appropriate
- [ ] Update tests to verify error handling
- [ ] Document error conditions in function documentation