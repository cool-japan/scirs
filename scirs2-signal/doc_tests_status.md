# Doc Tests Status for scirs2-signal

## Summary

Fixed doc tests that are now working:
1. `sswt::synchrosqueezed_cwt` - Fixed logspace function parameters
2. `hilbert::envelope` - Was already working, just needed to remove ignore
3. `reassigned::reassigned_spectrogram` - Fixed window function parameters

## Doc Tests with FIXME Comments

The following doc tests remain ignored due to underlying implementation issues:

### FFT/Complex Type Issues
- `wvd::wigner_ville` - FFT library expects f64 values but getting Complex64
- `wvd::cross_wigner_ville` - Same FFT type conversion issue
- `wvd::smoothed_pseudo_wigner_ville` - Same FFT type conversion issue

### DWT Module Issues
- `denoise::denoise_wavelet` - Approximation and detail coefficients length mismatch

### Type Conversion Issues
- `wavelets::cwt` (complex signal variant) - Complex64 type conversion error

### Integer Overflow Issues
- `nlm` module (3 tests) - Integer overflow in array indexing calculations

### Numerical Precision Issues  
- `swt::swt_reconstruct` - Numerical precision in SWT reconstruction
- `swt::iswt` - Similar precision issues in inverse SWT

### API/Validation Issues
- `parametric` module - AR coefficients validation issue

### Already Had FIXME Comments
- `lti_response` module (3 tests) - Marked for numerical accuracy issues

## Next Steps

These issues require deeper architectural fixes:
1. FFT library interface needs to handle complex types properly
2. DWT implementation needs coefficient length fixes
3. Integer arithmetic overflow protection needed
4. Numerical precision improvements in wavelet transforms
5. API consistency for AR coefficient formats