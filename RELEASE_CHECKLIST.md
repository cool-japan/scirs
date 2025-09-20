# Release Checklist for v0.1.0-beta.2

## Pre-Release Verification ✅

### Code Quality
- [x] All regular tests passing (~6,500 tests)
- [x] No compilation warnings in release build
- [x] Version numbers consistent across all modules (0.1.0-beta.2)
- [x] GPU tests have proper fallbacks for CPU-only systems

### Documentation
- [x] README.md updated with Known Limitations section
- [x] Release notes created (RELEASE_NOTES.md)
- [x] Release notes archived in docs/releases/v0.1.0-beta.2.md
- [x] Known issues documented

### Test Status
```
Total Tests: ~6,500+
Passing: All functional tests ✅
Ignored: ~595 (documented reasons)
  - Benchmarks: 404
  - Graph context: 11
  - Architecture limitations: 7
  - Unimplemented: 3
```

## Release Steps 🚀

### 1. Final Code Review
- [ ] Review critical path code changes
- [ ] Ensure no sensitive information in code
- [ ] Verify all TODOs are documented

### 2. Git Operations
```bash
# Ensure on correct branch
git checkout 0.1.0-beta.2

# Commit any remaining changes
git add -A
git commit -m "Prepare for 0.1.0-beta.2 release

- Added comprehensive Known Limitations to README
- Fixed GPU tests with adaptive fallbacks
- Created release documentation
- All tests passing"

# Create release tag
git tag -a v0.1.0-beta.2 -m "SciRS2 v0.1.0-beta.2 - First Beta Release"

# Push to origin
git push origin 0.1.0-beta.2
git push origin v0.1.0-beta.2
```

### 3. GitHub Release
- [ ] Go to https://github.com/cool-japan/scirs/releases
- [ ] Click "Create a new release"
- [ ] Select tag: v0.1.0-beta.2
- [ ] Title: "SciRS2 v0.1.0-beta.2 - First Beta Release"
- [ ] Copy content from RELEASE_NOTES.md
- [ ] Check "This is a pre-release"
- [ ] Publish release

### 4. Crates.io Publishing (Optional)
```bash
# Dry run first
cargo publish --dry-run

# If ready to publish
cargo publish
```

### 5. Post-Release
- [ ] Verify GitHub release page
- [ ] Update project website (if applicable)
- [ ] Announce on social media/forums
- [ ] Monitor issue tracker for user feedback

## Known Limitations Summary

### Must Communicate to Users
1. **Beta Status**: This is a beta release, not production-ready
2. **Autograd Limitations**: Some gradient operations have shape propagation issues
3. **Missing Features**: Cholesky decomposition, Thin Plate Splines pending
4. **GPU Requirements**: GPU features require compatible hardware/drivers
5. **Benchmark Tests**: Excluded from default test runs for CI optimization

## Rollback Plan

If critical issues are discovered:
```bash
# Delete the tag locally
git tag -d v0.1.0-beta.2

# Delete the tag remotely
git push origin :refs/tags/v0.1.0-beta.2

# Fix issues and re-tag when ready
```

## Success Metrics

- [ ] Release published successfully
- [ ] No critical bugs reported within 48 hours
- [ ] Community feedback collected
- [ ] Download/usage statistics tracked

## Next Release Planning (v0.2.0)

Priority items for next release:
1. Implement Cholesky decomposition
2. Add Thin Plate Spline solver
3. Address gradient shape propagation issues
4. Reduce number of ignored tests
5. Performance optimizations based on user feedback

---

**Release Manager Sign-off**: _________________

**Date**: _________________

**Notes**: First beta release of SciRS2 with 2M+ lines of code and comprehensive scientific computing capabilities.