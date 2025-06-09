<!-- 
🚀 Thank you for contributing to the NCS API!
Please fill out this template to help us review your pull request efficiently.
-->

## 📋 Pull Request Summary

**Type of Change:**
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring (no functional changes, no api changes)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test addition or improvement
- [ ] 🔐 Security enhancement
- [ ] 🛠️ Infrastructure/build changes
- [ ] 🔄 Dependency updates

**Brief Description:**
Provide a concise description of what this PR accomplishes.

**Related Issues:**
- Fixes #(issue number)
- Closes #(issue number) 
- Related to #(issue number)

## 🎯 What does this PR do?

**Detailed Description:**
Provide a detailed description of the changes made in this PR. Include:
- What problem does this solve?
- How does it solve the problem?
- What are the key changes made?

**Motivation and Context:**
- Why is this change required?
- What problem does it solve?
- What use cases does it enable?

## 🔄 Changes Made

**Code Changes:**
- [ ] Core algorithm modifications
- [ ] API endpoint changes
- [ ] Database schema changes
- [ ] Authentication/authorization changes
- [ ] Configuration updates
- [ ] Dependency updates
- [ ] Performance optimizations
- [ ] Bug fixes
- [ ] Code refactoring

**Files Modified:**
List the main files that were changed and briefly describe what was changed in each:
- `path/to/file1.py` - Description of changes
- `path/to/file2.py` - Description of changes
- `path/to/config.yml` - Description of changes

**New Files Added:**
- `path/to/new_file.py` - Purpose and functionality
- `docs/new_guide.md` - Documentation for new feature

## 🧪 Testing

**Test Coverage:**
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] End-to-end tests added/updated
- [ ] Performance tests added/updated
- [ ] Security tests added/updated
- [ ] Manual testing completed

**Testing Details:**
Describe the testing performed:
```bash
# Commands used for testing
pytest tests/test_new_feature.py -v
pytest tests/integration/ -k "new_feature"
python -m pytest --cov=app tests/
```

**Test Results:**
- All existing tests: ✅ Pass / ❌ Fail
- New tests: ✅ Pass / ❌ Fail
- Coverage: X% (provide coverage percentage)

**Manual Testing Scenarios:**
1. **Scenario 1:** Description of manual test
   - Steps taken
   - Expected result
   - Actual result

2. **Scenario 2:** Description of another test
   - Steps taken
   - Expected result
   - Actual result

## 📊 Performance Impact

**Performance Changes:**
- [ ] No performance impact
- [ ] Performance improvement (please quantify)
- [ ] Minor performance degradation (acceptable trade-off)
- [ ] Potential performance concerns (requires discussion)

**Benchmarks:**
If applicable, provide before/after performance metrics:
```
Before: 1000 requests/second, 50ms average latency
After:  1200 requests/second, 42ms average latency
Improvement: +20% throughput, -16% latency
```

**Memory Usage:**
- [ ] No memory impact
- [ ] Reduced memory usage
- [ ] Increased memory usage (justified by functionality)

## 🔐 Security Considerations

**Security Impact:**
- [ ] No security impact
- [ ] Security improvement
- [ ] New security considerations
- [ ] Requires security review

**Security Checklist:**
- [ ] Input validation implemented
- [ ] Authentication/authorization checked
- [ ] No sensitive data exposed
- [ ] SQL injection prevention verified
- [ ] XSS prevention verified
- [ ] CSRF protection maintained
- [ ] Encryption used where appropriate

**Secrets and Configuration:**
- [ ] No hardcoded secrets
- [ ] Environment variables used appropriately
- [ ] Configuration changes documented

## 📖 Documentation

**Documentation Updates:**
- [ ] API documentation updated
- [ ] User guide updated
- [ ] Developer documentation updated
- [ ] README updated
- [ ] Changelog updated
- [ ] Code comments added/updated
- [ ] SDK documentation updated

**Breaking Changes Documentation:**
If this is a breaking change, describe:
- What breaks and why
- Migration guide for users
- Timeline for deprecation (if applicable)

## 🔄 Database Changes

**Schema Changes:**
- [ ] No database changes
- [ ] New tables added
- [ ] Existing tables modified
- [ ] Database migrations included
- [ ] Data migrations needed

**Migration Details:**
If database changes are included:
```sql
-- Example migration SQL
ALTER TABLE users ADD COLUMN new_field VARCHAR(255);
CREATE INDEX idx_users_new_field ON users(new_field);
```

**Data Safety:**
- [ ] Migrations are reversible
- [ ] Data backup strategy considered
- [ ] Migration tested on sample data

## 🔧 Configuration Changes

**Environment Variables:**
List any new or changed environment variables:
- `NEW_CONFIG_VAR` - Description of purpose and default value
- `UPDATED_CONFIG_VAR` - Description of changes made

**Configuration Files:**
- [ ] No configuration changes
- [ ] New configuration options added
- [ ] Existing configuration modified
- [ ] Configuration documentation updated

## 🚀 Deployment Considerations

**Deployment Requirements:**
- [ ] No special deployment requirements
- [ ] Requires database migration
- [ ] Requires configuration updates
- [ ] Requires infrastructure changes
- [ ] Requires coordinated deployment

**Rolling Deployment:**
- [ ] Safe for rolling deployment
- [ ] Requires coordinated deployment
- [ ] Blue/green deployment recommended

**Rollback Plan:**
Describe how to rollback if issues are discovered:
1. Step 1 of rollback process
2. Step 2 of rollback process
3. Data restoration if needed

## 🔍 Code Quality

**Code Quality Checklist:**
- [ ] Code follows project style guidelines
- [ ] Code is properly commented
- [ ] Functions/methods have docstrings
- [ ] Type hints added where appropriate
- [ ] Error handling implemented
- [ ] Logging added where appropriate
- [ ] No TODO comments left in code
- [ ] Code is DRY (Don't Repeat Yourself)

**Static Analysis:**
- [ ] Linting passes (flake8, pylint)
- [ ] Type checking passes (mypy)
- [ ] Security scanning passes (bandit)
- [ ] Code formatting applied (black, isort)

## 📱 SDK Impact

**Python SDK:**
- [ ] No SDK changes needed
- [ ] Python SDK updated
- [ ] New SDK methods added
- [ ] Existing SDK methods modified
- [ ] SDK documentation updated
- [ ] SDK examples updated

**JavaScript SDK:**
- [ ] No SDK changes needed
- [ ] JavaScript SDK updated
- [ ] New SDK methods added
- [ ] Existing SDK methods modified
- [ ] SDK documentation updated
- [ ] SDK examples updated

## 🌐 Backward Compatibility

**Compatibility:**
- [ ] Fully backward compatible
- [ ] Minor breaking changes (with deprecation)
- [ ] Major breaking changes (version bump required)

**API Versioning:**
- [ ] No API changes
- [ ] New API endpoints added
- [ ] Existing API endpoints modified
- [ ] API versioning strategy followed

**Migration Guide:**
If breaking changes, provide migration instructions:
```python
# Before (deprecated)
client.old_method(param1, param2)

# After (new way)
client.new_method(param1=param1, param2=param2, new_param=default_value)
```

## 🎨 User Experience

**UX Changes:**
- [ ] No user-facing changes
- [ ] Improved user experience
- [ ] New user-facing features
- [ ] Modified existing functionality

**Error Messages:**
- [ ] Error messages are clear and helpful
- [ ] Error codes are documented
- [ ] Error handling provides recovery guidance

## ✅ Checklist

**Before Submitting:**
- [ ] I have read the [Contributing Guidelines](CONTRIBUTING.md)
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

**Code Quality:**
- [ ] Code follows the style guidelines of this project
- [ ] I have removed any debugging code and console logs
- [ ] I have checked for and removed any hardcoded values
- [ ] All methods and classes have appropriate docstrings
- [ ] Type hints are included where appropriate

**Testing:**
- [ ] I have thoroughly tested my changes
- [ ] I have run the full test suite locally
- [ ] I have tested edge cases and error conditions
- [ ] I have verified backward compatibility

**Documentation:**
- [ ] I have updated relevant documentation
- [ ] I have added examples where appropriate
- [ ] I have updated the changelog if needed
- [ ] I have verified all links work correctly

## 🔗 Additional Context

**Related PRs:**
- Link to any related pull requests
- Dependencies on other PRs

**Screenshots/Videos:**
If applicable, add screenshots or videos demonstrating the changes:
![Screenshot description](url-to-image)

**Additional Notes:**
Any additional information that reviewers should know:
- Specific areas that need extra attention
- Known limitations or trade-offs
- Future enhancements planned
- Alternative approaches considered

---

## 📝 Review Checklist (for Reviewers)

**Code Review:**
- [ ] Code is clean and follows project standards
- [ ] Logic is sound and efficient
- [ ] Error handling is appropriate
- [ ] Security considerations addressed
- [ ] Performance impact acceptable

**Testing Review:**
- [ ] Test coverage is adequate
- [ ] Tests are meaningful and comprehensive
- [ ] Manual testing scenarios are appropriate
- [ ] Edge cases are covered

**Documentation Review:**
- [ ] Documentation is clear and complete
- [ ] Examples are working and helpful
- [ ] API documentation is updated
- [ ] Breaking changes are clearly documented

**Deployment Review:**
- [ ] Deployment plan is sound
- [ ] Migration strategy is safe
- [ ] Rollback plan is viable
- [ ] Infrastructure impact considered

---

**Thank you for your contribution to the NCS API! 🙏**

Your pull request will be reviewed by the maintainers. Please be patient and responsive to feedback. For questions, feel free to comment on this PR or reach out on our community channels.