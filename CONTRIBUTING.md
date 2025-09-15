# Contributing to SmartTrade AI

Thank you for your interest in contributing to SmartTrade AI! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork the Repository
- Click the "Fork" button on the GitHub repository page
- Clone your forked repository to your local machine

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes
```bash
# Run the test suite
python3 test_enhanced.py

# Test the simple version
python3 test_analyzer.py

# Run the applications
streamlit run simple_trading_analyzer.py
streamlit run enhanced_trading_app.py
```

### 5. Commit Your Changes
```bash
git add .
git commit -m "Add: Brief description of your changes"
```

### 6. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 7. Create a Pull Request
- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your feature branch
- Provide a detailed description of your changes

## 📋 Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### Code Organization
- Keep related functionality together
- Separate concerns (data fetching, analysis, UI)
- Use meaningful file and directory names
- Add comments for complex logic

### Documentation
- Update README.md for new features
- Add docstrings to all functions
- Include examples in docstrings
- Update this CONTRIBUTING.md if needed

## 🎯 Areas for Contribution

### High Priority
- **New Technical Indicators**: Add more technical analysis tools
- **Data Sources**: Integrate additional free data sources
- **ML Models**: Implement new machine learning algorithms
- **UI Improvements**: Enhance the Streamlit interface
- **Performance**: Optimize data fetching and analysis speed

### Medium Priority
- **Testing**: Add more comprehensive test coverage
- **Documentation**: Improve code documentation
- **Error Handling**: Better error messages and recovery
- **Configuration**: Make more parameters configurable
- **Logging**: Add better logging throughout the system

### Low Priority
- **Visualization**: New chart types and visualizations
- **Export**: Data export functionality
- **Alerts**: Email/SMS notification system
- **Mobile**: Mobile-responsive improvements
- **Internationalization**: Multi-language support

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Detailed steps to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, dependencies
6. **Screenshots**: If applicable
7. **Logs**: Any error messages or logs

## 💡 Feature Requests

When requesting features, please include:

1. **Description**: Clear description of the feature
2. **Use Case**: Why this feature would be useful
3. **Implementation Ideas**: Any ideas on how to implement it
4. **Alternatives**: Other ways to achieve the same goal
5. **Priority**: How important this feature is to you

## 📝 Pull Request Guidelines

### Before Submitting
- [ ] Code follows the coding standards
- [ ] Tests pass successfully
- [ ] Documentation is updated
- [ ] No merge conflicts
- [ ] Commit messages are clear and descriptive

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass
- [ ] Manual testing completed
- [ ] No regressions

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

## 🏷️ Issue Labels

We use the following labels for issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority: high`: High priority
- `priority: medium`: Medium priority
- `priority: low`: Low priority

## 🚀 Release Process

1. **Version Bumping**: Update version numbers in relevant files
2. **Changelog**: Update CHANGELOG.md with new features and fixes
3. **Testing**: Ensure all tests pass
4. **Documentation**: Update README.md if needed
5. **Tagging**: Create a git tag for the release
6. **Release Notes**: Write comprehensive release notes

## 📞 Getting Help

If you need help or have questions:

1. **Check Documentation**: Read the README.md and code comments
2. **Search Issues**: Look through existing issues and discussions
3. **Create Issue**: Open a new issue with your question
4. **Contact**: Email mani.rastegari@gmail.com

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

## 📄 License

By contributing to SmartTrade AI, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to SmartTrade AI! 🚀📈
