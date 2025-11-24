# Developer Guide

Welcome to the MolBlender Developer Guide! This section contains comprehensive information for contributors, maintainers, and anyone interested in extending MolBlender.

## Quick Links

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🤝 **Contributing**
:link: contributing
:link-type: doc
How to contribute code, docs, and bug reports
:::

:::{grid-item-card} 🏗️ **Architecture**
:link: architecture
:link-type: doc
System design and core components
:::

:::{grid-item-card} 🛠️ **Setup**
:link: setup
:link-type: doc
Development environment configuration
:::

:::{grid-item-card} ✅ **Testing**
:link: testing
:link-type: doc
Testing guidelines and best practices
:::

::::

## Overview

MolBlender is designed with modularity, extensibility, and ease of use in mind. The codebase follows several key design principles:

- **Registry-Based Architecture**: Featurizers and models are registered dynamically
- **Lazy Loading**: Dependencies loaded only when needed
- **Modular Design**: High cohesion, low coupling between components
- **Dependency Management**: Unified system for optional dependencies
- **Type Safety**: Comprehensive type hints throughout

## Documentation Structure

```{toctree}
:maxdepth: 2
:hidden:

contributing
architecture
setup
style
testing
adding_features/index
release
```

## For New Contributors

If you're new to contributing, start here:

1. Read {doc}`contributing` - Understand our workflow
2. Set up your environment with {doc}`setup`
3. Learn the {doc}`architecture` - Understand how components fit together
4. Check {doc}`style` - Follow our code conventions
5. Write tests following {doc}`testing` guidelines

## Adding Features

Detailed guides for extending MolBlender:

- {doc}`adding_features/featurizers` - Add new molecular representations
- {doc}`adding_features/models` - Add new ML models
- {doc}`adding_features/modalities` - Add new data modalities

## Development Philosophy

### Code Quality Over Speed
We prioritize maintainable, well-tested code over quick hacks. Every PR should:
- Include tests
- Follow style guidelines
- Update relevant documentation
- Pass CI checks

### User-Centric Design
Features should be intuitive for computational chemists and ML practitioners:
- Clear, descriptive names
- Sensible defaults
- Comprehensive error messages
- Rich documentation with examples

### Scientific Rigor
Implementations should be scientifically sound:
- Reference original papers
- Validate against known benchmarks
- Handle edge cases properly
- Document limitations

## Communication

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Pull Requests**: Follow PR template and link related issues

## Getting Help

If you're stuck or have questions:

1. Check existing documentation
2. Search GitHub Issues
3. Ask in GitHub Discussions
4. Reach out to maintainers

Thank you for contributing to MolBlender! 🎉
