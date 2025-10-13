# Support Vector Machine (SVM) - Documentation

## 📋 Overview

SVM finds the optimal hyperplane that maximally separates different classes in the feature space[web:100][web:102].

**Key Characteristics:**
- **Type**: Supervised Learning - Classification
- **Algorithm**: Maximum margin classifier
- **Output**: Class label
- **Best For**: High-dimensional data, clear margins

## 🎯 Purpose and Use Cases

- **Text Classification**: Spam detection, sentiment analysis
- **Image Recognition**: Face detection, object classification
- **Bioinformatics**: Protein classification, gene expression
- **Financial**: Stock trend prediction
- **Medical**: Disease classification

## 📊 Key Parameters

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| **C** | Regularization | 1.0 | 0.1-100 |
| **kernel** | Kernel type | rbf | linear/rbf/poly |
| **gamma** | Kernel coefficient | scale | scale/auto |

## 💡 Kernel Selection

- **linear**: Linearly separable data, large features
- **rbf** (radial basis function): Default, most cases
- **poly** (polynomial): Specific polynomial relationships
- **sigmoid**: Neural network-like behavior

## 🔧 Parameter Tuning

### C (Regularization)
- **Low C**: Wider margin, more errors (underfitting)
- **High C**: Narrow margin, fewer errors (overfitting)
- **Start with**: 1.0, then try 0.1, 10, 100

### Gamma (RBF kernel)
- **Low gamma**: Far-reaching influence, smooth decision boundary
- **High gamma**: Close influence, complex decision boundary
- **Use**: 'scale' (default) or 'auto'

## 🐛 Troubleshooting

### Slow Training
- Use linear kernel for large datasets
- Reduce training data
- Scale features first

### Poor Performance
- Try different kernels
- Tune C and gamma
- Scale features (mandatory for SVM!)
- Check if data is separable

---

**Last Updated**: October 13, 2025  
**Version**: 1.0  
**Author**: Akshit  
**Hacktoberfest 2025 Contribution** 🎃
