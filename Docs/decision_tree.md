# Decision Tree - Documentation

## 📋 Overview

Decision Tree is a supervised learning algorithm that creates a tree-like model of decisions. It splits data based on feature values to make predictions for both classification and regression tasks[web:100][web:102].

**Key Characteristics:**
- **Type**: Supervised Learning - Classification or Regression
- **Output**: Class label or continuous value
- **Algorithm**: Recursive splitting based on information gain
- **Best For**: Non-linear relationships, interpretable models

## 🎯 Purpose and Use Cases

### Primary Use
Creating interpretable models that make decisions through a series of yes/no questions.

### Common Applications
- **Medical Diagnosis**: Decision pathways for treatment
- **Credit Approval**: Loan decision logic
- **Customer Segmentation**: Marketing strategy decisions
- **Fraud Detection**: Rule-based fraud identification
- **Product Recommendations**: Decision logic for suggestions

## 🚀 How to Run

### Step 1: Access the Model
1. Navigate to ML Simulator
2. Select **"Decision Tree"** from sidebar

### Step 2: Choose Data Source
- Upload CSV or use sample dataset
- For classification: binary or multi-class target
- For regression: continuous target

### Step 3: Configure Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **Max Depth** | Maximum tree depth | 5 | 1-20 |
| **Min Samples Split** | Minimum samples to split | 2 | 2-20 |
| **Min Samples Leaf** | Minimum samples in leaf | 1 | 1-10 |
| **Criterion** | Splitting metric | gini/mse | gini/entropy |

### Step 4: Train and Visualize
1. Configure parameters
2. Click **Train Model**
3. View tree structure and results

## 📊 What Each Plot Shows

### 1. Tree Visualization

**What You See:**
Visual representation of the decision tree structure.

**Components:**
- **Root node**: Top of tree (all data)
- **Internal nodes**: Decision points
- **Leaf nodes**: Final predictions
- **Branches**: Decision paths

**How to Read:**
- Each node shows:
  - Feature and threshold used for split
  - Number of samples
  - Class distribution or value
- Follow branches from top to bottom
- Leaf nodes contain predictions

### 2. Feature Importance

**What You See:**
Bar chart showing which features are most important[web:99][web:101].

**Interpretation:**
- Longer bars: More important for decisions
- Features at top of tree: Usually most important
- Zero importance: Feature not used

### 3. Confusion Matrix (Classification)

**Same as Logistic Regression**
Shows prediction accuracy breakdown.

### 4. Performance Metrics

**Classification:**
- Accuracy, Precision, Recall, F1-Score

**Regression:**
- R², MSE, RMSE, MAE

## 🔧 Model Parameters Explained

### max_depth
**Purpose**: Limit tree depth to prevent overfitting  
**Lower values**: Simpler, more general model  
**Higher values**: More complex, may overfit  
**Recommendation**: Start with 3-7

### min_samples_split
**Purpose**: Minimum samples required to split a node  
**Lower values**: More splits, complex tree  
**Higher values**: Fewer splits, simpler tree  
**Recommendation**: 2-10 depending on data size

### min_samples_leaf
**Purpose**: Minimum samples required in leaf node  
**Effect**: Smooths model, prevents overfitting  
**Recommendation**: 1-5

### criterion
**Classification:**
- **gini**: Gini impurity (default, faster)
- **entropy**: Information gain (more precise)

**Regression:**
- **mse**: Mean squared error (default)
- **mae**: Mean absolute error (robust to outliers)

## 💡 Tips and Best Practices

### Advantages
✅ Easy to understand and interpret  
✅ Handles non-linear relationships  
✅ No feature scaling required  
✅ Handles mixed data types  
✅ Provides feature importance

### Limitations
❌ Prone to overfitting  
❌ Unstable (small data changes affect tree)  
❌ Biased toward dominant classes  
❌ Not optimal for linear relationships

### Best Practices
- **Start shallow**: Begin with max_depth=3-5
- **Prune the tree**: Use min_samples parameters
- **Cross-validate**: Check performance on multiple splits
- **Ensemble methods**: Consider Random Forest for better stability
- **Visualize tree**: Understand decision logic

## 🐛 Troubleshooting

### Issue: Perfect Training Accuracy, Poor Test Accuracy

**Diagnosis:** Severe overfitting

**Solutions:**
1. Reduce max_depth (try 3-7)
2. Increase min_samples_split (try 10-20)
3. Increase min_samples_leaf (try 5-10)
4. Use Random Forest instead

### Issue: Tree Too Large to Visualize

**Solutions:**
1. Reduce max_depth
2. Export tree to graphical format
3. Focus on top levels only

### Issue: Low Accuracy

**Solutions:**
1. Increase max_depth (try up to 15)
2. Check feature quality
3. Add more relevant features
4. Try ensemble methods

## 📚 Additional Resources

- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Understanding Decision Trees](https://developers.google.com/machine-learning/decision-forests/decision-trees)
- [Tree Visualization Guide](https://mljar.com/blog/visualize-decision-tree/)

## 🎯 Example Use Case

### Scenario: Loan Approval System

**Features:**
- income, credit_score, debt_ratio, employment_years

**Tree might learn:**
