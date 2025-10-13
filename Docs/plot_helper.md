# Plot Helpers Documentation

## Overview

The `plot_helpers.py` module provides utility functions for creating matplotlib/seaborn visualizations optimized for Streamlit display.

## Functions

### `plot_roc_curve(y_true, y_pred_proba, title="ROC Curve", return_fig=True)`

Creates an ROC curve plot with AUC score.

**Parameters:**
- `y_true`: True binary labels (0 or 1)
- `y_pred_proba`: Predicted probabilities for positive class
- `title`: Plot title (optional)
- `return_fig`: Return figure object or base64 string

**Returns:** matplotlib Figure object

**Example:**
    from utils.plot_helpers import plot_roc_curve
import streamlit as st

fig = plot_roc_curve(y_test, y_pred_proba)
st.pyplot(fig)

text

### `plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix")`

Creates a confusion matrix heatmap.

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `labels`: Class labels (optional)
- `title`: Plot title (optional)

**Returns:** matplotlib Figure object

### `plot_feature_importance(feature_names, importance_scores, top_n=10, title="Feature Importance")`

Creates a horizontal bar chart of feature importance.

**Parameters:**
- `feature_names`: List of feature names
- `importance_scores`: Importance scores
- `top_n`: Number of top features to display
- `title`: Plot title (optional)

**Returns:** matplotlib Figure object

### `plot_prediction_distribution(y_pred_proba, y_true, title="Prediction Probability Distribution")`

Creates histogram of prediction probabilities by true class.

### `plot_residuals(y_true, y_pred, title="Residual Plot")`

Creates residual plots for regression models.

### `plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted")`

Creates scatter plot of actual vs predicted values.

## Styling

All plots use:
- Seaborn whitegrid style
- Consistent color scheme
- Bold labels and titles
- Grid for better readability
- High DPI for quality

## Author

Akshit - Hacktoberfest 2025
PR Comment for Issue #5
text
## 📊 ROC Curve Helper Function Complete!

Hi maintainers! 👋

I've successfully implemented a comprehensive **plot helper utility** for creating matplotlib/seaborn visualizations as requested in **issue #5**.

### 📁 Files Added/Modified:

✅ `utils/plot_helpers.py` - Complete plotting utility module  
✅ `docs/plot_helpers.md` - Comprehensive documentation  
✅ Updated `requirements.txt` with dependencies  
✅ Example integration code for Streamlit pages

### ✨ Features Implemented:

**1. ROC Curve Function:**
- Accepts true labels and predicted probabilities
- Calculates and plots ROC curve with AUC score
- Includes diagonal reference line (random classifier)
- Fills area under curve for better visualization
- Adds AUC score annotation box
- Returns matplotlib figure for Streamlit display

**2. Additional Helper Functions:**
- `plot_confusion_matrix()` - Heatmap visualization
- `plot_feature_importance()` - Horizontal bar chart
- `plot_prediction_distribution()` - Probability histograms
- `plot_residuals()` - For regression models
- `plot_actual_vs_predicted()` - Regression scatter plot

**3. Professional Styling:**
- Consistent color scheme
- Bold labels and titles
- Grid for readability
- High-quality DPI settings
- Seaborn whitegrid style

### 🎯 Addresses Issue Requirements:

✅ Added `plot_roc_curve()` function in `utils/plot_helpers.py`  
✅ Function accepts true labels and predicted scores  
✅ Returns matplotlib/seaborn figure for Streamlit  
✅ Includes comprehensive documentation  
✅ Easy to integrate with existing pages  
✅ Follows best practices for visualization

### 💡 Usage Example:

from utils.plot_helpers import plot_roc_curve
import streamlit as st

After model training
y_pred_proba = model.predict_proba(X_test)[:, 1]

Create and display ROC curve
roc_fig = plot_roc_curve(y_test, y_pred_proba,
title="ROC Curve - Logistic Regression")
st.pyplot(roc_fig)

text

### 📊 Features:

- Clean, professional visualizations
- Customizable titles
- AUC score calculation and display
- Reference line for random classifier
- Area under curve shading
- Annotation box with AUC value
- Grid and styling optimizations

### 🧪 Testing:

- ✅ Tested with binary classification data
- ✅ Verified AUC calculation accuracy
- ✅ Tested Streamlit integration
- ✅ Verified matplotlib figure compatibility
- ✅ Tested with different data sizes

The implementation is production-ready and can be easily integrated into all model pages for consistent, professional visualizations! 🚀

**Hacktoberfest 2025** 🎃

---

**Author**: Akshit  
**Issue**: #5  
**Type**: Enhancement