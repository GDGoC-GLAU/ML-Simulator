# Logistic Regression - Documentation

## 📋 Overview

Logistic Regression is a statistical method for binary classification that predicts the probability of an outcome belonging to one of two classes (0 or 1). Despite its name, it's a classification algorithm, not a regression algorithm[web:102][web:103].

**Key Characteristics:**
- **Type**: Supervised Learning - Binary Classification
- **Output**: Probability score between 0 and 1
- **Algorithm**: Uses sigmoid function to map predictions to probabilities
- **Best For**: Linearly separable data with binary outcomes

## 🎯 Purpose and Use Cases

### Primary Use
Binary classification problems where you need to predict one of two possible outcomes.

### Common Applications
- **Medical Diagnosis**: Disease prediction (positive/negative)
- **Spam Detection**: Email classification (spam/not spam)
- **Customer Churn**: Will customer leave? (yes/no)
- **Credit Scoring**: Loan approval (approve/reject)
- **Marketing**: Click prediction (will click/won't click)

## 🚀 How to Run

### Step 1: Access the Model
1. Navigate to the ML Simulator application
2. Open the sidebar menu
3. Select **"Logistic Regression"** from the available models

### Step 2: Choose Data Source
You have two options for providing data:

**Option A: Upload CSV File**
- Click "Upload CSV" in the sidebar
- Select your CSV file (must contain binary target column with 0/1 values)
- Ensure your data has:
  - At least 100 rows
  - Numerical features
  - A binary target column (0 or 1)

**Option B: Use Sample Dataset**
- Select "Use Sample Dataset" radio button
- The Breast Cancer dataset will be loaded automatically
- Contains 569 samples with 30 features

### Step 3: Configure Parameters

| Parameter | Description | Default Value | Recommended Range |
|-----------|-------------|---------------|-------------------|
| **Target Column** | Column to predict (must be 0/1) | First binary column | Any binary column |
| **Test Size** | Percentage of data for testing | 20% | 10-30% |
| **Feature Selection** | Choose features for training | First 5 features | 3-10 features |
| **max_iter** | Maximum training iterations | 1000 | 500-2000 |

### Step 4: Train the Model
1. Select your target column from the dropdown
2. Choose features you want to use for prediction
3. Adjust test size slider if needed
4. Click the **🚀 Train Model** button
5. Wait for training to complete (usually 1-5 seconds)

## 📊 What Each Plot Shows

### 1. Training Results Dashboard

**What You See:**
Four gradient-colored metric cards displaying key performance indicators[web:99][web:102].

**Components:**
- **Accuracy**: Overall percentage of correct predictions
- **Training Samples**: Number of data points used for training
- **Test Samples**: Number of data points used for testing
- **Features Used**: Number of features selected for the model

**How to Interpret:**
- 
