# pages/Logistic_Regression.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

# Page configuration
st.set_page_config(page_title="Logistic Regression Simulator", layout="wide", page_icon="📊")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
        padding-left: 15px;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Logistic Regression Simulator</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>🎯 About Logistic Regression</h3>
    <p>Logistic Regression is a statistical method for binary classification that predicts the probability 
    of an outcome belonging to a particular class. It's widely used in medical diagnosis, credit scoring, 
    and spam detection.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for data input
st.sidebar.header("⚙️ Configuration")
data_source = st.sidebar.radio("Choose Data Source:", ["Upload CSV", "Use Sample Dataset"])

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
else:
    # Sample dataset (you can use sklearn datasets)
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    st.sidebar.info("📊 Using Breast Cancer Dataset (sample)")

if df is not None:
    # Display dataset info
    st.markdown('<p class="sub-header">📁 Dataset Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    with st.expander("👀 View Dataset"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Feature selection
    st.markdown('<p class="sub-header">🎯 Model Configuration</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_column = st.selectbox("Select Target Column (0/1):", df.columns)
    
    with col2:
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
    
    # Select features
    available_features = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect(
        "Select Features for Training:",
        available_features,
        default=available_features[:min(5, len(available_features))]
    )
    
    if len(selected_features) > 0 and st.button("🚀 Train Model"):
        # Prepare data
        X = df[selected_features]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        with st.spinner('🔄 Training model...'):
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store in session state
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['features'] = selected_features
        
        st.success("✅ Model trained successfully!")
        
        # ==================== TRAINING RESULTS ====================
        st.markdown('<p class="sub-header">📈 Training Results</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>Accuracy</h3>
                <h2>{accuracy:.2%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h3>Training Samples</h3>
                <h2>{len(X_train)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h3>Test Samples</h3>
                <h2>{len(X_test)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-container">
                <h3>Features Used</h3>
                <h2>{len(selected_features)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # ==================== PREDICTIONS ====================
        st.markdown('<p class="sub-header">🔮 Predictions</p>', unsafe_allow_html=True)
        
        predictions_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Probability': y_pred_proba
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Sample Predictions:**")
            st.dataframe(predictions_df.head(10), use_container_width=True)
        
        with col2:
            # Prediction distribution
            fig_pred = px.histogram(
                predictions_df, 
                x='Probability',
                color='Actual',
                nbins=30,
                title='Prediction Probability Distribution',
                labels={'Probability': 'Predicted Probability', 'count': 'Frequency'},
                color_discrete_map={0: '#ff7675', 1: '#74b9ff'}
            )
            fig_pred.update_layout(height=400)
            st.plotly_chart(fig_pred, use_container_width=True)
        
        # ==================== CONFUSION MATRIX ====================
        st.markdown('<p class="sub-header">🎯 Confusion Matrix</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot using plotly for better interactivity
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues',
                showscale=True
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label',
                height=400
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Classification report
            st.write("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']), 
                        use_container_width=True)
        
        # ==================== ROC CURVE ====================
        st.markdown('<p class="sub-header">📉 ROC Curve</p>', unsafe_allow_html=True)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot ROC curve
            fig_roc = go.Figure()
            
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='#0984e3', width=3)
            ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='#d63031', width=2, dash='dash')
            ))
            
            fig_roc.update_layout(
                title='Receiver Operating Characteristic (ROC) Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                hovermode='x',
                legend=dict(x=0.6, y=0.1)
            )
            
            fig_roc.update_xaxes(range=[0, 1])
            fig_roc.update_yaxes(range=[0, 1])
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container" style="margin-top: 50px;">
                <h3>AUC Score</h3>
                <h1>{roc_auc:.4f}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h4>📚 Understanding AUC-ROC</h4>
                <ul>
                    <li><strong>AUC = 1.0:</strong> Perfect classifier</li>
                    <li><strong>AUC > 0.8:</strong> Excellent model</li>
                    <li><strong>AUC > 0.7:</strong> Good model</li>
                    <li><strong>AUC = 0.5:</strong> Random guess</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown('<p class="sub-header">⭐ Feature Importance</p>', unsafe_allow_html=True)
        
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Feature Coefficients',
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        fig_importance.update_layout(height=max(300, len(selected_features) * 30))
        st.plotly_chart(fig_importance, use_container_width=True)

else:
    st.info("👆 Please upload a dataset or select the sample dataset to get started!")
    
    st.markdown("""
    ### 📋 Instructions:
    1. Choose a data source from the sidebar (Upload CSV or use sample dataset)
    2. Select your target column (binary: 0/1)
    3. Choose features for training
    4. Adjust the test size if needed
    5. Click **Train Model** to see results
    
    ### ✨ Features:
    - 📊 Interactive confusion matrix
    - 📈 ROC curve with AUC score
    - 🎯 Detailed predictions with probabilities
    - ⭐ Feature importance visualization
    - 📉 Model performance metrics
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🎃 Hacktoberfest Contribution | Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
