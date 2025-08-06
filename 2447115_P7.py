import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

def main():
    st.title("🎓 Academic Performance Prediction Platform")
    
    # Sidebar Configuration
    st.sidebar.title("Configuration Panel")
    
    # File Upload Section
    st.sidebar.markdown("### Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing student academic data"
    )
    
    use_default = st.sidebar.checkbox("Use Sample Dataset", value=True if uploaded_file is None else False)
    
    if uploaded_file is not None or use_default:
        # Load data
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        else:
            df = create_sample_dataset()
            st.sidebar.info("Using sample dataset")
        
        # Data Overview
        st.markdown("## Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display data preview
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(10))
        
        # Configuration Options
        st.sidebar.markdown("### Processing Options")
        
        # Missing data handling
        missing_strategy = st.sidebar.selectbox(
            "Missing Data Strategy",
            ["mean", "zero", "median", "drop"],
            index=0,
            help="Choose how to handle missing values in numerical columns"
        )
        
        # Feature creation option
        create_features = st.sidebar.checkbox(
            "Enable Feature Engineering",
            value=True,
            help="Create AVG_SCORE and AT_RISK features automatically"
        )
        
        # Train-test split ratio
        train_ratio = st.sidebar.slider(
            "Training Data Percentage",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Percentage of data used for model training"
        )
        
        # Process data first to get all available targets including engineered features
        df_processed = preprocess_data(df, missing_strategy, create_features, show_output=False)
        
        # Target column selection - NOW INCLUDES ENGINEERED FEATURES
        st.sidebar.markdown("### Target Configuration")
        potential_targets = get_potential_targets(df_processed)
        
        if potential_targets:
            target_column = st.sidebar.selectbox(
                "Select Target Column",
                potential_targets,
                help="Choose the column to predict (includes engineered features)"
            )
            
            # Show target distribution
            if st.sidebar.checkbox("Show Target Distribution"):
                st.sidebar.write("**Target Value Counts:**")
                st.sidebar.write(df_processed[target_column].value_counts())
        else:
            st.sidebar.error("No suitable target column found.")
            return
        
        # Feature Selection Section
        st.sidebar.markdown("### Feature Selection")
        available_features = get_available_features(df_processed, target_column)
        
        if available_features:
            selected_features = st.sidebar.multiselect(
                "Select Features for Modeling",
                available_features,
                default=available_features,
                help="Choose which features to include in the model"
            )
            
            if not selected_features:
                st.sidebar.warning("Please select at least one feature for modeling")
                return
        else:
            st.sidebar.error("No suitable features found for modeling")
            return
        
        # Process button
        if st.sidebar.button("Run Analysis", type="primary"):
            run_analysis(df_processed, target_column, selected_features, train_ratio)
    
    else:
        st.markdown("""
        ## Welcome to the Academic Performance Prediction Platform
        
        This platform enables comprehensive student performance analysis with:
        
        - **Flexible Data Input**: Upload any CSV dataset or use our sample data
        - **Intelligent Preprocessing**: Advanced missing data handling strategies
        - **Custom Feature Selection**: Choose exactly which features to include
        - **Dynamic Target Selection**: Predict any column including engineered features
        - **Feature Engineering**: Automated creation of predictive features
        - **Model Comparison**: Logistic Regression vs Random Forest analysis
        
        **Get Started**: Upload your dataset using the sidebar or enable the sample dataset to begin analysis.
        """)

def create_sample_dataset():
    """Generate comprehensive sample academic dataset"""
    np.random.seed(42)
    n_students = 200
    
    data = {
        'student_id': range(1, n_students + 1),
        'gender': np.random.choice(['M', 'F'], n_students),
        'parental_education': np.random.choice(['high_school', 'bachelor', 'master', 'phd', 'associate'], n_students),
        'mathscore': np.random.randint(20, 100, n_students),
        'readingscore': np.random.randint(25, 95, n_students),
        'writingscore': np.random.randint(30, 90, n_students),
        'attendance_rate': np.random.uniform(0.5, 1.0, n_students),
        'study_hours': np.random.randint(1, 8, n_students),
        'status': np.random.choice(['pass', 'fail', 'dropout'], n_students, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'study_hours'] = np.nan
    
    return df

def get_potential_targets(df):
    """Get all potential target columns including engineered features"""
    potential_targets = []
    
    # Add categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].nunique() <= 10:  # Reasonable number of classes
            potential_targets.append(col)
    
    # Add binary numeric columns (like AT_RISK)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        unique_values = df[col].nunique()
        if unique_values <= 10:  # Could be categorical
            potential_targets.append(col)
    
    return potential_targets

def preprocess_data(df, missing_strategy, create_features, show_output=True):
    """Advanced data preprocessing pipeline"""
    df_processed = df.copy()
    
    if show_output:
        st.markdown("## Data Preprocessing")
    
    # Handle missing values
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    missing_info = df_processed[numeric_columns].isnull().sum()
    
    if missing_info.sum() > 0:
        if show_output:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Missing Values Before")
                st.dataframe(missing_info[missing_info > 0])
        
        # Apply missing value strategy
        for col in numeric_columns:
            if df_processed[col].isnull().sum() > 0:
                if missing_strategy == "mean":
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                elif missing_strategy == "zero":
                    df_processed[col].fillna(0, inplace=True)
                elif missing_strategy == "median":
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                elif missing_strategy == "drop":
                    df_processed.dropna(subset=[col], inplace=True)
        
        if show_output:
            with col2:
                st.markdown("### Missing Values After")
                after_missing = df_processed[numeric_columns].isnull().sum()
                if after_missing.sum() > 0:
                    st.dataframe(after_missing[after_missing > 0])
                else:
                    st.success("All missing values handled")
    
    # Feature engineering
    if create_features:
        if show_output:
            st.markdown("### Feature Engineering")
        
        # Identify score columns
        score_columns = [col for col in df_processed.columns if 'score' in col.lower()]
        
        if len(score_columns) >= 2:
            # Create average score
            df_processed['AVG_SCORE'] = df_processed[score_columns].mean(axis=1)
            
            # Create at-risk indicator (binary target candidate)
            threshold = df_processed['AVG_SCORE'].quantile(0.3)  # Bottom 30%
            df_processed['AT_RISK'] = (df_processed['AVG_SCORE'] < threshold).astype(int)
            
            if show_output:
                st.success(f"✅ Created AVG_SCORE and AT_RISK features using {len(score_columns)} score columns")
                st.info(f"📊 AT_RISK distribution: {df_processed['AT_RISK'].value_counts().to_dict()}")
        else:
            if show_output:
                st.info("Insufficient score columns for automatic feature creation")
    
    return df_processed

def get_available_features(df, target_column):
    """Get list of available features for selection (excluding target)"""
    df_encoded = df.copy()
    feature_columns = []
    
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != target_column]
    
    # Add encoded categorical features
    for col in categorical_columns:
        feature_columns.append(f"{col}_encoded")
    
    # Add numeric features (excluding target)
    numeric_columns = df_encoded.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col != target_column:
            feature_columns.append(col)
    
    return feature_columns

def prepare_features_target(df, target_column, selected_features):
    """Prepare features and target for modeling"""
    
    # Encode categorical variables
    df_encoded = df.copy()
    le_dict = {}
    
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col != target_column]
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le
    
    # Prepare feature matrix with selected features only
    available_features = []
    
    # Add encoded categorical features
    for col in categorical_columns:
        if f"{col}_encoded" in selected_features:
            available_features.append(f"{col}_encoded")
    
    # Add numeric features
    numeric_columns = df_encoded.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in selected_features and col != target_column:
            available_features.append(col)
    
    if not available_features:
        st.error("No valid features selected for modeling")
        return None, None, None, None
    
    X = df_encoded[available_features]
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_encoded[target_column].astype(str))
    target_mapping = dict(zip(le_target.classes_, range(len(le_target.classes_))))
    
    return X, y, available_features, target_mapping

def train_models(X, y, train_ratio, feature_names):
    """Train and evaluate multiple models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=42, stratify=y
    )
    
    results = {}
    
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    log_pred = log_reg.predict(X_test)
    log_accuracy = accuracy_score(y_test, log_pred)
    
    results['Logistic Regression'] = {
        'model': log_reg,
        'predictions': log_pred,
        'accuracy': log_accuracy,
        'confusion_matrix': confusion_matrix(y_test, log_pred),
        'classification_report': classification_report(y_test, log_pred, output_dict=True)
    }
    
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    results['Random Forest'] = {
        'model': rf_clf,
        'predictions': rf_pred,
        'accuracy': rf_accuracy,
        'confusion_matrix': confusion_matrix(y_test, rf_pred),
        'classification_report': classification_report(y_test, rf_pred, output_dict=True),
        'feature_importance': rf_clf.feature_importances_
    }
    
    results['test_data'] = {'X_test': X_test, 'y_test': y_test, 'feature_names': feature_names}
    
    return results

def run_analysis(df_processed, target_column, selected_features, train_ratio):
    """Execute comprehensive ML pipeline with preprocessed data"""
    
    with st.spinner("Training models and analyzing results..."):
        # Show selected configuration
        st.markdown("### Analysis Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Variable", target_column)
        with col2:
            st.metric("Selected Features", len(selected_features))
        with col3:
            st.metric("Training Split", f"{train_ratio:.0%}")
        
        # Prepare features and target
        X, y, feature_names, target_mapping = prepare_features_target(df_processed, target_column, selected_features)
        
        if X is None:
            return
        
        # Display feature summary
        st.markdown("### Selected Features")
        st.write(f"**Features**: {', '.join(feature_names)}")
        
        # Train models
        results = train_models(X, y, train_ratio, feature_names)
        
        # Display results
        display_results(results, target_mapping, feature_names, target_column)

def display_results(results, target_mapping, feature_names, target_column):
    """Display comprehensive analysis results"""
    
    st.markdown("## Model Performance Analysis")
    
    # Performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Accuracy Comparison")
        accuracy_data = {
            'Model': ['Logistic Regression', 'Random Forest'],
            'Accuracy': [results['Logistic Regression']['accuracy'], results['Random Forest']['accuracy']]
        }
        
        fig = px.bar(accuracy_data, x='Model', y='Accuracy', 
                     title="Model Accuracy Comparison",
                     color='Accuracy', color_continuous_scale='blues')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Performance Metrics")
        for model_name, result in results.items():
            if model_name != 'test_data':
                st.metric(
                    f"{model_name} Accuracy", 
                    f"{result['accuracy']:.1%}"
                )
    
    # Feature importance (Random Forest)
    if 'feature_importance' in results['Random Forest']:
        st.markdown("### Feature Importance Analysis")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': results['Random Forest']['feature_importance']
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', 
                     title="Feature Importance (Random Forest)",
                     orientation='h')
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 3 features
        top_features = importance_df.tail(3)['Feature'].tolist()
        st.info(f"**Top 3 Most Influential Features**: {', '.join(reversed(top_features))}")
    
    # Confusion matrices
    st.markdown("### Confusion Matrix Analysis")
    
    col1, col2 = st.columns(2)
    
    reverse_mapping = {v: k for k, v in target_mapping.items()}
    labels = [reverse_mapping[i] for i in sorted(reverse_mapping.keys())]
    
    with col1:
        st.markdown("#### Logistic Regression")
        fig = px.imshow(results['Logistic Regression']['confusion_matrix'], 
                        labels=dict(x="Predicted", y="Actual"),
                        x=labels, y=labels,
                        color_continuous_scale='Blues',
                        title="Logistic Regression Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Random Forest")
        fig = px.imshow(results['Random Forest']['confusion_matrix'],
                        labels=dict(x="Predicted", y="Actual"),
                        x=labels, y=labels,
                        color_continuous_scale='Blues',
                        title="Random Forest Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification reports
    with st.expander("Detailed Classification Reports"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Logistic Regression Report")
            report_df = pd.DataFrame(results['Logistic Regression']['classification_report']).transpose()
            st.dataframe(report_df.round(3))
        
        with col2:
            st.markdown("#### Random Forest Report")
            report_df = pd.DataFrame(results['Random Forest']['classification_report']).transpose()
            st.dataframe(report_df.round(3))
    
    # Business insights
    st.markdown("## Strategic Insights & Recommendations")
    
    better_model = "Random Forest" if results['Random Forest']['accuracy'] > results['Logistic Regression']['accuracy'] else "Logistic Regression"
    best_accuracy = max(results['Random Forest']['accuracy'], results['Logistic Regression']['accuracy'])
    
    if 'feature_importance' in results['Random Forest']:
        importance_df_calc = pd.DataFrame({
            'Feature': feature_names,
            'Importance': results['Random Forest']['feature_importance']
        }).sort_values('Importance', ascending=False)
        top_3_impact = importance_df_calc.head(3)['Importance'].sum()
    else:
        top_3_impact = 0
    
    insights = f"""
    ### Key Findings:
    
    1. **Model Performance**: {better_model} achieves superior accuracy at {best_accuracy:.1%}
    2. **Target Variable**: Predicting {target_column} using {len(feature_names)} selected features
    3. **Feature Impact**: Top 3 predictors contribute {top_3_impact:.1%} of model decisions
    4. **Risk Classification**: Model successfully identifies patterns in {target_column} outcomes
    
    ### Strategic Recommendations:
    
    - **Deploy {better_model}**: Use this model for production predictions
    - **Feature Optimization**: Consider feature selection impact on model performance  
    - **Continuous Monitoring**: Regular retraining ensures sustained accuracy
    - **Actionable Intelligence**: Focus interventions on highest-impact features
    """
    
    st.markdown(insights)

if __name__ == "__main__":
    main()
