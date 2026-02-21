import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, send_file, request, jsonify
import io
import base64
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ========== DATA LOADING AND CLEANING ==========
def load_and_clean_data():
    """Load and clean the dataset"""
    try:
        # Try multiple possible file locations
        file_path = None
        possible_names = [
            "dirty_data1.csv", 
            "data/dirty_data1.csv"
        ]
        
        for name in possible_names:
            if os.path.exists(name):
                file_path = name
                break
        
        if file_path is None:
            raise FileNotFoundError("CSV file not found in any expected location")
        
        df = pd.read_csv(file_path)
        print(f"✓ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Check if Student_Status column exists
        if 'Student_Status' not in df.columns:
            # Try to find target column with different names
            target_aliases = ['target', 'Target', 'status', 'Status', 'outcome', 'Outcome']
            for alias in target_aliases:
                if alias in df.columns:
                    df = df.rename(columns={alias: 'Student_Status'})
                    print(f"✓ Renamed '{alias}' column to 'Student_Status'")
                    break
        
        # ========== DATA CLEANING ==========
        print("Cleaning data...")
        
        # Clean Student_Status Variable
        if 'Student_Status' in df.columns:
            df['Student_Status'] = df['Student_Status'].astype(str).str.replace('_err', '', regex=False)
            df['Student_Status'] = df['Student_Status'].str.replace('nan_err', 'Dropout', regex=False)
            df['Student_Status'] = df['Student_Status'].str.strip()
            
            # Standardize status values (case-insensitive)
            def standardize_status(status):
                if pd.isna(status):
                    return np.nan
                status = str(status).lower().strip()
                if 'dropout' in status or 'drop' in status:
                    return 'Dropout'
                elif 'graduate' in status or 'grad' in status or 'completed' in status:
                    return 'Graduate'
                elif 'enrolled' in status or 'enroll' in status or 'current' in status:
                    return 'Enrolled'
                else:
                    return status.capitalize()  # Return as is but capitalized
            
            df['Student_Status'] = df['Student_Status'].apply(standardize_status)
            valid_statuses = ['Dropout', 'Graduate', 'Enrolled']
            df['Student_Status'] = df['Student_Status'].apply(lambda x: x if x in valid_statuses else np.nan)
        
        # Clean numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Clean Course column
        if 'Course' in df.columns:
            def clean_course(x):
                try:
                    if pd.notna(x):
                        x = float(x)
                        if 1000 < x < 100000:  # Reasonable course codes
                            return int(x)
                except:
                    pass
                return np.nan
            df['Course'] = df['Course'].apply(clean_course)
        
        # Clean Age column (handle Age_at_enrollment or similar names)
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        if age_cols:
            age_col = age_cols[0]  # Use first age-related column
            def clean_age(x):
                try:
                    if pd.notna(x):
                        x = float(x)
                        if 15 <= x <= 70:
                            return int(x)
                except:
                    pass
                return np.nan
            df[age_col] = df[age_col].apply(clean_age)
            # Rename to standard name
            df = df.rename(columns={age_col: 'Age_at_enrollment'})
        
        # Clean grades (should be between 0-200)
        grade_cols = [col for col in df.columns if 'grade' in col.lower()]
        for col in grade_cols:
            if col in df.columns:
                def clean_grade(x):
                    try:
                        if pd.notna(x):
                            x = float(x)
                            if 0 <= x <= 200:
                                return round(x, 2)
                    except:
                        pass
                    return np.nan
                df[col] = df[col].apply(clean_grade)
        
        # Remove rows with missing Student_Status
        if 'Student_Status' in df.columns:
            df = df.dropna(subset=['Student_Status'])
        
        print(f"✓ Cleaned dataset: {len(df)} rows remaining")
        print(f"✓ Student Status Distribution:")
        print(df['Student_Status'].value_counts())
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Load data once when app starts
df = load_and_clean_data()

# Global variable to store model accuracy
model_accuracy = "Not trained yet"

# ========== PREDICTION MODEL ==========
def train_prediction_model():
    """Train a model to predict student status"""
    global model_accuracy
    
    print("\n" + "="*50)
    print("TRAINING PREDICTION MODEL")
    print("="*50)
    
    if df.empty or 'Student_Status' not in df.columns:
        print("Cannot train model: No data or missing target column")
        return None, None, None, None
    
    # Prepare data for modeling
    model_df = df.copy()
    
    # Encode target variable
    le_target = LabelEncoder()
    model_df['Student_Status_encoded'] = le_target.fit_transform(model_df['Student_Status'])
    
    # Select features - using available columns
    feature_cols = []
    possible_features = [
        'Previous qualification (grade)', 'Admission grade', 
        'Age_at_enrollment', 'Scholarship holder', 'Debtor',
        'Tuition fees up to date', 'Application mode', 'Course',
        'Daytime/evening attendance', 'Marital status', 'International',
        'Curricular units 1st sem (credited)', 'Curricular units 2nd sem (credited)',
        'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)',
        'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',
        'Curricular units 1st sem (evaluations)', 'Curricular units 2nd sem (evaluations)',
        'Curricular units 1st sem (enrolled)', 'Curricular units 2nd sem (enrolled)',
        'Mother_qualification', 'Father_qualification'
    ]
    
    # Check which features exist in dataset
    for col in possible_features:
        if col in model_df.columns:
            feature_cols.append(col)
    
    # Add any other numerical columns
    numeric_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in feature_cols and col != 'Student_Status_encoded':
            # Avoid adding ID-like columns
            if model_df[col].nunique() > 2:  # More than binary
                feature_cols.append(col)
    
    if len(feature_cols) < 3:
        print(f"Not enough features for modeling (only {len(feature_cols)} found)")
        return None, None, None, None
    
    print(f"Using {len(feature_cols)} features for modeling")
    
    # Fill missing values
    for col in feature_cols:
        if col in model_df.columns and model_df[col].isna().any():
            if model_df[col].dtype in [np.float64, np.int64]:
                model_df[col] = model_df[col].fillna(model_df[col].median())
            else:
                model_df[col] = model_df[col].fillna(model_df[col].mode()[0] if not model_df[col].mode().empty else 0)
    
    # Encode categorical features
    for col in feature_cols:
        if col in model_df.columns and model_df[col].dtype == 'object':
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
    
    # Split data
    X = model_df[feature_cols]
    y = model_df['Student_Status_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Update global accuracy
    model_accuracy = f"{accuracy:.2%}"
    
    print(f"✅ Model Accuracy: {model_accuracy}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n🔝 Top 5 Most Important Features:")
    print(feature_importance.head(5).to_string(index=False))
    
    # Save model
    try:
        with open('student_status_model.pkl', 'wb') as f:
            pickle.dump({
                'model': model, 
                'encoder': le_target, 
                'features': feature_cols,
                'accuracy': accuracy
            }, f)
        print("💾 Model saved as 'student_status_model.pkl'")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")
    
    print("="*50 + "\n")
    
    return model, accuracy, feature_importance, le_target

# ========== HELPER FUNCTIONS ==========
def create_plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# ========== ROUTES ==========
@app.route('/')
def index():
    """Main page with all visualizations"""
    global model_accuracy
    
    if df.empty:
        return "Error: No data loaded. Please check your CSV file."
    
    if 'Student_Status' not in df.columns:
        return "Error: 'Student_Status' column not found in dataset. Please check your CSV file."
    
    # Store all plots in a dictionary
    plots = {}
    
    try:
        # ========== 1. STUDENT STATUS DISTRIBUTION ==========
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        status_counts = df['Student_Status'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Student Status Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(status_counts.index, status_counts.values, color=colors)
        ax2.set_title('Student Count by Status', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Status')
        ax2.set_ylabel('Number of Students')
        ax2.set_xticklabels(status_counts.index, rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plots['status_distribution'] = create_plot_to_base64(fig)
        
        # ========== 2. AGE DISTRIBUTION BY STATUS ==========
        if 'Age_at_enrollment' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create age groups
            df['Age Group'] = pd.cut(df['Age_at_enrollment'], 
                                     bins=[15, 20, 25, 30, 35, 40, 50, 100], 
                                     labels=['15-19', '20-24', '25-29', '30-34', '35-39', '40-49', '50+'])
            
            # Group by age group and status
            age_status = df.groupby(['Age Group', 'Student_Status']).size().unstack().fillna(0)
            age_status.plot(kind='bar', ax=ax, color=colors, width=0.8)
            
            ax.set_title('Age Distribution by Student Status', fontsize=14, fontweight='bold')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Number of Students')
            ax.legend(title='Status')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plots['age_distribution'] = create_plot_to_base64(fig)
        
        # ========== 3. GENDER DISTRIBUTION ==========
        gender_cols = [col for col in df.columns if 'gender' in col.lower() or 'sex' in col.lower()]
        if gender_cols:
            gender_col = gender_cols[0]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Overall gender distribution
            gender_counts = df[gender_col].value_counts()
            
            # Map numeric gender to labels if needed
            if gender_counts.index.dtype in [np.int64, np.float64]:
                gender_labels = []
                for val in gender_counts.index:
                    if val == 1 or val == '1':
                        gender_labels.append('Male')
                    elif val == 0 or val == '0':
                        gender_labels.append('Female')
                    else:
                        gender_labels.append(str(val))
            else:
                gender_labels = [str(x) for x in gender_counts.index]
            
            ax1.pie(gender_counts.values, labels=gender_labels, autopct='%1.1f%%',
                    colors=['#4ecdc4', '#ff6b6b'], startangle=90)
            ax1.set_title('Overall Gender Distribution', fontsize=14, fontweight='bold')
            
            # Gender by status
            gender_status = df.groupby([gender_col, 'Student_Status']).size().unstack()
            gender_status.index = gender_labels[:len(gender_status.index)]
            gender_status.plot(kind='bar', ax=ax2, color=colors, width=0.7)
            ax2.set_title('Gender Distribution by Status', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Gender')
            ax2.set_ylabel('Number of Students')
            ax2.legend(title='Status')
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plots['gender_distribution'] = create_plot_to_base64(fig)
        
        # ========== 4. ACADEMIC PERFORMANCE ==========
        grade_cols = [col for col in df.columns if 'grade' in col.lower()]
        if len(grade_cols) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Box plot of grades by status
            for i, grade_col in enumerate(grade_cols[:2]):  # Show first 2 grade columns
                data_to_plot = []
                labels = []
                for status in df['Student_Status'].unique():
                    data_to_plot.append(df[df['Student_Status'] == status][grade_col].dropna())
                    labels.append(status)
                
                axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True,
                               boxprops=dict(facecolor='lightblue', color='blue'),
                               medianprops=dict(color='red'))
                axes[i].set_title(f'{grade_col} by Status', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Status')
                axes[i].set_ylabel('Grade')
                axes[i].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plots['academic_performance'] = create_plot_to_base64(fig)
        
        # ========== 5. CORRELATION HEATMAP ==========
        if len(df.select_dtypes(include=[np.number]).columns) > 5:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Take top 10 most relevant columns if too many
            if len(num_cols) > 10:
                # Prioritize columns with less missing values
                num_cols = [col for col in num_cols if df[col].notna().sum() > 0.5 * len(df)]
                num_cols = num_cols[:10]
            
            if len(num_cols) > 1:
                corr_matrix = df[num_cols].corr()
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax, square=True)
                ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plots['correlation_heatmap'] = create_plot_to_base64(fig)
        
        # ========== 6. KEY STATISTICS ==========
        stats = {
            'total_students': len(df),
            'dropout_rate': f"{(df['Student_Status'] == 'Dropout').sum() / len(df) * 100:.1f}%",
            'graduation_rate': f"{(df['Student_Status'] == 'Graduate').sum() / len(df) * 100:.1f}%",
            'enrollment_rate': f"{(df['Student_Status'] == 'Enrolled').sum() / len(df) * 100:.1f}%",
            'model_accuracy': model_accuracy
        }
        
        if 'Age_at_enrollment' in df.columns:
            stats['avg_age'] = f"{df['Age_at_enrollment'].mean():.1f} years"
            stats['min_age'] = f"{df['Age_at_enrollment'].min()} years"
            stats['max_age'] = f"{df['Age_at_enrollment'].max()} years"
        
        # ========== 7. TOP COURSES ==========
        if 'Course' in df.columns:
            top_courses = df['Course'].value_counts().head(10)
            courses_data = {
                'labels': [str(c) for c in top_courses.index],
                'values': top_courses.values.tolist()
            }
            plots['top_courses'] = courses_data
        
        # ========== 8. STATUS BY COURSE TYPE ==========
        if 'Course' in df.columns:
            # Group courses into categories
            course_categories = {}
            for course in df['Course'].unique():
                if pd.notna(course):
                    course_str = str(course)
                    # Simple categorization based on course number
                    if course_str.startswith('9'):
                        course_categories[course] = '9000s'
                    elif course_str.startswith('8'):
                        course_categories[course] = '8000s'
                    elif course_str.startswith('1'):
                        course_categories[course] = '1000s'
                    else:
                        course_categories[course] = 'Other'
            
            df['Course_Category'] = df['Course'].map(course_categories)
            
            if 'Course_Category' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                course_status = df.groupby(['Course_Category', 'Student_Status']).size().unstack()
                course_status.plot(kind='bar', ax=ax, color=colors, width=0.8)
                ax.set_title('Student Status by Course Category', fontsize=14, fontweight='bold')
                ax.set_xlabel('Course Category')
                ax.set_ylabel('Number of Students')
                ax.legend(title='Status')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plots['course_status'] = create_plot_to_base64(fig)
        
        # Render template with all data
        return render_template('index.html', 
                             plots=plots, 
                             stats=stats,
                             data_head=df.head(10).to_html(classes='data-table'),
                             columns=list(df.columns))
    
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

@app.route('/data')
def show_data():
    """Show raw data table"""
    if df.empty:
        return "No data available"
    return df.to_html()

@app.route('/download_cleaned')
def download_cleaned():
    """Download cleaned dataset as CSV"""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='cleaned_student_data.csv',
                     mimetype='text/csv')

@app.route('/predict')
def predict_page():
    """Page to show prediction results"""
    try:
        # Train or load model
        model, accuracy, feature_importance, encoder = train_prediction_model()
        
        if model is None:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Model</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    body { padding: 40px; background-color: #f8f9fa; }
                </style>
            </head>
            <body>
                <div class="container text-center">
                    <h2 class="text-danger mb-4">❌ Prediction Model Not Available</h2>
                    <p class="lead">Could not train prediction model. Check if you have enough data.</p>
                    <a href='/' class='btn btn-primary'>← Back to Dashboard</a>
                </div>
            </body>
            </html>
            """
        
        # Prepare results for HTML
        feature_importance_html = feature_importance.head(10).to_html(
            classes='data-table table table-striped', 
            index=False
        )
        
        # Create feature importance plot
        fig, ax = plt.subplots(figsize=(12, 7))
        top_features = feature_importance.head(15)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        bars = ax.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color=colors[::-1])
        ax.set_xlabel('Feature Importance Score', fontsize=12)
        ax.set_title('Top 15 Most Important Features for Predicting Student Status', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        feature_plot = create_plot_to_base64(fig)
        
        # Create confusion matrix-like visualization
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        class_distribution = pd.Series(encoder.classes_).value_counts()
        colors2 = ['#ff9999', '#66b3ff', '#99ff99']
        ax2.pie(class_distribution.values, labels=class_distribution.index, 
                autopct='%1.1f%%', colors=colors2, startangle=90)
        ax2.set_title('Class Distribution in Training Data', fontsize=12, fontweight='bold')
        class_plot = create_plot_to_base64(fig2)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Model Results</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                body {{ padding: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 1200px; }}
                .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: none; }}
                .card-header {{ border-radius: 10px 10px 0 0 !important; }}
                .accuracy-badge {{ font-size: 1.2rem; }}
                .feature-plot {{ max-width: 100%; height: auto; border-radius: 8px; }}
                .badge-custom {{ background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%); }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mb-4"><i class="fas fa-robot me-2"></i>Student Status Prediction Model</h1>
                <p class="lead mb-4">Machine learning model that predicts student outcomes with <strong>{accuracy:.2%}</strong> accuracy</p>
                
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0"><i class="fas fa-tachometer-alt me-2"></i>Model Performance</h4>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Accuracy Score</h5>
                                <div class="display-4 text-success">{accuracy:.2%}</div>
                                <p>This model can predict whether a student will Graduate, Dropout, or remain Enrolled with high accuracy.</p>
                            </div>
                            <div class="col-md-6">
                                <h5>Model Details</h5>
                                <ul>
                                    <li><strong>Algorithm:</strong> Random Forest Classifier</li>
                                    <li><strong>Training Samples:</strong> {len(df)} students</li>
                                    <li><strong>Features Used:</strong> {len(feature_importance)} variables</li>
                                    <li><strong>Target Classes:</strong> {', '.join(encoder.classes_)}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-header bg-info text-white">
                                <h4 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Feature Importance Analysis</h4>
                            </div>
                            <div class="card-body">
                                <p>The most important factors for predicting student status:</p>
                                <img src="data:image/png;base64,{feature_plot}" class="feature-plot img-fluid" alt="Feature Importance">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header bg-warning text-dark">
                                <h4 class="mb-0"><i class="fas fa-chart-pie me-2"></i>Class Distribution</h4>
                            </div>
                            <div class="card-body">
                                <img src="data:image/png;base64,{class_plot}" class="img-fluid" alt="Class Distribution">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header bg-secondary text-white">
                        <h4 class="mb-0"><i class="fas fa-table me-2"></i>Top 10 Predictive Features</h4>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            {feature_importance_html}
                        </div>
                        <p class="text-muted mt-3"><small>The higher the importance score, the more the feature contributes to prediction accuracy.</small></p>
                    </div>
                </div>
                
                <div class="card mt-4">
                    <div class="card-body text-center">
                        <h4><i class="fas fa-lightbulb me-2"></i>Key Insights</h4>
                        <div class="row mt-3">
                            <div class="col-md-4">
                                <div class="card border-primary">
                                    <div class="card-body">
                                        <h5 class="text-primary">Academic Factors</h5>
                                        <p>Previous grades and admission scores are the strongest predictors of success.</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card border-success">
                                    <div class="card-body">
                                        <h5 class="text-success">Demographic Factors</h5>
                                        <p>Age and enrollment details significantly impact dropout risk.</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card border-info">
                                    <div class="card-body">
                                        <h5 class="text-info">Financial Factors</h5>
                                        <p>Scholarship status and tuition payment affect retention rates.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-4 text-center">
                    <a href="/" class="btn btn-primary btn-lg"><i class="fas fa-home me-2"></i>Back to Dashboard</a>
                    <a href="/predict_churn" class="btn btn-success btn-lg"><i class="fas fa-user-graduate me-2"></i>Predict Single Student</a>
                    <a href="/batch_predict" class="btn btn-warning btn-lg"><i class="fas fa-users me-2"></i>Batch Prediction</a>
                </div>
                
                <div class="mt-5 text-center text-muted">
                    <p><i class="fas fa-code me-1"></i> Model powered by Random Forest Algorithm | Flask Data Science App</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    except Exception as e:
        return f"""
        <div class="container text-center mt-5">
            <h2 class="text-danger">❌ Error in Prediction</h2>
            <p>Error: {str(e)}</p>
            <a href='/' class='btn btn-primary'>← Back to Dashboard</a>
        </div>
        """

@app.route('/train_model')
def train_model():
    """Simple endpoint to train model"""
    model, accuracy, _, _ = train_prediction_model()
    if model:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Training</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                body {{ 
                    padding: 40px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .success-card {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                    text-align: center;
                    max-width: 600px;
                }}
                .success-icon {{
                    font-size: 80px;
                    color: #28a745;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="success-card">
                <div class="success-icon">
                    <i class="fas fa-check-circle"></i>
                </div>
                <h2 class="text-success">✅ Model Training Complete!</h2>
                <div class="display-4 my-4">{accuracy:.2%}</div>
                <p class="lead">Model Accuracy Achieved</p>
                <p>The model has been successfully trained and saved as <code>student_status_model.pkl</code></p>
                <div class="mt-4">
                    <a href='/predict' class='btn btn-primary btn-lg'><i class="fas fa-chart-line me-2"></i>View Detailed Results</a>
                    <a href='/' class='btn btn-outline-primary btn-lg ms-2'><i class="fas fa-home me-2"></i>Back to Dashboard</a>
                </div>
            </div>
        </body>
        </html>
        """
    else:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Training</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { padding: 40px; background-color: #f8f9fa; }
            </style>
        </head>
        <body>
            <div class="container text-center">
                <h2 class="text-danger">❌ Model Training Failed</h2>
                <p class="lead">Check if you have enough data and features in your dataset.</p>
                <a href='/' class='btn btn-primary'>← Back to Dashboard</a>
            </div>
        </body>
        </html>
        """

# ========== NEW: CHURN PREDICTION ROUTES ==========
@app.route('/predict_churn')
def predict_churn_form():
    """Show form for predicting individual student churn"""
    return render_template('predict_form.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """API endpoint to predict student churn"""
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Load trained model
        try:
            with open('student_status_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                encoder = model_data['encoder']
                feature_cols = model_data['features']
        except:
            return jsonify({'error': 'Model not trained yet. Please visit /train_model first.'})
        
        # Create input dataframe
        input_data = {}
        
        # Set default values for all features
        for feature in feature_cols:
            input_data[feature] = 0  # Default value
        
        # Map form fields to model features
        field_mapping = {
            'age': 'Age_at_enrollment',
            'admission_grade': 'Admission grade',
            'previous_grade': 'Previous qualification (grade)',
            'tuition_paid': 'Tuition fees up to date',
            'scholarship': 'Scholarship holder',
            'debtor': 'Debtor',
            'course': 'Course',
            'sem1_approved': 'Curricular units 1st sem (approved)',
            'sem2_approved': 'Curricular units 2nd sem (approved)',
            'sem1_grade': 'Curricular units 1st sem (grade)',
            'sem2_grade': 'Curricular units 2nd sem (grade)',
            'sem1_evaluations': 'Curricular units 1st sem (evaluations)',
            'sem2_evaluations': 'Curricular units 2nd sem (evaluations)',
            'sem1_enrolled': 'Curricular units 1st sem (enrolled)',
            'sem2_enrolled': 'Curricular units 2nd sem (enrolled)',
            'sem1_credited': 'Curricular units 1st sem (credited)',
            'sem2_credited': 'Curricular units 2nd sem (credited)'
        }
        
        # Update with form values
        for form_field, model_field in field_mapping.items():
            if model_field in feature_cols:
                if form_field in data:
                    try:
                        input_data[model_field] = float(data[form_field])
                    except:
                        input_data[model_field] = 0
        
        # Special handling for binary fields
        binary_fields = ['Tuition fees up to date', 'Scholarship holder', 'Debtor']
        for field in binary_fields:
            if field in input_data:
                input_data[field] = 1 if input_data[field] > 0.5 else 0
        
        # Create prediction dataframe
        input_df = pd.DataFrame([input_data])
        
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training
        input_df = input_df[feature_cols]
        
        # Make prediction
        prediction_encoded = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get prediction
        prediction = encoder.inverse_transform([prediction_encoded])[0]
        
        # Calculate churn risk score (0-100%)
        dropout_prob = prediction_proba[encoder.transform(['Dropout'])[0]]
        if prediction == 'Dropout':
            churn_risk = dropout_prob * 100
        elif prediction == 'Enrolled':
            churn_risk = dropout_prob * 70  # Medium risk if enrolled but might dropout
        else:  # Graduate
            churn_risk = dropout_prob * 30  # Low risk
        
        # Ensure churn risk is between 0-100
        churn_risk = min(max(churn_risk, 0), 100)
        
        # Prepare response
        result = {
            'prediction': prediction,
            'churn_risk': round(churn_risk, 1),
            'probabilities': {
                'Dropout': round(prediction_proba[encoder.transform(['Dropout'])[0]] * 100, 1),
                'Graduate': round(prediction_proba[encoder.transform(['Graduate'])[0]] * 100, 1),
                'Enrolled': round(prediction_proba[encoder.transform(['Enrolled'])[0]] * 100, 1)
            },
            'confidence': round(max(prediction_proba) * 100, 1)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    """Batch prediction from uploaded CSV"""
    if request.method == 'GET':
        return render_template('batch_predict.html')
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return "No file uploaded"
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        
        # Load and process CSV
        df_batch = pd.read_csv(file)
        
        # Load model
        with open('student_status_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            encoder = model_data['encoder']
            feature_cols = model_data['features']
        
        # Prepare features - fill missing columns with defaults
        for col in feature_cols:
            if col not in df_batch.columns:
                df_batch[col] = 0
        
        # Ensure correct data types
        for col in feature_cols:
            df_batch[col] = pd.to_numeric(df_batch[col], errors='coerce').fillna(0)
        
        # Make predictions
        X_batch = df_batch[feature_cols]
        predictions_encoded = model.predict(X_batch)
        predictions = encoder.inverse_transform(predictions_encoded)
        
        # Add predictions to dataframe
        df_batch['Predicted_Status'] = predictions
        df_batch['Prediction_Confidence'] = [round(max(proba)*100, 1) for proba in model.predict_proba(X_batch)]
        
        # Calculate churn risk
        dropout_probs = model.predict_proba(X_batch)[:, encoder.transform(['Dropout'])[0]]
        
        def calculate_churn_risk(status, dropout_prob):
            if status == 'Dropout':
                return dropout_prob * 100
            elif status == 'Enrolled':
                return dropout_prob * 70
            else:
                return dropout_prob * 30
        
        df_batch['Churn_Risk_Score'] = [calculate_churn_risk(p, prob) for p, prob in zip(predictions, dropout_probs)]
        df_batch['Churn_Risk_Score'] = df_batch['Churn_Risk_Score'].round(1)
        
        # Add risk level
        def get_risk_level(score):
            if score < 30:
                return 'Low'
            elif score < 70:
                return 'Medium'
            else:
                return 'High'
        
        df_batch['Risk_Level'] = df_batch['Churn_Risk_Score'].apply(get_risk_level)
        
        # Save results
        output = io.BytesIO()
        df_batch.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name='student_predictions.csv',
            mimetype='text/csv'
        )
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    print("=" * 60)
    print("🎓 STUDENT CHURN PREDICTION DASHBOARD")
    print("=" * 60)
    print(f"📊 Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    if not df.empty and 'Student_Status' in df.columns:
        status_counts = df['Student_Status'].value_counts()
        print(f"📈 Student Status distribution:")
        for status, count in status_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   • {status}: {count} students ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("🤖 PREDICTION MODEL")
    print("="*60)
    print("The app includes a machine learning model that predicts student churn.")
    print("Model will train automatically when you visit /predict")
    print("\n📍 Available Routes:")
    print("   • /              - Main dashboard with visualizations")
    print("   • /predict_churn - Single student churn prediction")
    print("   • /batch_predict - Batch prediction from CSV")
    print("   • /predict       - Model analysis & details")
    print("   • /train_model   - Manually retrain model")
    print("   • /data          - View raw data")
    print("   • /download_cleaned - Download cleaned dataset")
    
    print("\n" + "="*60)
    print("🚀 Starting Flask server...")
    print("👉 Open your browser and go to: http://127.0.0.1:5000")
    print("👉 For churn prediction: http://127.0.0.1:5000/predict_churn")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)