# Job Skills Predictor - Run Guide

## 📋 Project Overview
This project predicts required skills from job descriptions using machine learning. It includes:
- ML model training (XGBoost, Random Forest, etc.)
- Web application for predictions
- Interview question database

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r src/requirements.txt

# Install additional packages for web app
pip install flask pandas numpy scikit-learn xgboost nltk joblib
```

### 2. Run the Web Application
```bash
# Start the Flask web server
python3 app.py
```

The application will be available at: **http://localhost:5000**

### 3. Use the Web Interface
1. Open http://localhost:5000 in your browser
2. Paste a job description in the text box
3. Click "Predict Skills" to see results

## 📊 Running the ML Pipeline

### Run the Complete ML Training
```bash
# Execute the main training script
python3 -c "
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print('Loading and preprocessing data...')

# Load data
try:
    job_skills_df = pd.read_csv('job_skills.csv')
    job_summary_df = pd.read_csv('job_summary.csv')
    linkedin_df = pd.read_csv('linkedin_job_postings.csv')
    
    # Merge datasets
    merged_df = pd.merge(job_skills_df, job_summary_df, on='job_link', how='inner')
    merged_df = pd.merge(merged_df, linkedin_df, on='job_link', how='inner')
    
    # Keep relevant columns
    relevant_columns = ['job_title', 'job_skills', 'job_summary']
    final_df = merged_df[relevant_columns].copy()
    
    # Clean text
    def clean_text(text):
        if pd.isna(text):
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
        text = re.sub(r'\\s+', ' ', text).strip()
        return text
    
    final_df['cleaned_summary'] = final_df['job_summary'].apply(clean_text)
    final_df['cleaned_skills'] = final_df['job_skills'].apply(clean_text)
    
    # Prepare features and labels
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X = tfidf.fit_transform(final_df['cleaned_summary'])
    
    # Extract top skills for multi-label classification
    all_skills = []
    for skills in final_df['cleaned_skills']:
        all_skills.extend(skills.split())
    
    from collections import Counter
    skill_counter = Counter(all_skills)
    top_skills = [skill for skill, count in skill_counter.most_common(30)]
    
    # Create binary labels
    def has_skill(skill_text, target_skill):
        return 1 if target_skill in skill_text.split() else 0
    
    y = np.zeros((len(final_df), len(top_skills)))
    for i, skill in enumerate(top_skills):
        y[:, i] = final_df['cleaned_skills'].apply(lambda x: has_skill(x, skill))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f'Data prepared: {X_train.shape} training samples, {X_test.shape} testing samples')
    print(f'Number of skills: {len(top_skills)}')
    print(f'Top skills: {top_skills[:10]}')
    
    # Train models
    models = {
        'Logistic Regression': MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=42)),
        'Random Forest': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
        'Naive Bayes': MultiOutputClassifier(MultinomialNB()),
        'Gradient Boosting': MultiOutputClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42)),
        'XGBoost': MultiOutputClassifier(XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))
    }
    
    results = {}
    
    for name, model in models.items():
        print(f'\\nTraining {name}...')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='micro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f'{name}:')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1-Score: {f1:.4f}')
    
    # Display results
    print('\\n' + '='*60)
    print('MODEL COMPARISON RESULTS:')
    print('='*60)
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_f1 = results[best_model_name]['f1_score']
    print(f'\\nBest model: {best_model_name} (F1-Score: {best_f1:.4f})')
    
    # Save best model
    best_model = models[best_model_name]
    import joblib
    joblib.dump(best_model, 'src/best_model.joblib')
    joblib.dump(tfidf, 'src/tfidf_vectorizer.joblib')
    
    print('\\nModel and vectorizer saved to src/best_model.joblib and src/tfidf_vectorizer.joblib')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
"
```

### 4. Test the Prediction Model
```bash
# Test the prediction script with sample data
python3 src/predict.py
```

## 📁 File Structure

### Essential Files (DO NOT DELETE):
```
Project/
├── app.py                 # Flask web application
├── src/
│   ├── best_model.joblib          # Trained ML model
│   ├── tfidf_vectorizer.joblib    # TF-IDF vectorizer
│   ├── top_skills.json            # Top skills list
│   ├── predict.py                 # Prediction script
│   ├── clean_questions.py         # Question cleaning utility
│   ├── fetch_github_questions.py   # GitHub question fetcher
│   └── requirements.txt            # Python dependencies
├── templates/
│   └── index.html                 # Web interface template
├── job_skills.csv                 # Original job skills data
├── job_summary.csv               # Original job summary data
├── linkedin_job_postings.csv     # Original LinkedIn data
└── RUN_GUIDE.md                  # This guide
```

### Files Safe to Delete (Intermediate/Generated):
- `cleaned_interview_questions.csv`
- `cleaned_job_data.csv`
- `cleaned_job_data_with_extracted_skills.csv`
- `cleaned_job_data_with_skills_and_questions.csv`
- `merged_job_data.csv`
- `tfidf_feature_names.csv`
- `tfidf_features.npz`
- `job_domain_classifier.joblib`
- `main.ipynb`
- `package.json`
- `server.js`
- `public/` directory (duplicate of templates)

## 🔧 Individual Scripts

### Run Prediction Only
```bash
python3 src/predict.py
```

### Fetch GitHub Questions
```bash
python3 src/fetch_github_questions.py
```

### Clean Interview Questions
```bash
python3 src/clean_questions.py
```

## 🐛 Troubleshooting

### Common Issues:

1. **ModuleNotFoundError: No module named 'flask'**
   ```bash
   pip install flask
   ```

2. **NLTK resources missing**
   ```bash
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

3. **Memory issues with large datasets**
   - Reduce dataset size in code
   - Use `max_features` parameter in TF-IDF

4. **Model file not found**
   - Run the training script first to generate model files

## 📈 Model Performance

The system currently uses XGBoost as the best performing model with:
- Micro F1-Score: ~0.59
- Accuracy: ~0.85
- Supports prediction of 20+ different skills

## 🔄 Re-training the Model

To re-train with different parameters:
1. Modify the training script parameters
2. Run the ML training command
3. New model will be saved automatically

## 🌐 Web API Usage

The web application provides a REST API:

```bash
# API endpoint for predictions
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"jobDescription": "Looking for software engineer with Python and cloud experience"}'
```

Response format:
```json
{
  "predicted_skills": ["python", "cloud", "software"],
  "count": 3,
  "input_length": 56
}
```

## 📝 Notes

- The web application uses port 5000 by default
- Model files are stored in the `src/` directory
- Original data files should be preserved for re-training
- The system can handle job descriptions of any length

## 🆘 Support

If you encounter issues:
1. Check all dependencies are installed
2. Verify model files exist in `src/` directory
3. Ensure original CSV data files are present
4. Check console for error messages

---

**Last Updated**: 2025-12-05
**Version**: 1.0.0