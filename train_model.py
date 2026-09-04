import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def generate_hypertension_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)
    
    # Generate realistic feature distributions
    age = np.random.randint(18, 85, size=n_samples)
    bmi = np.round(np.random.normal(27, 5, size=n_samples), 1)
    bmi = np.clip(bmi, 16.0, 48.0)
    
    resting_hr = np.random.randint(52, 105, size=n_samples)
    salt_intake = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.2, 0.4, 0.3, 0.1])
    # 1: Low (<3g), 2: Normal (3-6g), 3: High (6-9g), 4: Severe (>9g)
    
    physical_activity = np.round(np.random.exponential(scale=3.0, size=n_samples), 1)
    physical_activity = np.clip(physical_activity, 0.0, 14.0)
    
    stress_score = np.random.randint(1, 11, size=n_samples)
    smoking = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    alcohol = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])
    family_history = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])
    
    # Clinical Risk Formula (AHA Guidelines correlated)
    risk_score = (
        0.09 * (age - 35) +
        0.18 * (bmi - 23.5) +
        0.06 * (resting_hr - 68) +
        2.80 * (salt_intake - 1) +
        -1.30 * physical_activity +
        0.75 * stress_score +
        3.80 * smoking +
        2.10 * alcohol +
        4.20 * family_history +
        np.random.normal(0, 2.5, size=n_samples)
    )
    
    # Map continuous risk score to 4 clinical stages
    labels = np.zeros(n_samples, dtype=int)
    labels[risk_score < 4.0] = 0        # Normal
    labels[(risk_score >= 4.0) & (risk_score < 11.0)] = 1  # Pre-Hypertension
    labels[(risk_score >= 11.0) & (risk_score < 18.0)] = 2 # Stage 1
    labels[risk_score >= 18.0] = 3      # Stage 2
    
    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'resting_hr': resting_hr,
        'salt_intake': salt_intake,
        'physical_activity': physical_activity,
        'stress_score': stress_score,
        'smoking': smoking,
        'alcohol': alcohol,
        'family_history': family_history,
        'target': labels
    })
    
    return df

def train_and_save_model():
    print("Generating synthetic clinical hypertension dataset...")
    df = generate_hypertension_dataset(n_samples=6000)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training RandomForest Classifier...")
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Pre-Hypertension', 'Stage 1', 'Stage 2']))
    
    feature_names = list(X.columns)
    importances = dict(zip(feature_names, model.feature_importances_))
    
    saved_data = {
        'model': model,
        'feature_names': feature_names,
        'importances': importances,
        'risk_labels': {
            0: {'name': 'Normal', 'color': '#10B981', 'badge': 'LOW RISK', 'description': 'Blood pressure risk parameters are within optimal clinical bounds.'},
            1: {'name': 'Pre-Hypertension', 'color': '#F59E0B', 'badge': 'ELEVATED RISK', 'description': 'Early warning signs detected. Lifestyle adjustments recommended.'},
            2: {'name': 'Stage 1 Hypertension', 'color': '#F97316', 'badge': 'HIGH RISK', 'description': 'Moderate risk detected. Dietary modifications and medical consultation recommended.'},
            3: {'name': 'Stage 2 Hypertension', 'color': '#EF4444', 'badge': 'SEVERE RISK', 'description': 'Critical risk level. Immediate medical evaluation and clinical intervention advised.'}
        }
    }
    
    output_path = os.path.join(os.path.dirname(__file__), 'hypershield_model.joblib')
    joblib.dump(saved_data, output_path)
    print(f"Model successfully saved to {output_path}")

if __name__ == '__main__':
    train_and_save_model()
