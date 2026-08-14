import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings; warnings.filterwarnings('ignore')

def run(name, df, target, drop_cols=[]):
    df = df.drop(columns=drop_cols, errors='ignore').copy()
    df = df.dropna()
    y = df[target]
    X = df.drop(columns=[target])
    if not pd.api.types.is_numeric_dtype(y) or y.dtype == bool:
        y = LabelEncoder().fit_transform(y.astype(str))
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == bool:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    dt = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_test)
    proba_dt = dt.predict_proba(X_test)[:,1]
    results['DecisionTree'] = {
        'accuracy': accuracy_score(y_test, pred_dt),
        'precision': precision_score(y_test, pred_dt),
        'recall': recall_score(y_test, pred_dt),
        'f1': f1_score(y_test, pred_dt),
        'roc_auc': roc_auc_score(y_test, proba_dt),
        'top_features': sorted(zip(X.columns, dt.feature_importances_), key=lambda x:-x[1])[:5]
    }
    # lbfgs solver + light regularisation: more stable than adam on small tabular datasets
    mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=2000, random_state=42,
                         solver='lbfgs', alpha=0.01)
    mlp.fit(X_train_s, y_train)
    pred_mlp = mlp.predict(X_test_s)
    proba_mlp = mlp.predict_proba(X_test_s)[:,1]
    results['NeuralNet'] = {
        'accuracy': accuracy_score(y_test, pred_mlp),
        'precision': precision_score(y_test, pred_mlp),
        'recall': recall_score(y_test, pred_mlp),
        'f1': f1_score(y_test, pred_mlp),
        'roc_auc': roc_auc_score(y_test, proba_mlp),
    }
    print(f"\n=== {name} (n={len(df)}, positive rate={y.mean():.3f}) ===")
    for model, r in results.items():
        print(f"-- {model} --")
        for k,v in r.items():
            if k != 'top_features':
                print(f"   {k}: {v:.3f}")
        if 'top_features' in r:
            print(f"   top_features: {r['top_features']}")
    return results

cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
diab = pd.read_csv('diabetes.csv', header=None, names=cols)
r1 = run("Pima Diabetes", diab, target='Outcome')

hf = pd.read_csv('heart_failure.csv')
r2 = run("Heart Failure", hf, target='DEATH_EVENT')

import json
with open('health_results.json','w') as f:
    json.dump({'diabetes':r1,'heart_failure':r2}, f, indent=2, default=str)
print("\nsaved health_results.json")
