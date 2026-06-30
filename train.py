import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

print("🔄 Loading clean dataset locally...")
# 1. Load the clean data
df = pd.read_csv('cleaned_mldata.csv')

# 2. Split features and target
X = df.drop('Suggested Job Role', axis=1)
y = df['Suggested Job Role']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# 3. Build Preprocessor and Model Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

pipeline_dt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=15, min_samples_leaf=2, random_state=42))
])

# 4. Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("🚀 Training standalone Decision Tree on your local machine...")
pipeline_dt.fit(X_train, y_train)

# 5. Overwrite the broken pkl file with a perfectly compatible local one
joblib.dump(pipeline_dt, 'career_decision_tree_model.pkl')
print("💾 Fresh, compatible 'career_decision_tree_model.pkl' created successfully!")