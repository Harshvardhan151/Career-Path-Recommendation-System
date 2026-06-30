# Career Decision Support Pipeline & Advisor

An interactive Machine Learning-powered career recommendation application built with Python, Streamlit, and Scikit-Learn. This application uses a Decision Tree Classifier to predict potential career paths based on academic performance, technical skills, interests, and behavioral metrics.

## Features

- **Dynamic Machine Learning Backend**: Uses a pipeline with `scikit-learn` to process input features and output predictions using a trained `DecisionTreeClassifier`.
- **High-Fidelity UI**: Designed with a premium dark theme, animated gradient title headers, glassmorphism card layouts, and custom vertical timelines.
- **Interactive Visualizations**: Includes responsive Plotly charts representing:
  - **Feature Importance Distribution**: Visualizes key factors that influence the model's global decisions.
  - **Strengths & Development Opportunities**: Analyzes individual student profiles to categorize skill levels (Green for Strong, Gold for Average, Red for Development needed).
- **Comprehensive Career Advising**:
  - Predicts primary and alternative career paths.
  - Generates detailed career descriptions and profile match insights.
  - Outlines custom Phase-by-Phase next-step learning roadmaps.

## Project Structure

```text
├── cleaned_mldata.csv              # Student performance dataset
├── train.py                        # Model training and pipeline generation script
├── career_decision_tree_model.pkl  # Trained decision tree model artifact
├── app.py                          # Streamlit frontend & interactive dashboard
├── requirements.txt                # List of required packages
└── README.md                       # Documentation
```

## Installation & Setup

1. **Clone or navigate** to the project directory:
   ```bash
   cd C:\Users\Sameer\Downloads\proj_mvp_cuz_improved
   ```

2. **Install dependencies** using the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

4. *(Optional)* To retrain the decision tree classifier model:
   ```bash
   python train.py
   ```
