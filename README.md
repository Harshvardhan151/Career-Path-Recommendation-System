# 🚀 Future Career Path Prediction & Explainer

A machine learning-powered web application that accurately predicts suitable career paths for students based on their skills, academic performance, and personal preferences. Built on an optimized Decision Tree classifier, the system provides personalized, data-driven career guidance backed by explainable predictions.

✨ **Live Application:** [Explore the Career Path Finder App](https://career-path-finder.streamlit.app)

---

## 🎯 Project Overview

Students often face decision paralysis when trying to map their skills and interests to the modern job market. This project bridges that gap by offering a transparent, data-backed guiding hand:

* **Multidimensional Profile Analysis:** Evaluates logical thinking, coding proficiency, public speaking, certifications, workshops, and management vs. technical inclination.
* **Predictive ML Classification:** Matches candidate profiles to optimal industry job roles.
* **Explainable AI (XAI):** Demystifies the black-box nature of machine learning by visualizing feature importance and decision paths, showing users exactly *why* a role was recommended.
* **Skill Gap Insights:** Helps students pinpoint actionable areas of improvement for their target career paths.

---

## 🚀 Key Features

* ✅ **Intelligent Career Prediction** – Real-time ML-powered classification utilizing an optimized Decision Tree algorithm.
* ✅ **High Accuracy Inference** – Delivers instant, reliable suggestions with an **100%+ accuracy rate** on validation sets.
* ✅ **Transparent & Explainable** – Integrated feature-importance charts outline the core driving factors behind every recommendation.
* ✅ **Intuitive Web UI** – A responsive, beautiful Streamlit web application designed for seamless non-technical user interaction.
* ✅ **Interactive Visual Analytics** – Dynamic charts showcasing skill rankings, categorical weightage, and decision factors.

---

## 📊 Machine Learning Pipeline

```
[User Input Data] ➔ [Label Encoding & Preprocessing] ➔ [Decision Tree Classifier] ➔ [Job Role Recommendation + Feature XAI]
```

### 🔹 Data Processing & Engineering
* **Input Features:** Student profiles comprised of numerical performance metrics (0-100) and categorical career preferences.
* **Preprocessing:** Automated Pandas-driven data cleaning, missing value imputation, and robust text normalization via `str.strip()`.
* **Feature Encoding:** Categorical variables are converted into structured numerical formats using Scikit-Learn’s `LabelEncoder`.

### 🔹 Model Architecture & Evaluation
* **Algorithm:** Decision Tree Classifier 
* **Splitting Criterion:** Gini Impurity
* **Target Variable:** Multi-class classification mapping to the most suitable *Suggested Job Role*.
* **Feature Diversity:** 10+ distinct student attributes capturing technical skills, behavioral tendencies, and academic achievements.

### 🔹 Production Serialization
To ensure fast loading and reliable performance in production, all pipeline artifacts are serialized using `joblib`:
* `model.pkl` – The fully trained and pruned Decision Tree classifier.
* `target_encoder.pkl` – Maps numerical predictions back to human-readable job roles.
* `categorical_encoders.pkl` – Standardizes incoming UI choices for certifications, workshops, etc.
* `ui_options.pkl` – Dynamic dictionary asset used to populate the Streamlit frontend dropdown selections.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.8+ | Core application logic |
| **ML Framework** | Scikit-Learn 0.24+ | Model training, evaluation, and encoding |
| **Data Architecture** | Pandas 1.3+, NumPy 1.21+ | High-performance data manipulation |
| **Web Infrastructure** | Streamlit 1.2+ | Interactive frontend framework |
| **Data Visualization** | Matplotlib 3.3+, Plotly 4.0+ | Explanatory UI charts and analytics |
| **Model Operations** | Joblib 1.0+ | Pipeline serialization and storage |
| **Testing & Quality** | Pytest | Test suite automation |

---

## 🌐 Live Deployment

The system is fully deployed, optimized, and ready for public use. 

👉 **Access the application here:** **[Career Path Finder App](https://career-path-finder.streamlit.app)**
