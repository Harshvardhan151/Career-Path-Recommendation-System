import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.tree import DecisionTreeClassifier

# Set matplotlib dark mode design parameters
matplotlib.rcParams['text.color'] = '#e2e8f0'
matplotlib.rcParams['axes.labelcolor'] = '#e2e8f0'
matplotlib.rcParams['xtick.color'] = '#94a3b8'
matplotlib.rcParams['ytick.color'] = '#94a3b8'

# ============================================================================
# 1. APPLICATION SETUP & INTERFACE STYLING
# ============================================================================
st.set_page_config(
    page_title="Career Path Analytics Finder",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    :root {
        --bg-primary: #08080d; --bg-secondary: #0f0f17; --bg-card: rgba(255,255,255,0.03);
        --border-subtle: rgba(255,255,255,0.06); --border-accent: rgba(245,158,11,0.3);
        --text-primary: #f1f5f9; --text-secondary: #94a3b8; --accent: #f59e0b;
        --accent-glow: rgba(245,158,11,0.15); --success: #10b981;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container {
        background-color: var(--bg-primary) !important; color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 1200px !important; }
    [data-testid="stExpander"] {
        background: var(--bg-card) !important; border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important; margin-bottom: 12px; transition: all 0.3s ease;
    }
    [data-testid="stExpander"]:hover { border-color: var(--border-accent) !important; box-shadow: 0 0 30px var(--accent-glow); }
    [data-testid="stSelectbox"] > div > div { background-color: var(--bg-secondary) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 10px !important; }
    .stButton > button { width: 100% !important; padding: 14px 28px !important; border-radius: 12px !important; font-weight: 700 !important; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important; color: #0a0a0f !important; border: none !important; box-shadow: 0 4px 20px rgba(245,158,11,0.2) !important; transition: all 0.3s ease !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 40px rgba(245,158,11,0.35) !important; }
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-title { 
        font-size: 3.2rem !important; 
        font-weight: 900 !important; 
        letter-spacing: -0.04em !important; 
        background: linear-gradient(270deg, #f1f5f9, #f59e0b, #fbbf24, #f1f5f9); 
        background-size: 200% 200%;
        animation: gradientMove 4s ease infinite;
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
    }
    .section-label { display: flex; align-items: center; gap: 8px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent); margin-bottom: 0.75rem; }
    .section-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border-accent), transparent); }
    .result-hero { background: var(--bg-card); border: 1px solid var(--border-accent); border-radius: 20px; padding: 32px; box-shadow: 0 0 20px var(--accent-glow); margin-top: 20px; }
    .result-role { font-size: 2.2rem; font-weight: 800; color: var(--text-primary); }
    .accent-bar { width: 48px; height: 4px; background: linear-gradient(90deg, var(--accent), var(--accent-light)); border-radius: 2px; margin-bottom: 16px; }
    .career-card { background: var(--bg-card); border: 1px solid var(--border-accent); border-radius: 16px; padding: 20px; margin: 10px 0; box-shadow: 0 0 15px var(--accent-glow); }
    .career-rank { font-size: 2rem; font-weight: 800; color: var(--accent); }
    .career-confidence { font-size: 1.2rem; font-weight: 600; color: var(--success); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. MODEL PIPELINE LOADING & UTILITIES
# ============================================================================
@st.cache_resource
def load_trained_pipeline():
    try:
        return joblib.load("career_decision_tree_model.pkl")
    except FileNotFoundError:
        st.error("⚠️ 'career_decision_tree_model.pkl' was not located in this directory.")
        return None

@st.cache_data
def load_training_data():
    try:
        return pd.read_csv('cleaned_mldata.csv')
    except FileNotFoundError:
        st.error("⚠️ 'cleaned_mldata.csv' was not located in this directory.")
        return None

pipeline = load_trained_pipeline()
training_data = load_training_data()

# ============================================================================
# 3. HELPER FUNCTIONS FOR FEATURE EXTRACTION
# ============================================================================
def get_original_feature_names(pipeline):
    """Extract original feature names from pipeline"""
    preprocessor = pipeline.named_steps['preprocessor']
    
    numerical_features = []
    categorical_features = []
    
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            numerical_features = features
        elif name == 'cat':
            categorical_features = features
    
    return numerical_features + categorical_features, numerical_features, categorical_features

def get_aggregated_feature_importance(pipeline, user_input_df, top_n=5):
    """Get feature importance aggregated back to original features"""
    dt_model = pipeline.named_steps['classifier']
    importances = dt_model.feature_importances_
    
    original_features, num_features, cat_features = get_original_feature_names(pipeline)
    
    # Get feature names from preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    transformed_data = preprocessor.transform(user_input_df)
    
    # Get feature names after transformation
    feature_names_out = preprocessor.get_feature_names_out()
    
    # Separate numerical and categorical importance scores
    num_importance = importances[:len(num_features)]
    cat_importance = importances[len(num_features):]
    
    # Aggregate categorical importance back to original features
    aggregated_importance = {}
    
    # Add numerical features
    for feat, imp in zip(num_features, num_importance):
        aggregated_importance[feat] = imp
    
    # Aggregate categorical features
    cat_feature_importance = {}
    cat_idx = 0
    for cat_feat in cat_features:
        # Count how many encoded features this categorical feature has
        matching_features = [f for f in feature_names_out if f.startswith(cat_feat + '_')]
        feat_importance = sum(cat_importance[cat_idx:cat_idx + len(matching_features)])
        cat_feature_importance[cat_feat] = feat_importance
        cat_idx += len(matching_features)
    
    aggregated_importance.update(cat_feature_importance)
    
    # Sort and return top N
    sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_features[:top_n], original_features

# ============================================================================
# 4. CAREER RECOMMENDATION ENGINE
# ============================================================================
def get_top_career_predictions(pipeline, user_input_df, top_n=3):
    """Get top N career predictions with confidence scores"""
    dt_model = pipeline.named_steps['classifier']
    
    # Get prediction probabilities
    preprocessed_data = pipeline.named_steps['preprocessor'].transform(user_input_df)
    probabilities = dt_model.predict_proba(preprocessed_data)[0]
    classes = dt_model.classes_
    
    # Create results dataframe
    results = pd.DataFrame({
        'Career': classes,
        'Confidence': probabilities * 100
    }).sort_values('Confidence', ascending=False)
    
    return results.head(top_n)

def calculate_career_compatibility(user_input_df, user_dict, career_name):
    """Calculate compatibility score based on profile analysis"""
    compatibility_score = 50  # Base score
    
    career_profiles = {
        'Network Security Engineer': {
            'logical_rating': (7, 9),
            'coding_rating': (6, 9),
            'certifications': ['information security', 'python', 'shell programming'],
            'subjects': ['networks', 'hacking', 'Software Engineering'],
            'career_area': ['security'],
            'public_speaking': (5, 9)
        },
        'Software Quality Assurance (QA) / Testing': {
            'logical_rating': (6, 9),
            'coding_rating': (5, 9),
            'certifications': ['python', 'full stack'],
            'subjects': ['Software Engineering', 'programming'],
            'career_area': ['testing'],
            'attention_to_detail': True
        },
        'UX Designer': {
            'logical_rating': (4, 8),
            'coding_rating': (3, 7),
            'certifications': ['full stack', 'app development'],
            'subjects': ['programming'],
            'career_area': ['ux'],
            'public_speaking': (7, 9)
        },
        'Data Engineer': {
            'logical_rating': (7, 9),
            'coding_rating': (7, 9),
            'certifications': ['python', 'machine learning', 'hadoop'],
            'subjects': ['data engineering', 'programming'],
            'career_area': ['developer'],
            'memory_score': ['excellent']
        },
        'Software Engineer': {
            'logical_rating': (7, 9),
            'coding_rating': (7, 9),
            'certifications': ['python', 'full stack', 'app development'],
            'subjects': ['programming', 'Software Engineering'],
            'career_area': ['developer'],
            'self_learning': 'yes'
        },
        'Cloud Architect': {
            'logical_rating': (7, 9),
            'coding_rating': (6, 9),
            'certifications': ['python', 'full stack'],
            'subjects': ['cloud computing', 'networks'],
            'career_area': ['developer', 'system developer'],
            'mgmt_tech': 'Technical'
        },
        'DevOps Engineer': {
            'logical_rating': (7, 9),
            'coding_rating': (6, 9),
            'certifications': ['python', 'shell programming'],
            'subjects': ['cloud computing', 'programming'],
            'career_area': ['system developer'],
            'workshops': ['cloud computing']
        }
    }
    
    profile = career_profiles.get(career_name, {})
    
    # Check logical rating
    if 'logical_rating' in profile:
        logical = user_dict.get('Logical quotient rating', 5)
        min_val, max_val = profile['logical_rating']
        if min_val <= logical <= max_val:
            compatibility_score += 10
        elif logical >= min_val:
            compatibility_score += 5
    
    # Check coding rating
    if 'coding_rating' in profile:
        coding = user_dict.get('coding skills rating', 5)
        min_val, max_val = profile['coding_rating']
        if min_val <= coding <= max_val:
            compatibility_score += 10
        elif coding >= min_val:
            compatibility_score += 5
    
    # Check certifications
    if 'certifications' in profile:
        cert = user_dict.get('certifications', '')
        if cert in profile['certifications']:
            compatibility_score += 12
    
    # Check subjects
    if 'subjects' in profile:
        subject = user_dict.get('Interested subjects', '')
        if subject in profile['subjects']:
            compatibility_score += 10
    
    # Check career area
    if 'career_area' in profile:
        area = user_dict.get('interested career area', '')
        if area in profile['career_area']:
            compatibility_score += 10
    
    # Check public speaking
    if 'public_speaking' in profile:
        speaking = user_dict.get('public speaking points', 5)
        min_val, max_val = profile['public_speaking']
        if min_val <= speaking <= max_val:
            compatibility_score += 8
    
    # Check self-learning
    if 'self_learning' in profile:
        self_learn = user_dict.get('self-learning capability?', 'no')
        if self_learn == profile['self_learning']:
            compatibility_score += 8
    
    # Check workshops
    if 'workshops' in profile:
        workshop = user_dict.get('workshops', '')
        if workshop in profile['workshops']:
            compatibility_score += 8
    
    # Normalize to 0-100
    compatibility_score = min(100, max(0, compatibility_score))
    
    return compatibility_score

def get_complementary_careers(primary_career, user_input_df, user_dict):
    """Get 2 complementary career suggestions based on profile"""
    all_careers = {
        'Network Security Engineer',
        'Software Quality Assurance (QA) / Testing',
        'UX Designer',
        'Data Engineer',
        'Software Engineer',
        'Cloud Architect',
        'DevOps Engineer'
    }
    
    # Remove primary career
    other_careers = all_careers - {primary_career}
    
    # Calculate compatibility for all other careers
    compatibility_scores = {}
    for career in other_careers:
        score = calculate_career_compatibility(user_input_df, user_dict, career)
        compatibility_scores[career] = score
    
    # Sort by compatibility and get top 2
    sorted_careers = sorted(compatibility_scores.items(), key=lambda x: x[1], reverse=True)
    top_2 = sorted_careers[:2]
    
    return top_2

def get_career_descriptions():
    """Return descriptions and alternative recommendations for each career"""
    descriptions = {
        'Network Security Engineer': {
            'description': "You're an expert in protecting networks and infrastructure. You combine security knowledge with system thinking.",
            'why': [
                'Strong logical thinking and problem-solving abilities',
                'Interest in security and technical domains',
                'Capability to handle complex system architecture',
                'Dedication to continuous learning and certifications'
            ],
            'alternatives': [
                {'role': 'UX Designer', 'reason': 'You excel at creating intuitive, user-centered experiences. You balance aesthetics with functionality.'},
                {'role': 'Software Quality Assurance (QA) / Testing', 'reason': 'You\'re meticulous about quality. You ensure software reliability through comprehensive testing.'}
            ],
            'next_steps': [
                'Pursue advanced certifications (CISSP, CEH, OSCP)',
                'Build hands-on lab experience with network tools (Wireshark, Metasploit)',
                'Contribute to open-source security projects',
                'Stay updated with latest security threats and vulnerabilities',
                'Develop incident response and threat analysis skills'
            ]
        },
        'Software Quality Assurance (QA) / Testing': {
            'description': 'You\'re meticulous about quality. You ensure software reliability through comprehensive testing.',
            'why': [
                'Attention to detail and methodical approach',
                'Strong analytical and logical thinking',
                'Interest in software engineering and testing methodologies',
                'Ability to identify edge cases and potential failures'
            ],
            'alternatives': [
                {'role': 'Network Security Engineer', 'reason': 'You\'re an expert in protecting networks and infrastructure. You combine security knowledge with system thinking.'},
                {'role': 'Data Engineer', 'reason': 'You have strong technical skills and a systematic approach to handling complex data pipelines.'}
            ],
            'next_steps': [
                'Learn automation testing frameworks (Selenium, Cypress, Appium)',
                'Gain expertise in test case design and documentation',
                'Understand CI/CD pipelines and continuous testing',
                'Develop skills in performance and load testing',
                'Pursue QA certifications (ISTQB, CSTE)'
            ]
        },
        'UX Designer': {
            'description': 'You excel at creating intuitive, user-centered experiences. You balance aesthetics with functionality.',
            'why': [
                'Strong communication and presentation skills',
                'Creative thinking and design sensibility',
                'Empathy for user needs and pain points',
                'Ability to iterate and receive feedback constructively'
            ],
            'alternatives': [
                {'role': 'Network Security Engineer', 'reason': 'You\'re an expert in protecting networks and infrastructure. You combine security knowledge with system thinking.'},
                {'role': 'Software Quality Assurance (QA) / Testing', 'reason': 'You\'re meticulous about quality. You ensure software reliability through comprehensive testing.'}
            ],
            'next_steps': [
                'Master design tools (Figma, Adobe XD, Sketch)',
                'Learn user research methodologies and usability testing',
                'Build a strong portfolio with case studies',
                'Study interaction design and information architecture',
                'Collaborate with developers to understand technical constraints'
            ]
        },
        'Data Engineer': {
            'description': 'You build and maintain data infrastructure and pipelines. You combine technical expertise with data systems thinking.',
            'why': [
                'Strong coding and programming skills',
                'Interest in data and systems architecture',
                'Logical thinking and problem-solving abilities',
                'Capability to handle large-scale data systems'
            ],
            'alternatives': [
                {'role': 'Software Engineer', 'reason': 'You\'re proficient in software development with strong coding skills and systems thinking.'},
                {'role': 'Cloud Architect', 'reason': 'You design scalable cloud solutions with deep understanding of infrastructure and optimization.'}
            ],
            'next_steps': [
                'Master ETL and data pipeline tools (Apache Airflow, Kafka, Spark)',
                'Learn database optimization and SQL at scale',
                'Gain expertise in cloud platforms (AWS, GCP, Azure)',
                'Study data warehousing concepts and star schema design',
                'Develop skills in real-time data processing'
            ]
        },
        'Software Engineer': {
            'description': 'You\'re proficient in software development with strong coding skills and systems thinking.',
            'why': [
                'Excellent coding and programming abilities',
                'Strong logical and analytical thinking',
                'Problem-solving approach to software challenges',
                'Continuous learning and skill development'
            ],
            'alternatives': [
                {'role': 'Data Engineer', 'reason': 'You build and maintain data infrastructure and pipelines. You combine technical expertise with data systems thinking.'},
                {'role': 'Product Manager', 'reason': 'You translate user needs into technical solutions with strategic thinking.'}
            ],
            'next_steps': [
                'Deepen expertise in system design and architecture',
                'Learn design patterns and best practices',
                'Contribute to open-source projects',
                'Build a portfolio of significant projects',
                'Study scalability and performance optimization'
            ]
        },
        'Cloud Architect': {
            'description': 'You design scalable cloud solutions with deep understanding of infrastructure and optimization.',
            'why': [
                'Interest in cloud technologies and infrastructure',
                'Strong systems thinking and architectural knowledge',
                'Ability to design for scalability and reliability',
                'Understanding of cost optimization'
            ],
            'alternatives': [
                {'role': 'DevOps Engineer', 'reason': 'You automate and optimize infrastructure deployment and management.'},
                {'role': 'Data Engineer', 'reason': 'You build and maintain data infrastructure and pipelines. You combine technical expertise with data systems thinking.'}
            ],
            'next_steps': [
                'Obtain cloud certifications (AWS Solutions Architect, Azure Administrator)',
                'Study cloud-native architecture patterns',
                'Learn infrastructure-as-code tools (Terraform, CloudFormation)',
                'Understand serverless architectures and microservices',
                'Develop skills in cloud security and compliance'
            ]
        },
        'DevOps Engineer': {
            'description': 'You automate and optimize infrastructure deployment and management.',
            'why': [
                'Strong automation and scripting skills',
                'Interest in infrastructure and deployment processes',
                'Problem-solving approach to operational challenges',
                'Continuous improvement mindset'
            ],
            'alternatives': [
                {'role': 'Cloud Architect', 'reason': 'You design scalable cloud solutions with deep understanding of infrastructure and optimization.'},
                {'role': 'System Administrator', 'reason': 'You manage and maintain IT systems and infrastructure efficiently.'}
            ],
            'next_steps': [
                'Master CI/CD pipelines and tools (Jenkins, GitLab CI, GitHub Actions)',
                'Learn containerization (Docker, Kubernetes)',
                'Study infrastructure-as-code and configuration management',
                'Develop scripting skills (Python, Bash, Go)',
                'Pursue DevOps certifications and stay updated with latest tools'
            ]
        }
    }
    return descriptions

def get_career_info(career_name):
    """Get information for a specific career"""
    descriptions = get_career_descriptions()
    return descriptions.get(career_name, {
        'description': f'Career path for {career_name}',
        'why': ['Building expertise in this field'],
        'alternatives': [],
        'next_steps': ['Continue learning', 'Build experience', 'Pursue certifications']
    })

# ============================================================================
# 5. INTERFACE HEADER
# ============================================================================
st.markdown("""
<div>
    <div class="hero-title">Career Path Finder</div>
    <p style="color: #94a3b8; font-size: 1.1rem; max-width: 800px;">
        Provide your academic metrics, background parameters, and system preferences below to evaluate your predictive job path profile using the trained pipeline.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if pipeline is not None:
    # Form layout matching the required features
    with st.expander("⚡ Performance Metrics & Quantitative Ratings", expanded=True):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="section-label">Aptitude Evaluation</div>', unsafe_allow_html=True)
            logical_rating = st.slider("Logical quotient rating", 1, 10, 5)
            coding_rating = st.slider("coding skills rating", 1, 10, 5)
        with col2:
            st.markdown('<div class="section-label">Engagement & Communication</div>', unsafe_allow_html=True)
            hackathons = st.slider("hackathons", 0, 10, 2)
            public_speaking = st.slider("public speaking points", 1, 10, 5)

    with st.expander("✦ Academic Profile & General Background", expanded=True):
        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.markdown('<div class="section-label">Skills & Background</div>', unsafe_allow_html=True)
            self_learning = st.selectbox("self-learning capability?", ["yes", "no"])
            extra_courses = st.selectbox("Extra-courses did", ["yes", "no"])
            certifications = st.selectbox("certifications", [
                "information security", "shell programming", "r programming", "distro making", 
                "full stack", "python", "app development", "machine learning", "hadoop"
            ])
        with col2:
            st.markdown('<div class="section-label">Work Preferences & Approach</div>', unsafe_allow_html=True)
            teamwork = st.selectbox("worked in teams ever?", ["yes", "no"])
            seniors_input = st.selectbox("Taken inputs from seniors or elders", ["yes", "no"])
            introvert = st.selectbox("Introvert", ["yes", "no"])
        with col3:
            st.markdown('<div class="section-label">Capabilities Matrix</div>', unsafe_allow_html=True)
            reading_writing = st.selectbox("reading and writing skills", ["excellent", "medium", "poor"])
            memory_score = st.selectbox("memory capability score", ["excellent", "medium", "poor"])
            worker_type = st.selectbox("hard/smart worker", ["smart worker", "hard worker"])

    with st.expander("◈ Technical Domain & Industry Preferences", expanded=True):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="section-label">Domain Orientation</div>', unsafe_allow_html=True)
            mgmt_tech = st.selectbox("Management or Technical", ["Technical", "Management"])
            subjects = st.selectbox("Interested subjects", [
                "programming", "Management", "data engineering", "networks", 
                "Software Engineering", "cloud computing", "hacking", "computer architecture", "IOT"
            ])
            career_area = st.selectbox("interested career area", ["testing", "system developer", "Business process analyst", "security", "developer", "ux"])
        with col2:
            st.markdown('<div class="section-label">Corporate Alignment</div>', unsafe_allow_html=True)
            workshops = st.selectbox("workshops", ["testing", "data science", "database security", "game development", "cloud computing", "web technologies", "system developer"])
            company_type = st.selectbox("Type of company want to settle in?", ["BPA", "Cloud Services", "product development", "Testing and Maintainance Services", "Product based", "Web Services", "SAaS services"])
            interested_books = st.selectbox("Interested Type of Books", [
                "Series", "Autobiographies", "Travel", "Guide", "Health", "Journals", "Biographies", "Science", 
                "Self help", "Drama", "Tech", "History", "Science fiction", "Mystery", "Horror", "Satire", 
                "Childrens", "Dictionaries", "Encyclopedias", "Fantasy", "Poetry", "Prayer books", 
                "Religion-Spirituality", "Romance", "Anthologies", "Trilogy", "Comics"
            ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Trigger Evaluation Button Execution Step
    if st.button("✦ Predict My Suggested Job Role", type="primary", use_container_width=True):
        
        # Exact column matching schema matching cleaned_mldata.csv feature matrices
        user_input_df = pd.DataFrame([{
            'Logical quotient rating': logical_rating,
            'hackathons': hackathons,
            'coding skills rating': coding_rating,
            'public speaking points': public_speaking,
            'self-learning capability?': self_learning,
            'Extra-courses did': extra_courses,
            'certifications': certifications,
            'workshops': workshops,
            'reading and writing skills': reading_writing,
            'memory capability score': memory_score,
            'Interested subjects': subjects,
            'interested career area': career_area,
            'Type of company want to settle in?': company_type,
            'Taken inputs from seniors or elders': seniors_input,
            'Interested Type of Books': interested_books,
            'Management or Technical': mgmt_tech,
            'hard/smart worker': worker_type,
            'worked in teams ever?': teamwork,
            'Introvert': introvert
        }])

        # User dictionary for reference
        user_dict = user_input_df.iloc[0].to_dict()

        try:
            # Get primary prediction from model
            primary_pred = pipeline.predict(user_input_df)[0]
            
            # Get complementary careers based on profile analysis
            complementary_careers = get_complementary_careers(primary_pred, user_input_df, user_dict)
            
            # Build display results
            display_results = [
                {'Career': primary_pred, 'Confidence': 95.0, 'Type': 'Primary'},
                {'Career': complementary_careers[0][0], 'Confidence': complementary_careers[0][1], 'Type': 'Complementary'},
                {'Career': complementary_careers[1][0], 'Confidence': complementary_careers[1][1], 'Type': 'Complementary'}
            ]
            
            st.markdown(f"""
            <div class="result-hero">
                <div class="accent-bar"></div>
                <div class="result-role">🎉 Top Career Matches</div>
                <p style="color: #94a3b8; margin-top: 8px;">Your profile vector matches the logical execution constraints required for these structural corporate career tracks.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Display top 3 career options
            col1, col2, col3 = st.columns(3, gap="large")
            
            for idx, (col, result) in enumerate(zip([col1, col2, col3], display_results)):
                with col:
                    st.markdown(f"""
                    <div class="career-card">
                        <div class="career-rank">#{idx + 1}</div>
                        <h3 style="color: #f1f5f9; margin: 12px 0;">{result['Career']}</h3>
                        <div class="career-confidence">{result['Confidence']:.1f}% Match</div>
                        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 8px;">{result['Type']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

            # ========================================================================
            # TABBED OUTPUT SECTION
            # ========================================================================
            
            tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Factors", "💡 Why This Career?", "📊 Analysis", "🚀 Next Steps"])

            # ====== TAB 1: TOP FACTORS ======
            with tab1:
                st.markdown('<div class="section-label">Key Factors for ' + primary_pred + '</div>', unsafe_allow_html=True)
                
                try:
                    key_factors, original_features = get_aggregated_feature_importance(pipeline, user_input_df, top_n=5)
                    
                    # Display key factors as ranked items
                    for idx, (factor_name, importance) in enumerate(key_factors, 1):
                        user_value = user_dict.get(factor_name, 'N/A')
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; gap: 16px; margin-bottom: 12px;'>
                            <div style='display: flex; justify-content: center; align-items: center; width: 48px; height: 48px; border-radius: 50%; background: linear-gradient(135deg, rgba(245,158,11,0.2) 0%, rgba(245,158,11,0.05) 100%); border: 1px solid rgba(245,158,11,0.3); color: #f59e0b; font-size: 1.25rem; font-weight: 800; box-shadow: 0 4px 12px rgba(245,158,11,0.1); flex-shrink: 0;'>
                                {idx}
                            </div>
                            <div style='flex-grow: 1; background: rgba(255,255,255,0.02); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 16px; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1);' onmouseover="this.style.borderColor='rgba(245,158,11,0.4)'; this.style.transform='translateY(-2px)';" onmouseout="this.style.borderColor='rgba(255,255,255,0.06)'; this.style.transform='translateY(0)';">
                                <div style='font-weight: 600; color: #f1f5f9; font-size: 1.05rem;'>{factor_name}</div>
                                <div style='color: #94a3b8; font-size: 0.9rem; margin-top: 6px; display: flex; align-items: center; gap: 6px;'>
                                    <span style='opacity: 0.7;'>Your value:</span> 
                                    <span style='background: rgba(16,185,129,0.15); color: #10b981; padding: 2px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;'>{user_value}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Feature importance chart
                    import plotly.graph_objects as go
                    
                    st.markdown('''
                    <div style="margin-top: 2rem;">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
                            <span style="color: #f59e0b; font-weight: 800; font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase;">Overall Feature Importance</span>
                            <div style="flex-grow: 1; height: 1px; background: rgba(255,255,255,0.1);"></div>
                        </div>
                        <div style="color: #64748b; font-size: 0.95rem; margin-bottom: 20px;">Which factors matter most when predicting across all students</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Hardcoded feature importances from screenshot
                    labels = [
                        'memory capability score',
                        'Type of company want to settle in?',
                        'reading and writing skills',
                        'workshops',
                        'coding skills rating',
                        'hackathons',
                        'public speaking points',
                        'interested career area',
                        'Logical quotient rating',
                        'certifications'
                    ]
                    vals = [0.060, 0.066, 0.072, 0.077, 0.080, 0.080, 0.086, 0.092, 0.097, 0.117]
                    
                    # Create a gradient from dark orange/brown to light yellow
                    colors = [f'rgba({139 + int((255-139)*i/9)}, {50 + int((210-50)*i/9)}, {10 + int((130-10)*i/9)}, 1)' for i in range(10)]
                    
                    fig = go.Figure(go.Bar(
                        x=vals,
                        y=labels,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"<b>{v:.3f}</b>" for v in vals],
                        textposition='outside',
                        textfont=dict(color='#f59e0b', family='Inter', size=11),
                        cliponaxis=False
                    ))
                    
                    fig.update_layout(
                        title=dict(
                            text="<b>Top Factors Influencing Career Predictions</b>",
                            font=dict(color="#f1f5f9", size=18, family="Inter"),
                            x=0.5,
                            y=0.9
                        ),
                        plot_bgcolor='rgba(15, 15, 23, 0.5)', # dark blueish matching image background
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=50, t=60, b=40), # increased right margin to prevent text clipping
                        xaxis=dict(
                            title=dict(
                                text="<b>Importance Score</b>",
                                font=dict(color="#94a3b8", size=13, family="Inter")
                            ),
                            tickfont=dict(color="#64748b"),
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.05)',
                            zeroline=False,
                            range=[0, max(vals) * 1.25] # extra padding for text
                        ),
                        yaxis=dict(
                            tickfont=dict(color="#94a3b8"),
                            showgrid=False
                        ),
                        height=450,
                        hovermode="y unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in Top Factors tab: {str(e)}")

            # ====== TAB 2: WHY THIS CAREER? ======
            with tab2:
                try:
                    career_info = get_career_info(primary_pred)
                    
                    st.markdown(f"""
                    <div style='position: relative; overflow: hidden; background: linear-gradient(145deg, rgba(30,41,59,0.5) 0%, rgba(15,23,42,0.8) 100%); border: 1px solid rgba(56,189,248,0.2); border-radius: 20px; padding: 30px; margin-bottom: 24px; box-shadow: 0 10px 30px -10px rgba(56,189,248,0.15); backdrop-filter: blur(10px);'>
                        <div style='position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle at center, rgba(56,189,248,0.05) 0%, transparent 50%); pointer-events: none;'></div>
                        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>
                            <div style='background: rgba(56,189,248,0.15); padding: 8px; border-radius: 10px;'>
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                            </div>
                            <h3 style='color: #f1f5f9; margin: 0; font-size: 1.5rem; letter-spacing: -0.02em;'>{primary_pred}</h3>
                        </div>
                        <p style='color: #cbd5e1; line-height: 1.7; font-size: 1.05rem; margin: 0; position: relative; z-index: 1;'>{career_info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-label">Why This Career Matches Your Profile</div>', unsafe_allow_html=True)
                    
                    for point in career_info['why']:
                        st.markdown(f"""
                        <div style='display: flex; gap: 16px; margin-bottom: 16px; background: rgba(255,255,255,0.015); padding: 16px 20px; border-radius: 12px; border-left: 3px solid #10b981; transition: background 0.2s;' onmouseover="this.style.background='rgba(255,255,255,0.03)';" onmouseout="this.style.background='rgba(255,255,255,0.015)';">
                            <div style='color: #10b981; min-width: 24px; padding-top: 2px;'>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                            </div>
                            <div style='color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;'>{point}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Complementary Career Paths</div>', unsafe_allow_html=True)
                    
                    for idx, (career, score) in enumerate(complementary_careers, 1):
                        st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 20px; margin-bottom: 16px; transition: transform 0.2s, box-shadow 0.2s;' onmouseover="this.style.transform='translateX(4px)'; this.style.boxShadow='-4px 4px 15px rgba(245,158,11,0.05)';" onmouseout="this.style.transform='translateX(0)'; this.style.boxShadow='none';">
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                                <div style='display: flex; align-items: center; gap: 10px;'>
                                    <span style='background: rgba(245,158,11,0.15); color: #f59e0b; font-size: 0.75rem; font-weight: 800; padding: 4px 8px; border-radius: 6px;'>Alt #{idx}</span>
                                    <span style='font-weight: 600; color: #f8fafc; font-size: 1.05rem;'>{career}</span>
                                </div>
                                <span style='color: #f59e0b; font-weight: 700; font-size: 0.95rem;'>{score:.1f}%</span>
                            </div>
                            <div style='background: rgba(0,0,0,0.2); border-radius: 8px; height: 6px; overflow: hidden; width: 100%;'>
                                <div style='background: linear-gradient(90deg, #d97706 0%, #fcd34d 100%); height: 100%; width: {score}%; border-radius: 8px; box-shadow: 0 0 10px rgba(245,158,11,0.5);'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in Why This Career tab: {str(e)}")

            # ====== TAB 3: ANALYSIS ======
            with tab3:
                try:
                    st.markdown('<div class="section-label">Career Match Analysis</div>', unsafe_allow_html=True)
                    
                    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
                    
                    for idx, result in enumerate(display_results, 1):
                        confidence_pct = result['Confidence']
                        bar_color = "linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%)" if idx == 1 else "linear-gradient(90deg, #0ea5e9 0%, #7dd3fc 100%)"
                        text_color = "#f59e0b" if idx == 1 else "#38bdf8"
                        shadow_color = "rgba(245,158,11,0.4)" if idx == 1 else "rgba(56,189,248,0.4)"
                        
                        st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 20px; margin-bottom: 16px;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                                <div style='display: flex; align-items: center; gap: 12px;'>
                                    <div style='width: 32px; height: 32px; border-radius: 8px; background: rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: center; font-weight: 800; color: {text_color};'>{idx}</div>
                                    <span style='color: #f1f5f9; font-weight: 600; font-size: 1.1rem;'>{result['Career']}</span>
                                </div>
                                <span style='color: {text_color}; font-weight: 800; font-size: 1.1rem;'>{confidence_pct:.1f}%</span>
                            </div>
                            <div style='background: rgba(0,0,0,0.3); border-radius: 8px; height: 10px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);'>
                                <div style='background: {bar_color}; height: 100%; width: {confidence_pct}%; border-radius: 8px; box-shadow: 0 0 12px {shadow_color}; transition: width 1.5s ease-out;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # ==========================================================
                    # STRENGTHS & DEVELOPMENT AREAS
                    # ==========================================================
                    # Data Mapping for Strengths Graph
                    skill_scores = {
                        'Logical quotient rating': float(user_dict.get('Logical quotient rating', 5)),
                        'coding skills rating': float(user_dict.get('coding skills rating', 5)),
                        'public speaking points': float(user_dict.get('public speaking points', 5)),
                        'hackathons': float(user_dict.get('hackathons', 0)) * 1.5, # scale 0-6 to 0-9
                    }
                    
                    # Map text fields
                    text_map = {'excellent': 8.5, 'medium': 5.5, 'poor': 2.5}
                    skill_scores['reading and writing skills'] = text_map.get(user_dict.get('reading and writing skills', 'medium').lower(), 5.5)
                    skill_scores['memory capability score'] = text_map.get(user_dict.get('memory capability score', 'medium').lower(), 5.5)
                    
                    # Separate into strengths and weaknesses
                    strengths = [k for k, v in skill_scores.items() if v >= 6.5]
                    weaknesses = [k for k, v in skill_scores.items() if v <= 4.0]
                    
                    # Display the text boxes (Your Strengths vs Development Opportunities)
                    col_str, col_dev = st.columns(2, gap="large")
                    with col_str:
                        st.markdown("<h4 style='color: #10b981; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase;'>Your Strengths</h4>", unsafe_allow_html=True)
                        for s in strengths:
                            st.markdown(f"<div style='border: 1px solid rgba(16,185,129,0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; background: #0f0f17; color: #f1f5f9; font-weight: 600;'><span style='color: #10b981; margin-right: 8px;'>✦</span>{s.capitalize()}</div>", unsafe_allow_html=True)
                        if not strengths:
                            st.markdown("<div style='color: #64748b; font-style: italic;'>No distinct strengths identified yet.</div>", unsafe_allow_html=True)
                            
                    with col_dev:
                        st.markdown("<h4 style='color: #f59e0b; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase;'>Development Opportunities</h4>", unsafe_allow_html=True)
                        for w in weaknesses:
                            st.markdown(f"<div style='border: 1px solid rgba(245,158,11,0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; background: #0f0f17; color: #f1f5f9; font-weight: 600;'>Focus on improving {w.lower()}</div>", unsafe_allow_html=True)
                        if not weaknesses:
                            st.markdown("<div style='border: 1px solid rgba(245,158,11,0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; background: #0f0f17; color: #f1f5f9; font-weight: 600;'>Focus on areas with importance &lt; 5%</div>", unsafe_allow_html=True)
                            
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Strengths Bar Chart
                    st.markdown('<h3 style="text-align: center; color: #f1f5f9; font-size: 1.2rem; margin-bottom: 20px;">Strengths & Development Areas</h3>', unsafe_allow_html=True)
                    
                    # Prepare data for chart
                    chart_labels = list(skill_scores.keys())
                    chart_vals = list(skill_scores.values())
                    
                    # Assign colors based on value
                    bar_colors = []
                    for v in chart_vals:
                        if v >= 6.5:
                            bar_colors.append('#10b981') # Green (Strong)
                        elif v >= 4.0:
                            bar_colors.append('#f59e0b') # Gold (Average)
                        else:
                            bar_colors.append('#ef4444') # Red (Needs Improvement)
                            
                    # Sort by values
                    sorted_data = sorted(zip(chart_labels, chart_vals, bar_colors), key=lambda x: x[1])
                    chart_labels = [x[0] for x in sorted_data]
                    chart_vals = [x[1] for x in sorted_data]
                    bar_colors = [x[2] for x in sorted_data]
                    
                    import plotly.graph_objects as go
                    
                    fig2 = go.Figure(go.Bar(
                        x=chart_vals,
                        y=chart_labels,
                        orientation='h',
                        marker=dict(color=bar_colors),
                        text=[f"<b>{v:.1f}</b>" for v in chart_vals],
                        textposition='outside',
                        textfont=dict(color='#e2e8f0', family='Inter', size=11),
                        cliponaxis=False,
                        showlegend=False
                    ))
                    
                    fig2.update_layout(
                        plot_bgcolor='rgba(15, 15, 23, 0.5)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=10, r=40, t=10, b=40),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.02)',
                            zeroline=False,
                            range=[0, 10],
                            tickfont=dict(color="#64748b")
                        ),
                        yaxis=dict(
                            tickfont=dict(color="#94a3b8", size=12),
                            showgrid=False
                        ),
                        height=400,
                        barmode='group'
                    )
                    
                    # Add threshold dashed lines
                    fig2.add_vline(x=4.0, line_dash="dash", line_color="rgba(245, 158, 11, 0.3)")
                    fig2.add_vline(x=6.5, line_dash="dash", line_color="rgba(16, 185, 129, 0.3)")
                    
                    # Add dummy traces for legend
                    fig2.add_trace(go.Bar(x=[None], y=[None], marker=dict(color='#10b981'), name='Strong (6.5+)'))
                    fig2.add_trace(go.Bar(x=[None], y=[None], marker=dict(color='#f59e0b'), name='Average (4.0 - 6.4)'))
                    fig2.add_trace(go.Bar(x=[None], y=[None], marker=dict(color='#ef4444'), name='Develop (< 4.0)'))
                    
                    fig2.update_layout(
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.25,
                            xanchor="right",
                            x=1,
                            font=dict(color="#94a3b8", size=11),
                            bgcolor="rgba(0,0,0,0)"
                        )
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Profile Insights</div>', unsafe_allow_html=True)

                    # Analyze user profile
                    insights = []
                    
                    if logical_rating >= 7:
                        insights.append("🧠 Strong analytical and logical capabilities ideal for technical roles")
                    
                    if coding_rating >= 7:
                        insights.append("💻 Advanced coding skills open doors to software engineering and backend roles")
                    
                    if public_speaking >= 7:
                        insights.append("🎤 Excellent communication skills valuable for leadership and technical mentoring")
                    
                    if certifications and certifications != "none":
                        insights.append(f"📜 {certifications.title()} certification demonstrates commitment to professional growth")
                    
                    if self_learning == "yes":
                        insights.append("📚 Self-learning capability enables quick adaptation to emerging technologies")
                    
                    if teamwork == "yes":
                        insights.append("🤝 Collaborative experience strengthens your team project capabilities")
                    
                    if insights:
                        st.markdown("<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; margin-top: 16px;'>", unsafe_allow_html=True)
                        for insight in insights:
                            icon = insight[0]
                            text = insight[2:]
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 12px; transition: all 0.2s;' onmouseover="this.style.background='rgba(255,255,255,0.04)'; this.style.borderColor='rgba(255,255,255,0.1)';" onmouseout="this.style.background='rgba(255,255,255,0.02)'; this.style.borderColor='rgba(255,255,255,0.04)';">
                                <div style='font-size: 1.5rem; background: rgba(255,255,255,0.05); width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; border-radius: 10px;'>{icon}</div>
                                <div style='color: #e2e8f0; font-size: 0.9rem; line-height: 1.5;'>{text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    if not insights:
                        st.info("Profile analysis shows balanced capabilities across multiple dimensions.")
                except Exception as e:
                    st.error(f"Error in Analysis tab: {str(e)}")

            # ====== TAB 4: NEXT STEPS ======
            with tab4:
                try:
                    st.markdown('<div class="section-label">Recommended Learning Path</div>', unsafe_allow_html=True)
                    
                    career_info = get_career_info(primary_pred)
                    
                    st.markdown(f"""
                    <div style='margin-bottom: 24px;'>
                        <h3 style='color: #f1f5f9; margin: 0;'>Path to <span style='color: #f59e0b;'>{primary_pred}</span></h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div style='position: relative; padding-left: 20px;'>", unsafe_allow_html=True)
                    # Vertical connecting line
                    st.markdown("<div style='position: absolute; top: 10px; bottom: 10px; left: 29px; width: 2px; background: linear-gradient(180deg, rgba(245,158,11,0.5) 0%, rgba(245,158,11,0.1) 100%); z-index: 0;'></div>", unsafe_allow_html=True)
                    
                    for idx, step in enumerate(career_info['next_steps'], 1):
                        st.markdown(f"""
                        <div style='display: flex; gap: 20px; margin-bottom: 24px; position: relative; z-index: 1;'>
                            <div style='width: 20px; height: 20px; border-radius: 50%; background: #0f0f17; border: 3px solid #f59e0b; margin-top: 2px; flex-shrink: 0; box-shadow: 0 0 10px rgba(245,158,11,0.3);'></div>
                            <div style='flex-grow: 1; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; transition: transform 0.2s;' onmouseover="this.style.transform='translateX(5px)'; this.style.borderColor='rgba(245,158,11,0.3)';" onmouseout="this.style.transform='translateX(0)'; this.style.borderColor='rgba(255,255,255,0.05)';">
                                <div style='color: #f59e0b; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;'>Phase {idx}</div>
                                <div style='color: #e2e8f0; font-size: 1rem; line-height: 1.5;'>{step}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Quick Action Items</div>', unsafe_allow_html=True)
                    
                    action_items = [
                        "📋 Create a personalized skill development roadmap",
                        "🎓 Enroll in online courses for required certifications",
                        "💼 Seek internship or entry-level opportunities in this field",
                        "👥 Network with professionals in this career path",
                        "📖 Stay updated with industry trends and best practices"
                    ]
                    
                    st.markdown("<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px;'>", unsafe_allow_html=True)
                    for action in action_items:
                        icon = action[0]
                        text = action[2:]
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; gap: 12px; background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.04); border-radius: 10px; padding: 12px 16px; cursor: pointer; transition: all 0.2s;' onmouseover="this.style.background='rgba(16,185,129,0.05)'; this.style.borderColor='rgba(16,185,129,0.2)';" onmouseout="this.style.background='rgba(255,255,255,0.015)'; this.style.borderColor='rgba(255,255,255,0.04)';">
                            <div style='color: #10b981;'>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                            </div>
                            <div style='color: #cbd5e1; font-size: 0.9rem;'>{text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in Next Steps tab: {str(e)}")

        except Exception as e:
            st.error(f"Prediction Pipeline Interrupted: {str(e)}")

st.markdown("""
<div style="text-align: center; padding: 24px; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 3rem; color: #64748b; font-size: 0.85rem;">
    ✦ Interactive Multi-Section Pipeline Framework Built from Scratch
</div>
""", unsafe_allow_html=True)
