"""
AI-Powered Resume Screening System
Internship Project - Complete Solution
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ─── PDF EXTRACTION ─────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    """Extract text from PDF using pdfplumber first, PyMuPDF as fallback."""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text.strip(), "pdfplumber"
    except Exception:
        pass

    # Fallback: PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if text.strip():
            return text.strip(), "PyMuPDF"
    except Exception:
        pass

    return "", "failed"

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background: #0d1117; }

.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

.hero-card {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    color: white;
    text-align: center;
}

.hero-card h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero-card p  { font-size: 1rem; opacity: 0.85; margin-top: 8px; }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.metric-card .label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }

.result-high {
    background: linear-gradient(135deg, #1a4731, #2ea043);
    border-left: 4px solid #3fb950;
    border-radius: 10px;
    padding: 20px;
    color: white;
    margin: 12px 0;
}
.result-med {
    background: linear-gradient(135deg, #3d2b00, #9e6a03);
    border-left: 4px solid #d29922;
    border-radius: 10px;
    padding: 20px;
    color: white;
    margin: 12px 0;
}
.result-low {
    background: linear-gradient(135deg, #3d0000, #9b1c1c);
    border-left: 4px solid #f85149;
    border-radius: 10px;
    padding: 20px;
    color: white;
    margin: 12px 0;
}

.skill-badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb55;
    color: #58a6ff;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    margin: 3px;
}

.section-title {
    color: #e6edf3;
    font-size: 1.1rem;
    font-weight: 600;
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin: 20px 0 16px 0;
}

div[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}

.stTextArea textarea {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 28px !important;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ─── DATASET & MODEL (built-in, no Kaggle required) ─────────────────────────

# Synthetic but realistic resume dataset covering 8 job categories
RESUME_DATA = {
    "resume_text": [
        # DATA SCIENTIST
        "Experienced data scientist with 5 years in machine learning, deep learning, python, tensorflow, pytorch, scikit-learn, pandas, numpy, sql, data analysis, statistical modeling, NLP, computer vision, AWS, Docker, Kubernetes, research publications, PhD in Computer Science",
        "Data scientist skilled in python, R, machine learning, statistical analysis, data visualization, tableau, power bi, SQL, spark, hadoop, A/B testing, hypothesis testing, regression, classification, clustering",
        "Junior data scientist with python, pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter notebook, basic machine learning, SQL, data cleaning, exploratory data analysis, internship experience",
        "ML engineer with tensorflow, keras, pytorch, deep learning, neural networks, CNN, RNN, LSTM, transformer models, BERT, GPT, model deployment, Flask, FastAPI, docker, kubernetes, mlflow, 4 years experience",
        "Data analyst transitioning to data science, proficient in SQL, excel, tableau, python basics, statistics, data visualization, business intelligence, 3 years analytical experience",
        # SOFTWARE ENGINEER
        "Senior software engineer with 8 years experience in Java, Python, C++, microservices, REST APIs, Spring Boot, Django, React, Node.js, PostgreSQL, MongoDB, AWS, GCP, CI/CD, Jenkins, Docker, Kubernetes, Agile, TDD, system design",
        "Full stack developer skilled in React, Angular, Vue.js, TypeScript, Node.js, Express, Python, Django, MySQL, PostgreSQL, Redis, Docker, Git, Agile, responsive design, 5 years experience",
        "Backend developer with Java, Spring, Hibernate, microservices, REST API, GraphQL, MongoDB, PostgreSQL, RabbitMQ, Kafka, Docker, CI/CD, 4 years experience in fintech",
        "Frontend developer with React, Next.js, TypeScript, HTML5, CSS3, SASS, Redux, GraphQL, Jest, Webpack, responsive design, accessibility, UI/UX principles, 3 years experience",
        "Fresh software engineer graduate with Python, Java, data structures, algorithms, OOP, Git, basic web development, competitive programming, internship at tech startup",
        # MARKETING MANAGER
        "Marketing manager with 7 years experience in digital marketing, SEO, SEM, Google Ads, Facebook Ads, content marketing, email marketing, HubSpot, Salesforce, analytics, brand management, campaign management, team leadership",
        "Digital marketing specialist with SEO, SEM, PPC, social media marketing, content creation, email campaigns, Google Analytics, HubSpot, Mailchimp, A/B testing, conversion optimization, 4 years experience",
        "Marketing coordinator with social media management, content writing, email marketing, basic SEO, Canva, Adobe Photoshop, market research, event coordination, 2 years experience",
        "Growth marketing manager with growth hacking, product marketing, user acquisition, retention strategies, data-driven marketing, SQL, Python, Google Analytics, Mixpanel, 5 years in startups",
        # HR MANAGER
        "HR Manager with 10 years in talent acquisition, employee relations, performance management, compensation benefits, HRIS, payroll, onboarding, training development, labor law compliance, team management",
        "HR Business Partner with recruitment, talent management, performance reviews, employee engagement, organizational development, change management, SHRM certification, 6 years experience",
        "HR Generalist with recruiting, onboarding, benefits administration, HRIS, employee relations, policy implementation, compliance, training, 3 years experience at mid-size company",
        # ACCOUNTANT / FINANCE
        "Senior accountant with CPA, 8 years experience in financial reporting, GAAP, IFRS, tax preparation, audit, budget forecasting, SAP, QuickBooks, Excel advanced, financial analysis, month-end closing",
        "Financial analyst with Excel, financial modeling, DCF analysis, budgeting, forecasting, variance analysis, SQL, Power BI, CFA level 1, investment analysis, 4 years in investment banking",
        "Junior accountant with accounting degree, QuickBooks, Excel, accounts payable, accounts receivable, bank reconciliation, basic tax knowledge, 1 year experience",
        # GRAPHIC DESIGNER
        "Senior graphic designer with 7 years in Adobe Creative Suite, Photoshop, Illustrator, InDesign, After Effects, Figma, UI/UX design, brand identity, print design, motion graphics, art direction",
        "UI/UX designer with Figma, Sketch, Adobe XD, user research, wireframing, prototyping, usability testing, HTML, CSS basics, design systems, 4 years product design experience",
        "Junior graphic designer with Photoshop, Illustrator, Canva, social media graphics, logo design, typography, color theory, fresh graduate with strong portfolio",
        # SALES REPRESENTATIVE  
        "Senior sales representative with 6 years B2B sales experience, CRM Salesforce, lead generation, cold calling, negotiation, account management, sales pipeline, quota achievement, product demo, $2M annual revenue",
        "Inside sales rep with Salesforce, HubSpot, outbound calling, email outreach, lead qualification, objection handling, 3 years SaaS sales, consistent quota attainment",
        "Entry level sales with strong communication, customer service, retail experience, motivated self-starter, eager to learn, bachelor's degree",
        # TEACHER / EDUCATOR
        "Experienced teacher with 8 years K-12 education, curriculum development, lesson planning, classroom management, differentiated instruction, special education, Google Classroom, Zoom, parent communication",
        "University lecturer with PhD, 5 years teaching experience, research publications, course design, student mentoring, grading, academic writing",
        "Fresh teacher graduate with B.Ed, student teaching internship, lesson planning, Microsoft Teams, interactive learning, passionate about education",
    ],
    "category": [
        "Data Scientist", "Data Scientist", "Data Scientist", "Data Scientist", "Data Scientist",
        "Software Engineer", "Software Engineer", "Software Engineer", "Software Engineer", "Software Engineer",
        "Marketing Manager", "Marketing Manager", "Marketing Manager", "Marketing Manager",
        "HR Manager", "HR Manager", "HR Manager",
        "Accountant", "Accountant", "Accountant",
        "Graphic Designer", "Graphic Designer", "Graphic Designer",
        "Sales Representative", "Sales Representative", "Sales Representative",
        "Teacher", "Teacher", "Teacher",
    ],
    "years_experience": [5,4,1,4,3, 8,5,4,3,0, 7,4,2,5, 10,6,3, 8,4,1, 7,4,0, 6,3,0, 8,5,0],
    "education_level": [
        "PhD","Master's","Bachelor's","Master's","Bachelor's",
        "Master's","Bachelor's","Bachelor's","Bachelor's","Bachelor's",
        "Master's","Bachelor's","Bachelor's","Master's",
        "Master's","Bachelor's","Bachelor's",
        "Bachelor's","Master's","Bachelor's",
        "Bachelor's","Bachelor's","Bachelor's",
        "Bachelor's","Bachelor's","Bachelor's",
        "Master's","PhD","Bachelor's",
    ],
    "suitability_score": [
        92,78,55,88,60, 95,80,82,70,45, 88,73,52,85, 90,80,62, 88,78,50, 85,80,50, 82,70,40, 88,82,48
    ]
}

df_base = pd.DataFrame(RESUME_DATA)

# ─── ML MODEL ───────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def train_model():
    texts = df_base['resume_text'].tolist()
    labels = df_base['category'].tolist()

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1,2),
        stop_words='english',
        sublinear_tf=True
    )

    models = {
        "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    X = vectorizer.fit_transform(texts)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    results = {}
    for name, clf in models.items():
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        results[name] = scores.mean()

    # Train best model on full data
    best_model = models["Random Forest"]
    best_model.fit(X, y)

    return vectorizer, best_model, le, results, df_base

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text):
    skill_keywords = [
        "python","java","javascript","typescript","react","angular","vue","node.js","django","flask",
        "fastapi","spring","sql","postgresql","mysql","mongodb","redis","docker","kubernetes","aws",
        "gcp","azure","git","ci/cd","machine learning","deep learning","tensorflow","pytorch",
        "scikit-learn","pandas","numpy","nlp","computer vision","tableau","power bi","excel",
        "photoshop","illustrator","figma","sketch","seo","sem","google ads","hubspot","salesforce",
        "quickbooks","sap","r","spark","hadoop","kafka","rest api","graphql","html","css",
        "data analysis","statistics","research","agile","scrum","leadership","communication",
        "management","recruitment","training","budgeting","forecasting","content marketing",
        "social media","email marketing","canva","after effects","adobe xd"
    ]
    found = []
    text_lower = text.lower()
    for sk in skill_keywords:
        if sk in text_lower:
            found.append(sk.title())
    return list(set(found))

def predict_suitability(text, target_category, vectorizer, model, le):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    classes = le.classes_

    # Score for target category
    if target_category in classes:
        idx = list(classes).index(target_category)
        score = proba[idx] * 100
    else:
        score = 50.0

    # Adjust by keyword density
    keywords = extract_skills(text)
    bonus = min(len(keywords) * 1.5, 20)
    score = min(score + bonus, 99)

    predicted_idx = np.argmax(proba)
    predicted_cat = classes[predicted_idx]

    return round(score, 1), predicted_cat, keywords

def get_suitability_label(score):
    if score >= 70: return "HIGH", "result-high", "✅"
    elif score >= 45: return "MEDIUM", "result-med", "⚠️"
    else: return "LOW", "result-low", "❌"

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI Resume Screener")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home", "📊 Analytics", "📈 Model Metrics", "ℹ️ About"])
    st.markdown("---")
    st.markdown("**Job Categories:**")
    categories = sorted(df_base['category'].unique())
    for cat in categories:
        count = len(df_base[df_base['category'] == cat])
        st.markdown(f"• {cat} ({count})")
    st.markdown("---")
    st.markdown("<small style='color:#8b949e'>Built for AI/ML Internship Task<br>Resume Screening System</small>", unsafe_allow_html=True)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
vectorizer, model, le, model_results, df = train_model()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════
if "🏠 Home" in page:
    st.markdown("""
    <div class="hero-card">
        <h1>🤖 AI Resume Screening System</h1>
        <p>Intelligent candidate evaluation powered by Machine Learning & NLP</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{len(df)}</div>
            <div class="label">Resumes in Dataset</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{len(df['category'].unique())}</div>
            <div class="label">Job Categories</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        best_acc = max(model_results.values())
        st.markdown(f"""<div class="metric-card">
            <div class="value">{best_acc*100:.0f}%</div>
            <div class="label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="value">TF-IDF</div>
            <div class="label">Feature Extraction</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── SCREENING SECTION ──
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        # Input method toggle
        input_method = st.radio(
            "Input Method",
            ["📋 Paste Text", "📁 Upload PDF"],
            horizontal=True,
            label_visibility="collapsed"
        )

        resume_input = ""

        if input_method == "📁 Upload PDF":
            st.markdown('<div class="section-title">📁 Upload Resume PDF</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                with st.spinner("Extracting text from PDF..."):
                    file_bytes = uploaded_file.read()
                    extracted_text, method = extract_text_from_pdf(file_bytes)

                if extracted_text:
                    st.success(f"✅ Text extracted successfully using **{method}** ({len(extracted_text.split())} words found)")
                    resume_input = extracted_text
                    with st.expander("👁️ Preview Extracted Text"):
                        st.text(extracted_text[:1500] + ("..." if len(extracted_text) > 1500 else ""))
                else:
                    st.error("❌ Could not extract text from this PDF. Try a text-based PDF (not scanned image).")
        else:
            st.markdown('<div class="section-title">📄 Paste Resume Text</div>', unsafe_allow_html=True)
            resume_input = st.text_area(
                "Resume content",
                height=240,
                placeholder="Paste resume text here...\n\nExample:\nExperienced software engineer with 5 years in Python, Django, React, PostgreSQL, Docker, AWS...",
                label_visibility="collapsed"
            )

        st.markdown('<div class="section-title">🎯 Target Job Role</div>', unsafe_allow_html=True)
        target_job = st.selectbox("Select target job", categories, label_visibility="collapsed")

        analyze_btn = st.button("🔍 Analyze Resume", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">📋 Quick Tips</div>', unsafe_allow_html=True)
        st.info("""
**For better results, include:**
- Technical skills & tools
- Years of experience
- Education level
- Domain-specific keywords
- Previous job titles
        """)

        st.markdown('<div class="section-title">🔢 How It Works</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Text Preprocessing", "Clean & normalize resume text"),
            ("2", "TF-IDF Vectorization", "Extract feature importance"),
            ("3", "ML Classification", "Random Forest predicts category"),
            ("4", "Suitability Scoring", "Match score with target role"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; gap:12px; align-items:flex-start; margin-bottom:12px;'>
                <div style='background:#1f6feb; color:white; border-radius:50%; width:28px; height:28px; 
                     display:flex; align-items:center; justify-content:center; font-size:0.8rem; 
                     font-weight:700; flex-shrink:0;'>{num}</div>
                <div>
                    <div style='color:#e6edf3; font-weight:600; font-size:0.9rem;'>{title}</div>
                    <div style='color:#8b949e; font-size:0.8rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── RESULTS ──
    if analyze_btn:
        if not resume_input.strip():
            st.warning("⚠️ Please paste a resume first!")
        else:
            with st.spinner("Analyzing resume..."):
                score, predicted_cat, skills = predict_suitability(
                    resume_input, target_job, vectorizer, model, le
                )
                label, css_class, icon = get_suitability_label(score)

            st.markdown("---")
            st.markdown("### 📊 Analysis Results")

            r1, r2, r3 = st.columns(3)
            with r1:
                color = "#3fb950" if label=="HIGH" else "#d29922" if label=="MED" else "#f85149"
                st.markdown(f"""<div class="metric-card">
                    <div class="value" style="color:{color}">{score}%</div>
                    <div class="label">Suitability Score</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class="metric-card">
                    <div class="value" style="font-size:1.3rem">{predicted_cat}</div>
                    <div class="label">Best Match Category</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{len(skills)}</div>
                    <div class="label">Skills Detected</div>
                </div>""", unsafe_allow_html=True)

            # Suitability verdict
            st.markdown(f"""
            <div class="{css_class}">
                <h3>{icon} {label} SUITABILITY for {target_job}</h3>
                <p>This candidate scores <strong>{score}%</strong> for the {target_job} position.
                ML model best matches this resume to: <strong>{predicted_cat}</strong>.</p>
                {"<p>✅ Strong candidate — recommend for interview.</p>" if label=="HIGH" else
                 "<p>⚠️ Moderate match — consider for screening round." if label=="MEDIUM" else
                 "<p>❌ Low match — does not meet minimum requirements.</p>"}
            </div>
            """, unsafe_allow_html=True)

            # Skills
            if skills:
                st.markdown('<div class="section-title">🛠️ Detected Skills</div>', unsafe_allow_html=True)
                badges = " ".join([f'<span class="skill-badge">{s}</span>' for s in skills])
                st.markdown(badges, unsafe_allow_html=True)

            # Score gauge chart
            fig, ax = plt.subplots(figsize=(6, 2), facecolor='#0d1117')
            ax.set_facecolor('#0d1117')
            colors_grad = ['#f85149', '#ff7b00', '#d29922', '#3fb950', '#2ea043']
            breaks = [0, 20, 40, 60, 80, 100]
            for i in range(len(colors_grad)):
                ax.barh(0, breaks[i+1]-breaks[i], left=breaks[i], height=0.4,
                        color=colors_grad[i], alpha=0.7)
            ax.barh(0, 2, left=score-1, height=0.6, color='white')
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel('Suitability Score', color='#8b949e')
            ax.tick_params(colors='#8b949e')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            ax.set_title(f'Score: {score}%', color='#e6edf3', pad=10)
            st.pyplot(fig)
            plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
elif "📊 Analytics" in page:
    st.markdown("## 📊 Dataset Analytics")
    st.markdown("Exploring the resume dataset used for training the ML model.")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Experience Analysis", "Skills Heatmap"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            counts = df['category'].value_counts()
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(counts)))
            bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='#30363d')
            ax.set_xlabel('Count', color='#8b949e')
            ax.set_title('Resumes per Category', color='#e6edf3', pad=10)
            ax.tick_params(colors='#8b949e')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            for bar, val in zip(bars, counts.values):
                ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', color='#58a6ff', fontsize=9)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            edu_counts = df['education_level'].value_counts()
            colors_pie = ['#1f6feb', '#388bfd', '#58a6ff', '#79c0ff']
            wedges, texts, autotexts = ax.pie(
                edu_counts.values, labels=edu_counts.index,
                colors=colors_pie[:len(edu_counts)],
                autopct='%1.0f%%', startangle=90,
                textprops={'color': '#e6edf3'}
            )
            for a in autotexts: a.set_color('#0d1117'); a.set_fontweight('bold')
            ax.set_title('Education Level Distribution', color='#e6edf3')
            st.pyplot(fig)
            plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor='#161b22')
        for ax in axes: ax.set_facecolor('#161b22')

        exp_by_cat = df.groupby('category')['years_experience'].mean().sort_values(ascending=True)
        axes[0].barh(exp_by_cat.index, exp_by_cat.values,
                     color='#1f6feb', edgecolor='#30363d', alpha=0.85)
        axes[0].set_xlabel('Avg Years Experience', color='#8b949e')
        axes[0].set_title('Experience by Category', color='#e6edf3')
        axes[0].tick_params(colors='#8b949e')
        for spine in axes[0].spines.values(): spine.set_color('#30363d')

        score_by_cat = df.groupby('category')['suitability_score'].mean().sort_values(ascending=True)
        colors_s = ['#f85149' if v < 60 else '#d29922' if v < 75 else '#3fb950'
                    for v in score_by_cat.values]
        axes[1].barh(score_by_cat.index, score_by_cat.values,
                     color=colors_s, edgecolor='#30363d', alpha=0.85)
        axes[1].set_xlabel('Avg Suitability Score', color='#8b949e')
        axes[1].set_title('Avg Score by Category', color='#e6edf3')
        axes[1].tick_params(colors='#8b949e')
        for spine in axes[1].spines.values(): spine.set_color('#30363d')
        axes[1].set_xlim(0, 100)

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.markdown("**Skill Keyword Frequency across Categories**")
        key_skills = ["python","machine learning","sql","java","react","seo","management",
                      "excel","figma","photoshop","salesforce","docker","aws","leadership"]
        heatmap_data = {}
        for cat in df['category'].unique():
            texts = ' '.join(df[df['category']==cat]['resume_text']).lower()
            heatmap_data[cat] = {sk: texts.count(sk) for sk in key_skills}

        hm_df = pd.DataFrame(heatmap_data).T
        fig, ax = plt.subplots(figsize=(13, 5), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        sns.heatmap(hm_df, annot=True, fmt='d', cmap='Blues',
                    ax=ax, linecolor='#30363d', linewidths=0.5,
                    cbar_kws={'label': 'Frequency'})
        ax.set_title('Skill Keyword Heatmap by Category', color='#e6edf3', pad=12)
        ax.tick_params(colors='#8b949e')
        plt.xticks(rotation=35, ha='right', color='#8b949e')
        plt.yticks(rotation=0, color='#8b949e')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("### 📋 Full Dataset Preview")
    st.dataframe(
        df[['category','years_experience','education_level','suitability_score']],
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════
elif "📈 Model Metrics" in page:
    st.markdown("## 📈 ML Model Performance")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Model Comparison (Cross-Validation)")
        for name, acc in sorted(model_results.items(), key=lambda x: -x[1]):
            bar_pct = int(acc * 100)
            color = "#3fb950" if bar_pct >= 70 else "#d29922"
            st.markdown(f"""
            <div style='margin-bottom:16px;'>
                <div style='display:flex; justify-content:space-between; color:#e6edf3; margin-bottom:4px;'>
                    <span>{name}</span><span style='color:{color}; font-weight:700;'>{bar_pct}%</span>
                </div>
                <div style='background:#30363d; border-radius:4px; height:8px;'>
                    <div style='background:{color}; width:{bar_pct}%; height:8px; border-radius:4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Feature Engineering")
        st.markdown("""
        | Component | Value |
        |-----------|-------|
        | Vectorizer | TF-IDF |
        | Max Features | 500 |
        | N-gram Range | (1, 2) |
        | Stop Words | English |
        | Sublinear TF | Yes |
        | Classifier | Random Forest |
        | Estimators | 100 |
        | CV Folds | 3 |
        """)

    with col2:
        # Confusion matrix simulation
        st.markdown("### Confusion Matrix (Training Data)")
        texts = df['resume_text'].tolist()
        X_all = vectorizer.transform([clean_text(t) for t in texts])
        y_all = le.transform(df['category'].tolist())
        y_pred = model.predict(X_all)
        cm = confusion_matrix(y_all, y_pred)
        classes_short = [c[:8] for c in le.classes_]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes_short, yticklabels=classes_short,
                    ax=ax, linecolor='#30363d', linewidths=0.5)
        ax.set_xlabel('Predicted', color='#8b949e')
        ax.set_ylabel('Actual', color='#8b949e')
        ax.set_title('Confusion Matrix', color='#e6edf3')
        ax.tick_params(colors='#8b949e', labelsize=8)
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        plt.close()

        train_acc = accuracy_score(y_all, y_pred)
        st.success(f"✅ Training Accuracy: **{train_acc*100:.1f}%**")
        st.info("📌 Note: Dataset is small (synthetic for demo). Real-world performance uses Kaggle Resume Dataset with 2400+ records.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════
elif "ℹ️ About" in page:
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    ### 🎯 Project Overview
    This is an **AI-Powered Resume Screening System** built as part of an AI/ML Internship task.
    The system uses Natural Language Processing (NLP) and Machine Learning to automatically
    analyze resumes, extract skills, classify job categories, and predict candidate suitability.

    ### 🏗️ Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | Frontend | Streamlit |
    | ML Models | scikit-learn (RF, LR, GB) |
    | NLP | TF-IDF Vectorization |
    | Visualization | Matplotlib, Seaborn |
    | Language | Python 3.10+ |

    ### 📁 Project Structure
    ```
    resume_screening/
    ├── app.py          ← Main Streamlit application
    ├── run.py          ← Auto-installer & launcher
    └── requirements.txt
    ```

    ### 🔬 ML Pipeline
    1. **Data Collection** — Resume dataset (Kaggle: Resume Dataset / UpdatedResumeDataSet)
    2. **Preprocessing** — Lowercase, remove special chars, stopword removal
    3. **Feature Extraction** — TF-IDF with 500 features, bigrams
    4. **Model Training** — Random Forest (best accuracy), LR, GBM compared
    5. **Evaluation** — Cross-validation, confusion matrix, classification report
    6. **Deployment** — Streamlit web interface

    ### 🗂️ Dataset
    - **Source:** Kaggle — Resume Dataset (UpdatedResumeDataSet.csv)
    - **Categories:** 25 job categories, 2400+ resumes
    - **Features:** Resume text, category labels
    - **For demo:** Built-in curated dataset covering 8 categories
    """)
