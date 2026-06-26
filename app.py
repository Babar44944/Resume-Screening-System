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

# ─── DATASET ────────────────────────────────────────────────────────────────

RESUME_DATA = {
    "resume_text": [
        # ── DATA SCIENTIST ──
        "Experienced data scientist with 5 years in machine learning, deep learning, python, tensorflow, pytorch, scikit-learn, pandas, numpy, sql, data analysis, statistical modeling, NLP, computer vision, AWS, Docker, Kubernetes, research publications, PhD in Computer Science",
        "Data scientist skilled in python, R, machine learning, statistical analysis, data visualization, tableau, power bi, SQL, spark, hadoop, A/B testing, hypothesis testing, regression, classification, clustering",
        "Junior data scientist with python, pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter notebook, basic machine learning, SQL, data cleaning, exploratory data analysis, internship experience",
        "ML engineer with tensorflow, keras, pytorch, deep learning, neural networks, CNN, RNN, LSTM, transformer models, BERT, GPT, model deployment, Flask, FastAPI, docker, kubernetes, mlflow, 4 years experience",
        "Data analyst transitioning to data science, proficient in SQL, excel, tableau, python basics, statistics, data visualization, business intelligence, 3 years analytical experience",

        # ── SOFTWARE ENGINEER ──
        "Senior software engineer with 8 years experience in Java, Python, C++, microservices, REST APIs, Spring Boot, Django, React, Node.js, PostgreSQL, MongoDB, AWS, GCP, CI/CD, Jenkins, Docker, Kubernetes, Agile, TDD, system design",
        "Full stack developer skilled in React, Angular, Vue.js, TypeScript, Node.js, Express, Python, Django, MySQL, PostgreSQL, Redis, Docker, Git, Agile, responsive design, 5 years experience",
        "Backend developer with Java, Spring, Hibernate, microservices, REST API, GraphQL, MongoDB, PostgreSQL, RabbitMQ, Kafka, Docker, CI/CD, 4 years experience in fintech",
        "Frontend developer with React, Next.js, TypeScript, HTML5, CSS3, SASS, Redux, GraphQL, Jest, Webpack, responsive design, accessibility, UI/UX principles, 3 years experience",
        "Fresh software engineer graduate with Python, Java, data structures, algorithms, OOP, Git, basic web development, competitive programming, internship at tech startup",

        # ── AI ENGINEER ──
        "AI Engineer with 5 years experience building production AI systems, LLM fine-tuning, RAG pipelines, langchain, openai GPT-4, prompt engineering, vector databases, pinecone, chromadb, huggingface transformers, model serving, fastapi, python, mlops, azure ai, aws sagemaker",
        "AI Engineer specializing in generative AI, stable diffusion, midjourney APIs, text-to-image, DALL-E integration, LLM orchestration, langchain, llamaindex, embedding models, semantic search, python, 4 years experience in AI product development",
        "Junior AI Engineer with python, huggingface, transformers, fine-tuning BERT GPT, prompt engineering, openai API, basic RAG implementation, vector search, fastapi, 1 year experience building chatbots and AI tools",
        "Senior AI Engineer with expertise in multi-agent systems, autogen, crewai, tool use, function calling, AI safety, RLHF, reinforcement learning from human feedback, model evaluation, benchmark testing, 6 years in AI research and engineering",

        # ── COMPUTER VISION ENGINEER ──
        "Computer Vision Engineer with 5 years in OpenCV, YOLO, YOLOv8, object detection, image segmentation, instance segmentation, pose estimation, mediapipe, deep learning, CNN, pytorch, tensorflow, real-time video processing, edge deployment, TensorRT, ONNX, C++, python",
        "Computer Vision Engineer skilled in image classification, semantic segmentation, U-Net, Mask RCNN, Detectron2, OpenCV, PIL, albumentations, data augmentation, pytorch, custom dataset training, annotation tools, labelImg, Roboflow, 4 years experience in autonomous systems",
        "Junior Computer Vision Engineer with python, OpenCV, YOLO object detection, mediapipe hand tracking, face recognition, image preprocessing, pytorch, basic model training, webcam applications, 1 year experience",
        "Senior Computer Vision Engineer with expertise in 3D computer vision, depth estimation, stereo vision, point cloud, LiDAR, SLAM, autonomous driving, camera calibration, homography, optical flow, tracking algorithms, DeepSORT, ByteTrack, 7 years experience",

        # ── NLP ENGINEER ──
        "NLP Engineer with 5 years in natural language processing, text classification, named entity recognition NER, sentiment analysis, BERT, GPT, T5, transformers, huggingface, spacy, nltk, text summarization, machine translation, question answering, fine-tuning, python",
        "NLP Engineer skilled in information extraction, relation extraction, coreference resolution, dependency parsing, POS tagging, word embeddings, word2vec, GloVe, fasttext, topic modeling, LDA, text generation, seq2seq, attention mechanism, 4 years experience",
        "Junior NLP Engineer with python, nltk, spacy, basic text preprocessing, sentiment analysis, text classification using BERT, huggingface tutorials, chatbot development, regex, 1 year experience",

        # ── DEVOPS ENGINEER ──
        "Senior DevOps Engineer with 7 years in CI/CD pipelines, Jenkins, GitHub Actions, GitLab CI, Docker, Kubernetes, Helm, Terraform, Ansible, AWS, GCP, Azure, monitoring Prometheus Grafana, ELK stack, infrastructure as code, SRE practices, incident management",
        "DevOps Engineer skilled in containerization, Docker, Kubernetes, service mesh, Istio, cloud infrastructure, AWS EC2 S3 RDS Lambda, Terraform, Ansible, Python automation, bash scripting, CI/CD, Jenkins, CircleCI, 4 years experience",
        "Junior DevOps Engineer with Docker, basic Kubernetes, GitHub Actions, CI/CD concepts, Linux administration, bash scripting, AWS basics, monitoring tools, 1 year experience fresh graduate",

        # ── CYBERSECURITY ANALYST ──
        "Senior Cybersecurity Analyst with 8 years in penetration testing, vulnerability assessment, SIEM, SOC, incident response, malware analysis, threat hunting, OSCP certified, CEH, network security, firewall, IDS IPS, OWASP, CTF competitions, Python, Kali Linux",
        "Cybersecurity Analyst skilled in network security, vulnerability scanning, Nessus, Burp Suite, Metasploit, SIEM Splunk, log analysis, phishing analysis, endpoint security, EDR, threat intelligence, security audits, ISO 27001, 4 years experience",
        "Junior Cybersecurity Analyst with CompTIA Security+, basic network security, Wireshark, Nmap, vulnerability concepts, Linux fundamentals, SOC monitoring, incident ticketing, 1 year experience",

        # ── MOBILE DEVELOPER ──
        "Senior Mobile Developer with 6 years in iOS Swift SwiftUI, Android Kotlin Jetpack Compose, React Native, Flutter, cross-platform development, REST APIs, Firebase, push notifications, App Store deployment, Play Store, mobile UI UX, CoreML, TensorFlow Lite",
        "Mobile Developer skilled in Flutter Dart, React Native, iOS Android both platforms, state management, Redux BLoC, REST API integration, SQLite, Firebase Realtime Database, Firestore, authentication, 4 years experience",
        "Junior Mobile Developer with Flutter basics, Dart, simple Android Kotlin apps, React Native beginner, Firebase integration, mobile UI design, 1 year experience fresh graduate",

        # ── BUSINESS ANALYST ──
        "Senior Business Analyst with 8 years in requirements gathering, BRD FRD documentation, stakeholder management, process mapping, BPMN, Agile Scrum, JIRA, Confluence, SQL, Power BI, Tableau, gap analysis, user stories, UAT testing, Salesforce, CBAP certified",
        "Business Analyst skilled in data analysis, Excel advanced, SQL, Power BI, requirement elicitation, wireframing Balsamiq, workflow diagrams, stakeholder interviews, change management, process improvement, 4 years experience in banking sector",
        "Junior Business Analyst with SQL basics, Excel, data analysis concepts, Agile awareness, requirement documentation, business process understanding, fresh graduate with internship in consulting",

        # ── PRODUCT MANAGER ──
        "Senior Product Manager with 7 years in product strategy, roadmapping, user research, competitive analysis, A/B testing, analytics Google Analytics Mixpanel, stakeholder management, cross-functional teams, go-to-market strategy, agile, OKRs, SaaS products, fintech experience",
        "Product Manager skilled in product lifecycle management, user story writing, JIRA, product metrics, funnel analysis, customer interviews, MVP definition, prioritization frameworks, wireframing Figma, data-driven decisions, 4 years in e-commerce",
        "Junior Product Manager with product thinking, user research basics, Agile Scrum, JIRA familiarity, customer empathy, analytical mindset, fresh graduate with product internship",

        # ── MARKETING MANAGER ──
        "Marketing manager with 7 years experience in digital marketing, SEO, SEM, Google Ads, Facebook Ads, content marketing, email marketing, HubSpot, Salesforce, analytics, brand management, campaign management, team leadership",
        "Digital marketing specialist with SEO, SEM, PPC, social media marketing, content creation, email campaigns, Google Analytics, HubSpot, Mailchimp, A/B testing, conversion optimization, 4 years experience",
        "Marketing coordinator with social media management, content writing, email marketing, basic SEO, Canva, Adobe Photoshop, market research, event coordination, 2 years experience",

        # ── HR MANAGER ──
        "HR Manager with 10 years in talent acquisition, employee relations, performance management, compensation benefits, HRIS, payroll, onboarding, training development, labor law compliance, team management",
        "HR Business Partner with recruitment, talent management, performance reviews, employee engagement, organizational development, change management, SHRM certification, 6 years experience",
        "HR Generalist with recruiting, onboarding, benefits administration, HRIS, employee relations, policy implementation, compliance, training, 3 years experience at mid-size company",

        # ── ACCOUNTANT ──
        "Senior accountant with CPA, 8 years experience in financial reporting, GAAP, IFRS, tax preparation, audit, budget forecasting, SAP, QuickBooks, Excel advanced, financial analysis, month-end closing",
        "Financial analyst with Excel, financial modeling, DCF analysis, budgeting, forecasting, variance analysis, SQL, Power BI, CFA level 1, investment analysis, 4 years in investment banking",
        "Junior accountant with accounting degree, QuickBooks, Excel, accounts payable, accounts receivable, bank reconciliation, basic tax knowledge, 1 year experience",

        # ── GRAPHIC DESIGNER ──
        "Senior graphic designer with 7 years in Adobe Creative Suite, Photoshop, Illustrator, InDesign, After Effects, Figma, UI/UX design, brand identity, print design, motion graphics, art direction",
        "UI/UX designer with Figma, Sketch, Adobe XD, user research, wireframing, prototyping, usability testing, HTML, CSS basics, design systems, 4 years product design experience",
        "Junior graphic designer with Photoshop, Illustrator, Canva, social media graphics, logo design, typography, color theory, fresh graduate with strong portfolio",

        # ── SALES REPRESENTATIVE ──
        "Senior sales representative with 6 years B2B sales experience, CRM Salesforce, lead generation, cold calling, negotiation, account management, sales pipeline, quota achievement, product demo, $2M annual revenue",
        "Inside sales rep with Salesforce, HubSpot, outbound calling, email outreach, lead qualification, objection handling, 3 years SaaS sales, consistent quota attainment",
        "Entry level sales with strong communication, customer service, retail experience, motivated self-starter, eager to learn, bachelor's degree",

        # ── TEACHER ──
        "Experienced teacher with 8 years K-12 education, curriculum development, lesson planning, classroom management, differentiated instruction, special education, Google Classroom, Zoom, parent communication",
        "University lecturer with PhD, 5 years teaching experience, research publications, course design, student mentoring, grading, academic writing",
        "Fresh teacher graduate with B.Ed, student teaching internship, lesson planning, Microsoft Teams, interactive learning, passionate about education",
    ],
    "category": [
        # Data Scientist x5
        "Data Scientist","Data Scientist","Data Scientist","Data Scientist","Data Scientist",
        # Software Engineer x5
        "Software Engineer","Software Engineer","Software Engineer","Software Engineer","Software Engineer",
        # AI Engineer x4
        "AI Engineer","AI Engineer","AI Engineer","AI Engineer",
        # Computer Vision Engineer x4
        "Computer Vision Engineer","Computer Vision Engineer","Computer Vision Engineer","Computer Vision Engineer",
        # NLP Engineer x3
        "NLP Engineer","NLP Engineer","NLP Engineer",
        # DevOps Engineer x3
        "DevOps Engineer","DevOps Engineer","DevOps Engineer",
        # Cybersecurity Analyst x3
        "Cybersecurity Analyst","Cybersecurity Analyst","Cybersecurity Analyst",
        # Mobile Developer x3
        "Mobile Developer","Mobile Developer","Mobile Developer",
        # Business Analyst x3
        "Business Analyst","Business Analyst","Business Analyst",
        # Product Manager x3
        "Product Manager","Product Manager","Product Manager",
        # Marketing Manager x3
        "Marketing Manager","Marketing Manager","Marketing Manager",
        # HR Manager x3
        "HR Manager","HR Manager","HR Manager",
        # Accountant x3
        "Accountant","Accountant","Accountant",
        # Graphic Designer x3
        "Graphic Designer","Graphic Designer","Graphic Designer",
        # Sales Representative x3
        "Sales Representative","Sales Representative","Sales Representative",
        # Teacher x3
        "Teacher","Teacher","Teacher",
    ],
    "years_experience": [
        5,4,1,4,3,        # Data Scientist
        8,5,4,3,0,        # Software Engineer
        5,4,1,6,          # AI Engineer
        5,4,1,7,          # Computer Vision Engineer
        5,4,1,            # NLP Engineer
        7,4,1,            # DevOps Engineer
        8,4,1,            # Cybersecurity Analyst
        6,4,1,            # Mobile Developer
        8,4,1,            # Business Analyst
        7,4,1,            # Product Manager
        7,4,2,            # Marketing Manager
        10,6,3,           # HR Manager
        8,4,1,            # Accountant
        7,4,0,            # Graphic Designer
        6,3,0,            # Sales Representative
        8,5,0,            # Teacher
    ],
    "education_level": [
        "PhD","Master's","Bachelor's","Master's","Bachelor's",           # Data Scientist
        "Master's","Bachelor's","Bachelor's","Bachelor's","Bachelor's",  # Software Engineer
        "Master's","Master's","Bachelor's","PhD",                       # AI Engineer
        "Master's","Master's","Bachelor's","PhD",                       # Computer Vision Engineer
        "Master's","Master's","Bachelor's",                             # NLP Engineer
        "Bachelor's","Bachelor's","Bachelor's",                         # DevOps Engineer
        "Master's","Bachelor's","Bachelor's",                           # Cybersecurity Analyst
        "Bachelor's","Bachelor's","Bachelor's",                         # Mobile Developer
        "Master's","Bachelor's","Bachelor's",                           # Business Analyst
        "Master's","Bachelor's","Bachelor's",                           # Product Manager
        "Master's","Bachelor's","Bachelor's",                           # Marketing Manager
        "Master's","Bachelor's","Bachelor's",                           # HR Manager
        "Bachelor's","Master's","Bachelor's",                           # Accountant
        "Bachelor's","Bachelor's","Bachelor's",                         # Graphic Designer
        "Bachelor's","Bachelor's","Bachelor's",                         # Sales Representative
        "Master's","PhD","Bachelor's",                                  # Teacher
    ],
    "suitability_score": [
        92,78,55,88,60,   # Data Scientist
        95,80,82,70,45,   # Software Engineer
        91,85,58,94,      # AI Engineer
        90,82,56,95,      # Computer Vision Engineer
        88,80,54,         # NLP Engineer
        90,78,50,         # DevOps Engineer
        92,76,52,         # Cybersecurity Analyst
        85,78,48,         # Mobile Developer
        88,74,50,         # Business Analyst
        90,76,50,         # Product Manager
        88,73,52,         # Marketing Manager
        90,80,62,         # HR Manager
        88,78,50,         # Accountant
        85,80,50,         # Graphic Designer
        82,70,40,         # Sales Representative
        88,82,48,         # Teacher
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
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    X = vectorizer.fit_transform(texts)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    results = {}
    for name, clf in models.items():
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        results[name] = scores.mean()

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
        # General Programming
        "python","java","javascript","typescript","c++","c#","go","rust","scala","kotlin","swift","dart",
        # Web
        "react","angular","vue","node.js","django","flask","fastapi","spring","express","next.js",
        # Databases
        "sql","postgresql","mysql","mongodb","redis","elasticsearch","firebase","sqlite",
        # Cloud / DevOps
        "docker","kubernetes","aws","gcp","azure","terraform","ansible","jenkins","github actions",
        "ci/cd","helm","prometheus","grafana","elk stack",
        # AI / ML
        "machine learning","deep learning","tensorflow","pytorch","scikit-learn","keras","mlflow",
        "huggingface","transformers","bert","gpt","llm","langchain","llamaindex","rag","openai",
        "fine-tuning","vector database","pinecone","chromadb","embedding","prompt engineering",
        # Computer Vision
        "opencv","yolo","yolov8","object detection","image segmentation","mediapipe","cnn",
        "pose estimation","tensorrt","onnx","detectron2","labelimg","roboflow",
        # NLP
        "nlp","natural language processing","spacy","nltk","sentiment analysis","ner",
        "text classification","word2vec","bert","text summarization",
        # Data
        "pandas","numpy","matplotlib","seaborn","tableau","power bi","spark","hadoop","kafka",
        "a/b testing","statistics","data analysis","data visualization",
        # Mobile
        "flutter","react native","swift","swiftui","jetpack compose","firebase","android","ios",
        # Security
        "penetration testing","cybersecurity","burp suite","metasploit","kali linux","siem",
        "vulnerability assessment","oscp","wireshark","nmap","splunk",
        # Business / PM
        "jira","confluence","agile","scrum","product management","stakeholder management",
        "sql","power bi","tableau","business analysis",
        # Design
        "photoshop","illustrator","figma","sketch","adobe xd","canva","after effects","ui/ux",
        # Marketing / Sales
        "seo","sem","google ads","hubspot","salesforce","email marketing","content marketing",
        # HR / Finance
        "recruitment","payroll","quickbooks","sap","financial modeling","excel","cpa","cfa",
        # General
        "rest api","graphql","microservices","system design","git","agile","leadership","management",
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

    if target_category in classes:
        idx = list(classes).index(target_category)
        score = proba[idx] * 100
    else:
        score = 50.0

    keywords = extract_skills(text)
    bonus = min(len(keywords) * 1.5, 20)
    score = min(score + bonus, 99)

    predicted_idx = np.argmax(proba)
    predicted_cat = classes[predicted_idx]

    return round(score, 1), predicted_cat, keywords

def get_suitability_label(score):
    if score >= 70:   return "HIGH",   "result-high", "✅"
    elif score >= 45: return "MEDIUM", "result-med",  "⚠️"
    else:             return "LOW",    "result-low",  "❌"

# ─── SIDEBAR ────────────────────────────────────────────────────────────────

# Group categories for clean sidebar display
CATEGORY_GROUPS = {
    "🤖 AI / ML": ["AI Engineer","Computer Vision Engineer","Data Scientist","NLP Engineer"],
    "💻 Engineering": ["Software Engineer","DevOps Engineer","Mobile Developer","Cybersecurity Analyst"],
    "📊 Business": ["Business Analyst","Product Manager","Accountant","Marketing Manager"],
    "🎨 Creative & Other": ["Graphic Designer","HR Manager","Sales Representative","Teacher"],
}

with st.sidebar:
    st.markdown("### 🤖 AI Resume Screener")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Home", "📊 Analytics", "📈 Model Metrics", "ℹ️ About"])
    st.markdown("---")
    st.markdown("**Job Categories:**")
    for group, cats in CATEGORY_GROUPS.items():
        st.markdown(f"<small style='color:#58a6ff; font-weight:600;'>{group}</small>", unsafe_allow_html=True)
        for cat in cats:
            count = len(df_base[df_base['category'] == cat])
            st.markdown(f"<small style='color:#8b949e; margin-left:8px;'>• {cat} ({count})</small>", unsafe_allow_html=True)
    st.markdown("---")
    total_cats = len(df_base['category'].unique())
    st.markdown(f"<small style='color:#8b949e'>**{total_cats} categories** | {len(df_base)} resumes<br>Built for AI/ML Internship Task</small>", unsafe_allow_html=True)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
vectorizer, model, le, model_results, df = train_model()
categories = sorted(df['category'].unique())

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

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        input_method = st.radio(
            "Input Method",
            ["📋 Paste Text", "📁 Upload PDF"],
            horizontal=True,
            label_visibility="collapsed"
        )

        resume_input = ""

        if input_method == "📁 Upload PDF":
            st.markdown('<div class="section-title">📁 Upload Resume PDF</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
            if uploaded_file is not None:
                with st.spinner("Extracting text from PDF..."):
                    file_bytes = uploaded_file.read()
                    extracted_text, method = extract_text_from_pdf(file_bytes)
                if extracted_text:
                    st.success(f"✅ Text extracted via **{method}** ({len(extracted_text.split())} words)")
                    resume_input = extracted_text
                    with st.expander("👁️ Preview Extracted Text"):
                        st.text(extracted_text[:1500] + ("..." if len(extracted_text) > 1500 else ""))
                else:
                    st.error("❌ Could not extract text. Try a text-based PDF (not a scanned image).")
        else:
            st.markdown('<div class="section-title">📄 Paste Resume Text</div>', unsafe_allow_html=True)
            resume_input = st.text_area(
                "Resume content", height=240,
                placeholder="Paste resume text here...\n\nExample:\nAI Engineer with 3 years experience in LLM fine-tuning, LangChain, RAG pipelines, OpenAI API, HuggingFace...",
                label_visibility="collapsed"
            )

        st.markdown('<div class="section-title">🎯 Target Job Role</div>', unsafe_allow_html=True)

        # Group selectbox using categories dict
        group_labels = list(CATEGORY_GROUPS.keys())
        selected_group = st.selectbox("Category Group", group_labels, label_visibility="visible")
        group_cats = CATEGORY_GROUPS[selected_group]
        target_job = st.selectbox("Select Target Role", group_cats, label_visibility="visible")

        analyze_btn = st.button("🔍 Analyze Resume", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">📋 Quick Tips</div>', unsafe_allow_html=True)
        st.info("""
**For better results, include:**
- Technical skills & tools
- Years of experience
- Education level
- Domain-specific keywords
- Previous job titles & certifications
        """)

        st.markdown('<div class="section-title">🔢 How It Works</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Text Preprocessing",    "Clean & normalize resume text"),
            ("2", "TF-IDF Vectorization",  "Extract feature importance"),
            ("3", "ML Classification",     "Random Forest predicts category"),
            ("4", "Suitability Scoring",   "Match score with target role"),
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

        # Live category preview
        st.markdown('<div class="section-title">📂 All Categories</div>', unsafe_allow_html=True)
        for group, cats in CATEGORY_GROUPS.items():
            st.markdown(f"**{group}**")
            for c in cats:
                st.markdown(f"<small style='color:#8b949e; margin-left:8px;'>• {c}</small>", unsafe_allow_html=True)

    # ── RESULTS ──
    if analyze_btn:
        if not resume_input.strip():
            st.warning("⚠️ Please paste a resume or upload a PDF first!")
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
                color = "#3fb950" if label=="HIGH" else "#d29922" if label=="MEDIUM" else "#f85149"
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

            st.markdown(f"""
            <div class="{css_class}">
                <h3>{icon} {label} SUITABILITY for {target_job}</h3>
                <p>This candidate scores <strong>{score}%</strong> for the {target_job} position.
                ML model best matches this resume to: <strong>{predicted_cat}</strong>.</p>
                {"<p>✅ Strong candidate — recommend for interview.</p>" if label=="HIGH" else
                 "<p>⚠️ Moderate match — consider for screening round.</p>" if label=="MEDIUM" else
                 "<p>❌ Low match — does not meet minimum requirements.</p>"}
            </div>
            """, unsafe_allow_html=True)

            if skills:
                st.markdown('<div class="section-title">🛠️ Detected Skills</div>', unsafe_allow_html=True)
                badges = " ".join([f'<span class="skill-badge">{s}</span>' for s in skills])
                st.markdown(badges, unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(6, 2), facecolor='#0d1117')
            ax.set_facecolor('#0d1117')
            colors_grad = ['#f85149','#ff7b00','#d29922','#3fb950','#2ea043']
            breaks = [0,20,40,60,80,100]
            for i in range(len(colors_grad)):
                ax.barh(0, breaks[i+1]-breaks[i], left=breaks[i], height=0.4,
                        color=colors_grad[i], alpha=0.7)
            ax.barh(0, 2, left=score-1, height=0.6, color='white')
            ax.set_xlim(0,100)
            ax.set_yticks([])
            ax.set_xlabel('Suitability Score', color='#8b949e')
            ax.tick_params(colors='#8b949e')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
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
            fig, ax = plt.subplots(figsize=(7, 6), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            counts = df['category'].value_counts()
            colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(counts)))
            bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='#30363d')
            ax.set_xlabel('Count', color='#8b949e')
            ax.set_title('Resumes per Category', color='#e6edf3', pad=10)
            ax.tick_params(colors='#8b949e', labelsize=8)
            for spine in ax.spines.values(): spine.set_color('#30363d')
            for bar, val in zip(bars, counts.values):
                ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
                        str(val), va='center', color='#58a6ff', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='#161b22')
            ax.set_facecolor('#161b22')
            edu_counts = df['education_level'].value_counts()
            colors_pie = ['#1f6feb','#388bfd','#58a6ff','#79c0ff']
            wedges, texts, autotexts = ax.pie(
                edu_counts.values, labels=edu_counts.index,
                colors=colors_pie[:len(edu_counts)],
                autopct='%1.0f%%', startangle=90,
                textprops={'color':'#e6edf3'}
            )
            for a in autotexts: a.set_color('#0d1117'); a.set_fontweight('bold')
            ax.set_title('Education Level Distribution', color='#e6edf3')
            st.pyplot(fig)
            plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#161b22')
        for ax in axes: ax.set_facecolor('#161b22')

        exp_by_cat = df.groupby('category')['years_experience'].mean().sort_values()
        axes[0].barh(exp_by_cat.index, exp_by_cat.values, color='#1f6feb', edgecolor='#30363d', alpha=0.85)
        axes[0].set_xlabel('Avg Years Experience', color='#8b949e')
        axes[0].set_title('Experience by Category', color='#e6edf3')
        axes[0].tick_params(colors='#8b949e', labelsize=8)
        for spine in axes[0].spines.values(): spine.set_color('#30363d')

        score_by_cat = df.groupby('category')['suitability_score'].mean().sort_values()
        colors_s = ['#f85149' if v < 60 else '#d29922' if v < 75 else '#3fb950' for v in score_by_cat.values]
        axes[1].barh(score_by_cat.index, score_by_cat.values, color=colors_s, edgecolor='#30363d', alpha=0.85)
        axes[1].set_xlabel('Avg Suitability Score', color='#8b949e')
        axes[1].set_title('Avg Score by Category', color='#e6edf3')
        axes[1].tick_params(colors='#8b949e', labelsize=8)
        for spine in axes[1].spines.values(): spine.set_color('#30363d')
        axes[1].set_xlim(0,100)

        plt.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.markdown("**Skill Keyword Frequency across Categories**")
        key_skills = [
            "python","machine learning","sql","java","react","docker","aws",
            "langchain","opencv","yolo","nlp","kubernetes","figma",
            "salesforce","leadership","agile"
        ]
        heatmap_data = {}
        for cat in df['category'].unique():
            texts = ' '.join(df[df['category']==cat]['resume_text']).lower()
            heatmap_data[cat] = {sk: texts.count(sk) for sk in key_skills}

        hm_df = pd.DataFrame(heatmap_data).T
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        sns.heatmap(hm_df, annot=True, fmt='d', cmap='Blues',
                    ax=ax, linecolor='#30363d', linewidths=0.5,
                    cbar_kws={'label': 'Frequency'})
        ax.set_title('Skill Keyword Heatmap by Category', color='#e6edf3', pad=12)
        ax.tick_params(colors='#8b949e', labelsize=8)
        plt.xticks(rotation=35, ha='right', color='#8b949e')
        plt.yticks(rotation=0, color='#8b949e')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("### 📋 Full Dataset Preview")
    st.dataframe(df[['category','years_experience','education_level','suitability_score']], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════
elif "📈 Model Metrics" in page:
    st.markdown("## 📈 ML Model Performance")

    col1, col2 = st.columns([1,1])
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
        st.markdown("### Confusion Matrix (Training Data)")
        texts = df['resume_text'].tolist()
        X_all = vectorizer.transform([clean_text(t) for t in texts])
        y_all = le.transform(df['category'].tolist())
        y_pred = model.predict(X_all)
        cm = confusion_matrix(y_all, y_pred)
        classes_short = [c[:10] for c in le.classes_]

        fig, ax = plt.subplots(figsize=(9, 7), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes_short, yticklabels=classes_short,
                    ax=ax, linecolor='#30363d', linewidths=0.5)
        ax.set_xlabel('Predicted', color='#8b949e')
        ax.set_ylabel('Actual', color='#8b949e')
        ax.set_title('Confusion Matrix', color='#e6edf3')
        ax.tick_params(colors='#8b949e', labelsize=7)
        plt.xticks(rotation=40, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        train_acc = accuracy_score(y_all, y_pred)
        st.success(f"✅ Training Accuracy: **{train_acc*100:.1f}%**")
        st.info("📌 Note: Dataset is small (synthetic for demo). Real-world use Kaggle Resume Dataset with 2400+ records.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════
elif "ℹ️ About" in page:
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
### 🎯 Project Overview
This **AI-Powered Resume Screening System** is built as part of an AI/ML Internship task.
The system uses NLP and Machine Learning to analyze resumes, extract skills, classify job categories,
and predict candidate suitability.

### 🏗️ Tech Stack
| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| ML Models | scikit-learn (RF, LR, GB) |
| NLP | TF-IDF Vectorization |
| Visualization | Matplotlib, Seaborn |
| Language | Python 3.10+ |

### 📂 Supported Job Categories (16 total)
| Group | Categories |
|-------|-----------|
| 🤖 AI / ML | AI Engineer, Computer Vision Engineer, Data Scientist, NLP Engineer |
| 💻 Engineering | Software Engineer, DevOps Engineer, Mobile Developer, Cybersecurity Analyst |
| 📊 Business | Business Analyst, Product Manager, Accountant, Marketing Manager |
| 🎨 Creative & Other | Graphic Designer, HR Manager, Sales Representative, Teacher |

### 🔬 ML Pipeline
1. **Data Collection** — Synthetic + Kaggle Resume Dataset
2. **Preprocessing** — Lowercase, remove special chars, stopword removal
3. **Feature Extraction** — TF-IDF with 500 features, bigrams
4. **Model Training** — Random Forest (best), LR, GBM compared
5. **Evaluation** — Cross-validation, confusion matrix
6. **Deployment** — Streamlit web interface

### 📁 Project Structure
```
resume_screening/
├── app.py          ← Main Streamlit application
├── run.py          ← Auto-installer & launcher
└── requirements.txt
```
    """)
    
