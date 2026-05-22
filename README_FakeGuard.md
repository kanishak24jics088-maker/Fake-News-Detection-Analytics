# 🛡️ FakeGuard Analytics — Fake News Detection System
## Setup & Run Instructions

---

## 📦 Step 1: Install Required Libraries

Open your terminal and run:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly wordcloud nltk textblob Pillow
```

---

## 📥 Step 2: Download NLTK Data

Run this once in Python:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
```

---

## ▶️ Step 3: Run the App

```bash
streamlit run fake_news_app.py
```

The app will open at: **http://localhost:8501**

---

## 🗂️ App Pages (Use Sidebar to Navigate)

| Page | Content |
|------|---------|
| 📋 Problem Statement | Case study overview, objectives, analytics steps |
| 📊 Dataset Overview | Dataset structure, sample rows, feature descriptions |
| 🔍 Exploratory Analysis | EDA charts, word clouds, correlation heatmap |
| 🤖 ML Classification | 3 model results, confusion matrix, ROC curve |
| ⚡ Real-time Detector | Live fake news detector — paste any article |
| 📈 Trend Analysis | Time-series, hourly patterns, platform analysis |
| 📄 Full Report | Complete report: problem → dataset → insights → outcome |

---

## 📊 Models Used

- **Logistic Regression** — TF-IDF + L2 regularization
- **Naive Bayes** — Multinomial NB with TF-IDF
- **Random Forest** — 100 trees with TF-IDF

---

## 🔑 Key Features Analyzed

- `caps_ratio` — Proportion of UPPERCASE letters
- `clickbait_score` — Likelihood of clickbait headline
- `source_credibility` — Publisher trust score (0–1)
- `sentiment_score` — VADER sentiment (−1 to +1)
- `exclamations` — Number of ! marks in headline
- `word_count` — Article body length
- `shares` — Social media engagement

---

## ✅ Expected Results

- Best model accuracy: **~90–93%**
- Real-time inference: **< 100ms per article**
- 7 categories, 5 platforms analyzed
- Interactive detector with probability scores

---

*Generated for: Fake News Detection Analytics Case Study*
*Tools: Python, Streamlit, Scikit-learn, Plotly, NLTK, WordCloud*
