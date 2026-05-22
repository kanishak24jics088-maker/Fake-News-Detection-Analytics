# 🛡️ FakeGuard Analytics — Fake News Detection System

## 📌 Project Overview

Social media platforms face an unprecedented challenge — the rapid viral spread of **misinformation and fake news** that threatens public health, democratic processes, and social cohesion. Traditional fact-checking is too slow to keep pace with millions of articles posted daily.

**FakeGuard Analytics** is a data-driven solution that combines:
- 🤖 **Machine Learning** classification (Logistic Regression, Naive Bayes, Random Forest)
- 📝 **NLP text analysis** using TF-IDF vectorization and VADER sentiment
- 📊 **Interactive analytics dashboard** built with Streamlit and Plotly
- ⚡ **Real-time detector** — paste any article and get an instant verdict

> **Case Study Domain:** Data Analytics · Social Media Misinformation  
> **Academic Context:** This project was developed as a complete case study solution covering problem statement, dataset details, analytics methodology, business insights, and final outcomes.

---

## 🖥️ App Preview

| Page | Description |
|------|-------------|
| 📋 **Problem Statement** | Case study overview, objectives, 8-step analytics pipeline |
| 📊 **Dataset Overview** | 800-article dataset, 13 features, sample table, statistical summary |
| 🔍 **Exploratory Analysis** | Distribution charts, correlation heatmap, word clouds (fake vs real) |
| 🤖 **ML Classification** | 3 model comparison — accuracy, AUC-ROC, confusion matrix, ROC curve |
| ⚡ **Real-time Detector** | Live fake/real classifier with confidence score and feature breakdown |
| 📈 **Trend Analysis** | Daily volume, hourly patterns, platform and category analysis |
| 📄 **Full Report** | Complete academic report: problem → data → steps → insights → outcome |

---

## 🗂️ Dataset Details

| Property | Value |
|----------|-------|
| Total Articles | 800 (configurable up to 1200) |
| Fake Articles | ~42% of dataset |
| Real Articles | ~58% of dataset |
| Categories | Politics, Health, Science, Technology, Economy, Entertainment, Environment |
| Platforms | Twitter, Facebook, Reddit, WhatsApp, Telegram |
| Features | 13 (text + engagement + metadata) |

### Features Used

| Feature | Description |
|---------|-------------|
| `headline` | Article title text |
| `body` | Full article body |
| `label` | Target — 0 = Real, 1 = Fake |
| `shares` | Social media share count |
| `comments` | User comment count |
| `caps_ratio` | Proportion of UPPERCASE letters |
| `clickbait_score` | Clickbait likelihood score (0–1) |
| `source_credibility` | Publisher trust score (0–1) |
| `sentiment_score` | VADER sentiment (−1 to +1) |
| `word_count` | Article body word count |
| `exclamations` | Number of `!` marks in headline |
| `category` | News topic category |
| `platform` | Social media platform |

---

## 🧠 Models & Performance

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | ~91% | ~0.96 |
| Naive Bayes | ~88% | ~0.94 |
| Random Forest | ~90% | ~0.95 |

All models use **TF-IDF vectorization** (3000 features, unigrams + bigrams) on combined headline + body text with a 75/25 stratified train-test split.

---

## 📊 Analytics Steps Performed

1. **Data Collection & Simulation** — Labeled dataset with realistic engagement metrics across 7 categories and 5 platforms
2. **Text Preprocessing** — Lowercasing, punctuation removal, stopword filtering, TF-IDF vectorization
3. **Feature Engineering** — caps_ratio, exclamation count, clickbait_score, VADER sentiment, source credibility
4. **Exploratory Data Analysis** — Label distribution, engagement stats, word clouds, correlation heatmap
5. **Model Training** — Logistic Regression, Naive Bayes, Random Forest with cross-validation
6. **Model Evaluation** — Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix
7. **Trend Analysis** — Daily volume, hourly patterns, platform hotspots, share velocity
8. **Real-time Deployment** — Interactive article classifier with probability scores and feature explanation

---

## 💡 Key Business Insights

- 🔥 **Fake news gets shared ~2–3x more** than real news — viral amplification requires priority flagging
- ⚠️ **Health & Politics** categories have the highest fake news rates — stricter thresholds needed
- 🤖 **Caps ratio and clickbait score** are the strongest individual predictors of fake news
- 📱 **Facebook and WhatsApp** lead in fake news volume — priority platforms for API integration
- ⏰ **Fake news peaks in early morning hours** — automated off-hours detection is critical
- 📉 **Source credibility < 0.4** reliably predicts fake content — maintain a publisher trust database

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fakeguard-analytics.git
cd fakeguard-analytics
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib plotly wordcloud nltk Pillow
```

### 3. Download NLTK data (run once)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('punkt_tab')"
```

### 4. Run the app

```bash
streamlit run fake_news_app.py
```

Or if streamlit is not in PATH:

```bash
python -m streamlit run fake_news_app.py
```

App opens at **http://localhost:8501**

---

## 📁 Project Structure

```
fakeguard-analytics/
│
├── fake_news_app.py        # Main Streamlit application
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## 📋 Requirements

Create a `requirements.txt` with:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.15.0
wordcloud>=1.9.0
nltk>=3.8.0
Pillow>=10.0.0
```

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `'streamlit' is not recognized` | Use `python -m streamlit run fake_news_app.py` |
| `Invalid frequency: 3H` | Change `'3H'` to `'3h'` in `generate_dataset()` on line ~211 |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| NLTK errors | Run the NLTK download command in Step 3 above |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Interactive web dashboard |
| **Scikit-learn** | ML model training and evaluation |
| **NLTK + VADER** | NLP text processing and sentiment analysis |
| **TF-IDF** | Text feature extraction |
| **Plotly** | Interactive visualizations |
| **WordCloud** | Visual word frequency analysis |
| **Pandas / NumPy** | Data manipulation and analysis |
| **Matplotlib** | Static chart generation |

---

## 📄 Report Summary

This project delivers a **complete academic case study report** covering:

- ✅ Problem Statement
- ✅ Dataset Details and Feature Descriptions
- ✅ Step-by-step Analytics Methodology
- ✅ ML Model Training and Evaluation Results
- ✅ Business Insights from Data Analysis
- ✅ Final Outcome and Deployment Recommendations

All accessible directly from the **📄 Full Report** page inside the app.

---

## 🎓 Academic Context

> **Subject:** Data Analytics  
> **Case Study:** Fake News Detection Analytics  
> **Domain:** Social Media · NLP · Machine Learning · Content Moderation  
> **Unique Feature:** Real-time fake news detector with confidence scoring

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ using Python & Streamlit

⭐ Star this repo if you found it helpful!

</div>
