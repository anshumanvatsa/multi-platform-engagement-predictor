<h1 align="center">Multi-Platform Social Media Engagement Predictor</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/LightGBM-F1%3D0.87-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/AUC--ROC-0.94-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Dataset-800K%20rows-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Platforms-5-red?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

<p align="center">
  A production-grade machine learning system that classifies social media posts as <strong>HIGH</strong> or <strong>LOW</strong> engagement
  across five platforms — trained on 800K+ real posts with full SHAP explainability.
</p>

---

## What This Project Does

Most social media engagement predictors work on a single platform with a few hundred rows. This project does it properly:

- **800,953 posts** across Facebook, TikTok, Twitter, YouTube (×2 sources)
- **Per-platform binarisation** — "HIGH" means high *for that platform*, not globally. A post with 500 likes is high engagement on Facebook, but irrelevant on YouTube trending.
- **5 models compared** with 5-fold stratified cross-validation
- **LightGBM wins** — F1=0.8701, AUC=0.9444 on a 160K held-out test set
- **SHAP TreeExplainer** — 7 publication-ready explainability figures
- **FastAPI microservice** — serves predictions at `/predict`

---

## Results

### Model Comparison (5-Fold Cross-Validation)

| Model | CV F1 | CV AUC-ROC | Training Time |
|-------|-------|-----------|---------------|
| Logistic Regression | 0.8538 ± 0.0009 | 0.9341 | 17s |
| Random Forest | 0.8695 ± 0.0007 | 0.9433 | 176s |
| Gradient Boosting | 0.8684 ± 0.0009 | 0.9417 | 450s |
| XGBoost | 0.8701 ± 0.0009 | 0.9433 | 38s |
| **LightGBM** ⭐ | **0.8708 ± 0.0010** | **0.9445** | **38s** |

### Test Set Performance (160,191 held-out rows)

| Metric | Score |
|--------|-------|
| F1-Score | **0.8701** |
| Accuracy | **86.59%** |
| AUC-ROC | **0.9444** |
| Precision (HIGH) | 0.84 |
| Recall (HIGH) | 0.90 |

### Top-5 SHAP Features (Mean Absolute Impact)

| Rank | Feature | Mean \|SHAP\| |
|------|---------|-------------|
| 1 | total_views | 1.7405 |
| 2 | log_total_views | 1.6198 |
| 3 | post_hour | 0.2791 |
| 4 | hashtag_count | 0.2742 |
| 5 | platform | 0.1239 |

---

## Dataset Sources

| Platform | Dataset | Rows | Source | Type |
|----------|---------|------|--------|------|
| Facebook | UCI Facebook Metrics | 500 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/368/facebook+metrics) | ✅ Real |
| TikTok | TikTok User Engagement Data | 19,382 | [Kaggle](https://www.kaggle.com/datasets/yakhyojon/tiktok) | ✅ Real |
| Twitter | DMO Social Media Engagement | 23,006 | [Kaggle](https://www.kaggle.com/datasets/jocelyndumlao/dmo-social-media-engagement-dataset) | ✅ Real |
| YouTube | Video Trends & Non-Trends | 390,043 | [Kaggle](https://www.kaggle.com/datasets/muhammedchreiki/youtube-video-trends-and-non-trends-dataset) | ✅ Real |
| YouTube | YouTube Trending (10 countries) | 375,942 | [Kaggle](https://www.kaggle.com/datasets/datasnaek/youtube-new) | ✅ Real |

**Total after cleaning:** 800,953 rows · **Perfect 50/50 class balance** (per-platform median split)

---

## SHAP Explainability Figures

> All figures are saved in `outputs/figures/` and generated without rerunning the model.

| Figure | Description |
|--------|-------------|
| `shap_summary_beeswarm.png` | Feature impact distribution across all predictions |
| `shap_summary_bar.png` | Mean absolute SHAP — global feature ranking |
| `shap_waterfall_high.png` | Why a specific post was predicted HIGH |
| `shap_waterfall_low.png` | Why a specific post was predicted LOW |
| `shap_dependence_post_hour.png` | How posting hour shifts prediction |
| `shap_dependence_log_views.png` | How view count shifts prediction |
| `shap_platform_comparison.png` | What drives each platform differently |

---

## Project Structure

```
multi-platform-engagement-predictor/
│
├── normalise.py          # Loads 5 datasets → common schema → combined_dataset.csv
├── train.py              # ML pipeline: 5 models, 5-fold CV, saves best model
├── shap_analysis.py      # SHAP explainability: 7 figures + feature importance JSON
├── prediction_api.py     # FastAPI microservice: POST /predict
├── reason_generator.py   # SHAP values → plain English explanations
├── schemas.py            # Pydantic request/response models
├── config.py             # All paths, constants, feature lists
│
├── data/
│   ├── dataset.csv                  # UCI Facebook (original)
│   ├── combined_dataset.csv         # 800K combined (generate with normalise.py)
│   └── raw_datasets/                # Downloaded Kaggle datasets (gitignored)
│
├── models/
│   ├── best_model.joblib            # Trained LightGBM pipeline
│   └── preprocessor.joblib         # Fitted ColumnTransformer
│
├── outputs/
│   ├── model_results.json           # All CV and test scores
│   ├── shap_feature_importance.json # Mean |SHAP| per feature
│   └── figures/                     # 13 publication-ready figures
│
├── notebooks/
│   └── eda.ipynb                    # Exploratory data analysis
│
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | scikit-learn 1.5, LightGBM, XGBoost |
| Data | pandas 2.2, numpy 1.26 |
| Explainability | SHAP 0.45 (TreeExplainer) |
| Visualisation | matplotlib 3.9, seaborn 0.13 |
| API | FastAPI 0.111, uvicorn |
| Serialisation | joblib 1.4 |
| Validation | pydantic 2.7 |

---

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/anshumanvatsa/multi-platform-engagement-predictor.git
cd multi-platform-engagement-predictor
pip install -r requirements.txt
```

### 2. Download datasets

Download the 5 datasets listed above from Kaggle (requires a free Kaggle account) and place them in:
```
data/dataset.csv                                       ← UCI Facebook
data/raw_datasets/tiktok/tiktok_dataset.csv
data/raw_datasets/twitter/DMO social media.../Data LIWC 01 02 23.csv
data/raw_datasets/youtube_nontrends/Youtube_Videos.csv
data/raw_datasets/youtube_trending/USvideos.csv  (+ 9 other country files)
```

Or use the Kaggle CLI:
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d yakhyojon/tiktok -p data/raw_datasets/tiktok --unzip
kaggle datasets download -d jocelyndumlao/dmo-social-media-engagement-dataset -p data/raw_datasets/twitter --unzip
kaggle datasets download -d muhammedchreiki/youtube-video-trends-and-non-trends-dataset -p data/raw_datasets/youtube_nontrends --unzip
kaggle datasets download -d datasnaek/youtube-new -p data/raw_datasets/youtube_trending --unzip
```

### 3. Normalise datasets

```bash
python normalise.py
# Output: data/combined_dataset.csv (800K rows)
```

### 4. Train models

```bash
python train.py
# Output: models/best_model.joblib, outputs/model_results.json, outputs/figures/*.png
```

### 5. Run SHAP analysis

```bash
python shap_analysis.py
# Output: outputs/figures/shap_*.png, outputs/shap_feature_importance.json
```

### 6. Start the API

```bash
uvicorn prediction_api:app --reload --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

---

## API Usage

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "instagram",
    "post_type": "Reel",
    "post_hour": 19,
    "post_weekday": 5,
    "has_media": 1,
    "is_video": 1,
    "is_paid": 0,
    "caption_length": 45,
    "hashtag_count": 12,
    "follower_count": 45000
  }'
```

Response:
```json
{
  "engagement_class": 1,
  "label": "HIGH",
  "confidence": 0.82,
  "reason": "Strong prediction driven by peak posting hour (+0.28) and video format (+0.21). Consider adding more hashtags to improve discoverability."
}
```

---

## Honest Limitations

1. **YouTube dominates the dataset** (95% of rows). The model learned YouTube engagement dynamics most deeply. Per-platform binarisation corrects for this in the target variable, but feature weights still reflect YouTube-scale numbers.

2. **Post-publication features in training** — `total_views` and `log_total_views` are the top two SHAP features, but these aren't available before publishing. The API is intentionally restricted to pre-publication inputs only. A retrained model using only pre-publication features achieves F1 ≈ 0.75, which is the honest pre-publication performance.

3. **No Instagram native data** — Instagram API restrictions mean no first-party Instagram posts dataset was used. The TikTok Reels data partially proxies short-video Instagram behaviour.

---

## Methodology

### Per-Platform Binarisation
Rather than splitting HIGH/LOW at a global engagement median (which would make YouTube dominate the definition of "high"), we compute the median **per platform** and binarise within each platform:

```python
for platform in df["platform"].unique():
    mask = df["platform"] == platform
    median = df.loc[mask, "engagement_score"].median()
    df.loc[mask & (df["engagement_score"] >= median), "engagement_class"] = 1
```

This means "HIGH" on Facebook (median=125 interactions) and "HIGH" on YouTube (median=11,574 interactions) are both genuinely high **for their respective platforms**.

### Feature Engineering
Beyond raw dataset columns, the following features are derived:
- `log_total_views`, `log_follower_count`, `log_caption_length` — log-transforms for heavy-tailed distributions
- `video_x_peak` — interaction: video content posted in peak hours
- `paid_x_weekend` — interaction: paid promotion on weekends
- `media_x_hashtag` — interaction: media posts with hashtag counts
- `hour_bin` — categorical time-of-day bucket (night/morning/afternoon/evening)

---

## Citation

If you use this project or dataset construction methodology in your research:

```bibtex
@misc{mishra2026multiplatform,
  author       = {Anshuman Vatsa Mishra},
  title        = {Multi-Platform Social Media Engagement Predictor},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/anshumanvatsa/multi-platform-engagement-predictor}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with precision. Trained on reality.</p>
