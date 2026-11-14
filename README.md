# YouTube Comment Sentiment Analyzer

An end-to-end deep learning application that analyzes YouTube comments and classifies audience sentiment into **Positive, Negative, and Neutral** categories using a Bidirectional LSTM (Bi-LSTM) network.

## Features

- Fetches YouTube video information and comments using YouTube Data API v3.
- Classifies comments into positive, negative, and neutral sentiments.
- Generates prediction confidence scores and class probabilities.
- Displays sentiment distribution through interactive charts.
- Provides comment-level sentiment analysis.
- Allows filtering and downloading analyzed comments as CSV.

## Tech Stack

| Component | Technologies |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| NLP | NLTK, Keras Tokenizer |
| Model Architecture | Bidirectional LSTM |
| Backend | FastAPI |
| Frontend | Streamlit |
| Visualization | Plotly |
| Data Processing | NumPy, Pandas, Scikit-learn |
| External API | YouTube Data API v3 |

## Dataset

The model is trained using the [Twitter Sentiment Dataset by Saurabh Shahane](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset).

The dataset contains two primary columns:

- `clean_text`: Input text.
- `category`: Sentiment label.

| Original Label | Sentiment |
|---|---|
| -1 | Negative |
| 0 | Neutral |
| 1 | Positive |

The original dataset contains approximately 163K records. The training pipeline balances the three sentiment classes by retaining 35,000 positive samples, 35,000 neutral samples, and all 35,509 negative samples.

## Machine Learning Pipeline

### Text Preprocessing

The following preprocessing operations are applied:

1. Lowercasing
2. URL removal
3. HTML tag removal
4. Emoji removal
5. Chat abbreviation conversion
6. Punctuation removal
7. POS-based lemmatization
8. Tokenization and sequence padding

### Bi-LSTM Architecture

The sentiment classifier uses the following architecture:

| Layer | Configuration |
|---|---|
| Embedding | 128 dimensions |
| Bidirectional LSTM | 32 units, return sequences |
| Batch Normalization | Applied |
| Bidirectional LSTM | 64 units |
| Batch Normalization | Applied |
| Dense | 64 neurons, ReLU |
| Dropout | 0.2 |
| Output | 3 neurons, Softmax |

### Training Configuration

| Parameter | Value |
|---|---|
| Vocabulary size | 40,000 |
| Maximum sequence length | 100 |
| Optimizer | Adam |
| Loss function | Categorical Cross-Entropy |
| Batch size | 64 |
| Epochs | 5 |
| Train-test split | 80:20 |
| Random state | 42 |

## Model Evaluation

The model was evaluated on the held-out test portion of the Twitter sentiment dataset.

| Metric | Result |
|---|---:|
| Test Accuracy | **90.47%** |
| Macro F1-score | **0.905** |
| Test Loss | **0.3972** |

The model's classification performance was evaluated across all three sentiment classes using precision, recall, F1-score, and a confusion matrix.

These metrics represent performance on the held-out Twitter dataset, not independently measured performance on YouTube comments.

## Application Architecture

The application follows this workflow:

1. The user enters a YouTube video URL through the Streamlit interface.
2. FastAPI receives the request.
3. YouTube Data API v3 retrieves video information and comments.
4. Comments pass through the same preprocessing pipeline used during model training.
5. The saved Bi-LSTM model predicts the sentiment of each comment.
6. Predictions are aggregated into positive, negative, and neutral counts.
7. FastAPI returns the results as JSON.
8. Streamlit displays sentiment statistics, interactive charts, and individual comment predictions.

## Project Structure

```text
YouTube-Comment-Sentiment-Analyzer/
│
├── artifacts/
│   ├── sentiment_model.keras
│   ├── tokenizer.json
│   └── config.json
│
├── backend/
│   ├── main.py
│   └── services/
│       ├── predictor.py
│       ├── preprocessing.py
│       └── youtube_api.py
│
├── frontend/
│   └── app.py
│
├── notebooks/
│   └── Youtube-Comment-Sentiment-Analyzer.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YouTube-Comment-Sentiment-Analyzer
```

Install Git LFS and retrieve the trained model:

```bash
git lfs install
git lfs pull
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure NLTK resources

Download the required NLTK resources:

```bash
python -c 'import nltk; nltk.download("wordnet"); nltk.download("averaged_perceptron_tagger_eng")'
```

### 4. Configure YouTube API credentials

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key
```

Enable YouTube Data API v3 through [Google Cloud Console](https://console.cloud.google.com/).

Never commit your `.env` file or expose your API key.

### 5. Run the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Run the Streamlit frontend

Open another terminal:

```bash
streamlit run frontend/app.py
```

Open the local URL displayed in the terminal to access the application.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks backend and model status |
| POST | `/predict` | Predicts sentiment for individual text |
| POST | `/analyze` | Fetches and analyzes YouTube comments |

## Limitations

The initial classifier is trained on Twitter sentiment data and applied to YouTube comments during inference.

Differences in vocabulary, writing style, slang, sarcasm, and contextual expressions can affect prediction quality across platforms.

The reported accuracy and F1-score are based on the Twitter test dataset and should not be interpreted as YouTube-specific evaluation metrics.

The model also removes emojis during preprocessing, which can discard useful emotional information.

## Future Improvements

- Fine-tuning on YouTube-specific sentiment datasets.
- Improved sarcasm and contextual sentiment detection.
- Emoji-aware text preprocessing.
- Multilingual sentiment classification.
- Enhanced comment-level analytics.

---

**Project:** YouTube Comment Sentiment Analyzer  
**Model:** Bidirectional LSTM  
**Backend:** FastAPI  
**Frontend:** Streamlit