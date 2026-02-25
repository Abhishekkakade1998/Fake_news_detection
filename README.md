# Fake News Detection API

A BiLSTM-based fake news classifier served via a Flask REST API.

## Project Structure

```
fake_news_project/
├── app.py                        # Flask API
├── train.py                      # Model training script (run in Colab)
├── requirements.txt              # Python dependencies
├── fake_news_model.h5            # Trained model (generated after training)
└── tokenizer.pkl                 # Fitted tokenizer (generated after training)
```

---

## Setup & Deployment

### 1. Train the Model (Google Colab)
Upload `train.py` and your dataset (`fake_news_dataset_4000_rows.csv`) to Colab, then run:
```bash
python train.py
```
Download the generated `fake_news_model.h5` and `tokenizer.pkl` files.

### 2. Run Locally
```bash
pip install -r requirements.txt
python app.py
```
API will be available at `http://localhost:10000`

### 3. Deploy to Render
- Add `fake_news_model.h5` and `tokenizer.pkl` to the project root
- Set **Start Command** to: `gunicorn app:app`
- Set **Instance Type** to at least 512MB RAM

---

## API Endpoints

### `GET /`
Returns API status.

### `GET /health`
Health check endpoint.

### `POST /predict`
Classifies text as real or fake news.

**Request:**
```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "prediction": "Fake News",
  "probability": 0.8731,
  "confidence": 87.31
}
```

**Error Response:**
```json
{
  "error": "'text' field is required and cannot be empty"
}
```

---

## Notes
- Maximum input length: 10,000 characters
- Text is preprocessed (lowercased, punctuation removed) before prediction
- Probability > 0.5 = Fake News, ≤ 0.5 = Real News
