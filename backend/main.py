from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.services.predictor import SentimentPredictor
from backend.services.youtube_api import fetch_video_comments


app = FastAPI(title="YouTube Comment Sentiment Analyzer")

# Load model once when backend starts
predictor = SentimentPredictor()


class PredictRequest(BaseModel):
    text: str


class AnalyzeRequest(BaseModel):
    video_url: str
    max_comments: int = Field(default=1000, ge=1, le=10000)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        return predictor.predict(request.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
        video_data = fetch_video_comments(
            request.video_url,
            request.max_comments
        )

        results = []

        for comment in video_data["comments"]:
            try:
                prediction = predictor.predict(comment["text"])
                results.append({
                    **comment,
                    **prediction
                })
            except ValueError:
                continue

        counts = {
            label: sum(
                r["sentiment"] == label for r in results
            )
            for label in ["Negative", "Neutral", "Positive"]
        }

        return {
            "video_title": video_data["title"],
            "channel": video_data["channel"],
            "total_fetched": video_data["total_fetched"],
            "total_analyzed": len(results),
            "sentiment_counts": counts,
            "comments": results
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))