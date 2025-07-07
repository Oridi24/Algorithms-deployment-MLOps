from fastapi import FastAPI, Query
from transformers import pipeline

app = FastAPI(title="FastAPI con pipelines HF", version="1.0")

# Cargar pipelines una vez al iniciar la API
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

@app.get("/")
def root():
    return {"message": "API con 5 endpoints, 2 con Hugging Face pipelines"}

@app.get("/hello")
def hello(name: str = Query("Ori", description="Your name")):
    return {"greeting": f"Hello, {name}!"}

@app.get("/add")
def add(a: int, b: int):
    return {"result": a + b}

@app.get("/sentiment")
def sentiment(text: str = Query(..., description="Text to analyze sentiment")):
    result = sentiment_analyzer(text)
    return {"label": result[0]["label"], "score": result[0]["score"]}

@app.get("/summary")
def summary(text: str = Query(..., description="Text to summarize", max_length=1000)):
    result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return {"summary": result[0]["summary_text"]}

@app.get("/length")
def text_length(text: str = Query(..., description="Text to count length")):
    return {"length": len(text)}

