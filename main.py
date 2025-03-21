from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Small Business AI Analytics API!"}

from fastapi import UploadFile, File
import pandas as pd
import io

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # Read uploaded file
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))  # Convert to DataFrame
    return {"message": "File uploaded successfully", "columns": df.columns.tolist()}

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    data = {
        "Age": [25, 30, 35, 40, 45, 50, 55],
        "MonthlySpending": [100, 200, 150, 300, 250, 400, 350],
        "Churn": [0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    X = df.drop(columns=["Churn"])  # Features
    y = df["Churn"]  # Target

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, "churn_model.pkl")  # Save model
    return model

# Load model
try:
    model = joblib.load("churn_model.pkl")
except:
    model = train_model()

@app.post("/predict")
async def predict_churn(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])  # Remove actual churn values if present

    predictions = model.predict(df)
    df["Predicted Churn"] = predictions

    return {"message": "Predictions generated", "data": df.to_dict(orient="records")}

import openai

@app.post("/chat")
async def chat_with_ai(query: str):
    openai.api_key = "YOUR_OPENAI_API_KEY"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}]
    )
    return {"response": response["choices"][0]["message"]["content"]}
