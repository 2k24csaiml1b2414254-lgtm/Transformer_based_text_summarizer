
import re
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from transformers import T5ForConditionalGeneration, T5Tokenizer


# --------------------------------------------------
# App configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_summary_model"
TEMPLATE_DIR = BASE_DIR

app = FastAPI(
    title="Text Summarizer App",
    description="Text summarization using a fine-tuned T5 Transformer",
    version="1.0.0",
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
tokenizer = None


# --------------------------------------------------
# Load model once when the application starts
# --------------------------------------------------

@app.on_event("startup")
def load_model():
    global model, tokenizer

    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}. "
            "Place your saved T5 model inside saved_summary_model/."
        )

    try:
        tokenizer = T5Tokenizer.from_pretrained(str(MODEL_DIR))
        model = T5ForConditionalGeneration.from_pretrained(
            str(MODEL_DIR)
        )

        model.to(device)
        model.eval()

    except Exception as exc:
        raise RuntimeError(
            "Could not load the saved T5 model. "
            "Check that the model and tokenizer files are complete."
        ) from exc


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class DialogueInput(BaseModel):
    dialogue: str = Field(
        min_length=1,
        max_length=20000,
        description="Text to summarize",
    )


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_data(text: str) -> str:
    text = re.sub(r"\r\n?", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------
# Summarization
# --------------------------------------------------

def summarize_dialogue(dialogue: str) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("The summarization model is not loaded.")

    dialogue = clean_data(dialogue)

    if not dialogue:
        raise ValueError("Please enter some meaningful text.")

    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/", response_class=None)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
    }


@app.post("/summarize/")
async def summarize(payload: DialogueInput):
    if not payload.dialogue.strip():
        raise HTTPException(
            status_code=422,
            detail="Please enter text to summarize.",
        )

    try:
        summary = await run_in_threadpool(
            summarize_dialogue,
            payload.dialogue,
        )

        return {"summary": summary}

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        # Keep internal exception details out of the API response.
        raise HTTPException(
            status_code=500,
            detail="Summarization failed. Please try again.",
        ) from exc
