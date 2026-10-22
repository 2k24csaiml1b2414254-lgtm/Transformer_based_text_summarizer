
# Transformer-Based Text Summarizer

A text summarization web application built using a T5 Transformer model, Hugging Face Transformers, FastAPI, and HTML/CSS/JavaScript.

## Features

- Text summarization using a saved T5 model
- FastAPI backend
- Responsive web interface
- Input validation and error handling
- CPU and CUDA device selection
- REST API endpoint for summarization

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- T5
- FastAPI
- HTML, CSS, JavaScript

## Project Structure

```text
Transformer_based_text_summarizer/
├── Summarizer.py
├── index.html
├── requirements.txt
├── README.md
├── .gitignore
└── saved_summary_model/
```

The model directory must contain the model weights and tokenizer files required by Hugging Face Transformers.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/2k24csaiml1b2414254-lgtm/Transformer_based_text_summarizer.git
   cd Transformer_based_text_summarizer
   ```

2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Make sure `saved_summary_model/` exists and contains the required model files.

4. Start the application:

   ```bash
   uvicorn Summarizer:app --reload
   ```

5. Open the application:

   http://127.0.0.1:8000

## API

### POST `/summarize/`

Request:

```json
{
  "dialogue": "Paste the text you want to summarize here."
}
```

Response:

```json
{
  "summary": "The generated summary appears here."
}
```

### GET `/health`

Returns application and model-loading status.

## Important Notes

- The saved model must be available at `saved_summary_model/`.
- The tokenizer and model files must be compatible.
- Inputs are truncated to 512 tokens by the current implementation.
- Actual output quality depends on the model's training and fine-tuning.
- GPU availability depends on the installed PyTorch build and compatible hardware.

## License

Add the appropriate license for this project and its model.
