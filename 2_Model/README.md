# 2_Model — Trained model weights

This folder is populated automatically by the training notebook.

Run all cells in `1_Notebook/sentiment_model_training.ipynb`; the final cells save the
fine-tuned DistilBERT model and tokenizer to:

```
2_Model/distilbert-sentiment/
```

The API (`3_API/main.py`) loads the model from that path automatically.
Until the folder exists, the API falls back to a public pretrained 3-class sentiment
model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) so the demo still works
end-to-end before training.
