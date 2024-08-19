
# Text Summarizer

This project provides a Python-based solution for summarizing the content of PDF files using Natural Language Processing (NLP) techniques and transformer models. The summarizer can efficiently condense large texts into concise summaries.

## Features

- **Text Preprocessing**: The script removes stop words and tokenizes the text for better processing.
- **Summarization**: Utilizes the BART transformer model to generate summaries.
- **Customizable**: You can adjust parameters like the model, maximum and minimum length of the summary.

```python

text = "Your PDF content here"
processed_text = preprocess_text(text)


summary = generate_summary(processed_text)
print("Summary:", summary)
```
