
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(filtered_words)

def generate_summary(text, model_name="facebook/bart-large-cnn", max_length=400, min_length=200, do_sample=False):

    summarizer = pipeline("summarization", model=model_name)

    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
    return summary[0]['summary_text']

import PyPDF2
from fpdf import FPDF
from transformers import pipeline

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text(text, max_length=400):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

def generate_summary(text, model_name="facebook/bart-large-cnn", max_length=100, min_length=30, do_sample=False):
    summarizer = pipeline("summarization", model=model_name)
    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)
        summaries.append(summary[0]['summary_text'])

    return ' '.join(summaries)

def save_summary_to_text(summary, output_file):
    with open(output_file, 'w') as file:
        file.write(summary)

def main():
    text = read_pdf('document.pdf')

    print("Original Text Length:", len(text))

    summary = generate_summary(text)

    print("\nGenerated Summary:")
    print(summary)
    output_file = '/content/summary.txt'
    save_summary_to_text(summary, output_file)
    print("\nSummary saved to summary.txt")
if __name__ == "__main__":
    main()