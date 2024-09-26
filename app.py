from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import PyPDF2

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SUMMARY_FOLDER'] = 'summaries'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SUMMARY_FOLDER'], exist_ok=True)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(filtered_words)

def generate_summary(text, model_name="facebook/bart-large-cnn", max_length=150, min_length=50, do_sample=False):
    summarizer = pipeline("summarization", model=model_name)
    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)
        summaries.append(summary[0]['summary_text'])

    return ' '.join(summaries)


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

def save_summary_to_file(summary, filename):
    file_path = os.path.join(app.config['SUMMARY_FOLDER'], filename)
    with open(file_path, 'w') as file:
        file.write(summary)
    return file_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        text = read_pdf(file_path)
        summary = generate_summary(text)

        summary_filename = f"summary_{os.path.splitext(file.filename)[0]}.txt"
        summary_file_path = save_summary_to_file(summary, summary_filename)

        os.remove(file_path)

        return render_template('index.html', summary=summary, summary_filename=summary_filename)

@app.route('/download/<summary_filename>', methods=['GET'])
def download_summary(summary_filename):
    file_path = os.path.join(app.config['SUMMARY_FOLDER'], summary_filename)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
