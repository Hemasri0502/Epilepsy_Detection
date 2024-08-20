from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from chumma import *
import os 
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    if request.method == 'POST':
        if 'file' in request.files:
            edf_file = request.files['file']
            # Create the destination folder if it doesn't exist
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            # Save the uploaded file to a temporary location
            file_path = os.path.join(upload_folder, edf_file.filename)
            edf_file.save(file_path)
            # Perform detection
            category = detect(file_path)

    

    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run()

