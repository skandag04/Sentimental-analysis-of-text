from flask import Flask, render_template, request
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained emotion classification model
emotion_classifier = pipeline("text-classification", 
                              model="j-hartmann/emotion-english-distilroberta-base", 
                              return_all_scores=True)

@app.route('/', methods=['GET', 'POST'])
def analyze_emotion():
    result = None
    input_text = ""
    
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text.strip():  # Ensure input is not empty
            predictions = emotion_classifier(input_text)
            result = predictions[0]  # Get the emotion scores
    
    return render_template('index.html', input_text=input_text, result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
