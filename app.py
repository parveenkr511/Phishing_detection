from flask import Flask, request, render_template
import joblib
import pandas as pd
import re

app = Flask(__name__)
model = joblib.load("phishing_url_detector.pkl")

def extract_features(URL):
    return {
        'url_length': len(URL),
        'num_dots': URL.count('.'),
        'num_hyphens': URL.count('-'),
        'num_at': URL.count('@'),
        'has_https': int(URL.startswith("https")),
        'has_ip': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', URL))),
        'num_subdirs': URL.count('/'),
        'num_parameters': URL.count('='),
        'num_percent': URL.count('%'),
        'num_www': URL.count('www'),
        'num_digits': sum(c.isdigit() for c in URL),
        'num_letters': sum(c.isalpha() for c in URL)
    }

@app.route('/', methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        url = request.form["url"]
        features = pd.DataFrame([extract_features(url)])
        prediction = model.predict(features)[0]
        result = "Phishing ❌" if prediction == 1 else "Legitimate ✅"
    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
