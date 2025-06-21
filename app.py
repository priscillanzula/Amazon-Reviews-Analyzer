from flask import Flask, request, jsonify
from amazon_reviews_nlp import AmazonReviewsNLP

app = Flask(__name__)
analyzer = AmazonReviewsNLP()
@app.route('/')
def home():
    return "Flask API is running. Use POST /analyze with JSON {'text': ...}"
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    entities = analyzer.extract_entities(text)
    sentiment = analyzer.analyze_sentiment_rule_based(text)
    return jsonify({
        'entities': entities,
        'sentiment': sentiment
    })

if __name__ == '__main__':
    app.run()