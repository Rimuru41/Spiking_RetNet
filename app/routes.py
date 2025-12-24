from flask import Blueprint, render_template, request, jsonify
from .models.imdb.processor import run_imdb_inference

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model')
    user_input = request.form.get('input_data')

    if model_choice == 'imdb':
        if not user_input or len(user_input.strip()) < 5:
            return jsonify({"error": "Please enter a valid review"}), 400
        
        result = run_imdb_inference(user_input)
        return jsonify(result)

    return jsonify({"error": "Architecture not yet loaded"}), 400