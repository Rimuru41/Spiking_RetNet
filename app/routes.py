import os
import importlib
from flask import Blueprint, render_template, request, jsonify

main = Blueprint('main', __name__)

def get_filtered_model_structure():
    """
    Scans the app/models directory to find categories and datasets.
    Filters out system folders like __pycache__.
    """
    base_path = os.path.join(os.getcwd(), 'app', 'models')
    structure = {}
    
    # Folders to ignore in the UI
    IGNORED = {'__pycache__', '.ipynb_checkpoints', '.git', '.pytest_cache'}
    
    if os.path.exists(base_path):
        # 1. Identify Categories (video, audio, text, image, neuromorphic)
        categories = [d for d in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, d)) 
                      and d not in IGNORED 
                      and not d.startswith('.')]
        
        for cat in categories:
            cat_path = os.path.join(base_path, cat)
            # 2. Identify Datasets inside each category
            datasets = [d for d in os.listdir(cat_path) 
                        if os.path.isdir(os.path.join(cat_path, d)) 
                        and d not in IGNORED 
                        and not d.startswith('.')]
            
            # Only add the category to the UI if it actually contains a dataset
            if datasets:
                structure[cat] = datasets
                
    return structure

@main.route('/')
def index():
    """
    Renders the main dashboard and passes the dynamic model tree to the frontend.
    """
    models_tree = get_filtered_model_structure()
    return render_template('index.html', models_tree=models_tree)

@main.route('/predict', methods=['POST'])
def predict():
    """
    Handles inference requests. Dynamically loads the processor for the 
    selected dataset and runs the prediction logic.
    """
    category = request.form.get('category')
    dataset = request.form.get('model')
    
    if not category or not dataset:
        return jsonify({"error": "Missing category or dataset selection"}), 400

    try:
        # Construct the module path for dynamic loading
        # Example: app.models.text.imdb.processor
        module_path = f"app.models.{category}.{dataset}.processor"
        processor = importlib.import_module(module_path)
        
        # Branching logic based on category type
        if category == 'text':
            user_input = request.form.get('input_data')
            if not user_input or len(user_input.strip()) < 2:
                return jsonify({"error": "Please provide valid text input"}), 400
            
            # Call the specific inference function in the processor
            result = processor.run_imdb_inference(user_input)
            
        else:
            # Handle File Uploads for non-text categories
            uploaded_file = request.files.get('file_data')
            if not uploaded_file:
                return jsonify({"error": f"Please upload a valid {category} file"}), 400
            
            # Placeholder for other processors (Video/Image/Audio/Neuromorphic)
            # You would implement run_inference(uploaded_file) in those processors
            result = {
                "prediction": f"Processed {category} file: {uploaded_file.filename}",
                "confidence": "N/A",
                "metrics": {
                    "energy": "0.0000",
                    "sparsity": "0.00%",
                    "latency": "0.00",
                    "sops": "0"
                }
            }

        return jsonify(result)

    except ModuleNotFoundError:
        return jsonify({"error": f"Processor not found for {dataset}"}), 404
    except Exception as e:
        # Standard error handling to prevent the server from crashing
        return jsonify({"error": f"Inference Error: {str(e)}"}), 500