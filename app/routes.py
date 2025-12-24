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
            
            if datasets:
                structure[cat] = datasets
                
    return structure

@main.route('/')
def index():
    models_tree = get_filtered_model_structure()
    return render_template('index.html', models_tree=models_tree)

@main.route('/predict', methods=['POST'])
def predict():
    """
    Dynamically loads the processor and calls the specific dataset function.
    Matches 'imdb' -> 'run_imdb_inference' and 'cifar10' -> 'run_cifar_inference'.
    """
    category = request.form.get('category')
    dataset = request.form.get('model')
    
    if not category or not dataset:
        return jsonify({"error": "Missing category or dataset selection"}), 400

    try:
        # 1. Dynamically import the processor module
        # Path: app.models.{category}.{dataset}.processor
        module_path = f"app.models.{category}.{dataset}.processor"
        processor = importlib.import_module(module_path)
        
        # 2. Determine the function name (e.g., run_imdb_inference)
        # This matches the function names you used in your processor files
        func_name = f"run_{dataset}_inference"
        
        if not hasattr(processor, func_name):
            return jsonify({"error": f"Function {func_name} not found in {dataset} processor"}), 500
        
        inference_func = getattr(processor, func_name)

        # 3. Execute based on Input Type
        if category == 'text':
            user_input = request.form.get('input_data')
            if not user_input or len(user_input.strip()) < 2:
                return jsonify({"error": "Please provide valid text input"}), 400
            
            # Call processor for text datasets (expects a string)
            result = inference_func(user_input)
            
        else:
            # Handle File Uploads for Image, Video, Audio, Neuromorphic
            uploaded_file = request.files.get('file_data')
            if not uploaded_file:
                return jsonify({"error": f"Please upload a valid {category} file"}), 400
            
            # Call processor for file-based datasets (expects FileStorage object)
            result = inference_func(uploaded_file)

        return jsonify(result)

    except ModuleNotFoundError:
        return jsonify({"error": f"Architecture for '{dataset}' is configured but processor.py is missing."}), 404
    except Exception as e:
        # Detailed error for debugging, but keeps the server alive
        import traceback
        print(traceback.format_exc()) # Logs full error to console
        return jsonify({"error": f"Inference Engine Error: {str(e)}"}), 500