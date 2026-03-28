import os
import importlib
from flask import Blueprint, render_template, request, jsonify

main = Blueprint('main', __name__)

# ============================================
# PRE-LOAD ALL MODELS AT STARTUP
# ============================================
LOADED_PROCESSORS = {}

def preload_all_models():
    """
    Walks app/models and imports every processor.py eagerly at startup.
    So the first user request for ANY model is never slow.
    """
    base_path = os.path.join(os.getcwd(), 'app', 'models')
    IGNORED = {'__pycache__', '.ipynb_checkpoints', '.git', '.pytest_cache'}

    if not os.path.exists(base_path):
        print("!! [Startup] app/models directory not found, skipping preload.")
        return

    for category in os.listdir(base_path):
        cat_path = os.path.join(base_path, category)
        if not os.path.isdir(cat_path) or category in IGNORED or category.startswith('.'):
            continue

        for dataset in os.listdir(cat_path):
            ds_path = os.path.join(cat_path, dataset)
            if not os.path.isdir(ds_path) or dataset in IGNORED or dataset.startswith('.'):
                continue

            module_path = f"app.models.{category}.{dataset}.processor"
            try:
                mod = importlib.import_module(module_path)
                LOADED_PROCESSORS[f"{category}/{dataset}"] = mod
                print(f"✅ [Startup] Pre-loaded: {module_path}")
            except Exception as e:
                print(f"❌ [Startup] Failed to pre-load {module_path}: {e}")

# Runs ONCE when Flask imports routes.py — all models loaded before first request
preload_all_models()


# ============================================
# HELPER: SCAN MODEL FOLDER FOR UI
# ============================================
def get_filtered_model_structure():
    """
    Scans the app/models directory to find categories and datasets.
    Filters out system folders like __pycache__.
    """
    base_path = os.path.join(os.getcwd(), 'app', 'models')
    structure = {}
    IGNORED = {'__pycache__', '.ipynb_checkpoints', '.git', '.pytest_cache'}

    if os.path.exists(base_path):
        categories = [d for d in os.listdir(base_path)
                      if os.path.isdir(os.path.join(base_path, d))
                      and d not in IGNORED
                      and not d.startswith('.')]

        for cat in categories:
            cat_path = os.path.join(base_path, cat)
            datasets = [d for d in os.listdir(cat_path)
                        if os.path.isdir(os.path.join(cat_path, d))
                        and d not in IGNORED
                        and not d.startswith('.')]
            if datasets:
                structure[cat] = datasets

    return structure


# ============================================
# ROUTES
# ============================================
@main.route('/')
def index():
    models_tree = get_filtered_model_structure()
    return render_template('index.html', models_tree=models_tree)


@main.route('/predict', methods=['POST'])
def predict():
    """
    Uses pre-loaded processors from LOADED_PROCESSORS cache.
    Falls back to importlib if a processor was somehow missed at startup.
    """
    category = request.form.get('category')
    dataset = request.form.get('model')
    print(f"Received prediction request — Category: {category}, Dataset: {dataset}")

    if not category or not dataset:
        return jsonify({"error": "Missing category or dataset selection"}), 400

    try:
        # 1. Grab from cache (instant — no loading delay)
        key = f"{category}/{dataset}"
        processor = LOADED_PROCESSORS.get(key)

        if processor is None:
            # Fallback: import on-demand if missed during startup
            print(f"⚠️  [{dataset}] Not in cache, importing now (first-time delay expected)...")
            module_path = f"app.models.{category}.{dataset}.processor"
            processor = importlib.import_module(module_path)
            LOADED_PROCESSORS[key] = processor  # Cache it for next time

        # 2. Resolve the inference function (e.g. run_IMDB_inference)
        func_name = f"run_{dataset}_inference"
        if not hasattr(processor, func_name):
            return jsonify({"error": f"Function '{func_name}' not found in {dataset}/processor.py"}), 500

        inference_func = getattr(processor, func_name)

        # 3. Execute based on modality
        if category == 'text':
            user_input = request.form.get('input_data')
            if not user_input or len(user_input.strip()) < 2:
                return jsonify({"error": "Please provide valid text input"}), 400
            result = inference_func(user_input)

        else:
            # Image, Audio, Video, Neuromorphic — all file-based
            uploaded_file = request.files.get('file_data')
            if not uploaded_file:
                return jsonify({"error": f"Please upload a valid {category} file"}), 400
            result = inference_func(uploaded_file)

        return jsonify(result)

    except ModuleNotFoundError as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Module not found: {str(e)}"}), 404

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Inference Engine Error: {str(e)}"}), 500