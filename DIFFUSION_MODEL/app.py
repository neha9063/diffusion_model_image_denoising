"""
Diffusion Model for Image Denoising
Flask Web Application Backend
Routes: Home, Learn, Results, Upload, Forward Diffusion, Reverse Diffusion, Download
"""

import os
import time
import io
import base64
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch

from diffusion import DiffusionUNet, GaussianDiffusion, pil_to_tensor, tensor_to_pil, calculate_metrics

app = Flask(__name__)
app.config["SECRET_KEY"] = "diffusion-denoising-project-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max upload size
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(app.root_path, "outputs")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 300
MODEL_PATH = "models/diffusion_mnist.pth"

# Global model state
model = None
diffusion = GaussianDiffusion(timesteps=TIMESTEPS, device=DEVICE)
model_metadata = {
    "loaded": False,
    "device": DEVICE,
    "timesteps": TIMESTEPS,
    "epochs": 0,
    "final_loss": None,
    "trained_timestamp": None,
    "error": None
}


def load_trained_model():
    """
    Attempts to load the trained PyTorch DDPM weights from models/diffusion_mnist.pth.
    """
    global model, model_metadata
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[*] Loading model checkpoint from {MODEL_PATH}...")
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            loaded_model = DiffusionUNet(in_channels=1, out_channels=1, time_emb_dim=128).to(DEVICE)
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            loaded_model.eval()

            model = loaded_model
            model_metadata["loaded"] = True
            model_metadata["epochs"] = checkpoint.get("epochs", 5)
            model_metadata["timesteps"] = checkpoint.get("timesteps", TIMESTEPS)
            model_metadata["final_loss"] = round(checkpoint.get("final_loss", 0.0), 5)
            model_metadata["trained_timestamp"] = checkpoint.get("trained_timestamp", "Recent")
            model_metadata["error"] = None
            print(f"[+] Model loaded successfully on {DEVICE}!")
            return True
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            model_metadata["loaded"] = False
            model_metadata["error"] = str(e)
            return False
    else:
        model_metadata["loaded"] = False
        model_metadata["error"] = f"Model checkpoint '{MODEL_PATH}' not found. Please run 'python train.py' to train."
        return False


# Try to load on startup
load_trained_model()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    # Upscale display size smoothly for preview if 28x28
    if img.size == (28, 28):
        display_img = img.resize((224, 224), Image.Resampling.NEAREST)
    else:
        display_img = img
    display_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


# Store in-memory cache of recent active runs for results dashboard
recent_runs = []


# ============================================================================
# Page Routes
# ============================================================================

@app.route("/")
def index():
    # Refresh model status in case train.py just finished
    if not model_metadata["loaded"]:
        load_trained_model()
    return render_template("index.html", metadata=model_metadata)


@app.route("/learn")
def learn():
    return render_template("learn.html", metadata=model_metadata)


@app.route("/results")
def results():
    return render_template("results.html", metadata=model_metadata, recent_runs=recent_runs)


# ============================================================================
# API Routes
# ============================================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    if not model_metadata["loaded"]:
        load_trained_model()
    return jsonify(model_metadata)


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Returns the list of pre-generated sample digits available."""
    samples_dir = os.path.join(app.root_path, "static", "images", "samples")
    if not os.path.exists(samples_dir):
        return jsonify([])
    
    samples = []
    for f in sorted(os.listdir(samples_dir)):
        if f.endswith(".png"):
            digit_label = f.replace("digit_", "").replace(".png", "")
            samples.append({
                "filename": f,
                "label": f"Digit {digit_label}",
                "url": url_for("static", filename=f"images/samples/{f}")
            })
    return jsonify(samples)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Handles image uploads or sample digit selection.
    Preprocesses the image into 28x28 grayscale normalized tensor.
    """
    sample_file = request.form.get("sample_file")
    
    if sample_file:
        # Load sample from static/images/samples
        sample_path = os.path.join(app.root_path, "static", "images", "samples", secure_filename(sample_file))
        if not os.path.exists(sample_path):
            return jsonify({"error": "Sample file not found."}), 404
        try:
            img = Image.open(sample_path)
        except Exception as e:
            return jsonify({"error": f"Failed to open sample image: {str(e)}"}), 400
    else:
        # Check standard file upload
        if "image" not in request.files:
            return jsonify({"error": "No image file provided in upload request."}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed formats: PNG, JPG, JPEG."}), 400
        
        try:
            img = Image.open(file.stream)
        except Exception as e:
            return jsonify({"error": f"Could not decode image file: {str(e)}"}), 400

    # Process image: convert to grayscale and resize to 28x28
    run_id = uuid.uuid4().hex[:10]
    gray_28 = img.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    
    # Save original preview & 28x28 clean image
    clean_filename = f"clean_{run_id}.png"
    clean_path = os.path.join(app.config["UPLOAD_FOLDER"], clean_filename)
    gray_28.save(clean_path)

    # Convert to base64 preview
    preview_b64 = pil_to_base64(gray_28)

    return jsonify({
        "success": True,
        "run_id": run_id,
        "clean_filename": clean_filename,
        "preview_url": preview_b64,
        "resolution": "28 × 28 (Grayscale, Normalized to [-1, 1])",
        "message": "Image uploaded, converted to grayscale, and scaled to 28x28 input dimension."
    })


@app.route("/api/forward-diffuse", methods=["POST"])
def api_forward_diffuse():
    """
    Adds Gaussian noise to clean image using the forward diffusion equation:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    data = request.get_json() or {}
    clean_filename = data.get("clean_filename")
    timestep = int(data.get("timestep", 150))

    if not clean_filename:
        return jsonify({"error": "clean_filename is required"}), 400

    clean_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(clean_filename))
    if not os.path.exists(clean_path):
        return jsonify({"error": "Specified image file does not exist on server."}), 404

    # Bound timestep
    t = max(0, min(timestep, TIMESTEPS - 1))

    try:
        clean_img = Image.open(clean_path)
        x_0 = pil_to_tensor(clean_img, device=DEVICE)

        # Forward diffusion sampling
        noise = torch.randn_like(x_0)
        t_tensor = torch.tensor([t], device=DEVICE, dtype=torch.long)
        x_t = diffusion.q_sample(x_0, t_tensor, noise)

        noisy_img = tensor_to_pil(x_t)

        run_id = clean_filename.replace("clean_", "").replace(".png", "")
        noisy_filename = f"noisy_{run_id}_t{t}.png"
        noisy_path = os.path.join(app.config["OUTPUT_FOLDER"], noisy_filename)
        noisy_img.save(noisy_path)

        math_info = diffusion.get_noise_parameters(t)

        return jsonify({
            "success": True,
            "timestep": t,
            "noisy_filename": noisy_filename,
            "noisy_preview_url": pil_to_base64(noisy_img),
            "math_info": math_info
        })
    except Exception as e:
        return jsonify({"error": f"Forward diffusion failed: {str(e)}"}), 500


@app.route("/api/reverse-diffuse", methods=["POST"])
def api_reverse_diffuse():
    """
    Performs true iterative reverse diffusion using the trained PyTorch UNet model.
    Denoises x_t step by step down to x_0.
    """
    global model
    if not model_metadata["loaded"] or model is None:
        if not load_trained_model():
            return jsonify({
                "error": "Trained model checkpoint not found! Please train the model first by executing 'python train.py'."
            }), 400

    data = request.get_json() or {}
    clean_filename = data.get("clean_filename")
    noisy_filename = data.get("noisy_filename")
    timestep = int(data.get("timestep", 150))

    if not clean_filename or not noisy_filename:
        return jsonify({"error": "clean_filename and noisy_filename are required"}), 400

    clean_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(clean_filename))
    noisy_path = os.path.join(app.config["OUTPUT_FOLDER"], secure_filename(noisy_filename))

    if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
        return jsonify({"error": "Image files not found on server."}), 404

    try:
        clean_img = Image.open(clean_path)
        noisy_img = Image.open(noisy_path)

        x_0_tensor = pil_to_tensor(clean_img, device=DEVICE)
        x_t_tensor = pil_to_tensor(noisy_img, device=DEVICE)

        t = max(0, min(timestep, TIMESTEPS - 1))

        # Perform reverse diffusion with high-precision timing
        start_time = time.perf_counter()
        result = diffusion.reverse_diffusion(model, x_t_tensor, start_t=t, num_snapshots=5)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        denoised_tensor = result["final_tensor"]
        denoised_img = tensor_to_pil(denoised_tensor)

        run_id = clean_filename.replace("clean_", "").replace(".png", "")
        denoised_filename = f"denoised_{run_id}_t{t}.png"
        denoised_path = os.path.join(app.config["OUTPUT_FOLDER"], denoised_filename)
        denoised_img.save(denoised_path)

        # Generate snapshot thumbnails along reverse diffusion path
        snapshots_data = []
        for step, snap_tensor in result["snapshots"]:
            snap_img = tensor_to_pil(snap_tensor)
            snapshots_data.append({
                "step": step,
                "preview_url": pil_to_base64(snap_img)
            })

        # Calculate quantitative evaluation metrics
        metrics = calculate_metrics(x_0_tensor, denoised_tensor)
        metrics["inference_time_ms"] = inference_time_ms

        run_record = {
            "run_id": run_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "timestep": t,
            "clean_url": pil_to_base64(clean_img),
            "noisy_url": pil_to_base64(noisy_img),
            "denoised_url": pil_to_base64(denoised_img),
            "clean_file": clean_filename,
            "noisy_file": noisy_filename,
            "denoised_file": denoised_filename,
            "metrics": metrics,
            "snapshots": snapshots_data
        }

        # Keep recent runs
        recent_runs.insert(0, run_record)
        if len(recent_runs) > 10:
            recent_runs.pop()

        return jsonify({
            "success": True,
            "denoised_filename": denoised_filename,
            "denoised_preview_url": pil_to_base64(denoised_img),
            "metrics": metrics,
            "snapshots": snapshots_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Reverse diffusion failed: {str(e)}"}), 500


@app.route("/download/<folder>/<filename>")
def download_file(folder: str, filename: str):
    """Secure endpoint for downloading images."""
    if folder == "uploads":
        directory = app.config["UPLOAD_FOLDER"]
    elif folder == "outputs":
        directory = app.config["OUTPUT_FOLDER"]
    else:
        return "Invalid directory requested", 400

    return send_from_directory(directory, secure_filename(filename), as_attachment=True)


if __name__ == "__main__":
    print("[*] Starting Diffusion Model Web Application...")
    app.run(host="127.0.0.1", port=5000, debug=True)
