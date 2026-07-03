import os
import zipfile
import tempfile
import torch
import torch.nn as nn
import numpy as np
import pydicom
import pickle
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- GEMINI CONFIGURATION ---
# Professional Tip: Ensure this key is active in Google AI Studio
genai.configure(api_key="")
llm_model = genai.GenerativeModel('gemini-2.5-flash')

def generate_medical_narrative(data):
    # Adjust tone based on severity
    support_tone = "urgent and professional" if data['result'] == "MALIGNANT" else "brief and reassuring"
    
    prompt = f"""
    Context: Lung CT analysis for a {data['age_raw']}-year-old {data['gender_str']} ({data['smoking_str']}).
    Result: {data['result']} ({data['probability']}% confidence). Staging: {data['stage']}.
    
    Task: Write a concise clinical note in 3 short paragraphs:
    1. **Interpretation**: 1-2 sentences summarizing the findings objectively.
    2. **Recommended Action**: 3 bullet points of standard clinical next steps. 
    3. **Patient Note**: 1 short, {support_tone} sentence of support.

    Rules:
    - DO NOT use numbered headers (e.g., no "1.", "2.").
    - DO NOT use the word "Pathway" or "Support" in the headers.
    - Keep the total length under 100 words.
    - Professional, clinical tone.
    """
    try:
        # Note: If gemini-2.5-flash gives a 404, use 'gemini-1.5-flash'
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"--- LLM Error: {e} ---")
        return "Clinical interpretation currently unavailable."

# --- MODEL ARCHITECTURE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PulmoNetResNet(nn.Module):
    def __init__(self, num_classes=2, embedding_dim=128):
        super(PulmoNetResNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding_layer = nn.Sequential(
            nn.Linear(num_ftrs, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding_layer(features)
        logits = self.classifier(embedding)
        return logits, embedding

# Load Weights
cnn_model = PulmoNetResNet(num_classes=2).to(device)
cnn_model.load_state_dict(torch.load('best_pulmonet_standalone.pth', map_location=device))
cnn_model.eval()

# Load Fusion & Stats Models
with open('final_xgb_pca_model.pkl', 'rb') as f:
    xgb_fusion = pickle.load(f)
with open('final_xgb_survival_model.pkl', 'rb') as f:
    survival_model = pickle.load(f)
with open('arima_scaler.pkl', 'rb') as f:
    arima_scaler = pickle.load(f)
with open('arima_pca.pkl', 'rb') as f:
    arima_pca = pickle.load(f)

feature_columns = [f'Temporal_Growth_Vector_{i+1}' for i in range(10)] + ['age', 'gender', 'cigsmok']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def process_dicom_zip(zip_filepath):
    extracted_images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        dicom_files = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, f))
        
        slices_info = []
        for f in dicom_files:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            z_pos = float(ds.ImagePositionPatient[2]) if 'ImagePositionPatient' in ds else 0.0
            slices_info.append((f, z_pos))
        
        slices_info.sort(key=lambda x: x[1])
        selected = [s[0] for s in slices_info[int(len(slices_info)*0.15):int(len(slices_info)*0.85)]]
        
        for f in selected:
            ds = pydicom.dcmread(f)
            hu = ds.pixel_array * ds.get('RescaleSlope', 1) + ds.get('RescaleIntercept', 0)
            windowed = np.clip(hu, -1000, 400)
            normalized = (windowed + 1000) / 1400.0
            pil_img = Image.fromarray(np.uint8(normalized * 255)).convert('RGB')
            extracted_images.append(pil_img)
            
    return extracted_images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        age_val = float(request.form.get('age', 60))
        gender_val = float(request.form.get('gender', 1)) 
        smoking_val = float(request.form.get('smoking', 1)) 
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 1. Visual Feature Extraction
        pil_images = process_dicom_zip(filepath)
        tensors = torch.stack([transform(img) for img in pil_images]).to(device)
        
        with torch.no_grad():
            logits, embeddings = cnn_model(tensors)
            malignant_probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            worst_idx = torch.argmax(malignant_probs).item()
            visual_features = torch.max(embeddings, dim=0)[0].cpu().numpy()

        # 2. PCA Vector Simulation
        simulated_arima = []
        for val in visual_features:
            simulated_arima.append(val)  
            simulated_arima.append(0.0)  
            
        simulated_arima = np.array([simulated_arima])
        scaled_arima = arima_scaler.transform(simulated_arima)
        pca_vectors = arima_pca.transform(scaled_arima)[0]

        # 3. Multimodal Fusion
        input_data = {col: 0.0 for col in feature_columns}
        norm_age = (age_val - 54) / (74 - 54)
        input_data['age'] = float(np.clip(norm_age, 0.0, 1.0))
        input_data['gender'] = gender_val
        input_data['cigsmok'] = 1.0 if int(smoking_val) == 1 else 0.0
        
        for i, val in enumerate(pca_vectors):
            feat_name = f'Temporal_Growth_Vector_{i+1}'
            if feat_name in input_data:
                input_data[feat_name] = float(val)
            
        final_df = pd.DataFrame([input_data])[feature_columns]
        probs = xgb_fusion.predict_proba(final_df)[0]
        final_prob = float(probs[1]) * 100 
        final_result = "MALIGNANT" if final_prob >= 50.0 else "BENIGN"

        # 4. Prognosis & Staging
        if final_result == "MALIGNANT":
            surv_pred = survival_model.predict(final_df)[0]
            predicted_days = int(surv_pred * 2983)
            survival_display = str(max(0, predicted_days))
            estimated_stage = "Advanced (III/IV)" if final_prob > 85.0 else "Early (I/II)"
            treatment = "Immediate Oncology Referral"
        else:
            estimated_stage = "Normal / Benign"
            treatment = "Routine Annual Screening"
            survival_display = "N/A"

        # 5. LLM Synthesis
        narrative_payload = {
            'age_raw': age_val,
            'gender_str': 'Male' if int(gender_val) == 1 else 'Female',
            'smoking_str': 'Active' if int(smoking_val) == 1 else 'Non-Smoker',
            'result': final_result,
            'probability': round(float(final_prob), 1),
            'stage': estimated_stage,
            'survival_days': survival_display
        }
        llm_narrative = generate_medical_narrative(narrative_payload)

        # Image Display Logic
        best_img = pil_images[worst_idx]
        display_path = filepath.replace('.zip', '_analysis.png')
        best_img.save(display_path)

        return jsonify({
            'result': final_result,
            'probability': round(float(final_prob), 1),
            'image_url': f'/static/uploads/{os.path.basename(display_path)}',
            'treatment': treatment,
            'survival_days': survival_display,
            'stage': estimated_stage,
            'llm_report': llm_narrative 
        })

    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)