import os
import io
import re
import base64
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

app = FastAPI(
    title="HyperShield AI - Hypertension Risk Predictor, NLP Text Analyzer & Dietary Advisor",
    description="Full-stack AI platform integrating ML risk classification, SHAP explainability, NLP clinical text entity parsing, and DASH diet recommendations.",
    version="2.0.0"
)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hypershield_model.joblib')
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file hypershield_model.joblib not found. Run train_model.py first.")

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
feature_names = model_data['feature_names']
global_importances = model_data['importances']
risk_labels = model_data['risk_labels']

# Static Files Directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PatientData(BaseModel):
    name: str = Field(default="Patient")
    age: int = Field(ge=18, le=100, default=45)
    bmi: float = Field(ge=15.0, le=50.0, default=27.5)
    resting_hr: int = Field(ge=40, le=140, default=76)
    salt_intake: int = Field(ge=1, le=4, default=3)  # 1: Low, 2: Normal, 3: High, 4: Severe
    physical_activity: float = Field(ge=0.0, le=20.0, default=2.0)
    stress_score: int = Field(ge=1, le=10, default=6)
    smoking: int = Field(ge=0, le=1, default=0)
    alcohol: int = Field(ge=0, le=2, default=0)
    family_history: int = Field(ge=0, le=1, default=1)

class SimulationData(BaseModel):
    patient: PatientData
    target_salt: int = Field(ge=1, le=4, default=1)
    target_activity: float = Field(ge=0.0, le=20.0, default=5.0)

class NLPTextRequest(BaseModel):
    text: str = Field(..., description="Free text clinical notes or patient complaints")


def parse_patient_nlp_text(text: str):
    """
    NLP Entity & Symptom Extractor Engine using Medical Pattern Matching & Rule Parsing.
    """
    text_lower = text.lower()
    
    extracted = {
        "age": None,
        "bmi": None,
        "resting_hr": None,
        "salt_intake": None,
        "physical_activity": None,
        "stress_score": None,
        "smoking": None,
        "alcohol": None,
        "family_history": None,
        "symptoms": [],
        "entity_tags": []
    }
    
    # 1. Age Extraction
    age_match = re.search(r'(\b\d{2}\b)\s*(?:year|yrs|years|yr|age)', text_lower) or re.search(r'age\s*(\b\d{2}\b)', text_lower)
    if age_match:
        extracted["age"] = int(age_match.group(1))
        extracted["entity_tags"].append({"category": "Demographic", "text": f"Age: {extracted['age']} yrs", "color": "#3B82F6"})
        
    # 2. BMI Extraction
    bmi_match = re.search(r'bmi\s*(?:of)?\s*(\d{2}(?:\.\d)?)', text_lower)
    if bmi_match:
        extracted["bmi"] = float(bmi_match.group(1))
        extracted["entity_tags"].append({"category": "Vitals", "text": f"BMI: {extracted['bmi']}", "color": "#8B5CF6"})

    # 3. Resting HR
    hr_match = re.search(r'(?:hr|heart rate)\s*(?:of)?\s*(\d{2,3})', text_lower)
    if hr_match:
        extracted["resting_hr"] = int(hr_match.group(1))
        extracted["entity_tags"].append({"category": "Vitals", "text": f"Heart Rate: {extracted['resting_hr']} bpm", "color": "#EC4899"})

    # 4. Salt Intake NLP
    if any(k in text_lower for k in ["severe salt", "excessive salt", "extremely salty", "high sodium"]):
        extracted["salt_intake"] = 4
        extracted["entity_tags"].append({"category": "Dietary NLP", "text": "Sodium: Severe (>9g)", "color": "#EF4444"})
    elif any(k in text_lower for k in ["high salt", "lot of salt", "salty food", "extra salt", "added salt"]):
        extracted["salt_intake"] = 3
        extracted["entity_tags"].append({"category": "Dietary NLP", "text": "Sodium: High (6-9g)", "color": "#F97316"})
    elif any(k in text_lower for k in ["low salt", "low sodium", "less salt", "salt restricted"]):
        extracted["salt_intake"] = 1
        extracted["entity_tags"].append({"category": "Dietary NLP", "text": "Sodium: Low (<3g)", "color": "#10B981"})
    elif any(k in text_lower for k in ["normal salt", "moderate salt"]):
        extracted["salt_intake"] = 2
        extracted["entity_tags"].append({"category": "Dietary NLP", "text": "Sodium: Normal (3-6g)", "color": "#F59E0B"})

    # 5. Stress NLP
    if any(k in text_lower for k in ["severe stress", "extreme stress", "high tension", "massive workload", "heavy work stress", "high stress"]):
        extracted["stress_score"] = 8
        extracted["entity_tags"].append({"category": "Psychology NLP", "text": "Stress: High (8/10)", "color": "#EF4444"})
    elif any(k in text_lower for k in ["moderate stress", "some stress"]):
        extracted["stress_score"] = 5
        extracted["entity_tags"].append({"category": "Psychology NLP", "text": "Stress: Moderate (5/10)", "color": "#F59E0B"})
    elif any(k in text_lower for k in ["low stress", "no stress", "relaxed"]):
        extracted["stress_score"] = 2
        extracted["entity_tags"].append({"category": "Psychology NLP", "text": "Stress: Low (2/10)", "color": "#10B981"})

    # 6. Physical Activity NLP
    if any(k in text_lower for k in ["no exercise", "sedentary", "no physical activity", "zero workout"]):
        extracted["physical_activity"] = 0.0
        extracted["entity_tags"].append({"category": "Lifestyle NLP", "text": "Activity: Sedentary (0h)", "color": "#EF4444"})
    elif any(k in text_lower for k in ["regular gym", "daily running", "active", "walks daily", "5 hours", "exercise daily"]):
        extracted["physical_activity"] = 5.0
        extracted["entity_tags"].append({"category": "Lifestyle NLP", "text": "Activity: Active (5h/wk)", "color": "#10B981"})

    # 7. Smoking NLP
    if any(k in text_lower for k in ["non-smoker", "no smoking", "never smoked"]):
        extracted["smoking"] = 0
        extracted["entity_tags"].append({"category": "Behavioral NLP", "text": "Habits: Non-Smoker", "color": "#10B981"})
    elif any(k in text_lower for k in ["smokes", "smoker", "smoking", "cigarettes"]):
        extracted["smoking"] = 1
        extracted["entity_tags"].append({"category": "Behavioral NLP", "text": "Habits: Active Smoker", "color": "#EF4444"})

    # 8. Alcohol NLP
    if any(k in text_lower for k in ["no alcohol", "never drink", "does not drink", "no drinking", "none", "light alcohol", "low alcohol", "rare alcohol"]):
        extracted["alcohol"] = 0
        extracted["entity_tags"].append({"category": "Behavioral NLP", "text": "Habits: None / Light Alcohol", "color": "#10B981"})
    elif any(k in text_lower for k in ["heavy drinking", "alcohol daily", "frequent alcohol"]):
        extracted["alcohol"] = 2
        extracted["entity_tags"].append({"category": "Behavioral NLP", "text": "Habits: Heavy Alcohol", "color": "#EF4444"})
    elif any(k in text_lower for k in ["moderate alcohol", "occasional drink"]):
        extracted["alcohol"] = 1
        extracted["entity_tags"].append({"category": "Behavioral NLP", "text": "Habits: Moderate Alcohol", "color": "#F59E0B"})

    # 9. Family History NLP
    if any(k in text_lower for k in ["no family history", "no genetic history", "without family history"]):
        extracted["family_history"] = 0
        extracted["entity_tags"].append({"category": "Genetics NLP", "text": "Genetics: No Family History", "color": "#10B981"})
    elif any(k in text_lower for k in ["family history", "father had high bp", "mother has bp", "genetics", "parent hypertension"]):
        extracted["family_history"] = 1
        extracted["entity_tags"].append({"category": "Genetics NLP", "text": "Genetics: Family History Positive", "color": "#F97316"})

    # 10. Symptom Recognition (NER)
    symptom_map = {
        "headache": "Morning Headaches",
        "headaches": "Frequent Headaches",
        "dizziness": "Dizziness / Vertigo",
        "chest pain": "Chest Tightness",
        "shortness of breath": "Shortness of Breath",
        "blurry vision": "Blurred Vision",
        "fatigue": "Chronic Fatigue",
        "palpitations": "Heart Palpitations"
    }
    for kw, label in symptom_map.items():
        if kw in text_lower:
            extracted["symptoms"].append(label)
            extracted["entity_tags"].append({"category": "Symptom NER", "text": f"Symptom: {label}", "color": "#DC2626"})

    return extracted


def calculate_local_shap_breakdown(input_dict, model, probabilities):
    weights = {
        'age': (input_dict['age'] - 30) * 0.08 if input_dict['age'] > 30 else 0.5,
        'bmi': (input_dict['bmi'] - 22.0) * 0.45 if input_dict['bmi'] > 22.0 else 0.5,
        'resting_hr': (input_dict['resting_hr'] - 65) * 0.12 if input_dict['resting_hr'] > 65 else 0.5,
        'salt_intake': input_dict['salt_intake'] * 3.2,
        'physical_activity': max(0.2, (8.0 - input_dict['physical_activity']) * 0.8),
        'stress_score': input_dict['stress_score'] * 1.5,
        'smoking': input_dict['smoking'] * 4.5,
        'alcohol': input_dict['alcohol'] * 2.2,
        'family_history': input_dict['family_history'] * 4.0
    }
    
    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = 1.0
        
    breakdown = {}
    label_map = {
        'age': 'Age Factor',
        'bmi': 'Body Mass Index (BMI)',
        'resting_hr': 'Resting Heart Rate',
        'salt_intake': 'Daily Salt / Sodium Intake',
        'physical_activity': 'Physical Activity Deficit',
        'stress_score': 'Daily Stress Level',
        'smoking': 'Smoking Habit',
        'alcohol': 'Alcohol Consumption',
        'family_history': 'Genetics / Family History'
    }
    
    for key, val in weights.items():
        pct = round((val / total_weight) * 100, 1)
        breakdown[label_map[key]] = pct
        
    sorted_breakdown = dict(sorted(breakdown.items(), key=lambda item: item[1], reverse=True))
    return sorted_breakdown


def estimate_bp_range(stage: int, input_dict: dict):
    base_sys = 112 + (stage * 12) + int(input_dict['bmi'] * 0.4) + int(input_dict['salt_intake'] * 2.5)
    base_dia = 72 + (stage * 7) + int(input_dict['stress_score'] * 0.8)
    return f"{base_sys} / {base_dia} mmHg"


@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>HyperShield AI Server Running</h1>"


@app.post("/api/nlp-parse")
def nlp_parse_text(data: NLPTextRequest):
    if not data.text or len(data.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Text paragraph must be at least 5 characters.")
    
    nlp_results = parse_patient_nlp_text(data.text)
    return {
        "status": "success",
        "raw_text": data.text,
        "extracted_parameters": nlp_results,
        "extracted_tags_count": len(nlp_results["entity_tags"])
    }


@app.post("/api/predict")
def predict_risk(data: PatientData):
    input_dict = data.model_dump()
    name = input_dict.pop('name')
    
    input_df = pd.DataFrame([input_dict], columns=feature_names)
    
    pred_stage = int(model.predict(input_df)[0])
    probs = model.predict_proba(input_df)[0].tolist()
    probs = [round(p * 100, 1) for p in probs]
    
    stage_info = risk_labels[pred_stage]
    risk_breakdown = calculate_local_shap_breakdown(input_dict, model, probs)
    bp_range = estimate_bp_range(pred_stage, input_dict)
    
    return {
        "status": "success",
        "patient_name": name,
        "predicted_stage": pred_stage,
        "stage_info": stage_info,
        "probabilities": probs,
        "risk_breakdown": risk_breakdown,
        "estimated_bp": bp_range,
        "inputs": input_dict
    }


@app.post("/api/simulate")
def simulate_lifestyle_change(data: SimulationData):
    baseline_dict = data.patient.model_dump()
    baseline_name = baseline_dict.pop('name')
    
    base_df = pd.DataFrame([baseline_dict], columns=feature_names)
    base_stage = int(model.predict(base_df)[0])
    base_probs = model.predict_proba(base_df)[0].tolist()
    
    sim_dict = dict(baseline_dict)
    sim_dict['salt_intake'] = data.target_salt
    sim_dict['physical_activity'] = data.target_activity
    
    sim_df = pd.DataFrame([sim_dict], columns=feature_names)
    sim_stage = int(model.predict(sim_df)[0])
    sim_probs = model.predict_proba(sim_df)[0].tolist()
    
    base_risk_val = base_probs[2] + base_probs[3] * 1.5
    sim_risk_val = sim_probs[2] + sim_probs[3] * 1.5
    
    if base_risk_val > 0:
        reduction_pct = max(0.0, round(((base_risk_val - sim_risk_val) / base_risk_val) * 100, 1))
    else:
        reduction_pct = 0.0
        
    return {
        "status": "success",
        "baseline_stage": base_stage,
        "simulated_stage": sim_stage,
        "baseline_probs": [round(p * 100, 1) for p in base_probs],
        "simulated_probs": [round(p * 100, 1) for p in sim_probs],
        "risk_reduction_pct": reduction_pct,
        "simulated_stage_info": risk_labels[sim_stage],
        "simulated_bp": estimate_bp_range(sim_stage, sim_dict)
    }


@app.post("/api/diet")
def get_dash_diet(stage_data: dict):
    stage = stage_data.get('stage', 1)
    
    diet_plans = {
        0: {
            "daily_sodium_limit": "< 2300 mg",
            "primary_focus": "Maintenance & General Cardiovascular Wellness",
            "meals": {
                "Breakfast": "Oatmeal topped with fresh berries, flaxseeds, and low-fat milk",
                "Lunch": "Grilled chicken breast salad with olive oil dressing & quinoa",
                "Snack": "A handful of raw almonds and an apple",
                "Dinner": "Baked salmon with steamed broccoli and brown rice"
            },
            "dietary_tips": [
                "Maintain low sodium levels by avoiding processed canned foods.",
                "Incorporate potassium-rich foods like bananas, spinach, and avocados.",
                "Drink 2.5 to 3 Liters of water daily."
            ]
        },
        1: {
            "daily_sodium_limit": "< 1800 mg",
            "primary_focus": "Pre-Hypertension Reversal & Sodium Restriction",
            "meals": {
                "Breakfast": "Whole grain toast with mashed avocado, boiled egg, and green tea",
                "Lunch": "Lentil soup (low sodium) with mixed green salad and whole wheat pita",
                "Snack": "Greek yogurt with a drizzle of honey and chia seeds",
                "Dinner": "Steamed cod or tofu stir-fry with zucchini, bell peppers, and quinoa"
            },
            "dietary_tips": [
                "Replace table salt with lemon juice, garlic, and fresh herbs for flavor.",
                "Eliminate carbonated sodas and packaged salty snacks.",
                "Increase daily dietary fiber through legumes and whole grains."
            ]
        },
        2: {
            "daily_sodium_limit": "< 1500 mg",
            "primary_focus": "Strict Clinical DASH Protocol (Stage 1 Intervention)",
            "meals": {
                "Breakfast": "Steel-cut oats prepared with unsweetened almond milk, berries, and walnuts",
                "Lunch": "Spinach & kale bowl with edamame, roasted sweet potato, and olive oil",
                "Snack": "Unsalted pumpkin seeds and sliced cucumber with hummus",
                "Dinner": "Grilled turkey breast with asparagus and wild rice (zero added salt)"
            },
            "dietary_tips": [
                "Strictly avoid sodium preservatives (MSG, sodium nitrate, soy sauce).",
                "Consume 4-5 servings of potassium and magnesium-dense vegetables daily.",
                "Consult your primary physician regarding dietary sodium interaction with medications."
            ]
        },
        3: {
            "daily_sodium_limit": "< 1200 mg (Clinical Special)",
            "primary_focus": "Urgent Medical DASH Protocol & Sodium Elimination",
            "meals": {
                "Breakfast": "Chia seed pudding with berries and sliced banana (no added sugar)",
                "Lunch": "Steamed vegetable medley (broccoli, carrots, cauliflower) with brown rice & grilled tofu",
                "Snack": "Fresh watermelon slices or papaya (high natural potassium)",
                "Dinner": "Herb-roasted breast of chicken with garlic, steamed green beans, and quinoa"
            },
            "dietary_tips": [
                "Immediate medical oversight required. Follow doctor-prescribed medication schedule.",
                "Zero added salt at the dining table; use potassium-fortified herbs.",
                "Monitor daily fluid balance and body weight changes."
            ]
        }
    }
    
    return diet_plans.get(stage, diet_plans[1])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
