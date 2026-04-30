from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import logging
import os
import sys
import subprocess

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model
def load_or_train_model():
    try:
        if not os.path.exists("model.pkl"):
            raise FileNotFoundError("model.pkl not found")

        model = pickle.load(open("model.pkl", "rb"))

        # Fix sklearn compatibility issue
        if not hasattr(model, "multi_class"):
            model.multi_class = "auto"

        return model

    except Exception as e:
        logger.warning(f"Model load failed: {e} — retraining model...")

        try:
            subprocess.run([sys.executable, "train_model.py"], check=True)

            model = pickle.load(open("model.pkl", "rb"))

            if not hasattr(model, "multi_class"):
                model.multi_class = "auto"

            logger.info("Model retrained successfully")
            return model

        except Exception as retrain_error:
            logger.error(f"Retraining failed: {retrain_error}")

            # FINAL FALLBACK MODEL
            logger.warning("Using fallback dummy model")

            from sklearn.linear_model import LogisticRegression

            fallback_model = LogisticRegression()
            fallback_model.fit(
                [[30, 22, 120, 80, 1, 1],
                 [60, 30, 150, 95, 3, 0]],
                [0, 1]
            )

            return fallback_model


model = load_or_train_model()

# Set up SHAP explanations
SHAP_AVAILABLE = False
explainer      = None

try:
    import shap

    try:
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP: TreeExplainer initialised.")
    except Exception:
        # Background profiles for SHAP fallback
        _background = np.array([
            [30, 22.0, 110, 70, 1, 1],
            [40, 24.5, 118, 76, 1, 1],
            [45, 26.0, 128, 84, 2, 1],
            [55, 29.5, 138, 88, 2, 0],
            [62, 33.0, 148, 94, 3, 0],
            [58, 31.0, 155, 98, 3, 0],
        ])
        explainer = shap.KernelExplainer(model.predict_proba, _background)
        logger.info("SHAP: KernelExplainer initialised (fallback).")

    SHAP_AVAILABLE = True

except ImportError:
    logger.warning(
        "SHAP not installed. Run: pip install shap\n"
        "Using rule-based factor explanations until then."
    )
except Exception as exc:
    logger.warning(f"SHAP explainer init failed ({exc}). Using rule-based fallback.")


# Labels and advice text
FEATURE_META = {
    "age": {
        "label": "Age",
        "risk_advice": (
            "Age {val:.0f} yrs is contributing to elevated cardiovascular risk. "
            "Regular screening and preventive care are essential at this age."
        ),
        "safe_advice": (
            "Your age ({val:.0f} yrs) is currently a low cardiovascular risk factor. "
            "Maintain healthy habits to preserve this advantage long-term."
        ),
    },
    "bmi": {
        "label": "BMI",
        "risk_advice": (
            "BMI {val:.1f} is increasing your cardiovascular risk. "
            "A 5–10% weight reduction can meaningfully lower this risk."
        ),
        "safe_advice": (
            "BMI {val:.1f} is within a healthy range and is a protective factor. "
            "Maintain a balanced diet and regular physical activity."
        ),
    },
    "systolic": {
        "label": "Systolic BP",
        "risk_advice": (
            "Systolic BP of {val:.0f} mmHg is elevating your risk. "
            "Limit sodium to <2 g/day and discuss management with your physician."
        ),
        "safe_advice": (
            "Systolic BP of {val:.0f} mmHg is within a healthy range "
            "and is reducing your overall cardiovascular risk."
        ),
    },
    "diastolic": {
        "label": "Diastolic BP",
        "risk_advice": (
            "Diastolic BP of {val:.0f} mmHg is contributing to elevated risk. "
            "Regular monitoring and lifestyle changes are advised."
        ),
        "safe_advice": (
            "Diastolic BP of {val:.0f} mmHg is in a healthy range "
            "and is a protective factor in your current profile."
        ),
    },
    "cholesterol": {
        "label": "Cholesterol",
        "risk_advice": (
            "Elevated cholesterol (level {val:.0f}/3) is raising your cardiovascular risk. "
            "Reduce saturated fats, increase dietary fiber, and discuss statins with your doctor."
        ),
        "safe_advice": (
            "Your cholesterol level ({val:.0f}/3 = Normal) is a positive factor. "
            "Maintain a heart-healthy diet to keep it in this range."
        ),
    },
    "active": {
        "label": "Physical Activity",
        "risk_advice": (
            "Physical inactivity is significantly increasing your cardiovascular risk. "
            "Aim for at least 150 min of moderate-intensity exercise per week."
        ),
        "safe_advice": (
            "Your active lifestyle is a strong protective cardiovascular factor. "
            "Consistent exercise makes a measurable difference in long-term risk."
        ),
    },
}

# Model feature order
FEATURE_ORDER = ["age", "bmi", "systolic", "diastolic", "cholesterol", "active"]


# Read SHAP output safely
def _extract_shap_vector(sv, n_features: int) -> np.ndarray:
    """Return SHAP values for the risk class."""
    if isinstance(sv, list):
        return np.array(sv[1][0])

    sv = np.array(sv)

    if sv.ndim == 3:
        return sv[0, :n_features, 1]

    if sv.ndim == 2:
        return sv[0, :n_features]

    return sv[:n_features]


# Get clinical risk level using thresholds
def get_clinical_risk_level(feature, value):
    """Return clinical severity for a feature using fixed medical thresholds."""
    if feature == "age":
        if value < 30:
            return "Low"
        if value <= 50:
            return "Moderate"
        return "High"

    if feature == "bmi":
        if value < 18.5:
            return "Moderate"
        if value <= 25:
            return "Low"
        if value <= 30:
            return "Moderate"
        return "High"

    if feature == "systolic":
        if value < 120:
            return "Low"
        if value <= 139:
            return "Moderate"
        if value <= 179:
            return "High"
        return "Critical"

    if feature == "diastolic":
        if value < 80:
            return "Low"
        if value <= 89:
            return "Moderate"
        if value <= 119:
            return "High"
        return "Critical"

    if feature == "cholesterol":
        if int(value) == 1:
            return "Low"
        if int(value) == 2:
            return "Moderate"
        return "High"

    if feature == "active":
        return "Low" if int(value) == 1 else "High"

    return "Low"


# Convert SHAP values into UI factors
def shap_to_factors(input_array: np.ndarray, raw_shap: np.ndarray) -> list:
    """Build the factor list from SHAP values."""
    feature_vals = input_array[0]
    abs_vals     = np.abs(raw_shap)
    max_abs      = abs_vals.max()

    if max_abs == 0:
        scores = np.zeros(len(abs_vals))
    else:
        scores = (abs_vals / max_abs) * 100.0

    risk_weights = {
        "Low": 0.3,
        "Moderate": 0.6,
        "High": 0.85,
        "Critical": 1.0,
    }

    clinical_context = {
        "age": lambda v: f"Age {v:.0f} yrs",
        "bmi": lambda v: f"BMI {v:.1f}",
        "systolic": lambda v: f"Systolic BP {v:.0f} mmHg",
        "diastolic": lambda v: f"Diastolic BP {v:.0f} mmHg",
        "cholesterol": lambda v: f"Cholesterol level {int(v)}/3",
        "active": lambda v: "Active lifestyle" if int(v) == 1 else "Physical inactivity",
    }

    clinical_advice = {
        "Low": {
            "risk": (
                "{context} is clinically Low. It remains a normal finding, though the "
                "model sees a small upward contribution in this profile."
            ),
            "protective": (
                "{context} is clinically Low and is acting protectively in this profile."
            ),
            "neutral": (
                "{context} is clinically Low with no meaningful directional model contribution."
            ),
        },
        "Moderate": {
            "risk": (
                "{context} is clinically Moderate and is pushing risk upward. "
                "Address it with steady lifestyle improvement and routine monitoring."
            ),
            "protective": (
                "{context} is clinically Moderate. The model estimates a downward "
                "risk contribution here, but the clinical finding still deserves monitoring."
            ),
            "neutral": (
                "{context} is clinically Moderate with no meaningful directional model contribution. "
                "The clinical finding still deserves monitoring."
            ),
        },
        "High": {
            "risk": (
                "{context} is clinically High and is increasing risk. "
                "Prioritize medical review and targeted risk reduction."
            ),
            "protective": (
                "{context} is clinically High. The model estimates a downward "
                "risk contribution here, but the clinical severity remains High."
            ),
            "neutral": (
                "{context} is clinically High with no meaningful directional model contribution. "
                "The clinical severity remains High."
            ),
        },
        "Critical": {
            "risk": (
                "{context} is clinically Critical and requires prompt medical attention."
            ),
            "protective": (
                "{context} is clinically Critical. Even with a downward model contribution, "
                "clinical thresholds require urgent attention."
            ),
            "neutral": (
                "{context} is clinically Critical. Even without a directional model contribution, "
                "clinical thresholds require urgent attention."
            ),
        },
    }

    factors = []
    for i, feat_key in enumerate(FEATURE_ORDER):
        meta   = FEATURE_META[feat_key]
        val    = feature_vals[i]
        level  = get_clinical_risk_level(feat_key, val)
        base_score = float(scores[i])
        score  = int(round(base_score * risk_weights[level]))

        shap_value = float(raw_shap[i])
        if shap_value > 0:
            direction = "risk"
        elif shap_value < 0:
            direction = "protective"
        else:
            direction = "neutral"
        context = clinical_context[feat_key](float(val))
        advice = clinical_advice[level][direction].format(context=context)

        factors.append({
            "name":   meta["label"],
            "level":  level,
            "score":  score,
            "advice": advice,
        })

    factors.sort(key=lambda f: f["score"], reverse=True)
    return factors


def get_shap_factors(input_array: np.ndarray) -> list:
    """Compute SHAP values and return formatted factors. Raises on failure."""
    sv        = explainer.shap_values(input_array)
    raw_shap  = _extract_shap_vector(sv, len(FEATURE_ORDER))
    return shap_to_factors(input_array, raw_shap)


def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


def get_bp_category(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or (80 <= diastolic < 90):
        return "Stage 1 Hypertension"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension"
    else:
        return "Hypertensive Crisis"


# Fallback factors when SHAP is unavailable
def get_rule_based_factors(age, bmi, systolic, diastolic, cholesterol, active):
    factors = []

    if age > 60:
        factors.append({"name": "Age", "level": "High", "score": 80,
                         "advice": f"Age {int(age)} — Regular cardiovascular screening essential above 60"})
    elif age > 45:
        factors.append({"name": "Age", "level": "Moderate", "score": 48,
                         "advice": f"Age {int(age)} — Annual health check-up advised; risk rises with age"})
    else:
        factors.append({"name": "Age", "level": "Low", "score": 18,
                         "advice": f"Age {int(age)} — Low age-related risk; maintain healthy habits now"})

    bmi_cat = get_bmi_category(bmi)
    bmi_score_map  = {"Underweight": 30, "Normal": 10, "Overweight": 52, "Obese": 86}
    bmi_level_map  = {"Underweight": "Moderate", "Normal": "Low", "Overweight": "Moderate", "Obese": "High"}
    bmi_advice_map = {
        "Underweight": f"BMI {bmi:.1f} (Underweight) — Low body weight can indicate nutritional deficiency",
        "Normal":      f"BMI {bmi:.1f} (Normal) — Healthy weight range; maintain diet and exercise",
        "Overweight":  f"BMI {bmi:.1f} (Overweight) — 5–10% weight reduction can significantly lower risk",
        "Obese":       f"BMI {bmi:.1f} (Obese) — Obesity is a major cardiovascular risk factor; intervention needed",
    }
    factors.append({"name": "BMI", "level": bmi_level_map[bmi_cat],
                     "score": bmi_score_map[bmi_cat], "advice": bmi_advice_map[bmi_cat]})

    bp_cat = get_bp_category(systolic, diastolic)
    bp_score_map = {
        "Normal": 10, "Elevated": 32,
        "Stage 1 Hypertension": 62, "Stage 2 Hypertension": 84, "Hypertensive Crisis": 98
    }
    bp_level_map = {
        "Normal": "Low", "Elevated": "Moderate",
        "Stage 1 Hypertension": "High", "Stage 2 Hypertension": "High", "Hypertensive Crisis": "Critical"
    }
    factors.append({"name": "Blood Pressure", "level": bp_level_map[bp_cat],
                     "score": bp_score_map[bp_cat],
                     "advice": f"{bp_cat} ({int(systolic)}/{int(diastolic)} mmHg)"})

    pp = systolic - diastolic
    if pp > 60:
        factors.append({"name": "Pulse Pressure", "level": "High", "score": 74,
                         "advice": f"PP = {pp:.0f} mmHg (>60) — Elevated; indicates arterial stiffness"})
    elif pp < 25:
        factors.append({"name": "Pulse Pressure", "level": "Moderate", "score": 42,
                         "advice": f"PP = {pp:.0f} mmHg (<25) — Low; may indicate reduced cardiac output"})
    else:
        factors.append({"name": "Pulse Pressure", "level": "Low", "score": 14,
                         "advice": f"PP = {pp:.0f} mmHg — Within normal range (25–60 mmHg)"})

    chol_map = {
        1: ("Low",      10, "Normal — maintain a diet low in saturated and trans fats"),
        2: ("Moderate", 56, "Above normal — reduce dietary cholesterol and increase fiber"),
        3: ("High",     88, "High — consult a physician; statin therapy may be considered"),
    }
    lvl, score, advice = chol_map[cholesterol]
    factors.append({"name": "Cholesterol", "level": lvl, "score": score, "advice": advice})

    if active == 0:
        factors.append({"name": "Activity Level", "level": "High", "score": 72,
                         "advice": "Sedentary lifestyle — physical inactivity significantly raises cardiovascular risk"})
    else:
        factors.append({"name": "Activity Level", "level": "Low", "score": 12,
                         "advice": "Active lifestyle — regular exercise is a strong protective cardiovascular factor"})

    return factors


# Build patient recommendations
def build_recommendations(ctx, factors):
    recos   = []
    tier    = ctx["risk_tier"]
    bmi_cat = ctx["bmi_category"]
    bp_cat  = ctx["bp_category"]

    if tier == "Low":
        recos.append({"icon": "✓", "title": "Great health profile",
                       "text": "Your indicators are within healthy ranges. Schedule annual check-ups and maintain your current lifestyle."})

    if bmi_cat in ("Overweight", "Obese"):
        recos.append({"icon": "⊕", "title": "Weight management",
                       "text": "Aim for a 500 kcal/day deficit through diet and 150 min/week of moderate-intensity aerobic exercise."})

    if "Hypertension" in bp_cat:
        recos.append({"icon": "◉", "title": "Blood pressure control",
                       "text": "Limit sodium to <2 g/day, avoid excess alcohol, and discuss antihypertensive options with your physician."})

    chol_factor = next(
        (f for f in factors if f["name"] == "Cholesterol" and f["level"] != "Low"), None
    )
    if chol_factor:
        recos.append({"icon": "⊛", "title": "Cholesterol diet",
                       "text": "Increase soluble fiber (oats, legumes), reduce saturated fats, and consider plant sterols in your diet."})

    act_factor = next(
        (f for f in factors
         if f["name"] in ("Physical Activity", "Activity Level")
         and f["level"] in ("High", "Critical")),
        None
    )
    if act_factor:
        recos.append({"icon": "◈", "title": "Exercise prescription",
                       "text": "Begin with 30 min brisk walking 5×/week and progressively increase to 150 min moderate or 75 min vigorous/week."})

    if tier in ("High", "Critical"):
        recos.append({"icon": "⚕", "title": "Specialist consultation",
                       "text": "Given your elevated risk score, we strongly recommend scheduling an appointment with a cardiologist promptly."})

    return recos[:4]


# Load evaluation data
def load_health_dataset():
    import pandas as pd

    dataset_path = os.path.join(app.root_path, "data", "health.csv")
    return pd.read_csv(dataset_path, sep=";")


# Prepare dataset for the trained model
def prepare_model_dataset(df):
    feature_df = df.copy()
    feature_df["age"] = feature_df["age"] / 365.25
    height_m = feature_df["height"] / 100.0
    feature_df["bmi"] = feature_df["weight"] / (height_m ** 2)
    feature_df["systolic"] = feature_df["ap_hi"]
    feature_df["diastolic"] = feature_df["ap_lo"]

    model_df = feature_df[FEATURE_ORDER].replace([np.inf, -np.inf], np.nan)
    valid_rows = model_df.notna().all(axis=1) & feature_df["cardio"].notna()

    x_values = model_df.loc[valid_rows, FEATURE_ORDER].to_numpy(dtype=float)
    y_values = feature_df.loc[valid_rows, "cardio"].astype(int).to_numpy()
    return x_values, y_values


# Run model on the dataset
def get_model_predictions():
    df = load_health_dataset()
    x_values, y_values = prepare_model_dataset(df)
    predictions = model.predict(x_values)
    return y_values, predictions


# Calculate model metrics
def get_model_metrics(y_true=None, y_pred=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    if y_true is None or y_pred is None:
        y_true, y_pred = get_model_predictions()

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


# Calculate confusion matrix
def get_confusion_matrix(y_true=None, y_pred=None):
    from sklearn.metrics import confusion_matrix

    if y_true is None or y_pred is None:
        y_true, y_pred = get_model_predictions()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# Summarize dataset health
def get_dataset_info():
    df = load_health_dataset()
    null_counts = df.isnull().sum()
    class_counts = df["cardio"].value_counts(dropna=False).sort_index()
    total_rows = int(len(df))

    class_distribution = {
        str(int(label)) if not np.isnan(float(label)) else "null": int(count)
        for label, count in class_counts.items()
    }
    class_balance = {
        label: round(count / total_rows, 4) if total_rows else 0
        for label, count in class_distribution.items()
    }

    return {
        "number_of_samples": total_rows,
        "number_of_features": len(FEATURE_ORDER),
        "has_null_values": bool(null_counts.sum() > 0),
        "null_values": {col: int(count) for col, count in null_counts.items()},
        "class_distribution": class_distribution,
        "class_balance": class_balance,
    }


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Model information API
@app.route("/model-info", methods=["GET"])
def model_info():
    try:
        y_true, y_pred = get_model_predictions()
        return jsonify({
            "metrics": get_model_metrics(y_true, y_pred),
            "confusion_matrix": get_confusion_matrix(y_true, y_pred),
            "dataset_info": get_dataset_info(),
        })
    except Exception:
        logger.exception("Unexpected error in /model-info")
        return jsonify({"error": "Model info unavailable. Please try again."}), 500


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        age         = float(data["age"])
        bmi         = float(data["bmi"])
        systolic    = float(data["systolic"])
        diastolic   = float(data["diastolic"])
        cholesterol = int(data["cholesterol"])
        active      = int(data["active"])

        # Validate inputs
        if not (10 <= age <= 100):
            return jsonify({"error": "Age must be between 10 and 100 years"}), 400
        if not (12.0 <= bmi <= 60.0):
            return jsonify({"error": "BMI must be between 12 and 60"}), 400
        if not (70 <= systolic <= 200):
            return jsonify({"error": "Systolic BP must be between 70 and 200 mmHg"}), 400
        if not (40 <= diastolic <= 150):
            return jsonify({"error": "Diastolic BP must be between 40 and 150 mmHg"}), 400
        if systolic <= diastolic:
            return jsonify({"error": "Systolic BP must be greater than diastolic BP"}), 400
        if cholesterol not in (1, 2, 3):
            return jsonify({"error": "Cholesterol must be 1 (Normal), 2 (Elevated), or 3 (High)"}), 400
        if active not in (0, 1):
            return jsonify({"error": "Active must be 0 (No) or 1 (Yes)"}), 400

        # Calculate derived values
        pulse_pressure         = systolic - diastolic
        mean_arterial_pressure = round(diastolic + (pulse_pressure / 3.0), 1)

        # Calculate prediction
        input_array = np.array([[age, bmi, systolic, diastolic, cholesterol, active]])
        prediction  = model.predict(input_array)[0]
        prob        = model.predict_proba(input_array)[0][1]

        # Convert probability to risk tier
        if prob < 0.30:
            risk_tier = "Low"
        elif prob < 0.60:
            risk_tier = "Moderate"
        elif prob < 0.80:
            risk_tier = "High"
        else:
            risk_tier = "Critical"

        # Get factor explanations (clean clinical version)
        factors = get_rule_based_factors(
            age, bmi, systolic, diastolic, cholesterol, active
        )
        explanation_source = "clinical_rules"

        # Build response
        ctx = {
            "risk_tier":    risk_tier,
            "bmi_category": get_bmi_category(bmi),
            "bp_category":  get_bp_category(systolic, diastolic),
        }

        return jsonify({
            "risk":                   int(prediction),
            "probability":            round(float(prob), 4),
            "risk_tier":              risk_tier,
            "pulse_pressure":         round(float(pulse_pressure), 1),
            "mean_arterial_pressure": mean_arterial_pressure,
            "bmi_category":           ctx["bmi_category"],
            "bp_category":            ctx["bp_category"],
            "explanation_source":     explanation_source,
            "factors":                factors,
            "recommendations":        build_recommendations(ctx, factors),
        })

    except (KeyError, TypeError) as e:
        return jsonify({"error": f"Missing or invalid field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400
    except Exception:
        logger.exception("Unexpected error in /predict")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
