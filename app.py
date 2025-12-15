from flask import Flask, render_template, request, jsonify
import joblib, json
import pandas as pd
import numpy as np


from jobanalysis import engineer_features, encode_categorical_features, extract_skills_from_text


app = Flask(__name__)

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------

# Use the actual config file you showed in the tree: skills_config.json
SKILL_CONFIG_PATH = "skills_config.json"
with open(SKILL_CONFIG_PATH, "r") as f:
    SKILL_CONFIG = json.load(f)

BEST_MODEL = joblib.load("job_all_best_model.pkl")
SCALER = joblib.load("job_all_scaler.pkl")
with open("job_all_columns.json") as f:
    TRAIN_COLS = json.load(f)



# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def explain_skills_for_jd(job_description_text, role_name="backend_engineer"):
    """
    Return {skill: normalized importance} using SKILL_CONFIG only.
    """
    role_key = role_name.lower().replace(" ", "_")
    role_cfg = (
        SKILL_CONFIG.get(role_key)
        or SKILL_CONFIG.get("job_all")
        or SKILL_CONFIG.get("backend_engineer", {})
    )

    text = (job_description_text or "").lower()
    skill_scores = {}

    for _, info in role_cfg.items():
        skill_name = info["skill"]
        prob = float(info.get("probability", 0.0))
        keywords = info.get("keywords", [])
        detected = any(k in text for k in keywords)
        if detected:
            skill_scores[skill_name] = skill_scores.get(skill_name, 0.0) + prob

    if not skill_scores:
        return {}

    total = sum(skill_scores.values())
    normalized = {k: round(v / total, 3) for k, v in skill_scores.items()}
    normalized = dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))
    return normalized


def build_single_row_df(jd_text: str) -> pd.DataFrame:
    """Build a one-row DataFrame with at least job_description column."""
    return pd.DataFrame([{"job_description": jd_text}])


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    jd_text = data.get("job_description", "")
    role = data.get("role", "backend_engineer")

    if not jd_text.strip():
        return jsonify({"error": "Empty job_description"}), 400

    base_df = pd.DataFrame([{"job_description": jd_text}])

    df_fe = engineer_features("job_all", base_df.copy(), skill_config_path=SKILL_CONFIG_PATH)
    df_enc = encode_categorical_features(df_fe, max_categories=10)

    for col in TRAIN_COLS:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[TRAIN_COLS]

    df_scaled = SCALER.transform(df_enc)

    if "total_skill_prob" in df_enc.columns:
        col_idx = list(df_enc.columns).index("total_skill_prob")
        df_model = np.delete(df_scaled, col_idx, axis=1)
    else:
        df_model = df_scaled

    y_pred = float(BEST_MODEL.predict(df_model)[0])

    # existing importance from skills_config
    skill_importance = explain_skills_for_jd(jd_text, role_name=role)

    # NEW: other skills via SkillNer
    all_skills = extract_skills_from_text(jd_text)  # list of skill strings
    known_skills = set(skill_importance.keys())
    other_skills = [s for s in all_skills if s.lower() not in {k.lower() for k in known_skills}]

    return jsonify({
        "predicted_score": y_pred,
        "skill_importance": skill_importance,
        "other_skills": other_skills
    })




if __name__ == "__main__":
    app.run(debug=True , host="0.0.0.0", port=8000)
