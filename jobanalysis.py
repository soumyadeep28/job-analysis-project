import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(name, df, skill_config_path="skills_config.json"):
    """
    Create text, datetime and skill-based features using skill_config.json only.
    - name: dataset/role name, e.g. "job_all", "backend_engineer"
    - df:   pandas DataFrame with at least job_description or job_summary/job_skills
    """
    print(f"\n{'='*20} Feature Engineering for {name} {'='*20}")
    
    # -------- Generic text length features --------
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols[:3]:
        if df[col].notna().any() and df[col].astype(str).str.len().max() > 50:
            df[f"{col}_length"] = df[col].astype(str).str.len()
            df[f"{col}_word_count"] = df[col].astype(str).str.split().str.len()
            print(f"Created text features from: {col}")
    
    # -------- Specific features from job_summary --------
    if "job_summary" in df.columns:
        js = df["job_summary"].fillna("").astype(str)
        df["job_summary_length"] = js.str.len()
        df["job_summary_word_count"] = js.str.split().str.len()
        
        senior_words = ["senior", "lead", "principal", "manager"]
        junior_words = ["junior", "intern", "entry level"]
        df["has_senior_terms"] = js.str.lower().str.contains("|".join(senior_words)).astype(int)
        df["has_junior_terms"] = js.str.lower().str.contains("|".join(junior_words)).astype(int)
        print("Created text features from: job_summary")
    
    # -------- Skill features driven by skill_config.json --------
    # Choose the text column to scan for skills
    text_col = next((c for c in ["job_description", "job_summary", "job_skills"] if c in df.columns), None)
    if text_col is not None:
        job_text = df[text_col].fillna("").astype(str)
        print(f"Using '{text_col}' for skill detection")
        
        # Load skill configuration (must exist in production)
        try:
            with open(skill_config_path, "r") as f:
                all_skills_configs = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  '{skill_config_path}' not found. Skipping skill features.")
            return df
        
        role_key = name.lower().replace(" ", "_")
        # Try exact role or fall back to job_all/backend_engineer
        key_skills_config = (
            all_skills_configs.get(role_key)
            or all_skills_configs.get("job_all")
            or all_skills_configs.get("backend_engineer", {})
        )
        
        if not key_skills_config:
            print(f"⚠️  No usable config found in {skill_config_path}. Skipping skill features.")
            return df
        
        print(f"Loaded {len(key_skills_config)} skills from config for role '{role_key}'")
        
        lower_text = job_text.str.lower()
        total_prob = pd.Series(0.0, index=df.index)
        skill_detected_count = pd.Series(0, index=df.index)
        
        for flag_col, skill_info in key_skills_config.items():
            skill_keywords = skill_info.get("keywords", [])
            prob = float(skill_info.get("probability", 0.0))
            
            detected = lower_text.apply(
                lambda t, kws=skill_keywords: any(k in t for k in kws)
            ).astype(int)
            
            df[flag_col] = detected
            total_prob += detected * prob
            skill_detected_count += detected
        
        df["total_skill_prob"] = total_prob.clip(0.0, 1.0)
        df["skill_count"] = skill_detected_count
        
        # Map skill_count -> simple priority bins (0–4)
        if df["skill_count"].max() > 0:
            df["skill_priority"] = pd.cut(
                df["skill_count"],
                bins=[-1, 0, 1, 3, 5, np.inf],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(int)
        else:
            df["skill_priority"] = 0
        
        print(
            f"Skill features created: total_skill_prob, skill_count, skill_priority "
            f"(mean skill_count={df['skill_count'].mean():.2f})"
        )
    
    # -------- Datetime features --------
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]", "datetime64"]).columns
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        print(f"Created datetime features from: {col}")
    
    return df

# ============================================================
# CATEGORICAL ENCODING
# ============================================================

def encode_categorical_features(df, max_categories=10):
    """
    Encode categorical variables:
    - One-hot encode low-cardinality columns.
    - Label-encode high-cardinality columns.
    """
    print(f"\n{'='*20} Encoding Categorical Features {'='*20}")
    
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        
        if unique_count <= max_categories:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            print(f"One-hot encoded: {col} ({unique_count} categories)")
        else:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            df = df.drop(columns=[col])
            print(f"Label encoded: {col} ({unique_count} categories)")
    
    return df

# ============================================================
# OPTIONAL: SCALING HELPER (used in training, not in Flask)
# ============================================================

def scale_numeric_features(df):
    """
    Scale numeric features using StandardScaler.
    Returns transformed df and the fitted scaler.
    """
    print(f"\n{'='*20} Scaling Numeric Features {'='*20}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns to scale.")
        return df, None
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print(f"Scaled {len(numeric_cols)} numeric features.")
    return df, scaler


# ==============================================================================
# NEW CELL: Utility to extract skills from a JD string
# ==============================================================================

def extract_skills_from_text(jd_text: str):
    """
    Given a job description text, return a sorted list of unique skill names.
    """
    if not isinstance(jd_text, str) or jd_text.strip() == "":
        return []

    annotations = skill_extractor.annotate(jd_text)

    # SkillNER returns different groups; merge them into a simple set of names
    found_skills = set()

    for group_name in ["full_matches", "ngram_scored", "skill_mentions"]:
        group = annotations.get(group_name, [])
        for item in group:
            # Most variants have a 'doc_node_value' or 'skill_name' field
            name = item.get("doc_node_value") or item.get("skill_name")
            if isinstance(name, str):
                found_skills.add(name.strip())

    return sorted(found_skills)



import spacy
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB

# Load spaCy model once at module import
nlp = spacy.load("en_core_web_sm")

# Initialize SkillExtractor once
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
print("✓ Skill extractor initialized (jobanalysis.py).")

