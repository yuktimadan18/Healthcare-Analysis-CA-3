"""
feature_config.py
-----------------
Configuration for feature selection, leakage exclusions, survey metadata,
and train/validation/test partitioning for CDC BRFSS 2024 predictive modeling.
"""

from typing import Dict, List, Any

# Primary Prediction Target Configuration
TARGET_COL = "_MICHD"
TARGET_POS_CODES = [1]    # Reported having MI or CHD (Notes: CVDINFR4=1 OR CVDCRHD4=1)
TARGET_NEG_CODES = [2]    # Did not report having MI or CHD (Notes: CVDINFR4=2 AND CVDCRHD4=2)
TARGET_NAME = "Reported History of Coronary Heart Disease or Myocardial Infarction"

# Mandatory Leakage Exclusions (Direct Components of _MICHD)
MANDATORY_LEAKAGE_EXCLUSIONS: Dict[str, str] = {
    "CVDINFR4": (
        "DIRECT TARGET LEAKAGE: CDC defines _MICHD = 1 if CVDINFR4 = 1. "
        "Including history of heart attack directly reveals the prediction outcome."
    ),
    "CVDCRHD4": (
        "DIRECT TARGET LEAKAGE: CDC defines _MICHD = 1 if CVDCRHD4 = 1. "
        "Including history of angina directly reveals the prediction outcome."
    ),
}

# Survey Design Variables (Retained strictly in metadata, NOT in predictive feature matrix)
SURVEY_DESIGN_METADATA: Dict[str, str] = {
    "_LLCPWT": "Final survey expansion weight for landline and cellular combined.",
    "_STSTR": "Sample design stratification variable.",
    "_PSU": "Primary Sampling Unit cluster identifier.",
    "_STATE": "State FIPS code; excluded from feature matrix to prevent geographic overfitting.",
}

# Redundant / Collinear Features Excluded in Favor of Richer Equivalents
REDUNDANT_FEATURE_EXCLUSIONS: Dict[str, str] = {
    "_AGE65YR": "Binary age threshold (18-64 vs 65+); excluded in favor of continuous _AGE80 and 5-year _AGEG5YR.",
    "_RFHLTH": "Binary good/better vs fair/poor health; excluded in favor of 5-level ordinal GENHLTH.",
    "_PHYS14D": "3-level physical health summary; excluded in favor of continuous PHYSHLTH_DAYS (0-30 days).",
    "_MENT14D": "3-level mental health summary; excluded in favor of continuous MENTHLTH_DAYS (0-30 days).",
    "POORHLTH": "Poor health activity limitation days; excluded due to 41.5% structural skip missingness.",
    "_BMI5CAT": "4-level BMI categories; excluded in favor of continuous BMI_NUMERIC.",
    "_RFBMI5": "Binary overweight/obese indicator; excluded in favor of continuous BMI_NUMERIC.",
    "SMOKE100": "Lifetime 100 cigarettes indicator; excluded in favor of 4-level _SMOKER3.",
    "_RFSMOK3": "Binary current smoker indicator; excluded in favor of 4-level _SMOKER3.",
    "ECIGNOW3": "3-level e-cigarette frequency; excluded in favor of clean binary _CURECI3.",
    "EXERANY2": "Any physical activity; exact duplicate of calculated variable _TOTINDA.",
    "_DRNKWK3": "Calculated total drinks per week; excluded due to extreme skew and high non-response coding.",
    "_RFBING6": "Binge drinking calculated variable; collinear with DRNKANY6 and _RFDRHV9.",
    "PREDIAB2": "Pre-diabetes indicator; excluded due to 65.2% structural skip missingness.",
    "_ASTHMS1": "Calculated asthma status; excluded in favor of lifetime diagnosed ASTHMA3.",
    "PRIMINS2": "Detailed primary insurance source; high cardinality with state routing; replaced by _HLTHPL2.",
}

# Included Continuous Numeric Predictors (4 features)
NUMERIC_FEATURES: List[str] = [
    "_AGE80",          # Continuous age in years (imputed, collapsed at 80)
    "PHYSHLTH_DAYS",   # Cleaned days physical health not good in past 30 days (0-30, code 88 -> 0)
    "MENTHLTH_DAYS",   # Cleaned days mental health not good in past 30 days (0-30, code 88 -> 0)
    "BMI_NUMERIC",     # Continuous Body Mass Index (_BMI5 / 100.0)
]

# Included Categorical Predictors (34 features)
CATEGORICAL_FEATURES: List[str] = [
    # Demographics (9)
    "_SEX",            # Sex of respondent (1=Male, 2=Female)
    "_AGEG5YR",        # 14-level five-year age categories (1=18-24 ... 13=80+, 14=Missing)
    "_RACE",           # Race/ethnicity category (1=White NH ... 8=Hispanic, 9=Missing)
    "_EDUCAG",         # Education level completed (1=No HS ... 4=College Grad, 9=Missing)
    "_INCOMG1",        # Annual household income (1=<$15k ... 7=$200k+, 9=Missing)
    "MARITAL",         # Marital status (1=Married ... 6=Unmarried couple, 9=Refused)
    "EMPLOY1",         # Employment status (1=Employed ... 8=Unable to work, 9=Refused)
    "_METSTAT",        # Metropolitan status (1=Metropolitan, 2=Nonmetropolitan)
    "_URBSTAT",        # Urban/rural status (1=Urban, 2=Rural)
    # General Health (1)
    "GENHLTH",         # Self-rated health (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor, 7/9=DK/Ref)
    # Behaviors & Risk Factors (5)
    "_SMOKER3",        # Smoker status (1=Daily, 2=Some days, 3=Former, 4=Never, 9=DK/Ref)
    "_CURECI3",        # Current e-cigarette user (1=No, 2=Yes, 9=DK/Ref)
    "_TOTINDA",        # Leisure physical activity past 30 days (1=Yes, 2=No, 9=DK/Ref)
    "DRNKANY6",        # Drank alcohol past 30 days (1=Yes, 2=No, 7/9=DK/Ref)
    "_RFDRHV9",        # Heavy alcohol consumption (1=No, 2=Yes, 9=DK/Ref)
    # Healthcare Access (4)
    "_HLTHPL2",        # Have healthcare coverage (1=Yes, 2=No, 9=DK/Ref)
    "PERSDOC3",        # Personal healthcare provider (1=Yes one, 2=More than one, 3=No, 7/9=DK/Ref)
    "MEDCOST1",        # Could not see doctor due to cost past 12mo (1=Yes, 2=No, 7/9=DK/Ref)
    "CHECKUP1",        # Time since routine checkup (1=Past year ... 4=5+ yrs, 8=Never, 7/9=DK/Ref)
    # Chronic Conditions / Non-cardiac Comorbidities (9)
    "DIABETE4",        # Diagnosed diabetes (1=Yes, 2=Pregnancy only, 3=No, 4=Borderline, 7/9=DK/Ref)
    "ASTHMA3",         # Diagnosed asthma (1=Yes, 2=No, 7/9=DK/Ref)
    "CHCCOPD3",        # Diagnosed COPD/emphysema (1=Yes, 2=No, 7/9=DK/Ref)
    "ADDEPEV3",        # Diagnosed depressive disorder (1=Yes, 2=No, 7/9=DK/Ref)
    "CHCKDNY2",        # Diagnosed kidney disease (1=Yes, 2=No, 7/9=DK/Ref)
    "HAVARTH4",        # Diagnosed arthritis (1=Yes, 2=No, 7/9=DK/Ref)
    "CHCSCNC1",        # Diagnosed skin cancer (1=Yes, 2=No, 7/9=DK/Ref)
    "CHCOCNC1",        # Diagnosed other cancer (1=Yes, 2=No, 7/9=DK/Ref)
    "CVDSTRK3",        # Diagnosed stroke (1=Yes, 2=No, 7/9=DK/Ref) - Cerebrovascular comorbidity
    # Functional Disabilities (6)
    "DEAF",            # Serious hearing difficulty (1=Yes, 2=No, 7/9=DK/Ref)
    "BLIND",           # Serious vision difficulty (1=Yes, 2=No, 7/9=DK/Ref)
    "DECIDE",          # Serious cognitive difficulty (1=Yes, 2=No, 7/9=DK/Ref)
    "DIFFWALK",        # Serious mobility difficulty (1=Yes, 2=No, 7/9=DK/Ref)
    "DIFFDRES",        # Dressing/bathing difficulty (1=Yes, 2=No, 7/9=DK/Ref)
    "DIFFALON",        # Errands alone difficulty (1=Yes, 2=No, 7/9=DK/Ref)
]

# Categorical Missing Value Sentinel (guaranteed never to collide with non-negative BRFSS response codes)
CATEGORICAL_MISSING_SENTINEL: float = -1.0

# Partition Configuration
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
STRATIFY_COL = "target"
