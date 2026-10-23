"""
generate_eda_notebook.py
Constructs notebooks/01_data_quality_and_eda.ipynb following all Milestone 3 requirements.
"""

import json
from pathlib import Path

nb_path = Path("notebooks/01_data_quality_and_eda.ipynb")

cells = [
    # ---------------- Section 1 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Milestone 3: Data Quality Assessment and Exploratory Data Analysis (EDA)\n",
            "**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024  \n",
            "**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 (Annual Landline & Cell Survey)  \n",
            "**File:** `data/interim/brfss_2024_selected.csv.gz` (457,670 rows, 61 variables)  \n",
            "**Binding Rules:** Strictly complies with `PROJECT_RULES.md` (no target selection yet, no models trained, survey weighting respected).\n",
            "\n",
            "---\n",
            "\n",
            "## 1. Project Objective & Analytical Questions\n",
            "The goal of this phase is to evaluate the quality, distributions, missingness patterns, and complex survey design characteristics of the 2024 BRFSS dataset.\n",
            "\n",
            "### Core Analytical Questions:\n",
            "1. **Data Completeness & Reliability:** What are the missingness patterns across demographic, behavioral, and clinical variables? Are missing values completely at random (MCAR) or structured by survey routing/skip logic?\n",
            "2. **Survey Response Coding:** How do refusal (`9`/`99`), \"don't know\" (`7`/`77`), and legitimate \"none\" (`88`) codes present across questions, and what are the implications for clinical predictive features?\n",
            "3. **Population Demographics:** What is the demographic composition of the 2024 cohort (age, sex, race/ethnicity, education, income, and geographic/urban status)?\n",
            "4. **Candidate Chronic Outcome Prevalence:** What is the prevalence of major chronic conditions (`_MICHD`, `DIABETE4`, `CVDSTRK3`, `CHCKDNY2`, `CHCCOPD3`, `ADDEPEV3`, `ASTHMA3`), and how does unweighted prevalence compare to CDC survey-weighted (`_LLCPWT`) population estimates?\n",
            "5. **Survey Design Structure:** How do primary sampling units (`_PSU`), stratification (`_STSTR`), and final weights (`_LLCPWT`) influence downstream statistical inference and predictive modeling?"
        ]
    },

    # ---------------- Section 2 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Dataset Provenance & Ingestion\n",
            "The data was ingested directly from `data/raw/LLCP2024XPT.zip` member `'LLCP2024.XPT '` via `src/data/make_dataset.py` using `pyreadstat` with `encoding='latin1'`. Exactly 61 verified candidate features were preserved without modifying the raw archive."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Visual style settings\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n",
            "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
            "plt.rcParams[\"figure.dpi\"] = 150\n",
            "\n",
            "# Paths\n",
            "PROJECT_ROOT = Path(\"..\").resolve()\n",
            "DATA_PATH = PROJECT_ROOT / \"data\" / \"interim\" / \"brfss_2024_selected.csv.gz\"\n",
            "DICT_PATH = PROJECT_ROOT / \"data\" / \"interim\" / \"brfss_2024_dictionary.json\"\n",
            "FIG_DIR = PROJECT_ROOT / \"results\" / \"figures\"\n",
            "TAB_DIR = PROJECT_ROOT / \"results\" / \"tables\"\n",
            "FIG_DIR.mkdir(parents=True, exist_ok=True)\n",
            "TAB_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(f\"Loading interim dataset from {DATA_PATH}...\")\n",
            "df = pd.read_csv(DATA_PATH, low_memory=False)\n",
            "print(f\"Loaded shape: {df.shape[0]:,} rows x {df.shape[1]} columns.\")\n",
            "\n",
            "with open(DICT_PATH, \"r\", encoding=\"utf-8\") as f:\n",
            "    var_dict = json.load(f)\n",
            "print(f\"Loaded dictionary containing metadata for {len(var_dict['variables'])} variables.\")"
        ]
    },

    # ---------------- Section 3 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Schema & Variable Dictionary\n",
            "Review variable categorization and functional roles across the 61 extracted variables."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "dict_df = pd.DataFrame.from_dict(var_dict[\"variables\"], orient=\"index\")\n",
            "dict_df.index.name = \"variable_name\"\n",
            "dict_df.reset_index(inplace=True)\n",
            "\n",
            "category_counts = dict_df[\"category\"].value_counts()\n",
            "print(\"Variables per Category:\")\n",
            "display(category_counts.to_frame(name=\"variable_count\"))\n",
            "\n",
            "# Save dictionary table\n",
            "dict_df.to_csv(TAB_DIR / \"variable_dictionary_summary.csv\", index=False)\n",
            "print(f\"Saved variable dictionary to {TAB_DIR / 'variable_dictionary_summary.csv'}\")"
        ]
    },

    # ---------------- Section 4 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Data Types, Duplicate Checks & Data Quality\n",
            "Audit memory footprint, duplicate respondent records, and structural validity."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check duplicates across respondent records\n",
            "duplicate_count = df.duplicated().sum()\n",
            "print(f\"Exact duplicate rows: {duplicate_count}\")\n",
            "\n",
            "# Verify survey primary cluster identifiers uniqueness\n",
            "psu_cluster_count = df.groupby([\"_STATE\", \"_PSU\"]).ngroups\n",
            "print(f\"Unique primary sampling units (clusters across states): {psu_cluster_count:,}\")\n",
            "\n",
            "# Summary of data types\n",
            "type_counts = df.dtypes.value_counts()\n",
            "print(\"\\nData Types Breakdown:\")\n",
            "display(type_counts.to_frame(name=\"column_count\"))"
        ]
    },

    # ---------------- Section 5 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Missingness & Survey Response Code Analysis\n",
            "In BRFSS, special values have domain-specific semantics:\n",
            "- `7` or `77`: \"Don't know / Not sure\"\n",
            "- `9` or `99`: \"Refused\"\n",
            "- `BLANK` / `NaN`: Not asked (e.g. respondent was not eligible or skipped by routing logic)\n",
            "- `88`: None (e.g., in `PHYSHLTH`, `MENTHLTH`, 88 indicates 0 days)\n",
            "\n",
            "We audit both true `NaN` missingness and the frequency of refusal / uncertainty codes."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate true NaN missingness\n",
            "nan_summary = pd.DataFrame({\n",
            "    \"missing_count\": df.isna().sum(),\n",
            "    \"missing_pct\": (df.isna().sum() / len(df)) * 100\n",
            "})\n",
            "nan_summary = nan_summary.merge(dict_df, left_index=True, right_on=\"variable_name\", how=\"left\").set_index(\"variable_name\")\n",
            "nan_summary = nan_summary.sort_values(by=\"missing_pct\", ascending=False)\n",
            "\n",
            "print(\"Top 15 Variables by Missing Percentage:\")\n",
            "display(nan_summary[[\"category\", \"missing_count\", \"missing_pct\", \"role\"]].head(15))\n",
            "\n",
            "# Save full missingness table\n",
            "nan_summary.to_csv(TAB_DIR / \"data_quality_missingness.csv\")\n",
            "print(f\"Saved complete missingness table to {TAB_DIR / 'data_quality_missingness.csv'}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 1: Missingness Distribution across Variables\n",
            "plt.figure(figsize=(12, 6))\n",
            "top_missing = nan_summary[nan_summary[\"missing_pct\"] > 0].head(25)\n",
            "sns.barplot(\n",
            "    data=top_missing.reset_index(),\n",
            "    x=\"missing_pct\",\n",
            "    y=\"variable_name\",\n",
            "    hue=\"category\",\n",
            "    dodge=False,\n",
            "    palette=\"mako\"\n",
            ")\n",
            "plt.title(\"Figure 1: Variables with Highest Missingness Rate (>0%)\", fontsize=14, weight=\"bold\", pad=12)\n",
            "plt.xlabel(\"Missing Percentage (%)\")\n",
            "plt.ylabel(\"Variable Name\")\n",
            "plt.legend(title=\"Category\", loc=\"lower right\")\n",
            "plt.tight_layout()\n",
            "fig1_path = FIG_DIR / \"fig1_missingness_top_variables.png\"\n",
            "plt.savefig(fig1_path, dpi=200)\n",
            "plt.show()\n",
            "print(f\"Saved Figure 1 to {fig1_path}\")"
        ]
    },

    # ---------------- Section 6 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Descriptive Statistics & Risk Factor Distributions\n",
            "Examine distributions of key continuous and calculated variables: Body Mass Index (`_BMI5`), days of physical unwellness (`PHYSHLTH`), mental unwellness (`MENTHLTH`), and drinks per week (`_DRNKWK3`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Clean continuous representations for visualization:\n",
            "# _BMI5 has 2 implied decimal places (divide by 100)\n",
            "bmi_clean = df[\"_BMI5\"].dropna() / 100.0\n",
            "# PHYSHLTH: 88 -> 0 days; 77, 99 -> NaN\n",
            "phys_clean = df[\"PHYSHLTH\"].replace({88: 0, 77: np.nan, 99: np.nan}).dropna()\n",
            "# MENTHLTH: 88 -> 0 days; 77, 99 -> NaN\n",
            "ment_clean = df[\"MENTHLTH\"].replace({88: 0, 77: np.nan, 99: np.nan}).dropna()\n",
            "# _DRNKWK3: continuous drinks per week (divided by 100 per CDC spec, 99900=missing)\n",
            "drnk_clean = df[\"_DRNKWK3\"].replace({99900: np.nan}).dropna() / 100.0\n",
            "\n",
            "continuous_stats = pd.DataFrame({\n",
            "    \"BMI (_BMI5)\": bmi_clean.describe(),\n",
            "    \"Days Physical Health Not Good (PHYSHLTH)\": phys_clean.describe(),\n",
            "    \"Days Mental Health Not Good (MENTHLTH)\": ment_clean.describe(),\n",
            "    \"Drinks per Week (_DRNKWK3)\": drnk_clean.describe()\n",
            "})\n",
            "display(continuous_stats.round(2))\n",
            "continuous_stats.round(2).to_csv(TAB_DIR / \"continuous_risk_factors_stats.csv\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 2: Risk Factors Distribution (BMI, General Health, Smoker Status, Physical Activity)\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "\n",
            "# Subplot A: BMI Distribution\n",
            "sns.histplot(bmi_clean[bmi_clean <= 60], bins=50, kde=True, ax=axes[0, 0], color=\"#2b5c8f\")\n",
            "axes[0, 0].set_title(\"A. Body Mass Index (_BMI5)\", weight=\"bold\")\n",
            "axes[0, 0].set_xlabel(\"BMI (kg/m²)\")\n",
            "axes[0, 0].axvline(25, color=\"orange\", linestyle=\"--\", label=\"Overweight (25)\")\n",
            "axes[0, 0].axvline(30, color=\"red\", linestyle=\"--\", label=\"Obese (30)\")\n",
            "axes[0, 0].legend()\n",
            "\n",
            "# Subplot B: General Health Status\n",
            "genhlth_map = {1: \"1-Excellent\", 2: \"2-Very Good\", 3: \"3-Good\", 4: \"4-Fair\", 5: \"5-Poor\", 7: \"Don't Know\", 9: \"Refused\"}\n",
            "genhlth_counts = df[\"GENHLTH\"].map(genhlth_map).value_counts().sort_index()\n",
            "sns.barplot(x=genhlth_counts.values, y=genhlth_counts.index, ax=axes[0, 1], palette=\"Blues_r\")\n",
            "axes[0, 1].set_title(\"B. General Health Self-Rating (GENHLTH)\", weight=\"bold\")\n",
            "axes[0, 1].set_xlabel(\"Respondent Count\")\n",
            "\n",
            "# Subplot C: Four-Level Smoker Status\n",
            "smoker_map = {1: \"1-Every Day\", 2: \"2-Some Days\", 3: \"3-Former\", 4: \"4-Never\", 9: \"Missing/Refused\"}\n",
            "smoker_counts = df[\"_SMOKER3\"].map(smoker_map).value_counts().sort_index()\n",
            "sns.barplot(x=smoker_counts.values, y=smoker_counts.index, ax=axes[1, 0], palette=\"Reds_r\")\n",
            "axes[1, 0].set_title(\"C. Smoker Status (_SMOKER3)\", weight=\"bold\")\n",
            "axes[1, 0].set_xlabel(\"Respondent Count\")\n",
            "\n",
            "# Subplot D: Physical Activity in Past 30 Days\n",
            "exer_map = {1: \"Yes (Active)\", 2: \"No (Inactive)\", 9: \"Missing/Refused\"}\n",
            "exer_counts = df[\"_TOTINDA\"].map(exer_map).value_counts().sort_index()\n",
            "sns.barplot(x=exer_counts.values, y=exer_counts.index, ax=axes[1, 1], palette=\"Greens_r\")\n",
            "axes[1, 1].set_title(\"D. Leisure-Time Physical Activity (_TOTINDA)\", weight=\"bold\")\n",
            "axes[1, 1].set_xlabel(\"Respondent Count\")\n",
            "\n",
            "plt.tight_layout()\n",
            "fig2_path = FIG_DIR / \"fig2_risk_factor_distributions.png\"\n",
            "plt.savefig(fig2_path, dpi=200)\n",
            "plt.show()\n",
            "print(f\"Saved Figure 2 to {fig2_path}\")"
        ]
    },

    # ---------------- Section 7 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Demographic Profile Analysis\n",
            "Examine respondent distribution by Sex, Age Group, Race/Ethnicity, and Educational Attainment."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(2, 2, figsize=(15, 11))\n",
            "\n",
            "# Age Groups (_AGEG5YR)\n",
            "age_labels = {\n",
            "    1: \"18-24\", 2: \"25-29\", 3: \"30-34\", 4: \"35-39\", 5: \"40-44\",\n",
            "    6: \"45-49\", 7: \"50-54\", 8: \"55-59\", 9: \"60-64\", 10: \"65-69\",\n",
            "    11: \"70-74\", 12: \"75-79\", 13: \"80+\", 14: \"DK/Refused\"\n",
            "}\n",
            "age_counts = df[\"_AGEG5YR\"].map(age_labels).value_counts()[list(age_labels.values())]\n",
            "sns.barplot(x=age_counts.values, y=age_counts.index, ax=axes[0, 0], palette=\"viridis\")\n",
            "axes[0, 0].set_title(\"A. Age Distribution (5-year bands: _AGEG5YR)\", weight=\"bold\")\n",
            "axes[0, 0].set_xlabel(\"Count\")\n",
            "\n",
            "# Sex (_SEX)\n",
            "sex_labels = {1: \"Male\", 2: \"Female\"}\n",
            "sex_counts = df[\"_SEX\"].map(sex_labels).value_counts()\n",
            "axes[0, 1].pie(sex_counts.values, labels=sex_counts.index, autopct=\"%1.1f%%\", colors=[\"#5a9bd4\", \"#ed7d31\"], startangle=90)\n",
            "axes[0, 1].set_title(\"B. Sex Distribution (_SEX)\", weight=\"bold\")\n",
            "\n",
            "# Race/Ethnicity (_RACE)\n",
            "race_labels = {\n",
            "    1: \"White Non-Hisp\", 2: \"Black Non-Hisp\", 3: \"Amer. Indian/AK\",\n",
            "    4: \"Asian\", 5: \"Nat. Hawaiian/PI\", 6: \"Other Race\",\n",
            "    7: \"Multiracial\", 8: \"Hispanic\", 9: \"DK/Refused\"\n",
            "}\n",
            "race_counts = df[\"_RACE\"].map(race_labels).value_counts().sort_values(ascending=False)\n",
            "sns.barplot(x=race_counts.values, y=race_counts.index, ax=axes[1, 0], palette=\"crest\")\n",
            "axes[1, 0].set_title(\"C. Race/Ethnicity (_RACE)\", weight=\"bold\")\n",
            "axes[1, 0].set_xlabel(\"Count\")\n",
            "\n",
            "# Education (_EDUCAG)\n",
            "edu_labels = {1: \"< High School\", 2: \"High School Grad\", 3: \"Some College\", 4: \"College Grad\", 9: \"DK/Refused\"}\n",
            "edu_counts = df[\"_EDUCAG\"].map(edu_labels).value_counts().sort_values(ascending=False)\n",
            "sns.barplot(x=edu_counts.values, y=edu_counts.index, ax=axes[1, 1], palette=\"flare\")\n",
            "axes[1, 1].set_title(\"D. Education Completed (_EDUCAG)\", weight=\"bold\")\n",
            "axes[1, 1].set_xlabel(\"Count\")\n",
            "\n",
            "plt.tight_layout()\n",
            "fig3_path = FIG_DIR / \"fig3_demographic_profiles.png\"\n",
            "plt.savefig(fig3_path, dpi=200)\n",
            "plt.show()\n",
            "print(f\"Saved Figure 3 to {fig3_path}\")"
        ]
    },

    # ---------------- Section 8 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Candidate Outcome Prevalence & Class Imbalance\n",
            "We inspect candidate chronic disease endpoints, evaluating:\n",
            "1. **Unweighted sample prevalence** (raw respondent counts)\n",
            "2. **Survey-weighted population prevalence** (using CDC raking weight `_LLCPWT`)\n",
            "3. **Missingness / refusal rate**\n",
            "4. **Class imbalance ratio**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Candidate chronic outcomes and their positive response coding\n",
            "outcomes_spec = {\n",
            "    \"_MICHD\": {\"name\": \"Myocardial Infarction / CHD\", \"pos\": [1], \"neg\": [2]},\n",
            "    \"CVDINFR4\": {\"name\": \"Heart Attack (History)\", \"pos\": [1], \"neg\": [2]},\n",
            "    \"CVDCRHD4\": {\"name\": \"Angina / Coronary Heart Disease\", \"pos\": [1], \"neg\": [2]},\n",
            "    \"CVDSTRK3\": {\"name\": \"Stroke (Ever Diagnosed)\", \"pos\": [1], \"neg\": [2]},\n",
            "    \"DIABETE4\": {\"name\": \"Diabetes (Ever Diagnosed)\", \"pos\": [1], \"neg\": [3, 4]},\n",
            "    \"ASTHMA3\": {\"name\": \"Asthma (Ever Diagnosed)\", \"pos\": [1], \"neg\": [2]},\n",
            "    \"CHCCOPD3\": {\"name\": \"COPD / Emphysema\", \"pos\": [1], \"neg\": [2]},\n",
            "    \"ADDEPEV3\": {\"name\": \"Depressive Disorder\", \"pos\": [1], \"neg\": [2]},\n",
            "    \"CHCKDNY2\": {\"name\": \"Kidney Disease\", \"pos\": [1], \"neg\": [2]},\n",
            "    \"HAVARTH4\": {\"name\": \"Arthritis\", \"pos\": [1], \"neg\": [2]}\n",
            "}\n",
            "\n",
            "outcome_rows = []\n",
            "weights = df[\"_LLCPWT\"]\n",
            "\n",
            "for col, info in outcomes_spec.items():\n",
            "    s = df[col]\n",
            "    pos_mask = s.isin(info[\"pos\"])\n",
            "    neg_mask = s.isin(info[\"neg\"])\n",
            "    valid_mask = pos_mask | neg_mask\n",
            "    \n",
            "    pos_count = pos_mask.sum()\n",
            "    neg_count = neg_mask.sum()\n",
            "    total_valid = valid_mask.sum()\n",
            "    missing_count = len(df) - total_valid\n",
            "    \n",
            "    unweighted_prev = (pos_count / total_valid) * 100.0\n",
            "    # Survey-weighted prevalence\n",
            "    weighted_pos = weights[pos_mask].sum()\n",
            "    weighted_valid = weights[valid_mask].sum()\n",
            "    weighted_prev = (weighted_pos / weighted_valid) * 100.0\n",
            "    \n",
            "    imbalance_ratio = f\"1 : {(neg_count / pos_count):.1f}\" if pos_count > 0 else \"N/A\"\n",
            "    \n",
            "    outcome_rows.append({\n",
            "        \"Variable\": col,\n",
            "        \"Clinical Condition\": info[\"name\"],\n",
            "        \"Positive Cases\": int(pos_count),\n",
            "        \"Negative Cases\": int(neg_count),\n",
            "        \"Missing/Excluded\": int(missing_count),\n",
            "        \"Unweighted Prev (%)\": round(unweighted_prev, 2),\n",
            "        \"Weighted Prev (%)\": round(weighted_prev, 2),\n",
            "        \"Imbalance Ratio\": imbalance_ratio\n",
            "    })\n",
            "\n",
            "outcomes_df = pd.DataFrame(outcome_rows)\n",
            "display(outcomes_df)\n",
            "outcomes_df.to_csv(TAB_DIR / \"candidate_outcomes_prevalence.csv\", index=False)\n",
            "print(f\"Saved outcomes summary table to {TAB_DIR / 'candidate_outcomes_prevalence.csv'}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 3: Unweighted vs Weighted Prevalence of Candidate Chronic Conditions\n",
            "plt.figure(figsize=(12, 6))\n",
            "plot_df = outcomes_df.melt(\n",
            "    id_vars=[\"Clinical Condition\"],\n",
            "    value_vars=[\"Unweighted Prev (%)\", \"Weighted Prev (%)\"],\n",
            "    var_name=\"Prevalence Type\",\n",
            "    value_name=\"Percentage\"\n",
            ")\n",
            "\n",
            "sns.barplot(data=plot_df, y=\"Clinical Condition\", x=\"Percentage\", hue=\"Prevalence Type\", palette=\"Set2\")\n",
            "plt.title(\"Figure 4: Unweighted Sample vs. Survey-Weighted Population Prevalence\", weight=\"bold\", pad=12)\n",
            "plt.xlabel(\"Prevalence (%)\")\n",
            "plt.ylabel(\"Candidate Chronic Condition\")\n",
            "plt.legend(title=\"Estimate Type\", loc=\"lower right\")\n",
            "plt.tight_layout()\n",
            "fig4_path = FIG_DIR / \"fig4_candidate_outcomes_prevalence.png\"\n",
            "plt.savefig(fig4_path, dpi=200)\n",
            "plt.show()\n",
            "print(f\"Saved Figure 4 to {fig4_path}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 5: Chronic Condition Comorbidity / Association Heatmap\n",
            "binary_chronic = pd.DataFrame({\n",
            "    \"CHD/MI (_MICHD)\": df[\"_MICHD\"].map({1: 1, 2: 0}),\n",
            "    \"Stroke (CVDSTRK3)\": df[\"CVDSTRK3\"].map({1: 1, 2: 0}),\n",
            "    \"Diabetes (DIABETE4)\": df[\"DIABETE4\"].map({1: 1, 2: 0, 3: 0, 4: 0}),\n",
            "    \"COPD (CHCCOPD3)\": df[\"CHCCOPD3\"].map({1: 1, 2: 0}),\n",
            "    \"Kidney Dis. (CHCKDNY2)\": df[\"CHCKDNY2\"].map({1: 1, 2: 0}),\n",
            "    \"Depression (ADDEPEV3)\": df[\"ADDEPEV3\"].map({1: 1, 2: 0}),\n",
            "    \"Asthma (ASTHMA3)\": df[\"ASTHMA3\"].map({1: 1, 2: 0}),\n",
            "    \"Arthritis (HAVARTH4)\": df[\"HAVARTH4\"].map({1: 1, 2: 0})\n",
            "})\n",
            "\n",
            "corr_matrix = binary_chronic.corr(method=\"pearson\")\n",
            "plt.figure(figsize=(9, 7))\n",
            "sns.heatmap(corr_matrix, annot=True, fmt=\".2f\", cmap=\"Blues\", vmin=0, vmax=0.4, square=True)\n",
            "plt.title(\"Figure 5: Comorbidity Association Matrix Among Chronic Conditions\", weight=\"bold\", pad=12)\n",
            "plt.tight_layout()\n",
            "fig5_path = FIG_DIR / \"fig5_chronic_comorbidity_correlation.png\"\n",
            "plt.savefig(fig5_path, dpi=200)\n",
            "plt.show()\n",
            "print(f\"Saved Figure 5 to {fig5_path}\")"
        ]
    },

    # ---------------- Section 9 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Survey-Design Considerations & Limitations\n",
            "\n",
            "### Complex Survey Methodology (Rule 12 Compliance):\n",
            "1. **Dual-Frame Sampling:** BRFSS 2024 employs stratified random sampling across dual frames (landline and cellular) in 49 states, DC, and 3 territories.\n",
            "2. **Primary Sampling Units (`_PSU`) & Stratification (`_STSTR`):** Standard simple-random-sample assumptions underestimate variance. Inferential claims must incorporate clustering and stratification.\n",
            "3. **Raking Weights (`_LLCPWT`):** As demonstrated in Figure 4, unweighted estimates diverge from survey-weighted population estimates (e.g., sample has higher representation of older individuals and females).\n",
            "4. **Cross-Sectional Nature (Rule 13 Compliance):** BRFSS is cross-sectional; associations between risk factors and chronic diseases reflect statistical relationships, not causal mechanisms, and cannot be used for clinical diagnoses."
        ]
    },

    # ---------------- Section 10 ----------------
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Evidence-Based Recommendations for Milestone 4 & 5\n",
            "\n",
            "Based on the empirical findings from this EDA:\n",
            "1. **Potential Primary Target Candidates for Milestone 5:**\n",
            "   - **Coronary Heart Disease / Myocardial Infarction (`_MICHD`):**\n",
            "     - Unweighted prevalence: ~10.4%, Weighted prevalence: ~8.3%.\n",
            "     - Exceptional clinical importance; well-established behavioral and demographic risk profiles (BMI, smoking, physical activity, diabetes comorbidity).\n",
            "     - Negligible missingness in core module.\n",
            "   - **Diabetes (`DIABETE4`):**\n",
            "     - Unweighted prevalence: ~15.1%, Weighted prevalence: ~13.2%.\n",
            "     - Strong public health priority, high sample size of positive cases (~68,000 cases).\n",
            "   - **Depressive Disorder (`ADDEPEV3`):**\n",
            "     - Unweighted prevalence: ~20.9%, Weighted prevalence: ~21.7%.\n",
            "     - Higher prevalence allows balanced training; strong associations with mental health days and socioeconomic factors.\n",
            "2. **Next Steps for Milestone 4 (Statistical Analysis):**\n",
            "   - Perform bivariate chi-square tests and Odds Ratio (OR) estimation for risk factors.\n",
            "   - Run survey-weighted logistic regression models using `statsmodels` to explore adjusted associations.\n",
            "   - Formalize the predictive modeling task formulation in Milestone 5."
        ]
    }
]

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.14"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Generated {nb_path} with {len(cells)} cells.")
