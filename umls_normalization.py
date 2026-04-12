import os
import re
import json
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional


# =========================
# UMLS RRF column schemas
# =========================

MRCONSO_COLUMNS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI",
    "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"
]

MRSTY_COLUMNS = [
    "CUI", "TUI", "STN", "STY", "ATUI", "CVF"
]


# =========================
# Utility functions
# =========================

def clean_text(text: str) -> str:
    """
    Normalize a symptom string for matching.
    """
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[/,;]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def safe_read_rrf(path: str, columns: List[str]) -> pd.DataFrame:
    """
    Read a UMLS .RRF file.
    UMLS RRF files are pipe-delimited and usually end with an extra trailing pipe,
    so we use usecols to keep only the actual schema columns.
    """
    return pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns + ["EXTRA"],
        usecols=range(len(columns)),
        dtype=str,
        encoding="utf-8",
        low_memory=False
    )


# =========================
# Step 1: Read UMLS files
# =========================

def load_umls(mrconso_path: str, mrsty_path: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    mrconso = safe_read_rrf(mrconso_path, MRCONSO_COLUMNS)
    mrsty = safe_read_rrf(mrsty_path, MRSTY_COLUMNS) if mrsty_path else None
    return mrconso, mrsty


# =========================
# Step 2: Keep only symptom CUIs
# =========================

def get_symptom_cuis(mrsty: Optional[pd.DataFrame], mrconso: pd.DataFrame) -> Set[str]:
    """
    Keep CUIs whose semantic type is 'Sign or Symptom'.
    If mrsty is None, use all English CUIs.
    """
    if mrsty is not None:
        symptom_cuis = set(
            mrsty.loc[mrsty["STY"] == "Sign or Symptom", "CUI"].dropna().unique()
        )
    else:
        # Fallback: all English CUIs
        symptom_cuis = set(mrconso[mrconso["LAT"] == "ENG"]["CUI"].dropna().unique())
    return symptom_cuis


# =========================
# Step 3: Filter English terms
# =========================

def get_english_symptom_terms(
    mrconso: pd.DataFrame,
    symptom_cuis: Set[str],
    preferred_sources: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Extract English terms for symptom CUIs.

    preferred_sources can be used to restrict to specific vocabularies, e.g.
    {"SNOMEDCT_US", "MSH"}.
    If None, all English sources are kept.
    """
    df = mrconso[mrconso["CUI"].isin(symptom_cuis)].copy()
    df = df[df["LAT"] == "ENG"].copy()

    if preferred_sources:
        df = df[df["SAB"].isin(preferred_sources)].copy()

    df["STR_CLEAN"] = df["STR"].apply(clean_text)

    # Remove empty / weird rows
    df = df[df["STR_CLEAN"] != ""].copy()

    return df


# =========================
# Step 4: Build dictionaries
# =========================

def choose_canonical_term(group: pd.DataFrame) -> str:
    """
    Pick a canonical name for a CUI.

    Preference order:
    1. ISPREF == Y
    2. TTY in {"PT", "PN"}
    3. shortest cleaned string
    """
    preferred = group[group["ISPREF"] == "Y"]
    if not preferred.empty:
        group = preferred

    tty_pref = group[group["TTY"].isin(["PT", "PN"])]
    if not tty_pref.empty:
        group = tty_pref

    group = group.sort_values(by="STR_CLEAN", key=lambda s: s.str.len())
    return group.iloc[0]["STR_CLEAN"]


def build_umls_symptom_maps(
    symptom_terms: pd.DataFrame
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
    - term_to_cui
    - term_to_canonical
    - cui_to_terms
    """
    term_to_cui: Dict[str, str] = {}
    term_to_canonical: Dict[str, str] = {}
    cui_to_terms: Dict[str, List[str]] = {}

    canonical_by_cui: Dict[str, str] = {}

    for cui, group in symptom_terms.groupby("CUI"):
        unique_terms = sorted(set(group["STR_CLEAN"].dropna().tolist()))
        if not unique_terms:
            continue

        canonical = choose_canonical_term(group)
        canonical_by_cui[cui] = canonical
        cui_to_terms[cui] = unique_terms

    for cui, terms in cui_to_terms.items():
        canonical = canonical_by_cui[cui]
        for term in terms:
            term_to_cui[term] = cui
            term_to_canonical[term] = canonical

    return term_to_cui, term_to_canonical, cui_to_terms


# =========================
# Step 5: Extra manual synonym overrides
# =========================

def add_manual_overrides(
    term_to_canonical: Dict[str, str],
    manual_map: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Add or override mappings with your own dataset-specific symptom map.
    """
    updated = dict(term_to_canonical)
    if manual_map:
        for k, v in manual_map.items():
            updated[clean_text(k)] = clean_text(v)
    return updated


# =========================
# Step 6: Normalize symptom strings
# =========================

def normalize_symptom(symptom: str, term_to_canonical: Dict[str, str]) -> str:
    symptom_clean = clean_text(symptom)
    if not symptom_clean:
        return ""
    return term_to_canonical.get(symptom_clean, symptom_clean)


# =========================
# Step 7A: Normalize binary-column dataset
# =========================

def normalize_binary_dataset_columns(
    df: pd.DataFrame,
    label_col: str,
    term_to_canonical: Dict[str, str]
) -> pd.DataFrame:
    """
    For a dataset where columns are symptoms and values are 0/1.

    Example:
    fever, cough, headache, disease
    1,     0,     1,        flu

    If multiple original columns map to the same canonical symptom,
    they are merged by max().
    """
    feature_cols = [c for c in df.columns if c != label_col]

    canonical_groups: Dict[str, List[str]] = {}
    for col in feature_cols:
        canonical = normalize_symptom(col, term_to_canonical)
        canonical_groups.setdefault(canonical, []).append(col)

    normalized = pd.DataFrame(index=df.index)

    for canonical, cols in canonical_groups.items():
        if len(cols) == 1:
            normalized[canonical] = df[cols[0]]
        else:
            normalized[canonical] = df[cols].max(axis=1)

    normalized[label_col] = df[label_col]
    return normalized


# =========================
# Step 7B: Normalize string-symptom dataset
# =========================

def normalize_string_symptom_dataset(
    df: pd.DataFrame,
    symptom_cols: List[str],
    label_col: str,
    term_to_canonical: Dict[str, str]
) -> pd.DataFrame:
    """
    For a dataset like:
    Disease, Symptom_1, Symptom_2, Symptom_3

    It normalizes symptom strings but does not yet convert to binary.
    """
    out = df.copy()

    for col in symptom_cols:
        out[col] = out[col].apply(
            lambda x: normalize_symptom(x, term_to_canonical) if pd.notna(x) else x
        )

    out[label_col] = out[label_col].astype(str).str.strip()
    return out


# =========================
# Step 8: Convert string dataset to binary matrix
# =========================

def string_dataset_to_binary(
    df: pd.DataFrame,
    symptom_cols: List[str],
    label_col: str
) -> pd.DataFrame:
    """
    Convert normalized string-symptom dataset to binary matrix.
    """
    all_symptoms: Set[str] = set()

    for col in symptom_cols:
        vals = df[col].dropna().astype(str).tolist()
        vals = [clean_text(v) for v in vals if clean_text(v)]
        all_symptoms.update(vals)

    all_symptoms = sorted(all_symptoms)

    rows = []
    for _, row in df.iterrows():
        entry = {symptom: 0 for symptom in all_symptoms}

        for col in symptom_cols:
            val = row[col]
            if pd.notna(val):
                symptom = clean_text(val)
                if symptom:
                    entry[symptom] = 1

        entry[label_col] = row[label_col]
        rows.append(entry)

    return pd.DataFrame(rows)


# =========================
# Step 9: Align and merge two binary datasets
# =========================

def align_and_merge_binary_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label_col: str
) -> pd.DataFrame:
    feature_union = sorted(
        set(df1.columns) | set(df2.columns) - {label_col}
    )

    cols = feature_union + [label_col]

    df1_aligned = df1.reindex(columns=cols, fill_value=0)
    df2_aligned = df2.reindex(columns=cols, fill_value=0)

    merged = pd.concat([df1_aligned, df2_aligned], ignore_index=True)
    merged = merged.drop_duplicates()

    return merged


# =========================
# Step 10: Save dictionaries
# =========================

def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================
# Example main pipeline
# =========================

def main():
    # -------------------------
    # 1. Set your file paths
    # -------------------------
    umls_dir = "datasets"  # change this
    mrconso_path = os.path.join(umls_dir, "MRCONSO.RRF")
    mrsty_path = os.path.join(umls_dir, "MRSTY.RRF")

    # Your current binary dataset
    binary_dataset_path = os.path.join(umls_dir, "Final_Augmented_dataset_Diseases_and_Symptoms.csv")

    # Your second string-based dataset (if available)
    string_dataset_path = None

    # Output folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # 2. Load UMLS
    # -------------------------
    print("Loading UMLS files...")
    mrconso, mrsty = load_umls(mrconso_path, mrsty_path)

    # -------------------------
    # 3. Get symptom concepts
    # -------------------------
    print("Filtering symptom CUIs...")
    symptom_cuis = get_symptom_cuis(mrsty, mrconso)

    # Optional: restrict to selected vocabularies if you want
    # preferred_sources = {"SNOMEDCT_US", "MSH"}
    preferred_sources = None

    symptom_terms = get_english_symptom_terms(
        mrconso=mrconso,
        symptom_cuis=symptom_cuis,
        preferred_sources=preferred_sources
    )

    print(f"Symptom CUIs: {len(symptom_cuis)}")
    print(f"Symptom terms: {len(symptom_terms)}")

    # -------------------------
    # 4. Build maps
    # -------------------------
    print("Building normalization maps...")
    term_to_cui, term_to_canonical, cui_to_terms = build_umls_symptom_maps(symptom_terms)

    # Add your custom dataset-specific overrides here
    manual_map = {
        "high temperature": "fever",
        "tiredness": "fatigue",
        "coughing": "cough",
        "body pain": "body ache",
        "stomach pain": "abdominal pain",
        "throwing up": "vomiting",
    }

    term_to_canonical = add_manual_overrides(term_to_canonical, manual_map)

    save_json(term_to_cui, os.path.join(output_dir, "term_to_cui.json"))
    save_json(term_to_canonical, os.path.join(output_dir, "term_to_canonical.json"))
    save_json(cui_to_terms, os.path.join(output_dir, "cui_to_terms.json"))

    # -------------------------
    # 5. Normalize current binary dataset
    # -------------------------
    print("Normalizing binary-column dataset...")
    df_binary = pd.read_csv(binary_dataset_path)

    # Change this to your actual label column
    binary_label_col = "diseases"
    if binary_label_col not in df_binary.columns:
        if "prognosis" in df_binary.columns:
            binary_label_col = "prognosis"

    # Remove any accidental unnamed columns
    df_binary = df_binary.loc[:, ~df_binary.columns.astype(str).str.contains("^Unnamed")]

    df_binary_norm = normalize_binary_dataset_columns(
        df=df_binary,
        label_col=binary_label_col,
        term_to_canonical=term_to_canonical
    )

    df_binary_norm.to_csv(
        os.path.join(output_dir, "binary_dataset_normalized.csv"),
        index=False
    )

    # -------------------------
    # 6. Normalize second binary dataset
    # -------------------------
    if os.path.exists(string_dataset_path):
        print("Normalizing second binary dataset...")
        df_second_binary = pd.read_csv(string_dataset_path)

        # Assume label column is "diseases" or similar
        second_label_col = "diseases"
        if second_label_col not in df_second_binary.columns:
            if "Disease" in df_second_binary.columns:
                second_label_col = "Disease"
            elif "disease" in df_second_binary.columns:
                second_label_col = "disease"
            else:
                raise ValueError("Could not find disease label column in second dataset.")

        # Remove any accidental unnamed columns
        df_second_binary = df_second_binary.loc[:, ~df_second_binary.columns.astype(str).str.contains("^Unnamed")]

        df_second_norm = normalize_binary_dataset_columns(
            df=df_second_binary,
            label_col=second_label_col,
            term_to_canonical=term_to_canonical
        )

        df_second_norm.to_csv(
            os.path.join(output_dir, "second_binary_dataset_normalized.csv"),
            index=False
        )

        process_second = True
    else:
        print("Second dataset not found, skipping...")
        process_second = False

    # -------------------------
    # 8. Align label names before merge
    # -------------------------
    if binary_label_col != "disease":
        df_binary_norm = df_binary_norm.rename(columns={binary_label_col: "disease"})
    if process_second and second_label_col != "disease":
        df_second_norm = df_second_norm.rename(columns={second_label_col: "disease"})

    # Optional disease normalization
    disease_map = {
        "flu": "influenza",
        "common cold": "cold",
    }

    df_binary_norm["disease"] = (
        df_binary_norm["disease"].astype(str).str.strip().str.lower().replace(disease_map)
    )
    if process_second:
        df_second_norm["disease"] = (
            df_second_norm["disease"].astype(str).str.strip().str.lower().replace(disease_map)
        )

    # -------------------------
    # 9. Merge
    # -------------------------
    print("Merging datasets...")
    if process_second:
        merged = align_and_merge_binary_datasets(
            df1=df_binary_norm,
            df2=df_second_norm,
            label_col="disease"
        )
    else:
        merged = df_binary_norm.copy()

    merged.to_csv(os.path.join(output_dir, "merged_dataset.csv"), index=False)

    print("Done.")
    print(f"Normalized binary dataset shape: {df_binary_norm.shape}")
    if process_second:
        print(f"Second dataset shape:    {df_second_norm.shape}")
    print(f"Merged dataset shape:           {merged.shape}")
    print(f"Unique diseases:                {merged['disease'].nunique()}")


if __name__ == "__main__":
    main()