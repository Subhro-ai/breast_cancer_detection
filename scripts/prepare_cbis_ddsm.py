import pandas as pd
import shutil
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "RAW" / "CBIS-DDSM"
CSV_DIR = RAW_DIR / "csv"
JPEG_DIR = RAW_DIR / "jpeg"

OUT_DIR = BASE_DIR / "data" / "PROCESSED" / "CBIS-DDSM"
BENIGN_DIR = OUT_DIR / "BENIGN"
MALIGNANT_DIR = OUT_DIR / "MALIGNANT"

BENIGN_DIR.mkdir(parents=True, exist_ok=True)
MALIGNANT_DIR.mkdir(exist_ok=True)

# --- 1. LOAD DATA ---
print("Loading Data...")
df_train = pd.read_csv(CSV_DIR / "mass_case_description_train_set.csv")
df_test = pd.read_csv(CSV_DIR / "mass_case_description_test_set.csv")
df_labels = pd.concat([df_train, df_test], ignore_index=True)

# Load Meta and clean columns
df_meta = pd.read_csv(CSV_DIR / "meta.csv")
df_meta.columns = df_meta.columns.str.strip()

# --- 2. INDEX THE DATA ---
print("Indexing Meta.csv...")
# Map SeriesUID -> Folder Name
meta_series_map = set(df_meta['SeriesInstanceUID'].values)

# Map StudyUID -> List of SeriesUIDs belonging to that Study
# This helps us find the ROI folder if we only know the Study ID
meta_study_map = {}
for _, row in df_meta.iterrows():
    study_uid = row['StudyInstanceUID']
    series_uid = row['SeriesInstanceUID']
    if study_uid not in meta_study_map:
        meta_study_map[study_uid] = []
    meta_study_map[study_uid].append(series_uid)

print("Indexing Disk Folders...")
disk_folders = {p.name: p for p in JPEG_DIR.iterdir() if p.is_dir()}

# --- 3. EXECUTE COPY ---
copied = 0
missing = 0

print(f"Processing {len(df_labels)} cases...")

for _, row in df_labels.iterrows():
    # A. Get Label
    label = row["pathology"].strip()
    if label in ["BENIGN", "BENIGN_WITHOUT_CALLBACK"]:
        dst_folder = BENIGN_DIR
    elif label == "MALIGNANT":
        dst_folder = MALIGNANT_DIR
    else:
        continue

    # B. Get UIDs from CSV Path
    # Path format: "Mass... / <StudyUID> / <SeriesUID> / ..."
    # CAUTION: The position of Series/Study in the path varies by dataset version. 
    # We will try extracting ALL parts that look like UIDs.
    
    csv_path_roi = row["cropped image file path"]
    path_parts = csv_path_roi.split("/")
    
    # Extract readable ID for renaming
    case_id = path_parts[0].strip() # e.g. Mass-Training_P_00001_LEFT_CC_1
    
    # Candidate UIDs (parts that start with "1.3.6")
    candidate_uids = [p.strip() for p in path_parts if p.strip().startswith("1.3.6")]
    
    # Also get the "image file path" (Full Mammo) UID just in case
    csv_path_full = row["image file path"]
    candidate_uids += [p.strip() for p in csv_path_full.split("/") if p.strip().startswith("1.3.6")]

    # C. FIND THE FOLDER
    found_folder = None
    
    # Strategy 1: Direct Match (Is one of the CSV UIDs a folder on disk?)
    for uid in candidate_uids:
        if uid in disk_folders:
            found_folder = disk_folders[uid]
            break
            
    # Strategy 2: Study Match (Link via Meta.csv)
    # If Strategy 1 failed, maybe one of the candidates is a STUDY UID.
    # We check if that Study UID exists in meta.csv, and if so, pick its ROI Series.
    if not found_folder:
        for uid in candidate_uids:
            if uid in meta_study_map:
                # We found the Study! Now which Series is the ROI?
                # We look at all Series for this Study
                associated_series = meta_study_map[uid]
                
                for ser in associated_series:
                    # Check if this Series exists on disk
                    if ser in disk_folders:
                        # OPTIONAL: Check if it's actually an ROI?
                        # Since we don't have descriptions, we might just take the first one 
                        # or check file size. For now, we take the first valid match.
                        found_folder = disk_folders[ser]
                        break
            if found_folder: break

    # D. COPY
    if found_folder:
        # Find the jpg inside
        images = list(found_folder.glob("*.jpg"))
        if images:
            shutil.copy(images[0], dst_folder / f"{case_id}.jpg")
            copied += 1
        else:
            # Maybe deep search
            images = list(found_folder.rglob("*.jpg"))
            if images:
                shutil.copy(images[0], dst_folder / f"{case_id}.jpg")
                copied += 1
            else:
                missing += 1
    else:
        missing += 1
        if missing < 5:
            print(f"MISSING: {case_id} (Tried UIDs: {candidate_uids})")

print("\nDONE")
print(f"Copied: {copied}")
print(f"Missing: {missing}")