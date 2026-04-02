import os
import shutil
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from datetime import datetime
import glob

def sync_metadata(curated_dir, metadata_path):
    """Scans curated folder and adds missing files to metadata.csv."""
    if not os.path.exists(curated_dir):
        return

    columns = ['record_id', 'filename', 'date', 'subject', 'activity', 'context', 'clothing', 'relative_path']
    
    # Load existing filenames from metadata
    existing_files = set()
    if os.path.exists(metadata_path):
        try:
            df_existing = pd.read_csv(metadata_path)
            if 'filename' in df_existing.columns:
                existing_files = set(df_existing['filename'].tolist())
        except Exception as e:
            print(f"Warning: Could not read metadata.csv for sync: {e}")

    missing_rows = []
    # Scan all csv files in the folder
    for filepath in glob.glob(os.path.join(curated_dir, "*.csv")):
        filename = os.path.basename(filepath)
        
        # If file is in folder but not in metadata
        if filename not in existing_files:
            # Format: 001_20260329_ali_sitting_clean.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 5:
                try:
                    rec_id = int(parts[0])
                    date_str = parts[1]
                    
                    # Convert to ISO format (YYYY-MM-DD)
                    if len(date_str) == 8 and date_str.isdigit():
                        date_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    else:
                        date_iso = date_str
                        
                    subject = parts[2]
                    activity = parts[3]
                    context = "_".join(parts[4:])
                    clothing = "underclothes" if "underclothes" in filename else "overclothes"
                    
                    missing_rows.append({
                        'record_id': rec_id,
                        'filename': filename,
                        'date': date_iso,
                        'subject': subject,
                        'activity': activity,
                        'context': context,
                        'clothing': clothing,
                        'relative_path': f"curated_csv/{filename}"
                    })
                except Exception as e:
                    print(f"Could not parse {filename} for syncing: {e}")
    
    # Append missing records
    if missing_rows:
        df_missing = pd.DataFrame(missing_rows, columns=columns)
        if os.path.exists(metadata_path):
            df_missing.to_csv(metadata_path, mode='a', header=False, index=False)
        else:
            df_missing.to_csv(metadata_path, mode='w', header=True, index=False)
        print(f"🔄 SYNC: {len(missing_rows)} missing files found and added to metadata.csv!")
    else:
        print("🔄 SYNC: metadata.csv is already up to date.")

def get_next_record_id(curated_dir):
    """Finds the largest ID by checking the files in the curated_csv folder."""
    if not os.path.exists(curated_dir):
        os.makedirs(curated_dir) # Create the folder if it does not exist
        return 0
    
    max_id = -1
    for filepath in glob.glob(os.path.join(curated_dir, "*.csv")):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        if parts and parts[0].isdigit():
            file_id = int(parts[0])
            if file_id > max_id:
                max_id = file_id
                
    return max_id + 1

def append_to_metadata(metadata_path, new_row_dict):
    """Appends the new record to the metadata.csv file."""
    columns = ['record_id', 'filename', 'date', 'subject', 'activity', 'context', 'clothing', 'relative_path']
    df_new = pd.DataFrame([new_row_dict], columns=columns)
    
    if os.path.exists(metadata_path):
        df_new.to_csv(metadata_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(metadata_path, mode='w', header=True, index=False)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    curated_dir = os.path.join(current_dir, "curated_csv")
    metadata_path = os.path.join(current_dir, "metadata.csv")
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    today_str = datetime.now().strftime("%Y%m%d")
    last_subject = ""

    print("--- 🚀 Wearable Sensor Data Labeling Tool ---")
    print(f"Target Folder: {curated_dir}")
    print(f"Metadata File: {metadata_path}\n")

    # Run sync check before starting the main loop
    sync_metadata(curated_dir, metadata_path)
    print("\nReady to process new files...\n")

    while True:
        root.update_idletasks()
        
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Select the raw CSV file to be named",
            filetypes=[("CSV Files", "*.csv")]
        )
        
        if not file_path:
            print("Process terminated or no file selected.")
            break
            
        old_name = os.path.basename(file_path)
        print("-" * 50)
        print(f"Selected File: {old_name}")
        
        current_id = get_next_record_id(curated_dir)
        print(f"ID to be assigned: {current_id:03d}")
        
        date_input = input(f"Date (YYYYMMDD) [{today_str}]: ").strip()
        date_str = date_input if date_input else today_str
        
        if len(date_str) == 8:
            date_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            date_iso = date_str 

        subj_prompt = f"Subject Name [{last_subject}]: " if last_subject else "Subject Name: "
        subject = input(subj_prompt).strip().lower()
        if not subject and last_subject:
            subject = last_subject
        last_subject = subject 

        activity = input("Activity (sitting/walking/standing/running): ").strip().lower()
        
        context = input("Context (clean/noise/falsepositive etc.) [clean]: ").strip().lower()
        if not context: context = "clean"
            
        clothing = input("Clothing (overclothes/underclothes) [overclothes]: ").strip().lower()
        if not clothing: clothing = "overclothes"

        id_str = f"{current_id:03d}"
        new_name = f"{id_str}_{date_str}_{subject}_{activity}_{context}.csv"
        new_path = os.path.join(curated_dir, new_name)
        relative_path = f"curated_csv/{new_name}"
        
        try:
            shutil.move(file_path, new_path)
            print(f"✅ File moved and named: {new_name}")
            
            new_row = {
                'record_id': current_id,
                'filename': new_name,
                'date': date_iso,
                'subject': subject,
                'activity': activity,
                'context': context,
                'clothing': clothing,
                'relative_path': relative_path
            }
            append_to_metadata(metadata_path, new_row)
            print(f"✅ Processed into metadata!")
            
        except Exception as e:
            print(f"❌ Error occurred: {e}")

    root.destroy()

if __name__ == "__main__":
    main()