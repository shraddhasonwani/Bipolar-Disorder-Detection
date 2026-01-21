import os
import shutil

source_folder = "BP_Dataset"
organized_folder = "BP_Organized_Dataset"

# Aligning category names with your feature extraction script
file_types = {
    "audio": [".wav", ".mp3", "_COVAREP.csv", "_FORMANT.csv"],
    "openface_output": ["_CLNF_AUs.txt", "_CLNF_features.txt", "_CLNF_pose.txt", "_CLNF_gaze.txt"], 
    "transcript": ["_TRANSCRIPT.csv"]
}

# Create folders
for folder in list(file_types.keys()) + ["others"]:
    os.makedirs(os.path.join(organized_folder, folder), exist_ok=True)

# Walk through DAIC-WOZ folders
for root, dirs, files in os.walk(source_folder):
    # Skip the root folder itself if it has no files
    if root == source_folder:
        continue

    for file_name in files:
        file_path = os.path.join(root, file_name)
        
        # Determine the destination
        target_category = "others"
        for category, patterns in file_types.items():
            if any(p.lower() in file_name.lower() for p in patterns):
                target_category = category
                break
        
        # CRITICAL: Keep participant ID in filename or use subfolders
        # This prevents overwriting and makes feature extraction easier
        dest_path = os.path.join(organized_folder, target_category, file_name)
        
        shutil.copy2(file_path, dest_path)
        print(f"✅ Sorted: {file_name} -> {target_category}")

print("\nAll DAIC-WOZ files are organized and ready for preprocessing!")