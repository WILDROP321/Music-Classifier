import os
def delete_corrupted_files(folder_path):
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        if os.path.isfile(filepath):
            try:
                # Try to open the file in binary mode
                with open(filepath, 'rb') as f:
                    f.read()
                print(f"[OK] {filename} is fine.")
            except Exception as e:
                # If any error occurs, consider it corrupted and delete
                print(f"[CORRUPTED] {filename} is corrupted. Deleting... Reason: {e}")
                os.remove(filepath)

delete_corrupted_files('TRAINING_DATA_')