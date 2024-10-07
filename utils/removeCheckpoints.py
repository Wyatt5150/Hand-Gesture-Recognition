import os
import shutil

# Path to the checkpoints directory
checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'utils', 'checkpoints')

# Remove all checkpoint files
if os.path.exists(checkpoints_dir):
    shutil.rmtree(checkpoints_dir)  # This will delete the directory and all its contents
    print("All old checkpoints removed.")
else:
    print("Checkpoints directory does not exist.")
