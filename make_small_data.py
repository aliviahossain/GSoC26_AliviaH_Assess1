import os
import shutil
import random

def create_subset(source, target, num_samples=500):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target):
        os.makedirs(target, exist_ok=True)
    
    # Check if the source exists
    if not os.path.exists(source):
        print(f"❌ Error: Cannot find source folder at {source}")
        return

    for cls in os.listdir(source):
        src_cls_path = os.path.join(source, cls)
        dst_cls_path = os.path.join(target, cls)
        
        if os.path.isdir(src_cls_path):
            os.makedirs(dst_cls_path, exist_ok=True)
            files = [f for f in os.listdir(src_cls_path) if f.endswith('.npy')]
            
            # Pick a subset of images
            num_to_pick = min(num_samples, len(files))
            subset = random.sample(files, num_to_pick)
            
            print(f"Copying {num_to_pick} files for class: {cls}...")
            for f in subset:
                shutil.copy(os.path.join(src_cls_path, f), os.path.join(dst_cls_path, f))

# RUN THIS PART
# Based on your previous search, your data is at 'data/dataset'
create_subset('data/dataset/train', 'small_data/train', num_samples=200)
create_subset('data/dataset/val', 'small_data/val', num_samples=50)

print("✅ DONE! You now have a 'small_data' folder.")