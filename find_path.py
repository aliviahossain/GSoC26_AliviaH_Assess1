import os

print("--- Searching for your data folders ---")
for root, dirs, files in os.walk('.'):
    if 'train' in dirs and 'val' in dirs:
        print(f"✅ FOUND THEM! Your data is actually at: {root}")
        print(f"Inside '{root}', I found: {dirs}")
        break
else:
    print("❌ ERROR: I still can't find folders named 'train' and 'val'.")
    print("Current items in this folder:", os.listdir('.'))