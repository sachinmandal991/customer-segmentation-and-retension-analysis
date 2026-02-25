import os
import zipfile
import shutil
from pathlib import Path

print("Searching for Telecom dataset...")

# Common download locations
downloads = Path.home() / "Downloads"
desktop = Path.home() / "Desktop"

# Search for ZIP file
zip_file = None
for location in [downloads, desktop]:
    for file in location.glob("*.zip"):
        if "telco" in file.name.lower() or "churn" in file.name.lower():
            zip_file = file
            break
    if zip_file:
        break

if zip_file:
    print(f"✓ Found: {zip_file}")
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("data/raw")
    
    print("✓ Extracted to data/raw/")
    
    # List files
    print("\nFiles in data/raw/:")
    for file in os.listdir("data/raw"):
        print(f"  - {file}")
    
    print("\n✓ Dataset ready! Run: python main.py")
else:
    print("\nZIP file not found in Downloads or Desktop")
    print("\nManual steps:")
    print("1. Find your downloaded ZIP file")
    print("2. Extract it")
    print("3. Copy WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("4. Paste into: data\\raw\\ folder")
