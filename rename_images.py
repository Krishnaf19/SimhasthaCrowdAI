#!/usr/bin/env python3
"""
Rename images with meaningful names based on location, crowd density, and lighting.
Naming convention: Location_Density_Lighting_Split_Number.ext
Examples: Ujjain_High_Day_Train_001.jpg, Prayagraj_Low_Lowlight_Test_002.webp
"""

import os
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# Define project paths
project_root = Path(__file__).parent
images_dir = project_root / "Images"
data_images_dir = project_root / "data" / "images"
csv_file = project_root / "satark_dataset_metadata.csv"

# Mapping: Extract location from old filenames
location_mapping = {
    "ujjain": "Ujjain",
    "simhastha": "Simhastha",
    "prayagraj": "Prayagraj",
    "kumbh": "KumbhMela",
    "mahakumbh": "MahaKumbh",
    "maha-kumbh": "MahaKumbh",
    "getty": "Festival",
    "istockphoto": "Festival",
    "ganga_river": "GangaRiver",
    "pexels": "Festival",
    "hindu": "HinduGathering",
    "sadhu": "SadhuProcession",
    "naga": "NagaSadhu",
    "india": "India",
    "ram_ghat": "RamGhat",
    "ram ghat": "RamGhat",
}

def get_location(old_name):
    """Extract location from old filename."""
    old_name_lower = old_name.lower()
    for keyword, location in location_mapping.items():
        if keyword in old_name_lower:
            return location
    return "KumbhMela"  # Default

def abbreviate_density(density):
    """Convert density to abbreviation."""
    if density == "Low":
        return "Low"
    elif density == "Medium":
        return "Med"
    elif density == "High":
        return "High"
    return "Med"

def abbreviate_split(split):
    """Convert dataset split to abbreviation."""
    if split == "Train":
        return "Train"
    elif split == "Test":
        return "Test"
    elif split == "Validation":
        return "Val"
    return "Train"

def get_file_extension(filename):
    """Get file extension."""
    return Path(filename).suffix

# Read CSV and create rename mapping
rename_mapping = {}  # old_name -> new_name

print("Reading dataset metadata...")
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Initialize counters for tracking current count
location_counts = defaultdict(int)

# Second pass: create new names
print("\nGenerating new filenames...")
for row in rows:
    old_name = row['image_id']
    location = get_location(old_name)
    density = abbreviate_density(row['crowd_density'])
    lighting = "Day" if row['lighting'] == "Day" else "Lowlight"
    split = abbreviate_split(row['dataset_split'])
    ext = get_file_extension(row['image_path'])
    
    key = f"{location}_{density}_{lighting}_{split}"
    location_counts[key] += 1
    
    count_num = str(location_counts[key]).zfill(3)
    new_name = f"{location}_{density}_{lighting}_{split}_{count_num}{ext}"
    
    rename_mapping[old_name] = new_name
    print(f"  {old_name} -> {new_name}")

# Rename files in Images directory
print("\n" + "="*80)
print("Renaming files in Images directory...")
print("="*80)

rename_count = 0
for old_name, new_name in rename_mapping.items():
    old_path = images_dir / old_name
    new_path = images_dir / new_name
    
    if old_path.exists():
        old_path.rename(new_path)
        print(f"✓ Renamed: {old_name} -> {new_name}")
        rename_count += 1
    else:
        print(f"✗ NOT FOUND: {old_name}")

print(f"\nTotal files renamed in Images: {rename_count}")

# Also rename in data/images if they exist
print("\n" + "="*80)
print("Renaming files in data/images directory...")
print("="*80)

data_rename_count = 0
for old_name, new_name in rename_mapping.items():
    old_path = data_images_dir / old_name
    new_path = data_images_dir / new_name
    
    if old_path.exists():
        old_path.rename(new_path)
        print(f"✓ Renamed: {old_name} -> {new_name}")
        data_rename_count += 1

print(f"\nTotal files renamed in data/images: {data_rename_count}")

# Update CSV file
print("\n" + "="*80)
print("Updating CSV metadata...")
print("="*80)

updated_rows = []
for row in rows:
    old_name = row['image_id']
    if old_name in rename_mapping:
        new_name = rename_mapping[old_name]
        row['image_id'] = new_name
        row['image_path'] = f"data/images/{new_name}"
        updated_rows.append(row)
        print(f"✓ Updated CSV: {old_name} -> {new_name}")

# Write updated CSV
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"\n✓ CSV updated with {len(updated_rows)} entries")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total images renamed: {rename_count}")
print(f"Total data images renamed: {data_rename_count}")
print(f"CSV entries updated: {len(updated_rows)}")
print("\nNaming Convention: Location_Density_Lighting_Split_Number.ext")
print("Examples:")
for i, (old, new) in enumerate(list(rename_mapping.items())[:5]):
    print(f"  {old} -> {new}")
print(f"  ... and {len(rename_mapping) - 5} more")
