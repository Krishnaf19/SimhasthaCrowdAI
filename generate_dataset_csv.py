import os
import csv
import numpy as np
from PIL import Image
from pathlib import Path

def generate_dataset_csv():
    """Generate CSV metadata for SATARK dataset"""
    
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_FILE = BASE_DIR / 'satark_dataset_metadata.csv'
    
    # Define CSV headers (10 columns)
    headers = [
        'image_id',              # Unique image filename
        'image_path',            # Full path to image
        'image_width',           # Image width in pixels
        'image_height',          # Image height in pixels
        'head_count',            # Total heads detected
        'crowd_density',         # Low/Medium/High classification
        'headwear_type',         # Primary headwear detected
        'dataset_split',         # Train/Test split
        'annotation_confidence', # Confidence level (0-1)
        'lighting'               # Day/Night/Low-light
    ]
    
    # Categorize crowd density
    def get_crowd_density(head_count):
        if head_count <= 50:
            return 'Low'
        elif head_count <= 150:
            return 'Medium'
        else:
            return 'High'
    
    # Categorize primary headwear type
    def get_headwear_type(image_name):
        """Infer headwear type from image metadata or context"""
        name_lower = image_name.lower()
        
        if 'turban' in name_lower or 'saffron' in name_lower or 'ujjain' in name_lower:
            return 'Turban'
        elif 'veil' in name_lower or 'muslim' in name_lower or 'hijab' in name_lower:
            return 'Veil'
        elif 'cap' in name_lower or 'hat' in name_lower:
            return 'Cap'
        elif 'kumbh' in name_lower or 'simhasth' in name_lower:
            return 'Mixed'  # Religious gathering with diverse headgear
        else:
            return 'Mixed'
    
    # Categorize lighting
    def get_lighting(image_path):
        """Estimate lighting from image brightness"""
        try:
            img = Image.open(image_path).convert('RGB')
            # Calculate average brightness
            pixels = np.array(img)
            brightness = np.mean(pixels)
            
            if brightness > 180:
                return 'Day'
            elif brightness > 100:
                return 'Day'  # Overcast or shaded
            else:
                return 'Low-light'
        except:
            return 'Unknown'
    
    # Collect all images
    rows = []
    
    # Process all images in data/images directory
    images_dir = DATA_DIR / 'images'
    if images_dir.exists():
        for img_file in sorted(images_dir.glob('*')):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp']:
                try:
                    # Load image to get dimensions
                    img = Image.open(img_file)
                    width, height = img.size
                    
                    # Load heatmap to get head count
                    heatmap_file = DATA_DIR / 'heatmaps' / f"{img_file.stem}.npy"
                    head_count = 0
                    confidence = 0.85
                    
                    if heatmap_file.exists():
                        heatmap = np.load(heatmap_file)
                        head_count = int(np.round(heatmap.sum()))
                    else:
                        # Use average if heatmap doesn't exist
                        head_count = 200  # Default estimate
                        confidence = 0.60
                    
                    # Determine dataset split
                    train_dir = DATA_DIR / 'Train' / 'images' / img_file.name
                    test_dir = DATA_DIR / 'Test' / 'images' / img_file.name
                    
                    if train_dir.exists():
                        split = 'Train'
                    elif test_dir.exists():
                        split = 'Test'
                    else:
                        split = 'Validation'
                    
                    # Create row
                    row = {
                        'image_id': img_file.stem,
                        'image_path': f"data/images/{img_file.name}",
                        'image_width': width,
                        'image_height': height,
                        'head_count': head_count,
                        'crowd_density': get_crowd_density(head_count),
                        'headwear_type': get_headwear_type(img_file.name),
                        'dataset_split': split,
                        'annotation_confidence': f"{confidence:.2f}",
                        'lighting': get_lighting(img_file)
                    }
                    rows.append(row)
                    print(f"✓ Processed: {img_file.name} ({head_count} heads, {split})")
                    
                except Exception as e:
                    print(f"✗ Error processing {img_file.name}: {e}")
    
    # Write CSV file
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✅ Dataset CSV created: {OUTPUT_FILE}")
        print(f"📊 Total images: {len(rows)}")
        
        # Print statistics
        train_count = sum(1 for r in rows if r['dataset_split'] == 'Train')
        test_count = sum(1 for r in rows if r['dataset_split'] == 'Test')
        val_count = sum(1 for r in rows if r['dataset_split'] == 'Validation')
        
        print(f"   Train: {train_count} | Test: {test_count} | Validation: {val_count}")
        
        # Density distribution
        low = sum(1 for r in rows if r['crowd_density'] == 'Low')
        medium = sum(1 for r in rows if r['crowd_density'] == 'Medium')
        high = sum(1 for r in rows if r['crowd_density'] == 'High')
        
        print(f"\n📈 Crowd Density Distribution:")
        print(f"   Low: {low} | Medium: {medium} | High: {high}")
        
        # Headwear distribution
        headwear_counts = {}
        for r in rows:
            hw = r['headwear_type']
            headwear_counts[hw] = headwear_counts.get(hw, 0) + 1
        
        print(f"\n👜 Headwear Type Distribution:")
        for hw, count in sorted(headwear_counts.items()):
            print(f"   {hw}: {count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error writing CSV: {e}")
        return False


if __name__ == '__main__':
    print("🔄 Generating SATARK Dataset CSV...\n")
    generate_dataset_csv()
    print("\n✨ Done!")
