# import os
# import csv
# import xml.etree.ElementTree as ET
# import random
# import json

# def build_master_index(img_dir='Images', xml_dir='Annotations', output_csv='simhastha_master_index.csv', train_ratio=0.8):
#     print(" Step 1: Indexing & Preparing Simhastha Dataset...")

#     # Ensure output directories exist
#     os.makedirs('data/images', exist_ok=True)
#     os.makedirs('data/annotations', exist_ok=True)

#     if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
#         print(" Error: Could not find 'Images' or 'Annotations' folder.")
#         return

#     all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
#     xml_data = {}
    
#     # 1. Parse XMLs and Extract COORDINATES
#     for xml_file in xml_files:
#         xml_path = os.path.join(xml_dir, xml_file)
#         try:
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             for img_tag in root.findall('image'):
#                 name = img_tag.get('name')
#                 w = img_tag.get('width')
#                 h = img_tag.get('height')
                
#                 # Extract actual (x, y) points for heatmap generation
#                 points_list = []
#                 for p_tag in img_tag.findall('points'):
#                     # CVAT format is "x,y"
#                     label_points = p_tag.get('points')
#                     if label_points:
#                         x, y = map(float, label_points.split(','))
#                         points_list.append({'x': x, 'y': y})
                
#                 # Save coordinates to a small JSON file for Step 2
#                 json_name = os.path.splitext(name)[0] + '.json'
#                 with open(os.path.join('data/annotations', json_name), 'w') as jf:
#                     json.dump(points_list, jf)
                
#                 # Copy image to data folder for centralized access
#                 if os.path.exists(os.path.join(img_dir, name)):
#                     import shutil
#                     shutil.copy(os.path.join(img_dir, name), os.path.join('data/images', name))

#                 xml_data[name] = {
#                     'count': len(points_list),
#                     'xml_file': xml_file,
#                     'width': w,
#                     'height': h
#                 }
#         except Exception as e:
#             print(f"Warning: Could not read {xml_file}. Error: {e}")

#     # 2. Split Logic
#     labeled_list = [img for img in all_images if img in xml_data]
#     random.shuffle(labeled_list)
#     split_index = int(len(labeled_list) * train_ratio)
#     train_images = set(labeled_list[:split_index])

#     # 3. Write Master CSV
#     with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['image_name', 'status', 'head_count', 'source_xml', 'width', 'height', 'split_assignment'])

#         for img_name in all_images:
#             if img_name in xml_data:
#                 d = xml_data[img_name]
#                 split = 'Train' if img_name in train_images else 'Test'
#                 writer.writerow([img_name, 'Labeled', d['count'], d['xml_file'], d['width'], d['height'], split])
#             else:
#                 writer.writerow([img_name, 'Unlabeled', 0, 'None', 'Unknown', 'Unknown', 'Inference'])

#     print(f"\n Done! Labeled: {len(labeled_list)} images.")
#     print(f"📂 Centralized data ready in the 'data/' folder.")

# if __name__ == '__main__':
#     build_master_index()


import os
import csv
import xml.etree.ElementTree as ET
import random
import json
import shutil  # FIX 1: Moved out of the loop

RANDOM_SEED = 42  # FIX 2: Reproducible splits

def build_master_index(
    img_dir='Images',
    xml_dir='Annotations',
    output_csv='simhastha_master_index.csv',
    train_ratio=0.8
):
    print("Step 1: Indexing & Preparing Simhastha Dataset...")

    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/annotations', exist_ok=True)

    if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
        print(f"Error: Could not find '{img_dir}' or '{xml_dir}' folder.")
        return

    all_images = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    xml_data = {}

    # --- Step 1: Parse XMLs and extract coordinates ---
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for img_tag in root.findall('image'):
                name = img_tag.get('name')
                if not name:
                    continue

                # FIX 3: Warn on duplicate image entries across XMLs
                if name in xml_data:
                    print(f"  Warning: '{name}' already parsed from "
                          f"'{xml_data[name]['xml_file']}'. "
                          f"Skipping duplicate in '{xml_file}'.")
                    continue

                # FIX 4: Safe integer conversion for width/height
                try:
                    w = int(img_tag.get('width', 0))
                    h = int(img_tag.get('height', 0))
                except (ValueError, TypeError):
                    w, h = 0, 0
                    print(f"  Warning: Could not parse dimensions for '{name}'.")

                points_list = []
                for p_tag in img_tag.findall('points'):
                    raw = p_tag.get('points', '')
                    if not raw:
                        continue

                    # FIX 5: Handle both single "x,y" and multi "x1,y1;x2,y2" formats
                    for point_str in raw.split(';'):
                        point_str = point_str.strip()
                        if not point_str:
                            continue
                        parts = point_str.split(',')
                        if len(parts) == 2:
                            try:
                                x, y = float(parts[0]), float(parts[1])
                                points_list.append({'x': x, 'y': y})
                            except ValueError:
                                print(f"  Warning: Skipping malformed point "
                                      f"'{point_str}' in '{name}'.")
                        else:
                            print(f"  Warning: Unexpected point format "
                                  f"'{point_str}' in '{name}'. Skipping.")

                # Save coordinates to JSON
                json_name = os.path.splitext(name)[0] + '.json'
                json_path = os.path.join('data/annotations', json_name)
                with open(json_path, 'w') as jf:
                    json.dump(points_list, jf, indent=2)

                # Copy image to centralized data folder
                src = os.path.join(img_dir, name)
                dst = os.path.join('data/images', name)
                if os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    print(f"  Warning: Image file not found for '{name}'. "
                          f"Skipping copy.")

                xml_data[name] = {
                    'count': len(points_list),
                    'xml_file': xml_file,
                    'width': w,
                    'height': h
                }

        except ET.ParseError as e:
            print(f"  Warning: Could not parse XML '{xml_file}'. Error: {e}")
        except Exception as e:
            print(f"  Warning: Unexpected error reading '{xml_file}'. Error: {e}")

    # --- Step 2: Train/Test split with fixed seed ---
    labeled_list = [img for img in all_images if img in xml_data]
    random.seed(RANDOM_SEED)  # FIX 2: Reproducible every run
    random.shuffle(labeled_list)

    split_index = int(len(labeled_list) * train_ratio)
    train_images = set(labeled_list[:split_index])
    test_images  = set(labeled_list[split_index:])

    # --- Step 3: Write Master CSV ---
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'image_name', 'status', 'head_count',
            'source_xml', 'width', 'height', 'split_assignment'
        ])

        for img_name in all_images:
            if img_name in xml_data:
                d = xml_data[img_name]
                split = 'Train' if img_name in train_images else 'Test'
                writer.writerow([
                    img_name, 'Labeled', d['count'],
                    d['xml_file'], d['width'], d['height'], split
                ])
            else:
                writer.writerow([
                    img_name, 'Unlabeled', 0,
                    'None', 'Unknown', 'Unknown', 'Inference'
                ])

    # FIX 6: Detailed summary
    unlabeled_count = len(all_images) - len(labeled_list)
    print(f"\n Done! Master index saved to '{output_csv}'")
    print(f"  Total images found : {len(all_images)}")
    print(f"  Labeled            : {len(labeled_list)}")
    print(f"    -> Train          : {len(train_images)}")
    print(f"    -> Test           : {len(test_images)}")
    print(f"  Unlabeled (Inference): {unlabeled_count}")
    print(f"  Centralized data ready in 'data/' folder.")

if __name__ == '__main__':
    build_master_index()