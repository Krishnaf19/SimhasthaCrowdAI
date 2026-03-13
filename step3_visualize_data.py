# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# def generate_previews(data_dir='data', output_dir='previews'):
#     print("Step 3: Generating Visual Previews for Quality Assurance...")
    
#     os.makedirs(output_dir, exist_ok=True)

#     for split in ['Train', 'Test']:
#         img_folder = os.path.join(data_dir, split, 'images')
#         heat_folder = os.path.join(data_dir, split, 'heatmaps') # Matches Step 2
        
#         if not os.path.exists(heat_folder): 
#             print(f"Skipping {split}: Heatmaps folder not found.")
#             continue

#         for npy_name in os.listdir(heat_folder):
#             if not npy_name.endswith('.npy'): continue
            
#             # 1. Load the Heatmap Math
#             npy_path = os.path.join(heat_folder, npy_name)
#             density_map = np.load(npy_path)
            
#             # 2. Find the matching Image
#             base_name = os.path.splitext(npy_name)[0]
#             img_path = None
#             for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
#                 temp_path = os.path.join(img_folder, base_name + ext)
#                 if os.path.exists(temp_path):
#                     img_path = temp_path
#                     break
            
#             if img_path is None:
#                 print(f" Image not found for {npy_name}")
#                 continue

#             # 3. Plotting the comparison
#             plt.figure(figsize=(15, 7))
            
#             # Left: Original Image
#             plt.subplot(1, 2, 1)
#             plt.title(f"Original: {os.path.basename(img_path)}")
#             img = Image.open(img_path)
#             plt.imshow(img)
#             plt.axis('off')
            
#             # Right: Overlay (Heatmap on top of Image)
#             plt.subplot(1, 2, 2)
#             count = np.sum(density_map)
#             plt.title(f"Heatmap Overlay (Total Count: {count:.1f})")
            
#             # Show image first
#             plt.imshow(img)
#             # Show heatmap with transparency (alpha)
#             # We use 'jet' for the classic red/blue heat look
#             plt.imshow(density_map, cmap='jet', alpha=0.6) 
#             plt.axis('off')
            
#             # Save the comparison
#             save_name = f"{split}_{base_name}_preview.jpg"
#             save_path = os.path.join(output_dir, save_name)
#             plt.savefig(save_path, bbox_inches='tight', dpi=150)
#             plt.close()
#             print(f" Saved preview: {save_name}")

#     print(f"\n All previews saved to the '{output_dir}/' folder!")

# if __name__ == '__main__':
#     generate_previews()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def generate_previews(data_dir: str = 'data', output_dir: str = 'previews') -> None:
    print("Step 3: Generating Visual Previews for Quality Assurance...")
    os.makedirs(output_dir, exist_ok=True)

    # FIX 5: Added .JPEG and .PNG uppercase variants
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

    total_saved   = 0
    total_skipped = 0

    for split in ['Train', 'Test']:
        img_folder  = os.path.join(data_dir, split, 'images')
        heat_folder = os.path.join(data_dir, split, 'heatmaps')

        if not os.path.exists(heat_folder):
            print(f"  Skipping {split}: heatmaps folder not found.")
            continue
        if not os.path.exists(img_folder):
            print(f"  Skipping {split}: images folder not found.")
            continue

        npy_files = [f for f in os.listdir(heat_folder) if f.endswith('.npy')]
        if not npy_files:
            print(f"  Skipping {split}: no .npy files found.")
            continue

        print(f"\n  [{split}] — {len(npy_files)} heatmaps found.")

        for npy_name in sorted(npy_files):
            base_name = os.path.splitext(npy_name)[0]

            # ── Locate matching image ─────────────────────────────────────────
            img_path = None
            for ext in IMAGE_EXTENSIONS:
                candidate = os.path.join(img_folder, base_name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                print(f"    Skipping '{npy_name}' — no matching image found.")
                total_skipped += 1
                continue

            # ── Load data ─────────────────────────────────────────────────────
            npy_path    = os.path.join(heat_folder, npy_name)
            density_map = np.load(npy_path)

            # FIX 1: Use context manager so file handle is always released
            with Image.open(img_path) as pil_img:
                img_rgb = np.array(pil_img.convert('RGB'))   # ensure 3-channel

            img_h, img_w = img_rgb.shape[:2]
            map_h, map_w = density_map.shape[:2]

            # FIX 2: Resize density map to match image if dimensions differ
            if (map_h, map_w) != (img_h, img_w):
                print(f"    Note: resizing density map "
                      f"({map_w}×{map_h}) → ({img_w}×{img_h}) for '{npy_name}'.")
                from PIL import Image as PILImage
                # Use NEAREST so float values aren't interpolated into garbage
                dm_pil = PILImage.fromarray(density_map)
                dm_pil = dm_pil.resize((img_w, img_h), resample=PILImage.BILINEAR)
                density_map_display = np.array(dm_pil)
            else:
                density_map_display = density_map

            # FIX 4: Integer count display
            count     = float(np.sum(density_map))   # use original (not resized) for accuracy
            count_int = int(round(count))

            # ── Plotting ─────────────────────────────────────────────────────
            # FIX 6: Wrap in try/finally so figure is always closed on error
            fig = None
            try:
                fig, axes = plt.subplots(1, 2, figsize=(15, 7))
                fig.suptitle(
                    f"{split} — {os.path.basename(img_path)}",
                    fontsize=13, fontweight='bold'
                )

                # Left: Original image
                axes[0].set_title("Original Image", fontsize=11)
                axes[0].imshow(img_rgb)
                axes[0].axis('off')

                # Right: Heatmap overlay (correctly aligned)
                axes[1].set_title(
                    f"Density Map Overlay  |  Count: {count_int}",
                    fontsize=11
                )
                axes[1].imshow(img_rgb)
                axes[1].imshow(
                    density_map_display,
                    cmap='jet',
                    alpha=0.55,
                    interpolation='bilinear'
                )
                axes[1].axis('off')

                # Colourbar — shows density scale
                sm = cm.ScalarMappable(cmap='jet')
                sm.set_array(density_map_display)
                fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04,
                             label='Density')

                plt.tight_layout()

                save_name = f"{split}_{base_name}_preview.jpg"
                save_path = os.path.join(output_dir, save_name)
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                print(f"    Saved: '{save_name}'  (count={count_int})")
                total_saved += 1

            except Exception as e:
                print(f"    Error generating preview for '{npy_name}': {e}")
                total_skipped += 1

            finally:
                # FIX 6: Always close figure to release memory
                if fig is not None:
                    plt.close(fig)

    # FIX 3: Summary
    print(f"\n  Done! Previews saved : {total_saved}")
    print(f"         Skipped        : {total_skipped}")
    print(f"         Output folder  : '{output_dir}/'")


if __name__ == '__main__':
    generate_previews()