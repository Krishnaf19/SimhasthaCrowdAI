import os
import h5py
import numpy as np

def check_first_h5():
    gt_dir = os.path.join('data', 'Train', 'ground_truth')
    
    if not os.path.exists(gt_dir):
        print(f"Cannot find {gt_dir}")
        return
        
    h5_files = [f for f in os.listdir(gt_dir) if f.endswith('.h5')]
    
    if not h5_files:
        print("No .h5 files found in the folder!")
        return
        
    # Grab the very first .h5 file it finds
    target_file = os.path.join(gt_dir, h5_files[0])
    
    print(f"🔍 X-Raying file: {h5_files[0]}")
    
    # Open the binary file
    with h5py.File(target_file, 'r') as hf:
        # Extract the mathematical matrix
        density_map = np.array(hf['density'])
        
    print(f"-> Matrix Dimensions (Height x Width): {density_map.shape}")
    
    # The sum of all the "heat" should roughly equal the number of heads you clicked!
    total_heads = np.sum(density_map)
    print(f"-> Total heads counted by the math: {total_heads:.2f}")

if __name__ == '__main__':
    check_first_h5()
    