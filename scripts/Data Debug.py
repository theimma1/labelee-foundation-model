import os
import glob
import pandas as pd

def debug_csv_and_images():
    csv_path = '/Users/immanuelolajuyigbe/Desktop/Label Once/merging_e-commerce_dataset/e-commerce_cleaned.csv'
    
    print("=== CSV DEBUG ===")
    print(f"CSV Path: {csv_path}")
    print(f"CSV Exists: {os.path.exists(csv_path)}")
    
    if not os.path.exists(csv_path):
        print("ERROR: CSV file not found!")
        return
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Check for image-related columns
        print("\n=== IMAGE COLUMNS ===")
        image_cols = [col for col in df.columns if 'image' in col.lower() or 'path' in col.lower()]
        print(f"Image-related columns: {image_cols}")
        
        for col in image_cols:
            print(f"\n{col} sample values:")
            print(df[col].dropna().head().tolist())
        
        # Check LocalImagePath specifically
        if 'LocalImagePath' in df.columns:
            print("\n=== LOCAL IMAGE PATH ANALYSIS ===")
            local_paths = df['LocalImagePath'].dropna()
            print(f"Number of non-null LocalImagePath entries: {len(local_paths)}")
            
            if len(local_paths) > 0:
                sample_paths = local_paths.head(10).tolist()
                print("Sample paths:")
                for i, path in enumerate(sample_paths):
                    exists = os.path.exists(path) if isinstance(path, str) else False
                    print(f"  {i+1}. {path} - Exists: {exists}")
                
                # Check if any paths exist
                existing_count = sum(1 for path in sample_paths if isinstance(path, str) and os.path.exists(path))
                print(f"\nFound {existing_count}/{len(sample_paths)} existing images from sample")
                
                if existing_count == 0:
                    print("\n=== PATH ANALYSIS ===")
                    # Analyze the path structure
                    if sample_paths:
                        first_path = sample_paths[0]
                        print(f"First path: {first_path}")
                        print(f"Path components: {first_path.split('/') if isinstance(first_path, str) else 'N/A'}")
                        
                        # Check if it's a relative path issue
                        if isinstance(first_path, str) and not first_path.startswith('/'):
                            print("Paths appear to be relative!")
                            
                            # Try different base directories
                            base_dirs = [
                                '/Users/immanuelolajuyigbe/Desktop/Label Once',
                                '/Users/immanuelolajuyigbe/Desktop/Label Once/merging_e-commerce_dataset',
                                '/Users/immanuelolajuyigbe/Desktop',
                                '/Users/immanuelolajuyigbe'
                            ]
                            
                            for base_dir in base_dirs:
                                full_path = os.path.join(base_dir, first_path)
                                exists = os.path.exists(full_path)
                                print(f"  Trying: {full_path} - Exists: {exists}")
                                if exists:
                                    print(f"  ✓ FOUND! Base directory should be: {base_dir}")
                                    break
        
        # Search for image files in the CSV directory and subdirectories
        print("\n=== SEARCHING FOR IMAGE FILES ===")
        csv_dir = os.path.dirname(csv_path)
        print(f"Searching in CSV directory: {csv_dir}")
        
        # Search for common image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
        all_images = []
        
        for ext in image_extensions:
            # Search in current directory
            images = glob.glob(os.path.join(csv_dir, ext))
            all_images.extend(images)
            
            # Search in subdirectories
            images_sub = glob.glob(os.path.join(csv_dir, '**', ext), recursive=True)
            all_images.extend(images_sub)
        
        all_images = list(set(all_images))  # Remove duplicates
        print(f"Found {len(all_images)} image files")
        
        if all_images:
            print("Sample image files found:")
            for i, img_path in enumerate(all_images[:10]):
                rel_path = os.path.relpath(img_path, csv_dir)
                print(f"  {i+1}. {rel_path}")
            
            if len(all_images) > 10:
                print(f"  ... and {len(all_images) - 10} more")
        
        # Check directory structure
        print(f"\n=== DIRECTORY STRUCTURE ===")
        print(f"Contents of {csv_dir}:")
        try:
            items = os.listdir(csv_dir)
            for item in sorted(items):
                item_path = os.path.join(csv_dir, item)
                item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                print(f"  {item_type}: {item}")
        except Exception as e:
            print(f"Error listing directory: {e}")
            
    except Exception as e:
        print(f"Error loading CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_csv_and_images()