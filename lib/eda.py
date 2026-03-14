import pandas as pd
from typing import Optional
from pathlib import Path
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

class EDADataAggregator:
    """Extracts and computes geometric properties from the dataset for EDA."""
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.df: Optional[pd.DataFrame] = None

    def extract_metrics(self) -> pd.DataFrame:
        """Compiles annotations and calculates areas and aspect ratios."""
        print(f"Extracting EDA metrics from {self.raw_data_path}...")
        records = []
        img_dims_cache = {} # Cache to avoid reading the same image dimensions multiple times
        
        for csv_path in self.raw_data_path.rglob('annotations.csv'):
            df_ann = pd.read_csv(csv_path)
            image_dir = csv_path.parent
            
            for _, row in df_ann.iterrows():
                img_path = image_dir / row['filename']
                if not img_path.exists():
                    continue
                    
                if img_path not in img_dims_cache:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img_dims_cache[img_path] = img.shape[:2] # Stores (height, width)
                    
                img_h, img_w = img_dims_cache[img_path]
                
                box_w = row['xmax'] - row['xmin']
                box_h = row['ymax'] - row['ymin']
                
                records.append({
                    'track_id': row['track_id'],
                    'img_w': img_w,
                    'img_h': img_h,
                    'img_aspect_ratio': img_w / img_h if img_h > 0 else 0,
                    'img_area': img_w * img_h,
                    'box_w': box_w,
                    'box_h': box_h,
                    'box_aspect_ratio': box_w / box_h if box_h > 0 else 0,
                    'box_area': box_w * box_h,
                    'relative_box_area': (box_w * box_h) / (img_w * img_h) if (img_w * img_h) > 0 else 0
                })
                
        self.df = pd.DataFrame(records)
        print(f"Successfully compiled metrics for {len(self.df)} annotations.")
        return self.df

class PlotterUtility:
    """Utility class to ensure consistent styling across all EDA plots."""
    @staticmethod
    def set_style():
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({'figure.figsize': (10, 6), 'axes.titlesize': 14, 'axes.labelsize': 12})

def plot_relative_area(df: pd.DataFrame):
    """Plots the distribution of the bounding box area relative to the total image area."""
    plt.figure()
    sns.histplot(df['relative_box_area'], bins=50, kde=True, color='teal')
    plt.title('Distribution of Bounding Box Area Relative to Image Area')
    plt.xlabel('Relative Area (Box Area / Image Area)')
    plt.ylabel('Frequency (Number of Annotations)')
    
    # Add a vertical line for the median to quickly gauge the central tendency
    median_area = df['relative_box_area'].median()
    plt.axvline(median_area, color='red', linestyle='--', label=f'Median: {median_area:.4f}')
    
    plt.legend()
    plt.show()

def plot_image_aspect_ratios(df: pd.DataFrame):
    """Plots the distribution of image aspect ratios to assess resizing impact."""
    plt.figure()
    
    # We drop duplicates so we only count each image once, not once per annotation
    # unique_images_df = df.drop_duplicates(subset=['img_area', 'img_aspect_ratio'])
    
    sns.histplot(df['img_aspect_ratio'], bins=30, kde=False, color='coral')
    plt.title('Distribution of Image Aspect Ratios')
    plt.xlabel('Aspect Ratio (Width / Height)')
    plt.ylabel('Frequency (Number of Unique Images)')
    plt.show()


def plot_box_aspect_ratios(df: pd.DataFrame):
    """Plots the bounding box aspect ratios to understand the general shape of the targets."""
    plt.figure()
    sns.histplot(df['box_aspect_ratio'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Bounding Box Aspect Ratios (Target Shapes)')
    plt.xlabel('Aspect Ratio (Box Width / Box Height)')
    plt.ylabel('Frequency (Number of Annotations)')
    
    plt.axvline(x=1.0, color='red', linestyle='--', label='Square Box (Ratio = 1.0)')
    plt.legend()
    plt.show()
