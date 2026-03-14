import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import shutil
from sklearn.model_selection import train_test_split
import yaml


class DatasetConfig:
    """Configuration class to hold dataset paths and 3-way split settings."""
    def __init__(
            self, raw_data_path: str, 
            output_path: str, 
            val_size: float = 0.15, 
            test_size: float = 0.15
    ):
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.val_size = val_size
        self.test_size = test_size
        self.class_name = 'elasmobranch'
        self.class_id = 0


class YOLOFormatter:
    """Handles the conversion of bounding boxes to YOLO format."""
    
    @staticmethod
    def normalize_bbox(
        ymin: float, 
        xmin: float, 
        xmax: float,
        ymax: float, 
        img_w: int, 
        img_h: int
    ) -> Tuple[float, float, float, float]:
        """Converts absolute min/max coordinates to normalized YOLO center/width/height."""
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        
        return (
            max(0.0, min(1.0, x_center)), max(0.0, min(1.0, y_center)), 
            max(0.0, min(1.0, width)), max(0.0, min(1.0, height))
        )


class SharkTrackDatasetProcessor:
    """Main processor to convert SharkTrack CSV structure to a 3-way split Ultralytics YOLO format."""
    
    def __init__(self, config: DatasetConfig, seed: int = 45):
        self.config = config
        self._setup_directories()
        self.seed = seed

    def _setup_directories(self) -> None:
        """Creates the necessary YOLO directory structure for train, val, and test."""
        for split in ['train', 'val', 'test']:
            (self.config.output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.config.output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def _process_annotation_file(self, csv_path: Path) -> List[Dict]:
        """Reads a single annotation file and extracts bounding box and tracking data."""
        df = pd.read_csv(csv_path)
        annotations = []
        image_dir = csv_path.parent
        
        for _, row in df.iterrows():
            img_path = image_dir / row['filename']
            if not img_path.exists():
                continue
                
            annotations.append({
                'img_path': img_path,
                'track_id': row['track_id'],
                'bbox': (row['ymin'], row['xmin'], row['xmax'], row['ymax'])
            })
            
        return annotations

    def _write_yolo_labels(
            self, 
            grouped_annotations: Dict[Path, List[Tuple]], 
            split: str
    ) -> None:
        """Writes the YOLO format .txt files and copies images to the respective split folder."""
        for img_path, bboxes in grouped_annotations.items():
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            dest_img_path = self.config.output_path / 'images' / split / img_path.name
            dest_label_path = self.config.output_path / 'labels' / split / f"{img_path.stem}.txt"
            
            shutil.copy(img_path, dest_img_path)
            
            with open(dest_label_path, 'w') as f:
                for bbox in bboxes:
                    ymin, xmin, xmax, ymax = bbox
                    x_c, y_c, w, h = YOLOFormatter.normalize_bbox(ymin, xmin, xmax, ymax, img_w, img_h)
                    f.write(f"{self.config.class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    def process(self) -> pd.DataFrame:
        """Executes the pipeline, splitting into train/val/test by track_id without multi-track bleeding."""
        print(f"Scanning {self.config.raw_data_path} for annotations...")
        all_annotations = []
        
        for csv_path in self.config.raw_data_path.rglob('annotations.csv'):
            all_annotations.extend(self._process_annotation_file(csv_path))
            
        # Extract unique track IDs
        unique_tracks = list(set(ann['track_id'] for ann in all_annotations))
        
        # 1st Split: Separate Train from the rest (Val + Test)
        temp_size = self.config.val_size + self.config.test_size
        train_tracks, temp_tracks = train_test_split(
            unique_tracks, test_size=temp_size, random_state=self.seed
        )
        
        # 2nd Split: Separate Val and Test from the temp pool
        relative_test_size = self.config.test_size / temp_size
        val_tracks, test_tracks = train_test_split(
            temp_tracks, test_size=relative_test_size, random_state=self.seed
        )
        
        train_set, val_set, _ = set(train_tracks), set(val_tracks), set(test_tracks)
        
        train_data, val_data, test_data = {}, {}, {}
        image_split_map = {} # Maps an image to a split to prevent bleeding

        # Initialize statistics containers
        stats = {
            'train': {'images': 0, 'tracks': set(), 'bboxes': 0},
            'val': {'images': 0, 'tracks': set(), 'bboxes': 0},
            'test': {'images': 0, 'tracks': set(), 'bboxes': 0}
        }
        
        for ann in all_annotations:
            img_path = ann['img_path']
            track_id = ann['track_id']
            bbox = ann['bbox']
            
            # Determine target split based on track_id
            if track_id in train_set:
                target_split = 'train'
            elif track_id in val_set:
                target_split = 'val'
            else:
                target_split = 'test'
            
            # Safety check: Prevent image bleeding if multiple tracks exist in one frame
            if img_path in image_split_map:
                target_split = image_split_map[img_path]
            else:
                image_split_map[img_path] = target_split
                
            # Append bbox and update stats
            if target_split == 'train':
                train_data.setdefault(img_path, []).append(bbox)
            elif target_split == 'val':
                val_data.setdefault(img_path, []).append(bbox)
            else:
                test_data.setdefault(img_path, []).append(bbox)
            
            # Record statistics
            stats[target_split]['bboxes'] += 1
            stats[target_split]['tracks'].add(track_id)
        
        rows = []

        for split, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            s_key = split.lower()
            rows.append({
                "Split": split,
                "Images": len(data),
                "Tracks": len(stats[s_key]['tracks']),
                "Bboxes": stats[s_key]['bboxes']
            })
        self._write_yolo_labels(train_data, 'train')
        self._write_yolo_labels(val_data, 'val')
        self._write_yolo_labels(test_data, 'test')
        self._create_yaml()
        print("Data processing complete")
        return pd.DataFrame(rows)


    def _create_yaml(self) -> None:
        """Generates the data.yaml required by Ultralytics, now including the test set."""
        yaml_content = {
            'path': str(self.config.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': {self.config.class_id: self.config.class_name}
        }
        
        with open(self.config.output_path / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)