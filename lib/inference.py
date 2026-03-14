import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from ultralytics import YOLO, RTDETR
from typing import List, Tuple, Dict, cast
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2
import os
import random
import matplotlib.pyplot as plt



class InferenceConfig:
    """Configuration class for inference parameters."""
    def __init__(
        self, 
        model_path: str, 
        test_images_path: str, 
        conf_threshold: float = 0.2, 
        img_size: int = 640,
        batch_size: int = 64
    ):
        self.model_path = Path(model_path)
        self.test_images_path = Path(test_images_path)
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.batch_size = batch_size


class SharkTrackBaselineInferencer:
    """Handles loading the baseline YOLOv8 model and running inference on the test set."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = self._load_model()
        self.results_df: Optional[pd.DataFrame] = None

    def _load_model(self):
            """Dynamically loads YOLO or RT-DETR based on the file path string."""
            if not self.config.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.config.model_path}"
                )
                
            print(f"Loading model from {self.config.model_path}...")
            path_str = str(self.config.model_path).lower()
            
            # Route to the correct Ultralytics class based on the model name
            if 'rtdetr' in path_str:
                print('Loaded RT-DETR model')
                return RTDETR(str(self.config.model_path))
            else:
                print('Loaded YOLO model')
                return YOLO(str(self.config.model_path))

    def run_inference(self) -> pd.DataFrame:
        """Executes inference on all images in the test directory and records timing and predictions."""
        print(f"Starting inference on {self.config.test_images_path}...")
        
        # Support both standard image formats
        image_paths = sorted(
            list(self.config.test_images_path.glob('*.jpg')) + \
            list(self.config.test_images_path.glob('*.png'))
        )
        if not image_paths:
            raise ValueError(f"No images found in {self.config.test_images_path}")

        predictions = []
        total_inference_time_ms = 0.0
        
        results = self.model.predict(
            source=image_paths,
            conf=self.config.conf_threshold,
            imgsz=self.config.img_size,
            batch=self.config.batch_size,
            save=False,
            save_txt=False,
            stream=True,
            device=[1, 2]
        )
        
        for original_path, result in zip(image_paths, results):
            # img_name = original_path.name
            img_name = original_path.name
            
            # Extract pure inference speed from Ultralytics (ignores I/O bottlenecks for fair benchmarking)
            inference_ms = result.speed['inference']
            if inference_ms is None:
                raise RuntimeError("Inference ms for a result was None")
            total_inference_time_ms += inference_ms
            
            if result.boxes is None:
                raise RuntimeError("Inference boxes was None")
            boxes = result.boxes.cpu().numpy()
            
            # If no detections, we still record the frame to accurately calculate True Negatives and throughput
            if len(boxes) == 0:
                predictions.append({
                    'image_name': img_name, 
                    'class_id': None, 
                    'confidence': None,
                    'xmin': None, 
                    'ymin': None, 
                    'xmax': None, 
                    'ymax': None,
                    'inference_time_ms': inference_ms
                })
            else:
                for box in boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0]  # Absolute coordinates
                    predictions.append({
                        'image_name': img_name, 
                        'class_id': int(box.cls[0]), 
                        'confidence': float(box.conf[0]),
                        'xmin': x_min, 
                        'ymin': y_min, 
                        'xmax': x_max, 
                        'ymax': y_max,
                        'inference_time_ms': inference_ms
                    })
        
        self.results_df = pd.DataFrame(predictions)
        
        avg_time = total_inference_time_ms / len(image_paths)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        
        print(f"Inference complete! Processed {len(image_paths)} images.")
        print(f"Average Inference Time per Image: {avg_time:.2f} ms ({fps:.2f} FPS)")
        
        return self.results_df

    def save_results(self, output_csv: str) -> None:
        """Saves the inference results to a CSV file for downstream evaluation."""
        if self.results_df is not None:
            self.results_df.to_csv(output_csv, index=False)
            print(f"Predictions successfully saved to {output_csv}")
        else:
            print("No results to save. Run inference first.")

class EvaluationConfig:
    """Holds configuration parameters for the evaluation metrics."""
    def __init__(self, iou_threshold: float = 0.5, video_fps: int = 30):
        self.iou_threshold = iou_threshold
        self.video_fps = video_fps
        self.frames_per_hour = video_fps * 60 * 60


class BoundingBoxMatcher:
    """Handles geometric matching of predicted and ground truth boxes using IoU."""
    
    @staticmethod
    def compute_iou(boxA: List[float], boxB: List[float]) -> float:
        """
        Calculates Intersection over Union (IoU) for 
        two boxes [xmin, ymin, xmax, ymax].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea)


class SharkTrackEvaluator:
    """Orchestrates the calculation of mAP50, F1 Score, Inference Efficiency, and MaxN."""
    
    def __init__(self, config: EvaluationConfig, preds_csv: str, raw_data_path: str):
        self.config = config
        self.preds_df = pd.read_csv(preds_csv)
        self.raw_data_path = Path(raw_data_path)
        self.gt_df = self._load_ground_truth()

    def _load_ground_truth(self) -> pd.DataFrame:
        """Loads and filters original annotations to match only the test set images."""
        test_images = set(self.preds_df['image_name'].unique())
        all_gt = []
        
        for csv_path in self.raw_data_path.rglob('annotations.csv'):
            df = pd.read_csv(csv_path)
            # Filter only rows belonging to images in our test set
            df = df[df['filename'].isin(test_images)]
            all_gt.append(df)
            
        return pd.concat(all_gt, ignore_index=True) if all_gt else pd.DataFrame()

    def _calculate_f1_score(self) -> Tuple[float, float, float]:
        """
        Calculates Precision, Recall, and F1 Score at the 
        pre-configured confidence threshold.
        """
        tp, fp, fn = 0, 0, 0
        
        for img_name in self.preds_df['image_name'].unique():
            preds = self.preds_df[
                self.preds_df['image_name'] == img_name
            ].dropna(subset=['xmin'])
            gts = self.gt_df[
                self.gt_df['filename'] == img_name
            ]
            
            pred_boxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            gt_boxes = gts[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            
            matched_gts = set()
            for p_box in pred_boxes:
                match_found = False
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gts:
                        continue
                    if BoundingBoxMatcher.compute_iou(
                        p_box, 
                        gt_box
                    ) >= self.config.iou_threshold:
                        tp += 1
                        matched_gts.add(gt_idx)
                        match_found = True
                        break
                if not match_found:
                    fp += 1
            
            fn += len(gt_boxes) - len(matched_gts)
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

    def _calculate_map50(self) -> float:
        """Utilizes torchmetrics to calculate exact mAP50."""
        metric = MeanAveragePrecision(
            iou_type="bbox", 
            class_metrics=False
        )
        preds_list, target_list = [], []
        
        for img_name in self.preds_df['image_name'].unique():
            preds = self.preds_df[
                self.preds_df['image_name'] == img_name
            ].dropna(subset=['xmin'])
            gts = self.gt_df[
                self.gt_df['filename'] == img_name
            ]
            
            # Format Predictions
            if len(preds) > 0:
                preds_list.append({
                    "boxes": torch.tensor(
                        preds[['xmin', 'ymin', 'xmax', 'ymax']].values, 
                        dtype=torch.float32
                    ),
                    "scores": torch.tensor(
                        preds['confidence'].values, 
                        dtype=torch.float32
                    ),
                    "labels": torch.zeros(
                        len(preds), 
                        dtype=torch.int64
                    ) # Universal class 0
                })
            else:
                preds_list.append({
                    "boxes": torch.empty((0, 4)), 
                    "scores": torch.tensor([]), 
                    "labels": torch.tensor([])
                })
                
            # Format Ground Truth
            if len(gts) > 0:
                target_list.append({
                    "boxes": torch.tensor(
                        gts[['xmin', 'ymin', 'xmax', 'ymax']].astype(float).values, 
                        dtype=torch.float32
                    ),
                    "labels": torch.zeros(len(gts), dtype=torch.int64)
                })
            else:
                target_list.append({"boxes": torch.empty((0, 4)), "labels": torch.tensor([])})
                
        metric.update(preds_list, target_list)
        return metric.compute()['map_50'].item()

    def _calculate_efficiency(self) -> float:
        """Calculates ML Inference Time in minutes per video hour."""
        avg_ms_per_frame = self.preds_df['inference_time_ms'].mean()
        total_ms_per_hour = avg_ms_per_frame * self.config.frames_per_hour
        mins_per_hour = total_ms_per_hour / (1000 * 60)
        return mins_per_hour

    def _calculate_maxn_error(self) -> float:
        """Calculates Mean Absolute Error (MAE) for the MaxN metric across all video sources."""
        # Merge source data into predictions
        img_to_source = dict(zip(self.gt_df['filename'], self.gt_df['source']))
        self.preds_df['source'] = self.preds_df['image_name'].map(img_to_source)
        
        # Count boxes per frame
        gt_counts = cast(pd.Series, self.gt_df.groupby(['source', 'filename']).size())
        gt_counts = gt_counts.reset_index(name='count')
        pred_counts = self.preds_df.dropna(subset=['xmin']).groupby(
            ['source', 'image_name']
        ).size()
        pred_counts = cast(pd.Series, pred_counts).reset_index(name='count')
        
        # Get MaxN per source
        gt_maxn = gt_counts.groupby('source')['count'].max().to_dict()
        pred_maxn = pred_counts.groupby('source')['count'].max().to_dict()
        
        errors = []
        for src, true_max in gt_maxn.items():
            p_max = pred_maxn.get(src, 0)
            errors.append(abs(true_max - p_max))
            
        return float(np.mean(errors)) if errors else 0.0

    def evaluate(self) -> Dict[str, float]:
        """Runs all evaluations and returns a compiled dictionary of metrics."""
        print("Evaluating Baseline Model...")
        precision, recall, f1 = self._calculate_f1_score()
        map50 = self._calculate_map50()
        efficiency = self._calculate_efficiency()
        maxn_mae = self._calculate_maxn_error()
        
        metrics = {
            "mAP50": map50,
            "F1_Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Efficiency_Mins_Per_Video_Hour": efficiency,
            "MaxN_Mean_Absolute_Error": maxn_mae
        }
        
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        return metrics
    
def draw_box(image, box, label, color, is_yolo=False):
    """Draws a single box on the image."""
    h, w, _ = image.shape
    
    if is_yolo:
        # YOLO: [class, x_center, y_center, width, height] (normalized)
        _, x_c, y_c, wb, hb = box
        xmin = int((x_c - wb / 2) * w)
        ymin = int((y_c - hb / 2) * h)
        xmax = int((x_c + wb / 2) * w)
        ymax = int((y_c + hb / 2) * h)
    else:
        # Predictions: [xmin, ymin, xmax, ymax] (absolute)
        xmin, ymin, xmax, ymax = map(int, box)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(
        image, str(label), 
        (xmin, ymin - 5), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

def visualize_predictions(
        dataset_path, 
        preds_csv, 
        num_samples=3,
        seed: int = 45
):
    df_preds = pd.read_csv(preds_csv)
    
    # Paths
    img_dir = os.path.join(dataset_path, "images/test")
    lbl_dir = os.path.join(dataset_path, "labels/test")
    
    # Get random images from the test set
    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    random.seed(seed)
    sample_images = random.sample(
        all_images, 
        min(num_samples, len(all_images))
    )
    
    for img_name in sample_images:
        # 1. Load Image
        img_path = os.path.join(img_dir, img_name)
        img_gt = cv2.imread(img_path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_pred = img_gt.copy()
        
        # 2. Draw Ground Truth (YOLO Format)
        label_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = list(map(float, line.split()))
                    draw_box(img_gt, data, int(data[0]), (0, 255, 0), is_yolo=True)
        
        # 3. Draw Predictions (Absolute Format)
        img_preds_df = df_preds[df_preds['image_name'] == img_name]
        for _, row in img_preds_df.iterrows():
            box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            label = f"ID:{int(row['class_id'])} {row['confidence']:.2f}"
            draw_box(img_pred, box, label, (255, 0, 0), is_yolo=False)
            
        # 4. Plot Side-by-Side
        _, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(img_gt)
        ax[0].set_title(f"Ground Truth: {img_name}")
        ax[0].axis('off')
        
        ax[1].imshow(img_pred)
        ax[1].set_title(f"Predictions: {img_name}")
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.show()
