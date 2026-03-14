from ultralytics import YOLO, RTDETR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class SharkTrackTrainer:
    """Wrapper for standardized Ultralytics training across different architectures."""
    
    def __init__(self, data_yaml: str):
        self.data_yaml = data_yaml
        self.common_params = {
            'data': self.data_yaml,
            'epochs': 500,
            'imgsz': 640,
            'batch': 64,
            'conf': 0.2,
            'degrees': 45.0,   # Rotation augmentation
            'fliplr': 0.5,      # Horizontal Flip
            'val': True,        # Run validation every epoch
            'save': True,       # Save checkpoints
            'plots': True,      # Generate loss curves automatically
            'project': 'SharkTrack_Thesis',
            'exist_ok': True
        }

    def train_yolo11(self, model_variant: str = "yolo11n.pt"):
        print(f"Starting Training: YOLO11 ({model_variant})")
        model = YOLO(model_variant)
        results = model.train(**self.common_params, name="YOLO11_Experiment")
        return results

    def train_rtdetr(self, model_variant: str = "rtdetr-l.pt"):
        """Vision Transformer based Object Detection (RT-DETR)."""
        print(f"Starting Training: RT-DETR ({model_variant})")
        model = RTDETR(model_variant)
        results = model.train(**self.common_params, name="RTDETR_Experiment")
        return results
trainer = SharkTrackTrainer(data_yaml="./yolo_dataset/data.yaml")

def plot_yolo11_curves(yolo_csv: str):
    """Plots the loss curves comparing YOLO11 vs RT-DETR."""
    df_yolo = pd.read_csv(yolo_csv)
    
    # Clean column names (Ultralytics sometimes adds leading spaces)
    df_yolo.columns = [c.strip() for c in df_yolo.columns]

    plt.figure(figsize=(7, 7))
    
    # Plot Training Loss
    sns.lineplot(data=df_yolo, x='epoch', y='train/box_loss', label='YOLO11 Train Box Loss')
    
    # Plot Validation Loss
    sns.lineplot(data=df_yolo, x='epoch', y='val/box_loss', label='YOLO11 Val Box Loss')

    plt.title('YOLOv11 Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_rtdetr_curves(rtdetr_csv: str):
    df_detr = pd.read_csv(rtdetr_csv)
    
    # Clean column names
    df_detr.columns = [c.strip() for c in df_detr.columns]

    plt.figure(figsize=(7, 7))
    
    # Plot Training Loss (Box vs GIoU)
    sns.lineplot(data=df_detr, x='epoch', y='train/giou_loss', label='RT-DETR Train Box Loss')
    
    # Plot Validation Loss
    sns.lineplot(data=df_detr, x='epoch', y='val/giou_loss', label='RT-DETR Val Box Loss')

    plt.title('RT-DETR Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
