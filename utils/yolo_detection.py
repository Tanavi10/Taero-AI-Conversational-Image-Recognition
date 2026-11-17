from ultralytics import YOLO
import torch


class YOLOv12Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self):
        """Load YOLO only when detection is requested."""
        if self.model is None:
            try:
                print("🔄 Loading YOLOv12 model...")
                self.model = YOLO("yolo12s.pt").to(self.device)
                print("✅ YOLOv12 model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load YOLOv12 model: {e}")
                raise

    def detect(self, image_path):
        """Detect objects in an image."""
        try:
            self.load_model()
            results = self.model(image_path)
            return results[0].plot()[:, :, ::-1]
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            raise
