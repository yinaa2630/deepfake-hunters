# Path: ai/modules/predictor.py
# Desc: MobilenetV3 기반 딥페이크 예측기

import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

class DeepfakePredictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.mobilenet_v3_small(weights=None)

        # ✅ 2-class 분류 (real / fake)
        self.model.classifier[3] = torch.nn.Linear(self.model.classifier[3].in_features, 2)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image):
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
            prob = F.softmax(output, dim=1)[0][1].item()  # index 1 → fake 확률
        result = "딥페이크로 판단됨" if prob >= 0.5 else "실제 이미지로 판단됨"
        return prob, result
