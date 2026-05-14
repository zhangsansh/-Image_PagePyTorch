import sys
import torch
from torchvision import models, transforms
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["猫", "狗"]
MODEL_PATH = "best_model.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 无警告加载模型
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        confidences = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(confidences).item()
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = confidences[pred_idx].item() * 100

    return pred_class, pred_conf, confidences.tolist()

class ImageAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = load_model()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("图像分析工具")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.img_label = QLabel("请导入图片")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border: 2px dashed #ccc; min-height: 300px;")
        layout.addWidget(self.img_label)

        btn_layout = QHBoxLayout()
        self.import_btn = QPushButton("导入图片")
        self.import_btn.clicked.connect(self.import_image)
        btn_layout.addWidget(self.import_btn)
        layout.addLayout(btn_layout)

        self.result_label = QLabel("分析结果：等待图片导入")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.result_label)

        self.detail_label = QLabel("置信度数值：")
        self.detail_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detail_label)

        self.img_path = None

    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.img_path = file_path
            self.show_image(file_path)
            self.analyze_image()

    def show_image(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)

    def analyze_image(self):
        if not self.img_path:
            return
        pred_class, pred_conf, confs = predict_image(self.model, self.img_path)
        self.result_label.setText(f"分析结果：{pred_class}")
        self.detail_label.setText(f"猫：{confs[0]*100:.2f}% | 狗：{confs[1]*100:.2f}%")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())