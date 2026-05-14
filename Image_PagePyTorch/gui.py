import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
import matplotlib.pyplot as plt
import re

# 彻底修复 Matplotlib 中文字体警告
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ===================== 全局配置 =====================
device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自动创建模型保存文件夹
if not os.path.exists("best_model"):
    os.makedirs("best_model")

# ===================== 支持多级文件夹的数据集类 =====================
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_mode="folder"):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []
        self.label_mode = label_mode

        if not os.path.exists(root_dir):
            return

        # 模式1：支持 多级文件夹 分类（自动遍历所有子文件夹）
        if label_mode == "folder":
            self._load_multi_folder_dataset()

        # 模式2：按图片名称智能分类（汉字 + 英文前缀 自动识别）
        elif label_mode == "name":
            self._load_name_based_dataset()

    def _load_multi_folder_dataset(self):
        # 遍历所有子文件夹（无限级）
        class_names = set()
        data = []

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                    img_path = os.path.join(root, file)
                    # 取当前文件所在文件夹名称作为标签
                    label_name = os.path.basename(root)
                    class_names.add(label_name)
                    data.append((img_path, label_name))

        self.classes = sorted(list(class_names))
        label2idx = {c: i for i, c in enumerate(self.classes)}

        for img_path, label_name in data:
            self.image_paths.append(img_path)
            self.labels.append(label2idx[label_name])

    def _load_name_based_dataset(self):
        all_imgs = [f for f in os.listdir(self.root_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        if len(all_imgs) == 0:
            return

        label_set = set()
        img_label_map = []

        for img in all_imgs:
            name_no_ext = os.path.splitext(img)[0]
            chinese_words = re.findall(r'[\u4e00-\u9fff]+', name_no_ext)
            if chinese_words:
                label = ''.join(chinese_words).strip()
            else:
                label = name_no_ext.split('.')[0].strip()

            label_set.add(label)
            img_label_map.append((os.path.join(self.root_dir, img), label))

        self.classes = sorted(list(label_set))
        label2idx = {cls: i for i, cls in enumerate(self.classes)}
        for path, lbl in img_label_map:
            self.image_paths.append(path)
            self.labels.append(label2idx[lbl])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            return torch.zeros(3, 224, 224), label

# ===================== 训练线程 =====================
class TrainThread(QThread):
    log_signal = pyqtSignal(str)
    process_signal = pyqtSignal(str)
    chart_signal = pyqtSignal(list, list, list, list)
    finished_signal = pyqtSignal(dict)

    def __init__(self, batch_size, epochs, lr, data_dir, label_mode, test_split):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.data_dir = data_dir
        self.label_mode = label_mode
        self.test_split = test_split
        self.is_stopped = False

    def run(self):
        try:
            if not os.path.exists(self.data_dir):
                self.log_signal.emit(f"❌ 错误：路径不存在")
                return

            full_dataset = CustomDataset(self.data_dir, transform=transform, label_mode=self.label_mode)
            if len(full_dataset) == 0:
                self.log_signal.emit("❌ 未找到图片")
                return

            total = len(full_dataset)
            nc = len(full_dataset.classes)
            self.log_signal.emit(f"✅ 数据集加载成功！")
            self.log_signal.emit(f"🖼️  图片总数：{total} 张")
            self.log_signal.emit(f"📊 类别数量：{nc} 类 -> {full_dataset.classes}")

            test_size = int(total * self.test_split)
            train_size = total - test_size
            train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            ftrs = model.fc.in_features
            model.fc = nn.Linear(ftrs, nc)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            best_acc = 0
            epoch_list, loss_list, train_acc_list, test_acc_list = [], [], [], []

            for epoch in range(self.epochs):
                if self.is_stopped: break
                model.train()
                loss_sum = 0
                correct, total_b = 0, 0

                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()

                    loss_sum += loss.item() * x.size(0)
                    _, p = torch.max(pred, 1)
                    total_b += y.size(0)
                    correct += (p == y).sum().item()

                train_loss = loss_sum / train_size
                train_acc = correct / total_b if total_b else 0

                model.eval()
                t_correct, t_total = 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        pred = model(x)
                        _, p = torch.max(pred, 1)
                        t_total += y.size(0)
                        t_correct += (p == y).sum().item()
                test_acc = t_correct / t_total if t_total else 0

                epoch_list.append(epoch + 1)
                loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

                self.log_signal.emit(f"📌 Epoch {epoch + 1} | 损失：{train_loss:.3f} | 训练：{train_acc:.2f} | 测试：{test_acc:.2f}")
                self.chart_signal.emit(epoch_list, loss_list, train_acc_list, test_acc_list)

                if test_acc > best_acc:
                    best_acc = test_acc
                    save_path = "best_model/best_model.pth"
                    torch.save({
                        "model": model.state_dict(),
                        "classes": full_dataset.classes
                    }, save_path)
                    self.log_signal.emit(f"✅ 已自动保存最优模型 -> best_model/best_model.pth")

            self.log_signal.emit(f"🏁 训练完成！最佳准确率：{best_acc:.2%}")
            self.log_signal.emit(f"📁 最优模型已自动保存到：best_model/ 文件夹")
            self.finished_signal.emit({})
        except Exception as e:
            self.log_signal.emit(f"❌ 错误：{str(e)}")

    def stop(self):
        self.is_stopped = True

# ===================== 训练界面 =====================
class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.init_ui()

    def init_ui(self):
        self.setFont(QFont("Microsoft YaHei", 16))
        layout = QVBoxLayout()
        layout.setSpacing(22)
        layout.setContentsMargins(45, 35, 45, 35)

        title = QLabel("模型训练")
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(18)
        form.setLabelAlignment(Qt.AlignRight)

        self.batch = QSpinBox()
        self.batch.setValue(16)
        self.batch.setRange(1, 128)
        self.batch.setFont(QFont("Microsoft YaHei", 16))

        self.epoch = QSpinBox()
        self.epoch.setValue(5)
        self.epoch.setRange(1, 100)
        self.epoch.setFont(QFont("Microsoft YaHei", 16))

        self.lr = QLineEdit("0.001")
        self.lr.setFont(QFont("Microsoft YaHei", 16))

        self.path = QLineEdit()
        self.path.setFont(QFont("Microsoft YaHei", 16))

        self.mode = QComboBox()
        self.mode.addItems(["按文件夹分类(支持多级)", "按图片名称分类"])
        self.mode.setFont(QFont("Microsoft YaHei", 16))

        self.test_ratio = QDoubleSpinBox()
        self.test_ratio.setRange(0.05, 0.5)
        self.test_ratio.setValue(0.2)
        self.test_ratio.setSingleStep(0.05)
        self.test_ratio.setFont(QFont("Microsoft YaHei", 16))

        btn_style = """
        QPushButton {
            font-size:16px;
            padding:12px 18px;
            border-radius:8px;
            background-color:#409eff;
            color:white;
        }
        QPushButton:hover {
            background-color:#66b1ff;
        }
        """

        stop_style = """
        QPushButton {
            font-size:16px;
            padding:12px 18px;
            border-radius:8px;
            background-color:#f56c6c;
            color:white;
        }
        QPushButton:hover {
            background-color:#f78989;
        }
        """

        self.btn_choose = QPushButton("选择数据集")
        self.btn_start = QPushButton("开始训练")
        self.btn_stop = QPushButton("停止训练")

        self.btn_choose.setStyleSheet(btn_style)
        self.btn_start.setStyleSheet(btn_style)
        self.btn_stop.setStyleSheet(stop_style)

        self.btn_choose.clicked.connect(self.choose)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

        form.addRow("批次大小", self.batch)
        form.addRow("训练轮数", self.epoch)
        form.addRow("学习率", self.lr)
        form.addRow("分类模式", self.mode)
        form.addRow("测试集比例", self.test_ratio)
        form.addRow("数据路径", self.path)
        form.addWidget(self.btn_choose)
        form.addWidget(self.btn_start)
        form.addWidget(self.btn_stop)

        self.status = QLabel("准备就绪")
        self.status.setFont(QFont("Microsoft YaHei", 16))
        self.status.setStyleSheet("color:#2c3e50; padding:6px;")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Microsoft YaHei", 15))
        self.log.setMinimumHeight(240)
        self.log.setStyleSheet("background-color:#f8f9fa; border-radius:8px; padding:12px;")

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)

        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(QLabel("训练日志"))
        layout.addWidget(self.log)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def choose(self):
        p = QFileDialog.getExistingDirectory()
        if p:
            self.path.setText(p)

    def start(self):
        p = self.path.text().strip()
        if not p:
            QMessageBox.warning(self, "提示", "请选择路径")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.clear()

        mode = "folder" if "按文件夹" in self.mode.currentText() else "name"
        self.thread = TrainThread(
            self.batch.value(),
            self.epoch.value(),
            float(self.lr.text()),
            p,
            mode,
            self.test_ratio.value()
        )
        self.thread.log_signal.connect(self.log.append)
        self.thread.process_signal.connect(self.status.setText)
        self.thread.chart_signal.connect(self.update_chart)
        self.thread.finished_signal.connect(self.finish)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.thread.stop()

    def finish(self, _):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("训练完成")

    def update_chart(self, e, l, t, v):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(e, l, 'r-o', linewidth=2.5, markersize=7)
        self.ax2.plot(e, t, 'b-o', linewidth=2.5, markersize=7)
        self.ax3.plot(e, v, 'g-o', linewidth=2.5, markersize=7)

        self.ax1.set_title("Loss", fontsize=16)
        self.ax2.set_title("Train Acc", fontsize=16)
        self.ax3.set_title("Test Acc", fontsize=16)

        self.ax1.grid(True, alpha=0.3)
        self.ax2.grid(True, alpha=0.3)
        self.ax3.grid(True, alpha=0.3)

        self.canvas.draw()

# ===================== 预测界面 =====================
class PredictTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.classes = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(28)
        layout.setContentsMargins(45, 35, 45, 35)

        title = QLabel("图片预测")
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        btn_style = """
        QPushButton {
            font-size:17px;
            padding:14px 22px;
            border-radius:8px;
            background-color:#8e44ad;
            color:white;
        }
        QPushButton:hover {
            background-color:#9b59b6;
        }
        """

        pred_style = """
        QPushButton {
            font-size:17px;
            padding:14px 22px;
            border-radius:8px;
            background-color:#27ae60;
            color:white;
        }
        QPushButton:hover {
            background-color:#2ecc71;
        }
        """

        self.btn_load = QPushButton("加载模型")
        self.btn_pred = QPushButton("选择图片预测")
        self.btn_load.setStyleSheet(btn_style)
        self.btn_pred.setStyleSheet(pred_style)

        self.img_lab = QLabel("预览区")
        self.img_lab.setFont(QFont("Microsoft YaHei", 16))
        self.img_lab.setStyleSheet("border:2px dashed #bdc3c7; border-radius:10px;")
        self.img_lab.setAlignment(Qt.AlignCenter)
        self.img_lab.setMinimumSize(450, 450)

        self.res_lab = QLabel("结果：等待")
        self.prob_lab = QLabel("置信度：-")
        self.res_lab.setFont(QFont("Microsoft YaHei", 18))
        self.prob_lab.setFont(QFont("Microsoft YaHei", 18))
        self.res_lab.setAlignment(Qt.AlignCenter)
        self.prob_lab.setAlignment(Qt.AlignCenter)

        self.btn_load.clicked.connect(self.load)
        self.btn_pred.clicked.connect(self.pred)

        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_pred)
        layout.addWidget(self.img_lab)
        layout.addWidget(self.res_lab)
        layout.addWidget(self.prob_lab)
        self.setLayout(layout)

    def load(self):
        p, _ = QFileDialog.getOpenFileName()
        if not p:
            return
        ckpt = torch.load(p, map_location=device)
        self.classes = ckpt["classes"]
        nc = len(self.classes)
        self.model = models.resnet18()
        f = self.model.fc.in_features
        self.model.fc = nn.Linear(f, nc)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        QMessageBox.information(self, "成功", f"类别：{self.classes}")

    def pred(self):
        if not self.model:
            QMessageBox.warning(self, "提示", "先加载模型")
            return
        p, _ = QFileDialog.getOpenFileName()
        if not p:
            return
        im = Image.open(p).convert("RGB")
        im = transform(im).unsqueeze(0)
        with torch.no_grad():
            out = self.model(im)
            prob = torch.softmax(out, 1)
            val, idx = torch.max(prob, 1)
        self.res_lab.setText(f"结果：{self.classes[idx.item()]}")
        self.prob_lab.setText(f"置信度：{val.item():.2%}")
        self.img_lab.setPixmap(QPixmap(p).scaled(450, 450, Qt.KeepAspectRatio))

# ===================== 主窗口 =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像分类训练工具 — 智能识别版")
        self.setGeometry(100, 100, 1500, 950)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(245, 247, 250))
        self.setPalette(palette)

        tabs = QTabWidget()
        tabs.setFont(QFont("Microsoft YaHei", 17))
        tabs.setStyleSheet("""
        QTabWidget::pane {
            border:none;
            background-color:white;
            border-radius:12px;
        }
        QTabBar::tab {
            padding:14px 35px;
            font-size:17px;
            border-top-left-radius:8px;
            border-top-right-radius:8px;
        }
        QTabBar::tab:selected {
            background-color:#409eff;
            color:white;
        }
        """)

        tabs.addTab(TrainTab(), "🧠 模型训练")
        tabs.addTab(PredictTab(), "🔍 图片预测")
        self.setCentralWidget(tabs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
