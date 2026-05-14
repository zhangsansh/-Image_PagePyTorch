import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time

# 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 2

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集（适配 data/train/cat.123.jpg 格式）
class CatDogDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.folder_path, img_name)

        # 自动从文件名识别标签
        if "cat" in img_name.lower():
            label = 0
        elif "dog" in img_name.lower():
            label = 1
        else:
            label = 0

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# 模型（无警告写法）
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数（带完整训练过程输出）
def train_model():
    train_dir = r"data\train"

    if not os.path.exists(train_dir):
        print(f"❌ 错误：请创建文件夹 {train_dir}，并放入猫和狗的图片")
        return

    print("=" * 60)
    print("🚀 开始加载数据集...")
    train_dataset = CatDogDataset(train_dir, transform=transform)
    print(f"✅ 训练集图片总数：{len(train_dataset)} 张")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("🚀 开始训练模型...")
    print("=" * 60)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # 每10个batch输出一次进度
            if (batch_idx + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx+1}/{len(train_loader)} | 损失: {loss.item():.4f}")

        # 计算平均损失和训练准确率
        epoch_loss = running_loss / len(train_dataset)
        train_acc = correct_train / total_train

        # 验证阶段
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val

        # 输出一轮训练完整结果
        print("\n" + "-" * 60)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} 完成")
        print(f"📉 训练损失：{epoch_loss:.4f}")
        print(f"🎯 训练准确率：{train_acc:.4f} ({correct_train}/{total_train})")
        print(f"🎯 验证准确率：{val_acc:.4f} ({correct_val}/{total_val})")
        print("-" * 60)

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"💾 已保存最优模型 | 最佳准确率：{best_acc:.4f}\n")

    # 训练结束
    total_time = time.time() - start_time
    print("=" * 60)
    print("🏁 训练完成！")
    print(f"⏱️  总耗时：{total_time:.2f} 秒")
    print(f"🎯 最佳准确率：{best_acc:.4f}")
    print(f"💾 模型已保存为：best_model.pth")
    print("=" * 60)

if __name__ == "__main__":
    train_model()