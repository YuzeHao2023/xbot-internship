#!/usr/bin/env python3
"""
简化复现脚本：模拟 GoBA 的“视觉触发器注入 + 标签替换”流程（用于教学 / 验证思路）。

Usage:
  pip install torch torchvision pillow requests
  python report/周报5/goba_repro.py

主要输出：
  - 训练日志（每个 epoch 的平均损失）
  - 最终评估：无触发器的 benign accuracy 与 带触发器的 ASR（攻击成功率）
"""
import os
import random
from io import BytesIO
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# ----------------- 配置 -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 8
POISON_RATE = 0.02   # 注入率，示例值：2%
ATTACKER_LABEL = 0   # 攻击者希望在触发时被预测的标签（0-9）
TRIGGER_URL = "https://raw.githubusercontent.com/YuzeHao2023/xbot-internship/main/paper/Goal-oriented%20Backdoor%20Attack%20against%20Vision-Language-Action%20Models%20via%20Physical%20Objects/pics/object_test.jpg"
TRIGGER_PATH = "report/周报5/trigger_object_test.jpg"
# ----------------------------------------

def download_trigger(url=TRIGGER_URL, path=TRIGGER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        try:
            img = Image.open(path).convert("RGBA")
            return img
        except Exception:
            pass
    print("Downloading trigger from:", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGBA")
    img.save(path)
    return img

def overlay_trigger_on_pil(img_pil: Image.Image, trigger_pil: Image.Image, scale=0.25, pos=None):
    """
    将触发器（RGBA）叠加到输入 PIL RGB 图像上（返回 RGB 图像）。
    scale: 触发器相对于输入宽度的比例
    pos: 指定贴图位置 (x, y)，默认为右下角偏移
    """
    w, h = img_pil.size
    tw = max(1, int(w * scale))
    # 保持触发器长宽比
    tr_w, tr_h = trigger_pil.size
    if tr_w == 0 or tr_h == 0:
        return img_pil
    ratio = tr_h / tr_w
    th = max(1, int(tw * ratio))
    trigger_resized = trigger_pil.resize((tw, th))
    if pos is None:
        pos = (w - tw - 1, h - th - 1)
    out = img_pil.copy().convert("RGBA")
    out.paste(trigger_resized, pos, trigger_resized)
    return out.convert("RGB")

class PoisonedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform, download, trigger_pil, poison_rate=0.02, attacker_label=0):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.trigger_pil = trigger_pil
        self.poison_rate = poison_rate if train else 0.0
        self.attacker_label = attacker_label
        self.poison_idx = set()
        if train and self.poison_rate > 0:
            n = len(self.data)
            m = max(1, int(n * self.poison_rate))
            self.poison_idx = set(random.sample(range(n), m))
            print(f"Poisoning {len(self.poison_idx)} / {n} training samples ({self.poison_rate*100:.2f}%)")

    def __getitem__(self, index):
        img_arr, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img_arr)
        do_poison = index in self.poison_idx
        if do_poison:
            img = overlay_trigger_on_pil(img, self.trigger_pil, scale=0.25)
            target = self.attacker_label
        if self.transform:
            img = self.transform(img)
        return img, target

def make_dataloaders(trigger_pil):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    trainset = PoisonedCIFAR10(root='./data', train=True, download=True, transform=transform_train,
                               trigger_pil=trigger_pil, poison_rate=POISON_RATE, attacker_label=ATTACKER_LABEL)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return trainloader, testloader, trainset, testset

def build_model(num_classes=10):
    model = resnet18(pretrained=False)
    # adapt ResNet for CIFAR small images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

def train_one_epoch(model, trainloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")

def eval_benign_and_asr(model, testloader, testset, trigger_pil, attacker_label=ATTACKER_LABEL):
    model.eval()
    # benign accuracy (no trigger)
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    benign_acc = correct / total if total > 0 else 0.0

    # ASR: 对 testset 中所有真实标签 != attacker_label 的样本 overlay trigger，
    # 统计被模型预测为 attacker_label 的比例
    total_non_att = 0
    asr_success = 0
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    for idx in range(len(testset)):
        true_label = int(testset.targets[idx])
        if true_label == attacker_label:
            continue
        img_np = testset.data[idx]
        img_pil = Image.fromarray(img_np)
        img_triggered = overlay_trigger_on_pil(img_pil, trigger_pil, scale=0.25)
        img_t = transform_test(img_triggered).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img_t)
            pred = out.argmax(dim=1).item()
            if pred == attacker_label:
                asr_success += 1
        total_non_att += 1
    asr = asr_success / total_non_att if total_non_att > 0 else 0.0
    return benign_acc, asr

def main():
    random.seed(123)
    torch.manual_seed(123)

    trigger_pil = download_trigger()
    trainloader, testloader, trainset, testset = make_dataloaders(trigger_pil)
    model = build_model(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, trainloader, optimizer, criterion, epoch)

    benign_acc, asr = eval_benign_and_asr(model, testloader, testset, trigger_pil, ATTACKER_LABEL)
    print("Final evaluation:")
    print(f"  Benign accuracy (no trigger): {benign_acc*100:.2f}%")
    print(f"  Attack Success Rate (ASR) with trigger: {asr*100:.2f}%")

if __name__ == "__main__":
    main()
