# sanity_check.py
import torch, torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) recreate your architecture
model = resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 5)            # 5 classes
)
model.load_state_dict(torch.load("data/weights/cancer_resnet18.pth", map_location=device))
model.to(device).eval()

# 2) your normal test set transform
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
#test_ds = datasets.ImageFolder("data/source/CervicalCancer/JPEG/CROPPED", transform=tfm)
test_ds = datasets.ImageFolder("data\mid_CervicalCancer\\random\\05", transform=tfm)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

correct = 0
total = 0
with torch.no_grad():
    for imgs, labs in test_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        preds = model(imgs).topk(1,dim=1)[1].cpu().squeeze(1).tolist()
        # correct += (preds==labs)
        for i in range(len(preds)):
            if preds[i] == labs[i].item():
                correct += 1
        total += labs.size(0)

print(f"Baseline test accuracy: {100*correct/total:.2f}%")
