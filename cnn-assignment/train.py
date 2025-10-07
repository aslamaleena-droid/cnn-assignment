# train.py — CIFAR-10, 64x64x3 SpecCNN
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.model import SpecCNN

def main(epochs=3, batch=128, lr=1e-3, out="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),   # upscale CIFAR10 (32->64)
        transforms.ToTensor(),         # RGB -> [3,H,W], [0,1]
        # (optional) normalize if you like:
        # transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=256,   shuffle=False, num_workers=2)

    model = SpecCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def eval_acc():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in test_dl:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total   += y.numel()
        return correct/total

    for e in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen    += x.size(0)
        acc = eval_acc()
        print(f"epoch {e+1}: loss={running/seen:.4f}  test_acc={acc:.3f}")

    torch.save(model.state_dict(), out)
    print("saved", out)

if __name__ == "__main__":
    main()

