import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.model import SimpleCNN

def main(epochs=1, batch=64, lr=1e-3, out="model.pth"):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
    m = SimpleCNN()
    opt = optim.Adam(m.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    m.train()
    for e in range(epochs):
        tot, correct, n = 0.0, 0, 0
        for x,y in dl:
            opt.zero_grad(); logits = m(x); L = loss(logits,y); L.backward(); opt.step()
            tot += L.item()*x.size(0); n += x.size(0)
            correct += (logits.argmax(1)==y).sum().item()
        print(f"epoch {e+1}: loss={tot/n:.4f} acc={correct/n:.3f}")
    torch.save(m.state_dict(), out); print("saved", out)

if __name__ == "__main__":
    main(epochs=1)
