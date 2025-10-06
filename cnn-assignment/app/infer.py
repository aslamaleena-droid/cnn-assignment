import io, torch
from PIL import Image
from torchvision import transforms
from app.model import SimpleCNN

_pre = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_model(path="model.pth"):
    m = SimpleCNN()
    try:
        m.load_state_dict(torch.load(path, map_location="cpu"))
    except Exception:
        pass
    m.eval()
    return m

def predict_bytes(model, image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _pre(img).unsqueeze(0)  # [1,1,28,28]
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
        conf, pred = torch.max(p, dim=0)
    return {"pred_class": str(int(pred)), "confidence": float(conf)}
