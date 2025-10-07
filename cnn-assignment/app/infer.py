from app.model import SpecCNN
from torchvision import transforms
from PIL import Image
import io, torch, torch.nn.functional as F

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def load_model(path="model.pth"):
    m = SpecCNN().to(_device)
    m.load_state_dict(torch.load(path, map_location=_device))
    m.eval()
    return m

def predict_bytes(model, raw_bytes: bytes):
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    x = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = model(x)[0]
        probs = F.softmax(logits, dim=0)
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())
    return {"pred_class": str(pred), "confidence": conf}
