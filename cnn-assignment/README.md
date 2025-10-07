# CNN Image Classifier (MNIST)
Train a small CNN, serve it with FastAPI, and classify images.

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py                    # creates model.pth
python -m uvicorn app.main:app --reload --port 8015
# open:
# http://localhost:8015/healthz
# http://localhost:8015/docs
