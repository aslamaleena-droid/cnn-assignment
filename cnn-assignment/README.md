# CNN Image Classifier (MNIST)
Train a small CNN, serve it with FastAPI, and classify images.

## Quick start
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload

## Docker
docker build -t cnn-api .
docker run -p 8000:8000 cnn-api
