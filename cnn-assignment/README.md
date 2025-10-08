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



## Theory: Arithmetic of CNNs

**Q1.** Input `32×32×3`, conv with 8 filters, `5×5`, stride `1`, **no padding**  
Output spatial size: `(32-5+1) × (32-5+1) = 28×28`; channels `= 8`.  
**Answer:** `28×28×8`.

**Q2.** Same as Q1 but padding = “same” (stride `1`)  
“Same” keeps spatial size.  
**Answer:** `32×32×8`.

**Q3.** Apply `3×3` filter, stride `2`, **no padding** to `64×64`  
Spatial: `floor((64-3)/2) + 1 = 31`.  
**Answer:** `31×31` (channels depend on #filters).

**Q4.** Max-pool `2×2` with stride `2` on `16×16`  
Halves each spatial dim.  
**Answer:** `8×8` (per channel).

**Q5.** Two conv layers, each `3×3`, stride `1`, padding “same”, on `128×128`  
“Same” with stride `1` preserves spatial size each layer.  
**Answer:** `128×128`.

**Q6.** What if you remove `model.train()` before training?  
`model.train()` sets training mode (enables Dropout; BatchNorm uses batch stats and updates running stats).  
If omitted (and the model is in eval mode), Dropout stays off and BatchNorm won’t update → training degrades.  
Best practice: call `model.train()` for training and `model.eval()` for evaluation.

## Theory: Arithmetic of CNNs

**Q1.** Input `32×32×3`, conv with 8 filters, `5×5`, stride `1`, **no padding**  
Output spatial size: `(32-5+1) × (32-5+1) = 28×28`; channels `= 8`.  
**Answer:** `28×28×8`.

**Q2.** Same as Q1 but padding = “same” (stride `1`)  
“Same” keeps spatial size.  
**Answer:** `32×32×8`.

**Q3.** Apply `3×3` filter, stride `2`, **no padding** to `64×64`  
Spatial: `floor((64-3)/2) + 1 = 31`.  
**Answer:** `31×31` (channels depend on #filters).

**Q4.** Max-pool `2×2` with stride `2` on `16×16`  
Halves each spatial dim.  
**Answer:** `8×8` (per channel).

**Q5.** Two conv layers, each `3×3`, stride `1`, padding “same”, on `128×128`  
“Same” with stride `1` preserves spatial size each layer.  
**Answer:** `128×128`.

**Q6.** What if you remove `model.train()` before training?  
`model.train()` sets training mode (enables Dropout; BatchNorm uses batch stats and updates running stats).  
If omitted (and the model is in eval mode), Dropout stays off and BatchNorm won’t update → training degrades.  
Best practice: call `model.train()` for training and `model.eval()` for evaluation.
