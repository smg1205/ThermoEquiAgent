# PGSSI Collaborator Testing Handover Guide (Plain-Language Version)

> This document tells you: once you have a weights file, how to get PGSSI running and tested on your own
> computer.
> Just follow the steps in order; you do not need to understand the theory.

---

## 1. What PGSSI is (understand it in 30 seconds)

PGSSI is a **machine learning model** trained by the research group, and it does exactly one thing:

> **Given "solute + solvent + temperature", predict the infinite dilution activity coefficient γ∞**

(γ∞ is the physical quantity that measures "how a substance behaves when extremely diluted in another
substance"; it is an important thermodynamic property.)

- Input: the SMILES structural formulas of two molecules + temperature
- Output: γ∞ (and its logarithm ln γ∞)

It is **not** a bubble point/dew point/flash solver, but a "property predictor".

---

## 2. The 3 things you need to prepare

| # | Item | Description |
|---|---|---|
| 1 | **Project code** | The `ThermoEqui-Agent` repository, switched to the `feat/pgssi-gamma-curve` branch |
| 2 | **Weights file** | `all_merged_train_PGSSI_best.pth` (about 85 MB, given to you by someone else)|
| 3 | **PGSSI source code** | The research group's PGSSI repository (the directory containing `PGSSI_3D_architecture.py`)|

> ⚠️ **The weights are private: do not commit them to git, do not pass them on, and do not post them to a
> public repository.** Just put them anywhere on your local machine.

---

## 3. Environment setup (one-time)

### 3.1 Install dependencies

Using a Python 3.11/3.12 environment, install:

```bash
pip install "torch" torch-geometric rdkit pandas scikit-learn tqdm
```

> If `import torch_scatter` raises an error, run:
> ```bash
> pip install torch-scatter
> ```
> It does not matter if it will not install — the project includes a compatibility shim that handles it
> automatically.

### 3.2 Create the `.env` file (critical!)

In the project root, create a `.env` file (or edit it if it already exists) and fill in 4 lines:

```
PGSSI_CHECKPOINT=D:\your\path\all_merged_train_PGSSI_best.pth
PGSSI_SRC=D:\your\path\PGSSI\src\models\PGSSI
PGSSI_HIDDEN_DIM=512
PGSSI_ENABLE_CROSS_INTERACTION=1
```

**What each line means**:
- `PGSSI_CHECKPOINT`: the **full path** to the weights file (change it to your own)
- `PGSSI_SRC`: the full path to the `src/models/PGSSI` directory in the PGSSI source code (the folder
  containing `PGSSI_3D_architecture.py`)
- The last two lines are training hyperparameters; **just copy 512 and 1 as they are**

> ⚠️ `.env` is already excluded by `.gitignore` and will not be committed. But do **not** post real paths to
> the group chat.

---

## 4. Start the backend

```bash
cd ThermoEqui-Agent
python -m uvicorn apps.api.main:app --port 8000
```

> ⚠️ **Important**: if you have multiple Pythons on your computer, be sure to use the environment that has
> `torch_geometric`/`rdkit` installed (for example the conda environment
> `D:\Anaconda\envs\thermoequi-dev\python.exe`). Using the wrong environment will report
> "PGSSI requires optional dependencies".

Seeing `Application startup complete` means it succeeded.

---

## 5. Testing (just copy and paste)

### 5.1 Test the API directly (fastest validation)

Open a new terminal:

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read().decode())"
```

Returning `{"status":"ok",...}` = the backend is fine.

### 5.2 Test in the chat box (recommended)

Open the frontend `http://localhost:3000` (or use an API tool directly) and enter the following questions:

| Question | Expected result |
|---|---|
| **Calculate the infinite dilution activity coefficient of ethanol in water at 298K** | `γ∞(Ethanol→Water) ≈ 2.70`, `γ∞(Water→Ethanol) ≈ 1.96` |
| **Calculate the infinite dilution activity coefficient curve of ethanol in water from 280 to 360K** | A γ∞-T curve (2 directions), temperature 280→360K |
| **Predict γ∞ of ethane in propane at 310K** | A single value (order of magnitude ~1.x)|
| **Calculate the T-x-y curve of benzene-toluene at 101.325 kPa** | A normal T-x-y phase diagram (**not PGSSI**; it is Ideal/Raoult)|

**Judging criteria**:
- The γ∞ value should be **between 0.5 and 10** (a reasonable physical range)
- The curve should **change slowly** with temperature (not jump around)
- If it returns "PGSSI requires a trained checkpoint" → the weights path is not configured properly (check
  `.env`)
- If it returns "PGSSI requires optional dependencies" → the wrong Python environment is being used (see the
  warning in step 4)

---

## 6. Troubleshooting common problems

| Symptom | Cause | Solution |
|---|---|---|
| `PGSSI requires a trained checkpoint but none is configured` | The weights path in `.env` is wrong or did not take effect | Check the `PGSSI_CHECKPOINT` path + restart the backend |
| `PGSSI requires optional dependencies that are not installed` | The wrong Python environment was used | Start with an environment that has torch_geometric/rdkit installed |
| `PGSSI model architecture is unavailable` | The `PGSSI_SRC` path is wrong | Check that `PGSSI_SRC` points to the directory containing `PGSSI_3D_architecture.py` |
| The frontend shows "Failed to fetch" | The backend is not running / the port is wrong | Confirm the backend is up and that the frontend accesses 8000 |
| The value is -600 or 0.0000 | The weights do not match the code architecture | Switch to the weights that ship with this project (the AutoDL-trained version)|

---

## 7. Summary in one sentence

> **Configure `.env` (weights path + source path) → start with the right Python environment → ask about γ∞
> in the chat box and it works.**

If you run into problems, send the **complete backend terminal output** to the person in charge, along with
your operating system and Python version.
