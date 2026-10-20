# Synthetic Medical Record Generator

An end-to-end CPU-compatible project for tabular synthetic data research. The supplied dataset is `diabetic_data.csv` (101,766 rows, 50 columns). It remains unmodified and is ignored by Git. `encounter_id` and `patient_nbr` are excluded from model features. The model features are the other 48 columns: 11 numeric and 37 categorical as loaded by pandas. `?` and blank lab results are treated as missing. Numeric values are median-filled and scaled; categorical values are one-hot encoded and decoded to source values.

## Install and run

From PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Train a regular diffusion model and save it with its report:

```powershell
python train.py --data diabetic_data.csv --epochs 20 --batch-size 256 --steps 100 --output artifacts
```

Train with Opacus DP-SGD. The requested epsilon selects a noise multiplier; actual accountant epsilon is written to `artifacts/privacy_report.json`:

```powershell
python train.py --data diabetic_data.csv --epochs 20 --batch-size 256 --steps 100 --dp --epsilon 8 --delta 0.00001 --clip 1 --output artifacts
```

Launch the dashboard:

```powershell
streamlit run app.py
```

The dashboard also discovers CSV files in the project folder. Synthetic generation is done from a trained session model. The CLI checkpoint currently stores model weights and architecture; preprocessing metadata remains in the in-memory dashboard session and is not bundled into the checkpoint.

## Model and privacy

`models/diffusion.py` implements a DDPM noise-prediction objective and reverse Gaussian sampling over numeric scaled values plus categorical one-hot blocks. Sampling uses argmax categorical decoding and inverse numeric scaling. Conditional sampling is supported at the encoded-vector interface via `condition` and `condition_mask`; the dashboard currently exposes unconditional sampling.

`models/dp_trainer.py` trains the same PyTorch denoiser through Opacus `PrivacyEngine`, which clips per-record gradients and adds Gaussian noise. The accountant uses the actual Poisson-sampled batch rate and optimizer steps. Requested target ε is only used to solve for noise; reported ε comes from Opacus. Training/accounting exceptions are surfaced as failures, not as an epsilon guarantee. Opacus secure RNG is disabled by its default in this experimentation configuration; use a secure RNG configuration and retrain for a deployment-grade DP mechanism.

The formal definition is: for neighboring datasets D and D′ and every measurable output set S, M is (ε,δ)-DP when `Pr[M(D) ∈ S] ≤ exp(ε) Pr[M(D′) ∈ S] + δ`. The accountant's sampling guarantee is stated under record-level add/remove adjacency, Poisson sampling, and the Gaussian DP-SGD mechanism. **The current preprocessing fits scaling medians/ranges and categorical vocabularies on the CSV. Those data-dependent preprocessing outputs are not privatized. Therefore the accountant result is only the DP-SGD denoiser accounting result; this implementation does not claim end-to-end (ε,δ)-DP for the decoded records.** A valid end-to-end claim would require public/fixed preprocessing metadata or a separately analyzed private preprocessing mechanism. CTGAN is optional and is non-private. Empirical leakage probes do not prove differential privacy.

## Evaluation

The dashboard reports numeric Wasserstein distances and means, categorical total-variation distances, numeric correlations, exact row overlap, synthetic duplicate rate, nearest numeric neighbor distance, and optional train-synthetic/test-real classification utility. Evaluation measures are descriptive and are not privacy proofs. The synthetic output is not clinically validated and must not be used for care decisions.

The source file contains 101,766 encounters and 50 columns. Two columns are direct identifiers and omitted. Missingness: `max_glu_serum` has 96,420 missing; `A1Cresult` has 84,748 missing; the other columns show no pandas missing values, though many use `?` as a missing marker. Numeric features: admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses. The remaining 37 non-identifier fields are categorical.

## Optional CTGAN

Install separately with `pip install 'ctgan>=0.10,<1.0'`. `models/ctgan_baseline.py` wraps CTGAN and marks it non-private. It is an optional comparison model, never a replacement for the PyTorch diffusion model.
