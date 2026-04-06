"""
Explanation Stability Score (ESS) Evaluation

Computes ESS and Fidelity metrics across multiple transformer models,
datasets, and ablation conditions (original vs. paraphrased inputs).

Paper: "Empirical Characterization of Rationale Stability Under Controlled
       Perturbations for Explainable Pattern Recognition" (ICPR 2026)
"""

import random
import nltk
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet

# ── Configuration ──────────────────────────────────────────────────────────────

SEED = 42
MAX_EVALS = 100
MAX_LENGTH = 512

SAMPLE_SIZES = {"train": 500, "validation": 200, "test": 200}

DATASETS = {
    "sst2": {"loader": lambda split: load_dataset("glue", "sst2", split=split),
             "text_col": "sentence", "splits": ["train", "validation"]},
    "imdb": {"loader": lambda split: load_dataset("imdb", split=split),
             "text_col": "text", "splits": ["train", "test"]},
}

MODELS = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]

# ── Seed ───────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
nltk.download("wordnet", quiet=True)


# ── Core Metrics ───────────────────────────────────────────────────────────────

def compute_ess(shap_list, labels):
    """Compute pairwise cosine similarity of SHAP values within each class."""
    sims = []
    for lbl in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                sims.append(1 - cosine(shap_list[idxs[i]], shap_list[idxs[j]]))
    return sims


def compute_fidelity(texts, shap_flat, probs, preds, tokenizer, mask_token, k=5):
    """Measure prediction drop when top-k important tokens are masked."""
    fidelity_scores = []
    for i, txt in enumerate(texts):
        toks = tokenizer.tokenize(txt)
        L, M = len(toks), len(shap_flat[i])
        if L == 0 or M == 0:
            fidelity_scores.append(0.0)
            continue
        top_idxs = np.argsort(np.abs(shap_flat[i]))[-k:]
        masked_toks = toks.copy()
        for idx in top_idxs:
            tok_idx = min(int(idx * L / M), L - 1)
            masked_toks[tok_idx] = mask_token
        masked_text = tokenizer.convert_tokens_to_string(masked_toks)
        new_p = predict_fn(tokenizer, model, [masked_text])[0, preds[i]]
        fidelity_scores.append(probs[i, preds[i]] - new_p)
    return fidelity_scores


# ── Ablation ───────────────────────────────────────────────────────────────────

def paraphrase(text):
    """Simple WordNet-based synonym substitution (20% token replacement)."""
    words = text.split()
    for i, w in enumerate(words):
        syns = wordnet.synsets(w)
        if syns and random.random() < 0.2:
            words[i] = syns[0].lemmas()[0].name().replace("_", " ")
    return " ".join(words)


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_fn(tokenizer, model, texts):
    inputs = tokenizer(
        [str(t) for t in texts],
        return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1).cpu().numpy()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for model_name in MODELS:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        mask_token = tokenizer.mask_token or tokenizer.pad_token

        _predict = lambda texts: predict_fn(tokenizer, model, texts)
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(_predict, masker=masker, output_names=["NEG", "POS"])

        for ds_name, ds_cfg in DATASETS.items():
            for split in ds_cfg["splits"]:
                print(f"\n-- {ds_name} [{split}] --")
                ds = ds_cfg["loader"](split)
                texts_all = ds[ds_cfg["text_col"]]
                labels_all = ds["label"] if "label" in ds.features else \
                    np.argmax(_predict(texts_all), axis=1)

                n = SAMPLE_SIZES.get(split, 200)
                idxs = random.sample(range(len(texts_all)), n) \
                    if len(texts_all) > n else list(range(len(texts_all)))
                texts = [texts_all[i] for i in idxs]
                labels = [labels_all[i] for i in idxs]

                for condition, transform in [("original", lambda x: x), ("paraphrase", paraphrase)]:
                    texts_t = [transform(t) for t in texts]
                    shap_vals = explainer(texts_t, max_evals=MAX_EVALS)
                    probs = _predict(texts_t)
                    preds = np.argmax(probs, axis=1)
                    shap_flat = [shap_vals.values[i][preds[i]].flatten()
                                 for i in range(len(texts_t))]

                    ess_scores = compute_ess(shap_flat, labels)
                    fid_scores = compute_fidelity(
                        texts_t, shap_flat, probs, preds, tokenizer, mask_token)

                    results.append({
                        "model": model_name, "dataset": ds_name, "split": split,
                        "condition": condition,
                        "mean_ess": np.mean(ess_scores),
                        "std_ess": np.std(ess_scores),
                        "mean_fidelity_k5": np.mean(fid_scores),
                    })
                    print(f"  {condition}: ESS={np.mean(ess_scores):.4f}, "
                          f"Fidelity(k=5)={np.mean(fid_scores):.4f}")

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print(f"\nResults saved to results.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
