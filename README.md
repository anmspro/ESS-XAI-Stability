# ESS: Explanation Stability Score

Official implementation of **"Empirical Characterization of Rationale Stability Under Controlled Perturbations for Explainable Pattern Recognition"** (ICPR 2026).

## Overview

ESS (Explanation Stability Score) is a metric for evaluating the consistency of model explanations across similar inputs. It quantifies the cosine similarity of SHAP-based feature attributions for inputs sharing the same label, detecting inconsistent reasoning patterns in transformer-based classifiers.

Key features:
- Measures explanation consistency across same-label inputs using SHAP values
- Evaluates robustness via WordNet-based paraphrase perturbations
- Compares against standard fidelity metrics (top-k token masking)
- Supports multiple models (BERT, RoBERTa, DistilBERT) and datasets (SST-2, IMDB)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python ess_evaluation.py
```

This runs the full evaluation across all models, datasets, and conditions (original + paraphrased). Results are saved to `results.csv`.

## Method

1. Fine-tuned transformer models generate predictions on sentiment analysis datasets
2. SHAP values are computed for each input to obtain token-level attributions
3. **ESS** is calculated as the average pairwise cosine similarity of SHAP vectors within each class
4. **Fidelity** is measured by masking top-k important tokens and observing prediction drop
5. **Robustness** is assessed by comparing ESS on original vs. paraphrased inputs

## Citation

```bibtex
@inproceedings{sakib2026ess,
  title={Empirical Characterization of Rationale Stability Under Controlled Perturbations for Explainable Pattern Recognition},
  author={Sakib, Abu Noman Md and Wang, Zhensen and Roby, Merjulah and Zhang, Zijie},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2026}
}
```

## License

MIT
