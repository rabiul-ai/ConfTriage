# ConfTriage: Calibrated Generalist–LLM Triage for Lung Nodule Malignancy Prediction

## Abstract

ConfTriage is a confidence-calibrated clinical triage framework for pulmonary nodule malignancy prediction. When standard radiological nodule attributes are available, a generalist Large Language Model (LLM) can accurately assess malignancy risk from their natural-language description without requiring image-based model inference. ConfTriage calibrates the LLM's confidence estimates and selectively refers only uncertain cases to a specialist deep-learning (DL) model. Experiments on the LIDC-IDRI benchmark show that radiological descriptions contain sufficient diagnostic information for near-specialist performance, while confidence calibration enables reliable and efficient triage decisions.

---


## Highlights

* 🚀 **Efficient Triage** – 76.5% of cases were resolved by the LLM alone, requiring no image-based model inference.

* 📖 **Language-Driven Diagnosis** – Natural-language descriptions consistently provided the strongest diagnostic signal across all evaluated input types.

* 📊 **Confidence-Aware Decision Making** – Calibrated confidence scores enabled reliable referral of uncertain cases to a specialist model.

* 🔬 **Theoretically Grounded** – ConfTriage is supported by formal guarantees for calibration quality and selective referral performance.

* 🌍 **Model-Agnostic Framework** – Consistent results were observed across five frontier LLM families, including both open and closed models.

---

## Framework and Results

<img width="995" height="212" alt="image" src="https://github.com/user-attachments/assets/26e93724-bc30-4fa1-b27b-33edbdd5ace2" />

**Figure 1.** Overview of the ConfTriage framework. Structured radiological attributes are transformed into a deterministic natural-language description, analyzed by a generalist LLM, calibrated into a probabilistic prediction, and selectively referred to a specialist DL model when confidence is insufficient.


<img width="472" height="263" alt="image" src="https://github.com/user-attachments/assets/f246f51e-5745-439a-99b5-e1c7acb79805" />

**Figure 2.** Selection of the operating threshold (τ = 0.28), balancing predictive performance and LLM decision coverage.


<img width="830" height="358" alt="image" src="https://github.com/user-attachments/assets/2dff7622-6551-4c3c-b06b-5070e38ecd7c" />

**Figure 3.** Comparison of average precision (AP) across five LLMs under seven input regimes.

---

## Key Findings

### 1. Language Descriptions Are Surprisingly Powerful

Across all evaluated LLMs, language-only input consistently achieved the highest diagnostic performance.

| Input Regime              | Observation                            |
| ------------------------- | -------------------------------------- |
| Language Only             | Best overall performance               |
| Radiology Attributes Only | Moderate performance                   |
| Image Statistics Only     | Near-random performance                |
| Language + Other Inputs   | Comparable but not consistently better |

This finding suggests that structured radiological descriptions already encode most of the clinically relevant information required for malignancy prediction.

---

### 2. Near-Specialist Performance Without Image Training

| Method                                | AUC   |
| ------------------------------------- | ----- |
| Gemini 3.1 Flash Lite (Language Only) | 0.907 |
| Mistral Large (Language Only)         | 0.901 |
| Certain-Net Specialist DL             | 0.933 |

The best language-only LLM configuration approaches specialist deep-learning performance while requiring no image-level training.

---

### 3. Selective Referral Improves Overall Performance

At the selected operating threshold:

* 76.5% of cases were resolved directly by the LLM.
* Only 23.5% of uncertain cases required specialist DL analysis.
* Overall F1 score reached **88.22%**.
* Overall AUC reached **92.03%**.

This demonstrates that selective referral effectively combines the strengths of generalist and specialist AI systems.

---

### 4. Calibration Matters

Probability calibration:

* Reduced Expected Calibration Error (ECE) by approximately 50%.
* Increased LLM decision coverage.
* Improved overall triage performance.
* Enabled principled confidence-based routing.

---

## Reproducibility

This repository contains:

* Evaluation scripts
* Calibration pipeline
* Statistical analysis code
* Result generation scripts

All experiments are conducted on the publicly available LIDC-IDRI dataset.

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{islam2026conftriage,
  title={ConfTriage: Calibrated Generalist--LLM Triage for Lung Nodule Malignancy Prediction with a Selective Specialist Backstop},
  author={Islam, Md Rabiul and Abdaljalil, Samir and Serpedin, Erchin and Kurban, Hasan},
  journal={Under Review},
  year={2026}
}
```

---

## Acknowledgements

We thank the maintainers of the LIDC-IDRI dataset and the developers of the open- and closed-source LLMs evaluated in this study.

Special thanks to **[Prof. Dr. Hasan Kurban](https://github.com/KurbanIntelligenceLab)**, **[Samir Abdaljalil](https://www.linkedin.com/in/samir-abdaljalil/)**, and **[Prof. Dr. Erchin Serpedin](https://engineering.tamu.edu/electrical/profiles/eserpedin.html)** for their continuous support, collaboration, and valuable contributions throughout this research.

---


## Contact

**Md Rabiul Islam**

Ph.D. Researcher, Texas A&M University

📧 **Email:** [rabiul_islam@tamu.edu](mailto:rabiul_islam@tamu.edu)

🎓 **Google Scholar:** https://scholar.google.com/citations?user=GfveUqIAAAAJ&hl=en

🌐 **Personal Website:** https://sites.google.com/view/rabiuleeekuet/home

💼 **LinkedIn:** https://www.linkedin.com/in/rabiulkuet/

For questions, collaborations, or research discussions, please open an issue in this repository or contact me directly.


