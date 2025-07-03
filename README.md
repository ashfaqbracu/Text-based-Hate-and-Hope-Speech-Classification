# Arabic Text Classification Notebooks

This repository contains **Jupyter notebooks** for Arabic text classification using transformer models, including multi-task and single-label classification. The notebooks were executed in different environments as follows:

- **Multi-task notebook**: Executed on **Kaggle**.
- **Single-label and Qwen3 notebooks**: Executed locally on a **PC with RTX 4090**.

---

## 📂 Contents

| Notebook Filename                                                                                 | Description                                                                                   | Execution Environment | Model Used                         |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------|------------------------------------|
| `Emotion-Offensive-Language-and-Hate-Detection-Multi-Task-from-text-classification.ipynb`         | Multi-task classification for emotion, offensive language, and hate speech in Arabic          | Kaggle (GPU)          | `aubmindlab/bert-base-arabertv2`   |
| `Arabic_Bert_finetuning_for_text_classification.ipynb`                                            | Fine-tuning AraBERT for single-label text classification                                     | PC (RTX 4090)         | `aubmindlab/bert-base-arabertv2`   |
| `Qwen3_8B_non-reasoning-_-fine-tuning.ipynb`                                                      | Fine-tuning Qwen3 8B on custom tasks                                                          | PC (RTX 4090)         | Qwen3 8B                           |

---

## 📘 1. Multi-Task Arabic Text Classification (Kaggle)

**Notebook:** `Emotion-Offensive-Language-and-Hate-Detection-Multi-Task-from-text-classification.ipynb`

- **Model:** `aubmindlab/bert-base-arabertv2` (used as the main transformer backbone, despite some code referencing AraELECTRA).
- **Tasks:**
  - **Emotion** classification (12 classes: anger, joy, sadness, etc.)
  - **Offensive language** detection (binary: yes/no)
  - **Hate speech** detection (binary: hate/not_hate)
- **Features:**
  - Custom PyTorch `Dataset` and `nn.Module` for multi-task outputs.
  - Handles missing labels and encodes them properly.
  - Reads input from CSV or Excel files.
  - Trains on full dataset (no validation split).
  - Saves trained model and prediction results (with confidence scores) to Excel.
  - Displays sample predictions for inspection.

---

## 📘 2. Single-Label AraBERT Fine-Tuning (PC RTX 4090)

**Notebook:** `Arabic_Bert_finetuning_for_text_classification.ipynb`

- **Model:** `aubmindlab/bert-base-arabertv2`
- **Features:**
  - Standard single-label classification setup using HuggingFace Trainer.
  - Uses a clean and minimal pipeline to fine-tune the model on user-provided Arabic datasets.
  - Easy to adapt for binary or multi-class classification tasks.

---

## 📘 3. Qwen3 8B Fine-Tuning (PC RTX 4090)

**Notebook:** `Qwen3_8B_non-reasoning-_-fine-tuning.ipynb`

- **Model:** Qwen3 8B
- **Features:**
  - Demonstrates fine-tuning of a large language model on task-specific Arabic data.
  - Includes commented setup instructions for running on Google Colab (Tesla T4).
  - Logs training loss per step for monitoring convergence and stability.

---

## 💻 Requirements

Make sure you have the following installed:

- Python 3.7+
- PyTorch
- HuggingFace Transformers
- `scikit-learn`
- `pandas`, `numpy`
- `tqdm`, `matplotlib`, `seaborn` (optional for visualizations)

For Qwen3, ensure your system has enough GPU memory (e.g., RTX 4090 or Google Colab Pro).

**Install dependencies:**

```bash
pip install torch transformers scikit-learn pandas numpy tqdm matplotlib seaborn

```
##  citations :
```bash
@article{antoun2020arabert,
  title = {AraBERT: Transformer-based Model for Arabic Language Understanding},
  author = {Wissam Antoun and Fady Baly and Hazem Hajj},
  journal = {arXiv preprint arXiv:2003.00104},
  year = {2020}
}
@article{yang2025qwen3,
  title = {Qwen3 Technical Report},
  author = {An Yang and Anfeng Li and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Gao and Chengen Huang and Chenxu Lv and Chujie Zheng and Dayiheng Liu and Fan Zhou and Fei Huang and Feng Hu and Hao Ge and Haoran Wei and Huan Lin and Jialong Tang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and…},
  journal = {arXiv preprint arXiv:2505.09388},
  year = {2025}
}
@article{zhang2025qwen3embedding,
  title = {Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author = {Yanzhao Zhang and Mingxin Li and Dingkun Long and Xin Zhang and Huan Lin and Baosong Yang and Pengjun Xie and An Yang and Dayiheng Liu and Junyang Lin and Fei Huang and Jingren Zhou},
  journal = {arXiv preprint arXiv:2506.05176},
  year = {2025}
}
```
