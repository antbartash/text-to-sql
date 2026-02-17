# SQL Generation through LLM Fine-tuning

A comprehensive comparison of LLM fine-tuning methods for text-to-SQL generation using Qwen3-0.6B. This project evaluates multiple parameter-efficient fine-tuning techniques and reinforcement learning approaches to generate SQL queries from natural language questions.

---

## 📋 Task Description

Generate accurate SQL queries from natural language prompts given database schema context.

**Example:**

```json
{
  "sql_prompt": "What is the total volume of timber sold by each salesperson, sorted by salesperson?",
  "sql_context": "CREATE TABLE salesperson (salesperson_id INT, name TEXT, region TEXT); INSERT INTO salesperson (salesperson_id, name, region) VALUES (1, 'John Doe', 'North'), (2, 'Jane Smith', 'South'); CREATE TABLE timber_sales (sales_id INT, salesperson_id INT, volume REAL, sale_date DATE); INSERT INTO timber_sales (sales_id, salesperson_id, volume, sale_date) VALUES (1, 1, 120, '2021-01-01'), (2, 1, 150, '2021-02-01'), (3, 2, 180, '2021-01-01');",
  "sql": "SELECT salesperson_id, name, SUM(volume) as total_volume FROM timber_sales JOIN salesperson ON timber_sales.salesperson_id = salesperson.salesperson_id GROUP BY salesperson_id, name ORDER BY total_volume DESC;"
}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU
- Access to the text-to-SQL dataset (see Dataset section)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/antbartash/text-to-sql.git
cd text-to-sql
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
cp _config.example.py _config.py
# Edit _config.py to add your WANDB and Groq API keys
```

### Quick Start

For the clean example notebooks, navigate to the `examples/` directory:

- **Full Fine-tuning:** `examples/train_fullft.ipynb`
- **LoRA:** `examples/train_lora.ipynb`
- **QLoRA (memory-efficient):** `examples/train_qlora.ipynb`
- **DPO:** `examples/train_dpo.ipynb`
- **GRPO:** `examples/train_grpo.ipynb`
- **Reward Model:** `examples/train_rm.ipynb`

### Reproducing Results

To reproduce the full experimental results:

1. Start with baseline evaluation: `baseline/baseline.ipynb`
2. For hyperparameter search experiments, navigate to the corresponding method directory (`sft/`, `dpo/`, `grpo/`)
3. The numbered/suffixed notebooks (e.g., `grpo_lr1e5.ipynb`, `lora.ipynb`) contain the actual experiments referenced in the Results section

---

## 📊 Dataset

- **Size:** 97,500 training / 2,500 validation / 5,851 test examples
- **Format:** Natural language questions paired with SQL contexts and target queries
- **Message structure:**

```python
messages = [
    {"role": "system", "content": f"The user asks a question. Your task is to generate the SQL query to answer that question. Return SQL query only. The context of the question is the following: '{context}'"},
    {"role": "user", "content": prompt}
]
```

---

## 🎯 Methodology

### Model info

 - Base model: Qwen3-0.6B
 - Architecture: decoder-only transformer
 - Parameters: 596M
 - Quantization: NF4 for QLoRA

### Training Pipeline

Unless specified otherwise, training follows a 3-stage approach:

1. **Stage 1:** Small dataset (4,096 train / 1,024 validation) for large hyperparameter search
2. **Stage 2:** Medium dataset (8,192 train / 2,048 validation) with best candidates
3. **Stage 3:** Full dataset (97.5k train) for final model evaluation

### Training Configuration

| Parameter | Value |
|---|---|
| Hardware | NVIDIA Tesla T4 / NVIDIA H100 |
| Schedule | Cosine LR with linear warmup |
| Regularization | Gradient clipping at 1.0 (unless specified) |
| Generation | `max_new_tokens=512` |
| Batch size | 2 per device with gradient accumulation (unless specified) |
| Precision | FP16 (unless specified) |

### Hyperparameters Tuned

- Optimizer type
- Effective batch size
- Learning rate & weight decay
- Optimizer betas
- Warmup ratio
- Method-specific parameters (e.g., LoRA rank)

---

## 📈 Evaluation Metric

Predictions are scored using a hybrid approach:

- **Exact execution match:** Score = `1.0`
- **Mismatch:** Score = `0.7 × ROUGE-L`

This rewards correct results while giving partial credit for structurally similar SQL. The 0.7 weight was chosen to penalize incorrect execution while still rewarding structural similarity.

---

## 🧪 Methods Evaluated

- Full Fine-tuning (FullFT)
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Prompt Tuning
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- Various combinations (SFT+DPO, SFT+GRPO)

---

## 📊 Results

### Training Progression by Method

| Training Method | PEFT Method | Stage | Train Size | Valid Size | Candidates | Best Eval Loss | Notes |
|---|---|---|---|---|---|---|---|
| SFT | FullFT | 1 | 4,096 | 1,024 | 59 | 0.46566 | |
| | | 2 | 8,092 | 2,048 | 12 | 0.44965 | |
| | | 3 | 97,500 | 2,500 | 1 | 0.34353 | |
| SFT | LoRA | 1 | 4,096 | 1,024 | 71 | 0.52727 | |
| | | 2 | 8,092 | 2,048 | 72 | 0.62897 | |
| | | 3 | 97,500 | 2,500 | 1 | 0.49189 | r=16, alpha=64, target: q_proj, k_proj |
| | | 4 | 97,500 | 2,500 | 1 | 0.44037 | r=16, alpha=64, target: all-linear |
| | | 5 | 97,500 | 2,500 | 1 | 0.41089 | r=64, alpha=256, target: all-linear |
| SFT | QLoRA | 5 | 97,500 | 2,500 | 1 | 0.41261 | r=64, alpha=256, all-linear |
| SFT | Prompt Tuning | 1 | 4,096 | 1,024 | 30 | 7.57019 | Low virtual tokens worked best |
| DPO | QLoRA | 1 | 6,248 | 695 | 1 | 0.11618 | |
| SFT+DPO | QLoRA | 1 | 6,248 | 695 | 1 | 0.14759 | |
| GRPO | LoRA | 1 | 9,750 | 250 | 1 | - | lr=1e-5, score=0.756 |
| | | 2 | 9,750 | 250 | 1 | - | lr=1e-6, score=0.757 |
| | | 3 | 9,750 | 250 | 1 | - | lr=1e-8, gradient_clip=0.3, score=0.757 |
| | | 4 | 19,500 | 500 | 1 | - | lr=1e-6, num_generations=4, score=0.758 |
| SFT+GRPO | LoRA | 5 | 9,750 | 250 | 1 | - | lr=1e-8, num_generations=8, score=0.756 |

### Final Model Comparison

| Training Method | PEFT Method | Samples/sec (training) | VRAM Usage | Trainable Params | Score | Notes |
|---|---|---|---|---|---|---|
| baseline | - | - | 7.4GB | - | 0.679 | Inference only |
| baseline-nf4 | - | - | 5.6GB | - | 0.648 | 1.7x slower |
| baseline-8B-nf4 | - | - | 11.5GB | - | 0.784 | 4.9x slower, larger model |
| SFT | FullFT | 25.0 | 15.8GB | 596.06M | **0.793** | Best performance |
| SFT | LoRA | 18.6 | 10.4GB | 4.04M | 0.774 | Stage 5 result |
| SFT | QLoRA | 16.6 | 8.8GB | 4.04M | 0.755 | |
| SFT | Prompt Tuning | 22.5 | 9.6GB | 10,240 | 0.052 | Poor performance |
| DPO | QLoRA | 1.52 | 11.4GB | 4.04M | 0.612 | |
| SFT+DPO | QLoRA | 1.51 | 11.9GB | 4.04M | 0.702 | |
| GRPO | LoRA | 1.4 | ~8.9GB* | 4.04M | 0.758 | |
| SFT+GRPO | LoRA | 7.4 | ~7.7GB* | 4.04M | 0.68 | |

\* per_device_batch_size=8 (4x larger compared to SFT and DPO)

<img width="1484" height="900" alt="1000001026" src="https://github.com/user-attachments/assets/99c1eaf9-a5d0-4f4d-b9e9-715275b5c985" />

---

## 🔍 Key Findings

1. **Full fine-tuning** achieves the best performance (0.793) but requires the most resources (15.8GB VRAM)
2. **LoRA** provides excellent trade-offs: 97% of FullFT performance with only 0.68% of trainable parameters
3. **QLoRA** further reduces memory (8.8GB) with minimal performance loss (0.755)
4. Larger LoRA rank (64) and targeting all linear layers improved performance significantly
5. **GRPO** shows promise (0.758) without requiring SFT pretraining
6. **DPO** underperforms when used alone but improves when combined with SFT
7. **Prompt tuning** failed for this task (0.052 score)
8. The 8B baseline outperforms most fine-tuned 0.6B models but at 4.9x slower inference

---

## ⚠️ Limitations & Future Work

- Evaluated on a single text-to-SQL dataset; schema generalization untested
- Small base model (0.6B params); larger models may show different PEFT trade-offs
- Limited training budget; longer training may improve further

---

## 📁 Repository Structure

```
├── baseline/                          # Baseline models evaluation
│   ├── baseline.ipynb
│   ├── baseline-8B.ipynb
├── sft/
│   ├── fullft.ipynb
│   ├── lora.ipynb
│   ├── qlora.ipynb
│   ├── prompt_tuning_random.ipynb     # Prompt tuning with a random initialization
│   ├── prompt_tuning_text.ipynb       # Prompt tuning with the text initialization
├── dpo/
│   ├── dpo.ipynb
│   ├── sft_dpo.ipynb                  # Post-training of the SFT model
├── grpo/
│   ├── reward_model/
│       ├── rm_data_generation.ipynb   # Synthetic data generation for reward modeling
│       ├── reward_model.ipynb         # Reward modeling
│   ├── grpo_lr1e5.ipynb
│   ├── grpo_lr1e6.ipynb
│   ├── grpo_lr1e8.ipynb
│   ├── grpo_lr1e6_ngen4.ipynb
│   ├── grpo_sft.ipynb                 # Post-training of the SFT model
│   ├── grpo_rm.ipynb                  # GRPO with the reward model
├── examples/                          # Clean training scripts
│   ├── train_fullft.ipynb
│   ├── train_lora.ipynb
│   ├── train_qlora.ipynb
│   ├── train_dpo.ipynb
│   ├── train_grpo.ipynb
│   ├── train_rm.ipynb
├── _config.example.py                 # WANDB and Groq API keys (placeholder values)
├── requirements.txt 

```
