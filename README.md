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

### Synthetic Data for DPO & Reward Model

An additional **6,248 preference examples** were synthetically generated to train the DPO model and the reward model used in GRPO. A random sample of the original training dataset was used as seed examples, with several LLMs prompted to produce alternative (rejected) SQL queries.

| Model | Samples Generated |
|---|---|
| llama-3.1-8b-instant | 1,700 |
| moonshotai/kimi-k2-instruct | 1,000 |
| meta-llama/llama-4-scout-17b-16e-instruct | 1,000 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 999 |
| moonshotai/kimi-k2-instruct-0905 | 999 |
| openai/gpt-oss-120b | 347 |
| qwen/qwen3-32b | 337 |
| llama-3.3-70b-versatile | 312 |
| openai/gpt-oss-20b | 249 |
| **Total** | **6,248** |

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
| GRPO | LoRA | 1 | 9,750 | 250 | 1 | - | scoring func as the reward, lr=1e-5, score=0.756 |
| | | 2 | 9,750 | 250 | 1 | - | scoring func as the reward, lr=1e-6, score=0.757 |
| | | 3 | 9,750 | 250 | 1 | - | scoring func as the reward, lr=1e-8, gradient_clip=0.3, score=0.696 |
| | | 4 | 19,500 | 500 | 1 | - | scoring func as the reward, lr=1e-6, num_generations=4, longer training, score=0.758 |
| | | RM | 9,750 | 250 | 1 | - | scoring func + RM as the reward, lr=1e-8, num_generations=8, score=0.68 |
| SFT+GRPO | LoRA | 5 | 9,750 | 250 | 1 | - | scoring func as the reward, lr=1e-8, num_generations=8, score=0.755 |

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
| SFT+GRPO | LoRA | 7.4 | ~7.7GB* | 4.04M | 0.755 | |

\* per_device_batch_size=8 (4x larger compared to SFT and DPO)

<img width="1484" height="900" alt="177132266266725325225405841586" src="https://github.com/user-attachments/assets/09a72d56-431a-4711-a3f2-fb70b4696e8a" />

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
9. **8B model without quantization** exceeded available VRAM (OOM on T4 test config)

---

## ⚠️ Limitations & Future Work

- Synthetic data quality for DPO and GRPO-RM likely bottlenecks performance; improving the generation pipeline is the most promising avenue for further gains
- Small base model (0.6B params); larger models may show different PEFT trade-offs
- Limited training budget; longer training may improve further

---

## 📁 Repository Structure

```
├── baseline/                          # Baseline models evaluation
│   ├── baseline.ipynb
│   ├── baseline-8B.ipynb
├── data/
│   ├── data_generation.ipynb          # Synthetic preference data generation for DPO and RM 
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
│   ├── reward_model.ipynb             # RM for grpo_rm.ipynb
│   ├── grpo_lr1e5.ipynb
│   ├── grpo_lr1e6.ipynb
│   ├── grpo_lr1e8.ipynb
│   ├── grpo_lr1e6_ngen4.ipynb
│   ├── grpo_sft.ipynb                 # Post-training of the SFT model
│   ├── grpo_rm.ipynb                  # GRPO with the reward model
├── examples/                          # Clean training scripts
│   ├── train_fullft_ddp.py            # Training script for multi-GPU DDP
│   ├── train_fullft.ipynb
│   ├── train_lora.ipynb
│   ├── train_qlora.ipynb
│   ├── train_dpo.ipynb
│   ├── train_grpo.ipynb
│   ├── train_rm.ipynb
│   ├── utils.py                       # GPU-aware training config detection and resource monitoring utilities
├── _config.example.py                 # WANDB and Groq API keys (placeholder values)
├── requirements.txt 

```

---

## 📊 Appendix: GPU & Training Configuration Benchmarks

Benchmarking results for full fine-tuning (FullFT) on Qwen3-0.6B across different hardware and training configurations. All runs use the full training dataset (97,500 examples) with identical hyperparameters except where noted.

| GPU | Precision | TF32 | Attention | Liger Kernel | Batch Size | VRAM Usage | Training Time | Eval Loss |
|---|---|---|---|---|---|---|---|---|
| H100 SXM | bf16 | ✓ | FA3 | ✓ | 256 | 70.7GB | 22m 44s | 0.3531 |
| H200 NVL | bf16 | ✓ | FA3 | ✓ | 512 | 87.7GB | 20m 5s | 0.3690 |
| H200 SXM | bf16 | ✓ | FA3 | ✓ | 512 | 87.7GB | 19m 34s | 0.3715 |
| B200 | bf16 | ✓ | FA3 | ✓ | 1024 | 166.9GB | 18m 24s | 0.3969 |
| 2×H100 PCIe (DDP) | bf16 | ✓ | FA3 | ✓ | 256 | 74.7GB | 17m 2s | 0.3667 |
| 2×H100 SXM (DDP) | bf16 | ✓ | FA3 | ✓ | 256 | 74.3GB | **13m 24s** | 0.3684 |

### Precision & Attention Implementation Ablations (2×H100 SXM DDP)

| Precision | TF32 | Attention | Liger Kernel | Batch Size | VRAM Usage | Training Time | Eval Loss |
|---|---|---|---|---|---|---|---|
| fp32 | ✗ | SDPA | ✗ | 32 | 55.3GB | 43m 54s | 0.3407 |
| fp32 | ✗ | SDPA | ✓ | 32 | 18.9GB | 43m 48s | 0.3407 |
| fp32 | ✗ | FA3 | ✓ | 32 | 18.8GB | 43m 19s | **0.3403** |
| fp16 | ✗ | FA3 | ✓ | 32 | 20.2GB | 15m 8s | 0.3491 |
| bf16 | ✗ | FA3 | ✓ | 32 | 74.5GB | 11m 49s | 0.3663 |
| bf16 | ✓ | FA3 | ✓ | 32 | 20.0GB | 15m 4s | 0.3412 |
| bf16 | ✓ | Eager | ✓ | 32 | 82.9GB | 15m 8s | 0.3670 |
| bf16 | ✓ | SDPA | ✓ | 32 | 74.5GB | 13m 5s | 0.3662 |
| bf16 | ✓ | FA3 | ✗ | 32 | 65.0GB | 14m 16s | 0.3409 |

### Key Observations

1. **DDP scaling:** 2×H100 SXM achieves 1.7× speedup over single H100 (13m 24s vs 22m 44s)
2. **Best precision/speed trade-off:** bf16 + TF32 + FA3 + Liger achieves near-FP32 quality (0.3412 vs 0.3403) at 2.9× faster training
3. **Liger kernel impact:** Reduces VRAM by 66% (55.3GB → 18.9GB) with FP32, no performance loss
4. **FlashAttention-3:** Consistently faster than SDPA and Eager across all configurations
5. **Batch size scaling:** Larger batches degrade eval loss on this task (0.3531 @ bs=256 vs 0.3969 @ bs=1024 on similar hardware)
