# Context-Aware Safety Guardrails via Latent Space Steering

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-Qwen--14B-yellow)](https://huggingface.co/Qwen)

> **Dynamic Safety Policy Switching for LLMs without Fine-tuning.** > **파인튜닝 없이, 잠재 공간 제어(Steering)만으로 LLM의 안전 정책을 실시간으로 조절하는 개인 프로젝트입니다.**


## 🚀 English Description

### 1. Introduction
Current Large Language Models (LLMs) often suffer from **"Safety Over-refusal"** due to rigid RLHF alignment. They tend to block queries involving conflict or violence even in fictional contexts (e.g., game scenarios, creative writing), while a single generic safety standard is insufficient for high-risk domains like finance.

This project implements **Inference-time Latent Space Steering** to dynamically modulate the model's safety guardrails. By manipulating the internal activation vectors, we can switch the model between **Game Mode (Relaxed Safety)** and **Finance Mode (Strict Safety)** in real-time without expensive fine-tuning.

### 2. Methodology
We utilized **Representation Engineering (RepE)** techniques:
1.  **Diagnosis (Linear Probing):** Constructed a contrastive dataset (Game Safe vs. Finance Unsafe) to identify the "Safety Direction" vector in the latent space.
2.  **Control (Activation Steering):** Applied arithmetic operations to the hidden states during the forward pass.
    * $$H' = H + (\alpha \cdot v_{safety})$$
    * **Negative Steering ($\alpha < 0$):** Suppresses safety mechanisms (Game Mode).
    * **Positive Steering ($\alpha > 0$):** Enhances safety mechanisms (Finance Mode).

### 3. Key Results
* **Optimal Layer:** The safety concept is most distinct at **Layer 20** of Qwen-14B.
* **Performance:**
    * **Baseline ($\alpha=0$):** 92.5% Refusal Rate (Highly biased).
    * **Game Mode ($\alpha=-40$):** **70.0% Refusal Rate**.
    * Achieved a **22.5%p improvement** in responsiveness for fictional contexts.

### 4. How to Run
```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_ID/LLM-Latent-Space-Steering.git](https://github.com/YOUR_ID/LLM-Latent-Space-Steering.git)
cd LLM-Latent-Space-Steering

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Steering Demo
# --coeff: Steering coefficient (negative for game mode)
# --layer: Target layer index
python steer_main.py --model "Qwen/Qwen-14B-Chat" --coeff -40 --layer 20
```
# --layer: Target layer index
python steer_main.py --model "Qwen/Qwen-14B-Chat" --coeff -40 --layer 20
```
