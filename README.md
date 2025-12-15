# Context-Aware Safety Guardrails via Latent Space Steering

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-Qwen--14B-yellow)](https://huggingface.co/Qwen)

> **Dynamic Safety Policy Switching for LLMs without Fine-tuning.** > **파인튜닝 없이, 잠재 공간 제어(Steering)만으로 LLM의 안전 정책을 실시간으로 조절하는 개인 프로젝트입니다.**

<br>

## 🌍 Language
* [English](#-english-description)
* [한국어 (Korean)](#-project-description-in-korean)

---

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
## 🇰🇷 Project Description in Korean

### 1. 프로젝트 개요 (Overview)
최신 LLM은 RLHF(인간 피드백 강화학습)를 통해 안전성이 강화되었지만, 이로 인해 게임이나 소설 창작과 같은 허구적 상황에서도 문맥을 파악하지 못하고 답변을 거절하는 '과잉 방어(Over-refusal)' 문제가 발생합니다.

본 프로젝트는 Representation Engineering (RepE) 기술을 활용하여, 모델의 재학습(Fine-tuning) 없이 추론 단계에서 실시간으로 안전 가드레일의 높낮이를 조절하는 기술을 구현했습니다. 이를 통해 하나의 모델로 '자유도가 높은 게임 NPC'와 '엄격한 금융 전문가'의 페르소나를 자유롭게 오갈 수 있습니다.

### 2. 핵심 기술 및 방법론 (Methodology)
   * 데이터셋 구축: '공격적이지만 허용 가능한 게임 용어' vs '실제 위험한 범죄 모의'를 대조하는 데이터셋을 구축하여 모델 내부의 **안전 벡터(Safety Vector)**를 추출했습니다.

   * 선형 탐침 (Linear Probing): 각 레이어의 Hidden State를 분석하여 안전/위험을 구분하는 최적의 결정 경계(Decision Boundary)를 찾았습니다.

   * 액티베이션 제어 (Activation Steering): Forward Pass 도중 특정 레이어에 벡터를 주입하여 모델의 성향을 제어했습니다.

      * Game Mode (Negative Steering): 안전 벡터를 뺄셈하여 규제를 완화합니다.

      * Compliance Mode (Positive Steering): 안전 벡터를 덧셈하여 규제를 강화합니다.
### 3. 실험 결과 및 인사이트 (Results & Insights)
   Layer 20의 발견: 모든 레이어가 아닌, 네트워크 중반부인 Layer 20에서 안전 판단이 형성됨을 규명했습니다.

   정량적 성과:

      기본 상태(Baseline) 거절률: 92.5%

      게임 모드(Coeff -40) 적용 시 거절률: 70.0%

      결과: 22.5%p의 응답성 개선을 달성했습니다.

   핵심 인사이트: "0은 중립이 아니다."

      아무런 개입이 없는 상태(Coefficient 0)에서도 모델은 이미 안전 편향이 심합니다. 따라서 진정한 중립이나 게임 모드를 구현하려면 단순한 가드레일 해제가 아닌 **능동적인 벡터 뺄셈(Negative Steering)**이 필수적임을 입증했습니다.

### 4. 실행 방법 (Usage)
이 코드는 Qwen-14B 모델을 기반으로 동작합니다. (GPU 메모리 24GB 이상 권장)

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
