# Reinforcement Learning with Chain of Thought (RLCoT) 및 RL-STaR

이 README는 RLCoT와 RL-STaR 구현에 대한 상세한 개요와 사용 방법을 제공합니다. 이 프로젝트는 언어 모델과 강화학습을 결합하여 사고의 연쇄(chain-of-thought) 추론을 통해 추론 작업을 해결합니다.

## 개요
이 프로젝트는 다음을 목표로 설계되었습니다:

- 강화학습(RL)을 사용하여 문제 해결을 위한 단계별 추론 궤적 생성
- 텍스트 응답 생성을 위한 트랜스포머 기반 언어 모델(예: GPT-2) 활용
- 추론 과정을 효과적으로 안내하는 정책 네트워크 훈련

구현에는 다음이 포함됩니다:

- RLCoT: 추론 궤적을 생성하는 프레임워크
- RL-STaR: 정책 네트워크와 보상 기반 훈련을 활용한 단계별 추론 강화학습

## 주요 기능

### 언어 모델 통합:
- 추론을 위한 사전 훈련된 트랜스포머 모델 사용
- 기본적으로 GPT-2를 사용하지만 다른 모델로 대체 가능

### 정책 네트워크:
- 상태 임베딩을 기반으로 행동을 선택하여 추론 과정을 안내하는 신경망

### 궤적 생성:
- 단계별로 사고의 연쇄 구성
- 더 나은 텍스트 생성을 위한 온도 제어 빔 서치 사용

### 보상 시스템:
- 목표 답변과의 텍스트 유사도를 기반으로 생성된 답변 평가
- 수치적 및 단어 수준 유사도 통합

### 훈련 프레임워크:
- 강화학습을 통한 정책 네트워크 업데이트
- 정확도, 보상, 손실과 같은 훈련 지표 추적

## 설치

### 필수 조건
- Python 3.8+
- PyTorch
- Hugging Face Transformers

### 필요 라이브러리
다음 명령어로 필요한 라이브러리를 설치하세요:

```bash
pip install torch transformers numpy
```

## 사용 방법

### 1. 모델 훈련
스크립트는 제공된 훈련 데이터를 사용하여 RL-STaR 모델을 훈련합니다.

예시 훈련 데이터:
```python
training_data = [
    ("What is 2+3?", "The answer is 5"),
    ("What is the capital of France?", "The capital of France is Paris")
]
```

훈련 과정 실행:
```bash
python rl_star.py
```

훈련 출력:
- 각 에피소드의 평균 손실, 정확도, 보상
- 성능이 보상 임계값을 초과하면 조기 중단 (기본값: 0.8)

### 2. 예측
훈련 후, predict 메소드를 사용하여 새로운 질문에 대한 답변을 생성합니다.

예시:
```python
question = "What is 5+7?"
prediction = rl_star.predict(question)
print(f"Generated Answer: {prediction}")
```

## 코드 구조

### 1. RLCoT 클래스
추론 궤적 생성을 담당:

- `generate_trajectory()`: 사고의 연쇄 궤적 생성
- `generate_next_state()`: 현재 상태를 기반으로 다음 추론 단계 생성
- `is_final_state()`: 최종 답변에 도달했는지 확인

### 2. RL-STaR 클래스
강화학습 처리:

- `update_policy()`: 궤적과 보상을 기반으로 정책 네트워크 업데이트
- `train()`: 제공된 데이터로 모델 훈련
- `calculate_reward()`: 텍스트 유사도 지표를 사용하여 보상 계산

### 3. 정책 네트워크
추론 과정을 안내하는 정책 정의:

- 입력: 현재 상태의 임베딩
- 출력: 추론 단계에 대한 행동 로짓

## 출력 예시

### Training Example:
```plaintext
Episode 1
-----------------------------------
Example 1:
Q: What is 2+3?
Expected: The answer is 5
Generated: The answer is 5
Reward: 1.0000

Episode Summary:
Average Loss: 0.0123
Accuracy: 1.00
Average Reward: 1.0000
```

### Prediction Example:
```plaintext
Question: What is the capital of Japan?
Generated Answer: The capital of Japan is Tokyo.
```

## 참고사항 및 팁

### 커스텀 모델:
- "gpt2"를 Hugging Face가 지원하는 다른 모델로 대체 가능 (예: GPT-3, LLaMA)

### 정책 네트워크 차원:
- 선택한 언어 모델의 임베딩 크기에 맞게 input_dim, hidden_dim, output_dim 조정

### 보상 함수:
- 보상 함수는 수치적 및 의미적 정확성을 보장하는 텍스트 유사도 사용

### GPU 사용:
- 더 빠른 훈련과 추론을 위해 GPU 사용 가능 여부 확인

### 디버깅:
- 중간 상태와 궤적을 검사하기 위해 print 문이나 로깅 사용

## 라이센스

MIT License

Copyright (c) 2024 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 인용

이 프로젝트를 인용하실 때는 다음 논문을 참조해 주세요:

```bibtex
@article{rl2023learning,
      title={Learning to Generate Step-By-Step Reasoning via Recursive Reward Modeling},
      author={Fang Liu and Hanxun Huang and Han Wang and Liangming Pan and Zhixing Tan and Yang Liu and Yizhong Wang and Chengxiang Zhai},
      journal={arXiv preprint arXiv:2410.23912},
      year={2023}
}
```

질문이나 추가 지원이 필요한 경우 언제든 문의해주세요!