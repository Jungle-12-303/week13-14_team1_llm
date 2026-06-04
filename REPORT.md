# mini GPT 구현 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | SW-AI 12기 303호 |
| 팀명 | 1조 |
| 팀원 | 고유진, 김희준, 정범진, 황정연 |

---

## 1. 구현 현황

| 단계 | 구현 내용 | 구현 파일 | 담당자 |
| --- | --- | --- | --- |
| 1 | UTF-8 byte-level BPE tokenizer | `src/bpe.py` | all |
| 2 | GPTDataset, create_dataloader, InputEmbedding | `src/dataset.py`, `src/embeddings.py` | all |
| 3 | MultiHeadAttention, causal mask | `src/attention.py` | all |
| 4 | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple | `src/model.py` | all |
| 5 | loss 계산, checkpoint, generate, train_model | `src/train.py` | all |
| 6 | NSMC 감성 분류 Dataset과 classifier | `src/finetune.py` | all |

---

## 2. 테스트 통과 현황

| 실행 명령 | 결과 | 비고 |
| --- | --- | --- |
| `pytest tests/test_bpe.py -v` | 통과 |  |
| `pytest tests/test_dataset.py -v` | 통과 |  |
| `pytest tests/test_attention.py -v` | 통과 |  |
| `pytest tests/test_model.py -v` | 통과 |  |
| `pytest tests/test_train.py -v` | 통과 |  |
| `pytest tests/test_finetune.py -v` | 통과 |  |
| `pytest tests/ -v` | 통과 |  |


---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation 분리 |
| 사용한 데이터 크기 | Light |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | 1000/2000/3000/4000/5000 |
| 학습 corpus 크기 | `corpus[:400000]` |
| 어휘 학습 시간 | CPU 기준 약 10분 |
| vocabulary 저장 경로 | `data/nsmc_bpe_vocab_{vocab_size}.json` |
| 인코딩/디코딩 복원 예시 | `decode(encode("이 영화는 좋았다")) == 원문` |

---

## 5. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/model.py` |
| 전체 구조 | InputEmbedding -> N x TransformerBlock -> LayerNorm -> LM head |
| vocab_size | 2000 |
| context_length | 64 |
| emb_dim | 128 |
| n_heads | 4 |
| n_layers | 4 |
| drop_rate | 0.2 |
| qkv_bias | False |
| 총 파라미터 수 | token_emb(2000×128) + pos_emb(64×128) + transformer_block(197,760×4) + final_norm(128×2) + lm_head(128×2000) + classifier(128×2+2) = 1,311,746개|

---

## 6. 사전 학습

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 모델 | vocab_size | 2000 |
| 모델 | context_length | 64 |
| 모델 | emb_dim | 128 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 4 |
| 학습 | batch_size | 256 |
| 학습 | num_epochs | 15 |
| 학습 | eval_freq, eval_iter | 기본값 |
| 최적화 | lr, weight_decay | 기본값 |


## 7. 미세 조정

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | 64 |
| batch_size | 	256 |
| backbone learning rate | 3e-4 |
| classifier learning rate | 3e-4 |
| validation loss / accuracy | 0.437 / 0.806 |
| test loss / accuracy | 0.453 / 0.788 |
| 오류 예시 | 과접합 문제 |

---

## 8. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | Python 3.11 |
| PyTorch | PyTorch 2.12 |
| 실행 환경 | Colab GPU와 로컬 두 가지 병행|
| GPU/CPU 정보 | CPU: Ryzen 3500x, GPU: GTX 1650 |
| 총 학습 소요 시간 | 약 70분 |

---

## 9. 고찰

**초기값 설정**
토큰 길이가 128일 시, p95=63, p99=87로.. 로 99% 커버 가능하나, attention 비용이 sequence length 제곱에 비례, 학습 속도를 고려해 `max_length=64`를 선택.  
vocab size 비교 결과 `vocab_size=3000`에서 p95=64로, 전체 문장의 95%를 커버할 수 있었음을 확인.
<br>

**과적합**  
그래프 예시:
![image](https://github.com/user-attachments/assets/6566b286-5c7d-44f5-aca0-20dee4e5cc0c)
테스트 과정에서 val이 train을 따라가지 못하는 현상이 관찰되었다.  
val: 훈련되지 않은 데이터로 모델에 epoch 별 테스트 시행  
train: 훈련용 데이터로 모델의 파라미터 조정에 사용  
<br>

**과적합 정의**
모델이 패턴을 학습하기 보단, train 데이터 자체를 암기하는 현상  
<img src="https://github.com/user-attachments/assets/9ac17d91-8ca8-4dea-baf9-c2438522daad" width="900" />
<br>

**실험 환경 설정**  
정의: checkpoint를 활용한 *동적 파라미터 조정 모델*로 실험을 진행했다.  
동적으로 조정이 가능한 인자값: `learning rate, batch size, train data size, epoch num, drop rate`  
반대로, 조정 지양할 인자값: 모델 자체 구조를 재설계하는 인자값들(layer num, emb dim, n_heads, vocab size)  

동적 조정 수행 예시: 
과적합 감지 epoch → check point로 이동 → 3가지 시나리오 수행 →  "val_loss"기준 택일  
<br>

**시나리오 그래프**  
<img src="https://github.com/user-attachments/assets/a8c5c949-17d6-4a79-9415-163735366695" width="900" />

다음 epoch에서 *동적 파라미터 조정*이 발생하였다: epoch 10, epooh 13, epoch 15  
조정된 내역 정리를 표로 요약하자면 다음과 같다:  
| Epoch 분기 | 발생 요인 | 시나리오 3가지 | val_loss | 택일 |
| :--- | :--- | :--- | :--- | :--- |
| 10 회차 | overfit | 1. drop out 0.3로 증가 <br> 2. batch size 128로 감소 <br> 3. 둘 다 적용 | 0.498<br>0.446<br>0.457 | 2 번 |
| 13 회차 | overfit | 1. drop out 0.3로 증가 <br> 2. batch size 64로 감소 <br> 3. 둘 다 적용 | 0.437<br>0.460<br>0.451 | 1 번 |
| 14 회차 | overfit | 1. drop out 0.4로 증가 <br> 2. batch size 64로 감소 <br> 3. 둘 다 적용 | 0.452<br>0.449<br>0.472| 2 번 |

<br>

**조정 인자값 설명**  
`drop_rate`: 모델의 암기력 강제 억제, 지엽적인 패턴 암기를 강제로 차단하는 효과  
`batch_size`: Gradient Noise(수학적 노이즈)를 주입하여, "과적합 곡면"을 약화시킴  
<br>

*왜 epoch 후반기에 "과적합"발생 빈도가 증가하였나?*  
모델은 "대중적인 규칙 → 지엽적인 규칙" 순으로 학습을 한다.  
why? → **미분** 특성 상, 지엽적인 규칙은 큰 곡면에 의해 가려지기 때문  
<br>

**회고**  
고유진: 조금 더 다양한 실험을 하고 싶었으나, 시간 제약 상 아쉬움이 있다  
김희준: 개념공부 시간의 비중이 높아, 구현하는데 부족함이 있었다.  
정범상: 상호 간의 이해도 파악에 어려움이 있어, 협업 조율에 어려움이 있었다. 서로 간 강의 시간이 유익했음.  
황정연: 팀원과 합을 맞추는 것과 개인 학습 병행 방식에 차질이 있었다. 좀 더 협업에 집중하고 싶다.  

종합: 개념 학습과 구현 병행으로 진행에 아쉬움이 있었다. 이후에는 역할 분담과 실험을 더 체계화하고자 한다.  
