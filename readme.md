# 병합 셀 기반 HWP 문서 검색을 위한 구조 인식 컨텍스트 패딩 기법

Structure-Aware Context Padding for Retrieval on Merged-Cell HWP Documents

---

## 개요

한국 공공·교육기관에서 광범위하게 사용되는 HWP 문서는 행 병합 표 구조를 포함하는 경우가 많다.
기존 RAG 파이프라인에서 이를 단순 텍스트나 마크다운으로 직렬화하면 병합 셀의 상위 범주 정보가 하위 행 청크에 전파되지 않아 검색 성능이 크게 저하된다.
본 프로젝트는 이 문제를 해결하는 **컨텍스트 패딩(Context Padding)** 전처리 기법을 제안하고, 소거 실험으로 검증한다.

- 전처리 방식 3종: Plain Text / Standard Markdown / Context Padding
- 검색 방식 2종: Vector Only / Hybrid (BM25 + Vector, RRF)
- 임베딩 모델 2종: `nomic-embed-text`, `bge-m3` (via Ollama)
- 평가 지표: Hit@1, Hit@3, Hit@5, MRR
- 골드셋: 125문항 (standard 50 / homonym 25 / no_grade 25 / no_name 25)

---

## 프로젝트 구조

```
hwp-rag-experiment/
│
├── data/
│   ├── raw/                        # HWP 원본 파일
│   ├── processed/                  # 전처리 완료 마크다운
│   │   ├── case1.md                # Plain Text 직렬화
│   │   ├── case2_3.md              # Standard Markdown (패딩 없음)
│   │   └── case4.md                # Context Padding 적용
│   └── gold_dataset.json           # 평가용 골드셋 125문항
│
├── src/
│   ├── hwp_handler.py              # HWP → DataFrame (그리드 기반 파서)
│   ├── preprocessor.py             # Plain / Markdown / Context Padding 직렬화
│   ├── retriever.py                # BM25 + Vector 하이브리드 리트리버
│   └── evaluator.py                # Hit@k, MRR 계산
│
├── main.py                         # 전처리 실행 (HWP → case*.md 생성)
├── run_experiment.py               # 소거 실험 실행 및 결과 출력
├── generate_mockdata.py            # 실험용 장학 명단 CSV 생성
└── requirements.txt                # 의존성 패키지
```

---

## 핵심 모듈 설명

| 파일 | 역할 |
|------|------|
| `hwp_handler.py` | HWP를 HTML로 변환 후 그리드 기반 파싱. rowspan·colspan 처리로 열 밀림 방지 |
| `preprocessor.py` | 파싱된 DataFrame을 세 가지 방식으로 직렬화. Context Padding은 열 방향 ffill 적용 |
| `retriever.py` | Kiwi 형태소 분석 기반 BM25와 bge-m3 벡터 검색을 RRF(0.5/0.5)로 결합 |
| `evaluator.py` | 검색 결과 상위 k개 내 학과명·정답 금액 동시 포함 여부로 Hit@k, MRR 계산 |

---

## 골드셋 구성 (`data/gold_dataset.json`)

| query_type | 문항 수 | 설명 |
|---|---|---|
| `standard` | 50 | 학과·학년·성명·장학유형 4조건 완전 명시 |
| `homonym` | 25 | 동일 이름이 전 학과에 존재, 학과 문맥이 핵심 단서 |
| `no_grade` | 25 | 학년 미명시, 학과·성명·장학유형 3조건 질의 |
| `no_name` | 25 | 이름 미명시, 학과·학년·장학유형 조합으로 정답 행 특정 |

---

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

Ollama 설치 후 모델 준비:

```bash
ollama pull nomic-embed-text
ollama pull bge-m3
ollama serve
```

### 2. 합성 데이터 생성

```bash
python generate_mockdata.py
```

### 3. 전처리 실행 (HWP → 마크다운)

```bash
python main.py
```

### 4. 소거 실험 실행

```bash
python run_experiment.py
```

---

## 주요 실험 결과

골드셋: 125문항 (standard 50 / homonym 25 / no_grade 25 / no_name 25), (학과, 금액) 충돌 없음 보장

### 소거 실험 (Table 2)

| 케이스 | 전처리 | 검색 | nomic Hit@1 | nomic Hit@5 | nomic MRR | bge-m3 Hit@1 | bge-m3 Hit@5 | bge-m3 MRR |
|---|---|---|:-----------:|:-----------:|:---------:|:------------:|:------------:|:----------:|
| C1 | Plain Text | Vector | 0.000 | 0.000 | 0.000 | 0.032 | 0.032 | 0.032 |
| C2 | Markdown | Vector | 0.000 | 0.008 | 0.004 | 0.032 | 0.032 | 0.032 |
| C3 | Markdown | Hybrid | 0.008 | 0.016 | 0.015 | 0.016 | 0.032 | 0.027 |
| C4 | **Context Padding** | **Vector** | 0.000 | 0.008 | 0.003 | **0.976** | **1.000** | **0.988** |
| C5 | **Context Padding** | **Hybrid** | 0.008 | 0.616 | 0.293 | 0.896 | 1.000 | 0.941 |

### 질의 유형별 성능 (bge-m3, C4 기준)

| 질의 유형 | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| standard (50) | 1.000 | 1.000 | 1.000 |
| homonym (25) | 1.000 | 1.000 | 1.000 |
| no_grade (25) | 0.960 | 1.000 | 0.980 |
| no_name (25) | 0.920 | 1.000 | 0.953 |
