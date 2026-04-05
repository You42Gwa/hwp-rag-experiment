import json
import os
import pandas as pd
import time
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.retriever import get_hybrid_retriever
from src.evaluator import calculate_metrics
import random
import numpy as np

# 1. 환경 설정
EMBEDDING_MODEL = "nomic-embed-text"
#EMBEDDING_MODEL = "bge-m3"
GOLD_DATA_PATH = "data/gold_dataset.json"
PROCESSED_DIR = "data/processed"
TOP_K = 5  # 논문 변별력을 위해 k=5로 설정

# def get_row_level_documents(file_path):
#     """파일 형식에 따라 적절하게 행 단위 문서를 생성합니다."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
    
#     docs = []
#     # 파일명이 case1을 포함하는지 확인 (Plain Text 케이스)
#     is_plain_text = "case1" in file_path.lower()
    
#     for line in lines:
#         clean_line = line.strip()
#         if not clean_line:
#             continue
            
#         if is_plain_text:
#             # Case 1: 일반 텍스트이므로 모든 유효한 줄을 추가
#             docs.append(Document(page_content=clean_line))
#         else:
#             # Case 2~5: 마크다운 표이므로 헤더와 구분선 제외하고 파이프가 있는 행만 추가
#             if "|" in line and "학과" not in line and ":---" not in line:
#                 docs.append(Document(page_content=clean_line))
    
#     # [방어 코드] 만약 결과가 비어있다면 에러 방지를 위해 로그 출력
#     if not docs:
#         print(f"⚠️ 경고: {file_path}에서 추출된 문서가 없습니다.")
        
#     return docs

def get_row_level_documents(file_path):
    """인덱스 기반 필터링으로 '학과'명이 포함된 데이터의 유실을 방지합니다."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    docs = []
    is_plain_text = "case1" in file_path.lower()
    
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line:
            continue
            
        if is_plain_text:
            docs.append(Document(page_content=clean_line))
        else:
            # [수정] 첫 2줄(헤더, 구분선)만 제외하고 나머지는 모두 데이터로 인정
            if i >= 2 and "|" in line:
                docs.append(Document(page_content=clean_line))
    
    return docs

def set_seed(seed=42):
    """실험 재현성을 위해 모든 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def run_case_experiment(file_name, is_hybrid=False):
    set_seed(42) # 실험 재현성을 위해 시드 고정
    file_path = os.path.join(PROCESSED_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"⚠️ 파일을 찾을 수 없음: {file_path}")
        return None

    # 문서 로드 및 분할 (500행 데이터이므로 청크를 작게 유지하여 변별력 강화)
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    
    # get_row_level_documents 함수를 사용하여 각 행을 독립된 Document로 변환
    split_docs = get_row_level_documents(file_path)
    
    if not split_docs:
        return None

    # 최신 langchain-ollama 패키지 사용
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Vector DB 생성 (실험마다 초기화되도록 설정)
    vectorstore = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings,
        collection_name=f"coll_{file_name.replace('.', '_')}"
    )

    # 리트리버 설정
    if is_hybrid:
        # 하이브리드 검색 (BM25 + Vector)
        retriever = get_hybrid_retriever(split_docs, vectorstore)
    else:
        # 단순 벡터 검색
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # 골드 데이터셋 로드
    with open(GOLD_DATA_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # 지표 초기화 (Hit@1, Hit@3, Hit@5, MRR 추가)
    total_metrics = {"Hit@1": 0, "Hit@3": 0, "Hit@5": 0, "MRR": 0}
    
    for item in gold_data:
        # 검색 수행
        retrieved_docs = retriever.invoke(item["question"])
        
        # 기존: metrics = calculate_metrics(retrieved_docs, item["target_keyword"], k_list=[1, 3, 5])
        metrics = calculate_metrics(
            retrieved_docs, 
            item["target_keyword"], 
            item["answer"],          # <-- 정답 금액(Fact) 추가
            k_list=[1, 3, 5]
        )
        
        for key in total_metrics:
            total_metrics[key] += metrics.get(key, 0)

    # 평균값 계산
    count = len(gold_data)
    avg_metrics = {k: v / count for k, v in total_metrics.items()}
    
    # 사용한 컬렉션 삭제 (메모리 정리)
    vectorstore.delete_collection()
    
    return avg_metrics

if __name__ == "__main__":
    set_seed(42) # 재현성을 위한 시드 고정
    
    test_cases = [
        {"title": "Case 1 (Plain+Vec)", "file": "case1.md",   "hybrid": False},
        {"title": "Case 2 (Std+Vec)",   "file": "case2_3.md", "hybrid": False},
        {"title": "Case 3 (Std+Hyb)",   "file": "case2_3.md", "hybrid": True},
        {"title": "Case 4 (Pad+Vec)",   "file": "case4.md",   "hybrid": False},
        {"title": "Case 5 (Pad+Hyb)",   "file": "case4.md",   "hybrid": True},
    ]

    results = {}

    print("\n🚀 5단계 소거 실험(Ablation Study) 시작 (500명 대규모 데이터)")
    print(f"📊 설정: Top-k={TOP_K}, Embedding={EMBEDDING_MODEL}")

    for case in test_cases:
        title = case["title"]
        file_name = case["file"]
        is_hybrid = case["hybrid"]
        
        print(f"\n🔎 {title} 실행 중...")
        res = run_case_experiment(file_name, is_hybrid=is_hybrid)
        if res:
            results[title] = res

    print("\n" + "="*60)
    print(" [최종 5단계 소거 실험 결과 보고서] ")
    print("="*60)
    
    df_results = pd.DataFrame(results).T
    print(df_results)
    
    print("\n" + "="*60)
    print("✅ 모든 실험이 완료되었습니다. 이 데이터를 논문의 메인 결과로 사용하세요.")