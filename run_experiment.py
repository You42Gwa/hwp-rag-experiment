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
#EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL = "bge-m3"
GOLD_DATA_PATH = "data/gold_dataset.json"
PROCESSED_DIR = "data/processed"
TOP_K = 5 

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

    total_metrics = {"Hit@1": 0, "Hit@3": 0, "Hit@5": 0, "MRR": 0}
    type_metrics = {}

    for item in gold_data:
        retrieved_docs = retriever.invoke(item["question"])
        metrics = calculate_metrics(
            retrieved_docs,
            item["target_keyword"],
            item["answer"],
            k_list=[1, 3, 5]
        )

        for key in total_metrics:
            total_metrics[key] += metrics.get(key, 0)

        q_type = item.get("query_type", "standard")
        if q_type not in type_metrics:
            type_metrics[q_type] = {"Hit@1": 0, "Hit@3": 0, "Hit@5": 0, "MRR": 0, "_count": 0}
        type_metrics[q_type]["_count"] += 1
        for key in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:
            type_metrics[q_type][key] += metrics.get(key, 0)

    count = len(gold_data)
    avg_metrics = {k: v / count for k, v in total_metrics.items()}
    avg_metrics["_type_breakdown"] = {
        qt: {k: v / d["_count"] for k, v in d.items() if k != "_count"}
        for qt, d in type_metrics.items()
    }

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

    print("\n5단계 소거 실험(Ablation Study) 시작 (500명 대규모 데이터)")
    print(f"설정: Top-k={TOP_K}, Embedding={EMBEDDING_MODEL}")

    for case in test_cases:
        title = case["title"]
        file_name = case["file"]
        is_hybrid = case["hybrid"]
        
        print(f"\n[{title}] 실행 중...")
        res = run_case_experiment(file_name, is_hybrid=is_hybrid)
        if res:
            results[title] = res

    print("\n" + "="*60)
    print(" [최종 5단계 소거 실험 결과 보고서] ")
    print("="*60)

    # 전체 지표 테이블
    summary = {title: {k: v for k, v in res.items() if k != "_type_breakdown"}
               for title, res in results.items()}
    df_results = pd.DataFrame(summary).T
    print(df_results.to_string())

    # 유형별 분리 테이블 (Case 4 기준)
    best_case = "Case 4 (Pad+Vec)"
    if best_case in results and "_type_breakdown" in results[best_case]:
        print("\n" + "="*60)
        print(f" [질의 유형별 성능 분석 - {best_case}] ")
        print("="*60)
        breakdown = results[best_case]["_type_breakdown"]
        type_labels = {"standard": "standard (50)", "homonym": "homonym (25)", "no_grade": "no_grade (25)", "no_name": "no_name (25)"}
        rows = {}
        for qt, label in type_labels.items():
            if qt in breakdown:
                rows[label] = breakdown[qt]
        df_type = pd.DataFrame(rows).T
        print(df_type.to_string())

    print("\n" + "="*60)
    print("All experiments complete.")