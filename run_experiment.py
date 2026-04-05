# import json
# import os
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings
# from src.retriever import get_hybrid_retriever
# from src.evaluator import calculate_metrics

# # 1. 환경 설정
# EMBEDDING_MODEL = "nomic-embed-text"
# GOLD_DATA_PATH = "data/gold_dataset.json"
# PROCESSED_DIR = "data/processed"

# def run_case_experiment(file_name, is_hybrid=False):
#     # 문서 로드 및 분할
#     loader = TextLoader(os.path.join(PROCESSED_DIR, file_name), encoding='utf-8')
#     docs = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator="\n")
#     split_docs = text_splitter.split_documents(docs)

#     # Vector DB 생성 (메모리 상에 임시 생성)
#     embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
#     vectorstore = Chroma.from_documents(split_docs, embeddings)

#     # 리트리버 설정
#     if is_hybrid:
#         retriever = get_hybrid_retriever(split_docs, vectorstore)
#     else:
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     # 실험 수행
#     with open(GOLD_DATA_PATH, "r", encoding="utf-8") as f:
#         gold_data = json.load(f)

#     total_metrics = {"Hit@1": 0, "Hit@3": 0, "MRR": 0}
    
#     for item in gold_data:
#         # 검색 수행
#         retrieved_docs = retriever.invoke(item["question"])
        
#         # 지표 계산
#         metrics = calculate_metrics(retrieved_docs, item["target_keyword"], k_list=[1, 3])
#         for k in total_metrics:
#             total_metrics[k] += metrics[k]

#     # 평균 계산
#     count = len(gold_data)
#     return {k: v / count for k, v in total_metrics.items()}

# if __name__ == "__main__":
#     # Case별 실험 실행
#     results = {
#         "Case 1 (Vector)": run_case_experiment("case1.md", is_hybrid=False),
#         "Case 2 (Vector)": run_case_experiment("case2_3.md", is_hybrid=False),
#         "Case 3 (Hybrid)": run_case_experiment("case2_3.md", is_hybrid=True),
#         "Case 4 (Proposed)": run_case_experiment("case4.md", is_hybrid=True)
#     }

#     print("\n" + "="*50)
#     print(" [최종 성능 비교 결과] ")
#     print("="*50)
#     for case, metric in results.items():
#         print(f"{case}: {metric}")

import json
import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.retriever import get_hybrid_retriever
from src.evaluator import calculate_metrics

# 1. 환경 설정
EMBEDDING_MODEL = "nomic-embed-text"
GOLD_DATA_PATH = "data/gold_dataset.json"
PROCESSED_DIR = "data/processed"
TOP_K = 5  # 논문 변별력을 위해 k=5로 설정

def run_case_experiment(file_name, is_hybrid=False):
    file_path = os.path.join(PROCESSED_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"⚠️ 파일을 찾을 수 없음: {file_path}")
        return None

    # 문서 로드 및 분할 (500행 데이터이므로 청크를 작게 유지하여 변별력 강화)
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    # 표의 한 행씩 검색되도록 separator를 \n으로 설정
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator="\n")
    split_docs = text_splitter.split_documents(docs)

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
        
        # 지표 계산 (calculate_metrics 내부에서 k_list를 지원하도록 src/evaluator.py 확인 필요)
        metrics = calculate_metrics(retrieved_docs, item["target_keyword"], k_list=[1, 3, 5])
        
        for key in total_metrics:
            total_metrics[key] += metrics.get(key, 0)

    # 평균값 계산
    count = len(gold_data)
    avg_metrics = {k: v / count for k, v in total_metrics.items()}
    
    # 사용한 컬렉션 삭제 (메모리 정리)
    vectorstore.delete_collection()
    
    return avg_metrics

if __name__ == "__main__":
    # 실험 대상 케이스 정의
    test_cases = {
        "Case 1 (Plain Text)": "case1.md",
        "Case 2 (Standard Markdown)": "case2_3.md",
        "Case 3 (Hybrid Search)": "case2_3.md",
        "Case 4 (Proposed - Padding)": "case4.md"
    }

    results = {}

    print("\n🚀 전처리 케이스별 성능 비교 실험 시작 (500명 대규모 데이터)")
    print(f"📊 설정: Top-k={TOP_K}, Embedding={EMBEDDING_MODEL}")

    for title, file_name in test_cases.items():
        is_hybrid = "Hybrid" in title or "Proposed" in title
        print(f"🔎 {title} 실행 중...")
        res = run_case_experiment(file_name, is_hybrid=is_hybrid)
        if res:
            results[title] = res

    # 최종 결과 출력
    print("\n" + "="*60)
    print(" [최종 성능 비교 결과 보고서] ")
    print("="*60)
    
    # 결과를 표 형태로 보기 좋게 출력
    df_results = pd.DataFrame(results).T
    print(df_results)
    
    print("\n" + "="*60)
    print("✅ 실험이 완료되었습니다. 결과를 엑셀에 기록하고 그래프를 생성하세요.")