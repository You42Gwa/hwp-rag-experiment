# import json
# import os
# from langchain_community.document_loaders import TextLoader
# from langchain_core.documents import Document
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
# from src.retriever import get_hybrid_retriever

# # 터미널 색상 및 스타일 설정
# class Color:
#     PURPLE = '\033[95m'
#     BLUE = '\033[94m'
#     GREEN = '\033[92m'
#     YELLOW = '\033[93m'
#     RED = '\033[91m'
#     CYAN = '\033[96m'
#     BOLD = '\033[1m'
#     END = '\033[0m'

# EMBEDDING_MODEL = "nomic-embed-text"
# GOLD_DATA_PATH = "data/gold_dataset.json"
# PROCESSED_DIR = "data/processed"

# def get_row_level_documents(file_path):
#     """표의 헤더를 제외하고 각 행을 독립된 문서로 만듭니다."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         lines = f.readlines()
    
#     docs = []
#     for line in lines:
#         # 데이터가 있는 행만 추출 (헤더와 구분선인 --- 은 제외)
#         if "|" in line and "학과" not in line and ":---" not in line:
#             clean_line = line.strip()
#             if clean_line:
#                 # 각 행을 하나의 Document 객체로 생성
#                 docs.append(Document(page_content=clean_line))
#     return docs

# def get_top_result(file_name, question, is_hybrid=False):
#     loader = TextLoader(os.path.join(PROCESSED_DIR, file_name), encoding='utf-8')
#     docs = loader.load()
    
#     split_docs = get_row_level_documents(os.path.join(PROCESSED_DIR, file_name))
    
#     embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
#     # 실험 독립성을 위해 매번 새로운 컬렉션 생성
#     vectorstore = Chroma.from_documents(
#         split_docs, 
#         embeddings, 
#         collection_name=f"verify_{'hyb' if is_hybrid else 'vec'}"
#     )
    
#     if is_hybrid:
#         retriever = get_hybrid_retriever(split_docs, vectorstore)
#     else:
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
#     results = retriever.invoke(question)
#     top_result = results[0].page_content if results else "검색 결과 없음"
    
#     vectorstore.delete_collection()
#     return top_result

# def print_compare_box(title, content, target_dept, target_ans):
#     has_dept = target_dept in content
#     has_ans = target_ans in content
#     is_success = has_dept and has_ans
    
#     status = f"{Color.GREEN}✅ SUCCESS{Color.END}" if is_success else f"{Color.RED}❌ FAIL{Color.END}"
#     print(f" {Color.BOLD}[{title}]{Color.END} {status}")
#     print(f"  - 검색 내용: {content.replace('\n', ' ')}")
#     print(f"  - 검증 지표: 학과정보({'⭕' if has_dept else '❌'}), 정답금액({'⭕' if has_ans else '❌'})")
#     return is_success

# def main():
#     if not os.path.exists(GOLD_DATA_PATH):
#         print(f"❌ '{GOLD_DATA_PATH}'가 없습니다.")
#         return

#     with open(GOLD_DATA_PATH, "r", encoding="utf-8") as f:
#         samples = json.load(f)[:5]

#     print(f"\n{Color.CYAN}{Color.BOLD}📊 [Ablation Study: Case 4(Vector) vs Case 5(Hybrid) 정밀 비교]{Color.END}")
    
#     for i, item in enumerate(samples, 1):
#         q = item["question"]
#         a = item["answer"]
#         dept = item["target_keyword"]
        
#         print("\n" + "═"*100)
#         print(f"{Color.YELLOW}📍 질문 {i}:{Color.END} {Color.BOLD}{q}{Color.END}")
#         print(f"{Color.YELLOW}🎯 목표:{Color.END} 학과[{dept}] / 금액[{a}]")
#         print("─"*100)
        
#         # Case 4 (Vector Only)
#         res4 = get_top_result("case4.md", q, is_hybrid=False)
#         success4 = print_compare_box("Case 4 (Padding + Vector Only)", res4, dept, a)
        
#         print("-" * 50)
        
#         # Case 5 (Hybrid)
#         res5 = get_top_result("case4.md", q, is_hybrid=True)
#         success5 = print_compare_box("Case 5 (Padding + Hybrid Search)", res5, dept, a)
        
#         # 상세 기술 분석
#         print(f"\n{Color.BLUE}🔍 기술적 분석:{Color.END}")
#         if success5 and not success4:
#             print(f"  > {Color.BOLD}하이브리드 시너지 확인:{Color.END} 벡터 검색은 의미적 유사성으로 인해 동명이인이나")
#             print(f"    유사 금액 행을 혼동했으나, {Color.BOLD}BM25(키워드)가 이름/학년/장학유형을 정확히 래칭(Latching){Color.END}함.")
#         elif success5 and success4:
#             print(f"  > 두 방식 모두 성공했으나, 하이브리드 방식이 검색 순위(MRR) 면에서 더 안정적인 경향을 보임.")
#         else:
#             print(f"  > {Color.RED}예외 발생:{Color.END} 두 방식 모두 실패. 질문의 모호성 혹은 데이터 분할(Chunking) 한계 가능성 검토 필요.")

# if __name__ == "__main__":
#     main()

import json
import os
import time  # 시간 정보 활용을 위해 추가
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.retriever import get_hybrid_retriever
from langchain_core.documents import Document

# 터미널 색상 설정
class Color:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

EMBEDDING_MODEL = "nomic-embed-text"
GOLD_DATA_PATH = "data/gold_dataset.json"
PROCESSED_DIR = "data/processed"

def get_row_level_documents(file_path):
    """[개선] 인덱스 기반으로 헤더를 건너뛰어 '데이터과학과' 등의 행 누락 방지"""
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
            # [수정] 문자열 매칭 대신 인덱스(i >= 2)를 사용하여 헤더와 구분선 스킵
            if i >= 2 and "|" in line:
                docs.append(Document(page_content=clean_line))
    return docs

def get_top_result(file_name, question, is_hybrid=False, q_idx=0):
    """[개선] q_idx와 타임스탬프를 조합하여 컬렉션 충돌 방지"""
    file_path = os.path.join(PROCESSED_DIR, file_name)
    split_docs = get_row_level_documents(file_path)
    
    if not split_docs:
        return "데이터 추출 실패"

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # [수정] 유니크한 컬렉션 네이밍 적용
    unique_coll_name = f"verify_{'hyb' if is_hybrid else 'vec'}_{q_idx}_{int(time.time())}"
    
    vectorstore = Chroma.from_documents(
        split_docs, 
        embeddings, 
        collection_name=unique_coll_name
    )
    
    if is_hybrid:
        retriever = get_hybrid_retriever(split_docs, vectorstore)
    else:
        # Vector Only는 k=1로 설정하여 Top-1 확인
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    results = retriever.invoke(question)
    top_result = results[0].page_content if results else "검색 결과 없음"
    
    vectorstore.delete_collection()
    return top_result

def print_result_box(case_name, content, target_dept, target_ans):
    has_dept = str(target_dept) in content
    has_ans = str(target_ans) in content
    is_success = has_dept and has_ans
    
    status = f"{Color.GREEN}✅ SUCCESS{Color.END}" if is_success else f"{Color.RED}❌ FAIL{Color.END}"
    print(f" {Color.BOLD}[{case_name}]{Color.END} {status}")
    print(f"  - 검색 내용: {content.replace('\n', ' ')}")
    print(f"  - 검증 지표: 학과정보({'⭕' if has_dept else '❌'}), 정답금액({'⭕' if has_ans else '❌'})")
    return is_success

def main():
    if not os.path.exists(GOLD_DATA_PATH):
        print(f"❌ '{GOLD_DATA_PATH}'가 없습니다.")
        return

    with open(GOLD_DATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)[:5]

    print(f"\n{Color.CYAN}{Color.BOLD}📊 [Ablation Study: Case 4(Vector) vs Case 5(Hybrid) 정밀 검증]{Color.END}")
    
    for i, item in enumerate(samples, 1):
        q = item["question"]
        a = item["answer"]
        dept = item["target_keyword"]
        
        print("\n" + "═"*100)
        print(f"{Color.YELLOW}📍 질문 {i}:{Color.END} {Color.BOLD}{q}{Color.END}")
        print(f"{Color.YELLOW}🎯 목표:{Color.END} 학과[{dept}] / 금액[{a}]")
        print("─"*100)
        
        # [수정] q_idx=i 를 인자로 전달하여 에러 해결
        res4 = get_top_result("case4.md", q, is_hybrid=False, q_idx=i)
        success4 = print_result_box("Case 4 (Padding + Vector Only)", res4, dept, a)
        
        print("-" * 50)
        
        res5 = get_top_result("case4.md", q, is_hybrid=True, q_idx=i)
        success5 = print_result_box("Case 5 (Padding + Hybrid Search)", res5, dept, a)

if __name__ == "__main__":
    main()