import json
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.retriever import get_hybrid_retriever

# 터미널 색상 설정
class Color:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

EMBEDDING_MODEL = "nomic-embed-text"
GOLD_DATA_PATH = "data/gold_dataset.json"
PROCESSED_DIR = "data/processed"

def get_top_result(file_name, question):
    loader = TextLoader(os.path.join(PROCESSED_DIR, file_name), encoding='utf-8')
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator="\n")
    split_docs = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    retriever = get_hybrid_retriever(split_docs, vectorstore)
    
    # 검색 수행
    results = retriever.invoke(question)
    top_result = results[0].page_content if results else "검색 결과 없음"
    
    vectorstore.delete_collection()
    return top_result

def print_result_box(case_name, content, target_dept, target_ans):
    # 성공 여부 판단
    has_dept = target_dept in content
    has_ans = target_ans in content
    is_success = has_dept and has_ans
    
    status = f"{Color.GREEN}✅ SUCCESS{Color.END}" if is_success else f"{Color.RED}❌ FAIL{Color.END}"
    dept_check = "⭕" if has_dept else "❌"
    ans_check = "⭕" if has_ans else "❌"
    
    print(f" {Color.BOLD}[{case_name}]{Color.END} {status}")
    print(f"  - 검색된 내용: {content.replace('\n', ' ')}")
    print(f"  - 검증: 학과정보({dept_check}), 정답금액({ans_check})")
    return is_success

def main():
    if not os.path.exists(GOLD_DATA_PATH):
        print(f"❌ '{GOLD_DATA_PATH}'가 없습니다. 먼저 데이터를 생성하세요.")
        return

    with open(GOLD_DATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)[:5] # 5개 샘플만 확인

    print(f"\n{Color.PURPLE}{Color.BOLD}🚀 [RAG 검색 품질 정밀 검증 보고서]{Color.END}")
    
    for i, item in enumerate(samples, 1):
        q = item["question"]
        a = item["answer"]
        dept = item["target_keyword"]
        
        print("\n" + "═"*100)
        print(f"{Color.YELLOW}📍 질문 {i}:{Color.END} {Color.BOLD}{q}{Color.END}")
        print(f"{Color.YELLOW}🎯 목표:{Color.END} 학과[{dept}] / 정답[{a}]")
        print("─"*100)
        
        # Case 3 실행 및 출력
        res3 = get_top_result("case2_3.md", q)
        success3 = print_result_box("Case 3 (Standard Markdown)", res3, dept, a)
        
        print("-" * 50)
        
        # Case 4 실행 및 출력
        res4 = get_top_result("case4.md", q)
        success4 = print_result_box("Case 4 (Proposed Padding) ", res4, dept, a)
        
        # 종합 분석
        print("\n" + f"{Color.BLUE}🔍 분석 결과:{Color.END}", end=" ")
        if success4 and not success3:
            print(f"{Color.BOLD}제안 기법(Case 4)이 누락된 문맥(학과)을 보정하여 검색에 성공했습니다.{Color.END}")
        elif not success4 and not success3:
            print("두 방식 모두 실패했습니다. (동명이인 혼선 또는 임베딩 한계)")
        else:
            print("두 방식 모두 유효한 검색 결과를 보여줍니다.")

if __name__ == "__main__":
    main()