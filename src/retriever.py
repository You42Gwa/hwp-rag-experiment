from langchain_classic.retrievers import EnsembleRetriever    # classic 패키지에서 불러오기
from langchain_community.retrievers import BM25Retriever      # 커뮤니티 패키지 유지
from langchain_chroma import Chroma                           # 최신 전용 패키지 사용
from kiwipiepy import Kiwi
kiwi = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]

def get_hybrid_retriever(documents, vectorstore):
    """
    BM25(키워드)와 Vector(의미) 검색을 결합한 하이브리드 리트리버를 생성합니다.
    v1에서는 EnsembleRetriever가 classic 모듈로 이동했습니다.
    """
    # 1. 키워드 기반 (Sparse)
    bm25_retriever = BM25Retriever.from_documents(
    documents, 
    preprocess_func=kiwi_tokenize  # 토크나이저 지정
    )
    bm25_retriever.k = 5 # BM25 검색 시 상위 5개 결과 반환
    
    # 2. 벡터 기반 (Dense)
    # Chroma 객체에서 리트리버 인터페이스 추출
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 3. 하이브리드 결합 (앙상블)
    # RRF(Reciprocal Rank Fusion) 알고리즘으로 결과 통합
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], 
        weights=[0.4, 0.6]  # BM25의 비중을 높임
    )