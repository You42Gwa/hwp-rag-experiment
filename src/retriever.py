from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.vectorstores import Chroma

def get_hybrid_retriever(documents, vectorstore):
    # 키워드 기반 (Sparse)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3
    
    # 벡터 기반 (Dense)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 하이브리드 결합 (가중치 설정 0.7 : 0.3)
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], 
        weights=[0.7, 0.3]
    )