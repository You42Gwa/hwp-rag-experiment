def calculate_metrics(retrieved_docs, gold_answer_id, k_list=[1, 3, 5]):
    """
    retrieved_docs: 검색된 문서 리스트
    gold_answer_id: 실제 정답 문서의 고유 ID 또는 핵심 키워드
    """
    results = {}
    
    # 1. Hit@k 계산
    for k in k_list:
        hit = any(gold_answer_id in doc.page_content for doc in retrieved_docs[:k])
        results[f"Hit@{k}"] = 1 if hit else 0
        
    # 2. MRR 계산
    mrr = 0
    for idx, doc in enumerate(retrieved_docs):
        if gold_answer_id in doc.page_content:
            mrr = 1 / (idx + 1)
            break
    results["MRR"] = mrr
    
    return results