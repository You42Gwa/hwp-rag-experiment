def calculate_metrics(retrieved_docs, target_keyword, gold_answer, k_list=[1, 3, 5]):
    import re
    results = {}
    # 숫자와 한글만 남기고 모두 제거하여 비교의 정확도를 높입니다.
    def normalize(text):
        return re.sub(r'[^0-9가-힣]', '', str(text))

    gold_ans_norm = normalize(gold_answer)
    target_norm = normalize(target_keyword)

    for k in k_list:
        hit = 0
        for doc in retrieved_docs[:k]:
            content_norm = normalize(doc.page_content)
            # 학과명과 금액이 동시에 포함되어 있는지 확인
            if target_norm in content_norm and gold_ans_norm in content_norm:
                hit = 1
                break
        results[f"Hit@{k}"] = float(hit)
    
    # MRR 계산 (Strict 기준)
    mrr = 0.0
    for idx, doc in enumerate(retrieved_docs):
        content_norm = normalize(doc.page_content)
        if gold_ans_norm in content_norm and target_norm in content_norm:
            mrr = 1 / (idx + 1)
            break
    results["MRR"] = mrr
    return results