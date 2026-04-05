# import pandas as pd
# import random
# import json
# import os

# # 1. 설정값
# DEPARTMENTS = [
#     "인공지능융합공학부", "금융IT소프트웨어학부", "바이오화학공학과", 
#     "첨단기계설계학과", "스마트건설공학과", "전자제어공학부", 
#     "신소재공학부", "에너지시스템공학과", "컴퓨터학부", "데이터과학과"
# ]
# SCHOLARSHIP_TYPES = ["성적우수", "가계곤란", "근로장학", "리더십", "글로벌", "기업매칭", "산학협력"]

# data = []

# # 2. 500명 데이터 생성 (학과당 50명씩)
# for dept in DEPARTMENTS:
#     for i in range(1, 51):
#         # [실험 핵심] 이름을 고유하게 만들어 검색 모호성을 제거합니다.
#         # 예: 인공_01, 금융_15 등
#         unique_name = f"{dept[:2]}_{i:02d}" 
#         grade = f"{(i % 4) + 1}학년"
#         stype = random.choice(SCHOLARSHIP_TYPES)
#         # 금액은 50만원 ~ 400만원 사이 랜덤
#         amount = f"{random.randint(50, 400) * 10000:,}원"
        
#         data.append([dept, grade, unique_name, stype, amount])

# df = pd.DataFrame(data, columns=["학과", "학년", "성명", "장학유형", "금액"])

# # 3. CSV 저장 (엑셀에서 열어서 HWP로 복사용)
# csv_path = "실험용_장학명단_500명ver2.csv"
# df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# # 4. 골드 데이터셋(질문지) 생성
# os.makedirs("data", exist_ok=True)
# gold_dataset = []

# # 실험의 변별력을 위해 학과별로 골고루 질문 생성
# for dept in DEPARTMENTS:
#     # 각 학과에서 무작위로 2명씩 추출 (총 20개 질문)
#     sample_students = df[df["학과"] == dept].sample(2)
    
#     for _, s in sample_students.iterrows():
#         gold_dataset.append({
#             "question": f"{s['학과']} 소속 {s['성명']} 학생의 장학금액은 얼마인가요?",
#             "answer": s["금액"],
#             "target_keyword": s["학과"] # 검색 성공 여부 판단 기준
#         })

# with open("data/gold_dataset.json", "w", encoding="utf-8") as f:
#     json.dump(gold_dataset, f, indent=4, ensure_ascii=False)

# print(f"✅ 데이터 생성 완료!")
# print(f"1. 표 데이터: {csv_path} (이 파일을 HWP로 옮겨 '학과' 셀 병합을 진행하세요)")
# print(f"2. 평가 데이터: data/gold_dataset.json (총 {len(gold_dataset)}개의 고유 질문 생성)")


### 중복 이름 데이터

# import pandas as pd
# import random
# import json

# # 1. 설정값
# DEPARTMENTS = [
#     "인공지능융합공학부", "금융IT소프트웨어학부", "바이오화학공학과", 
#     "첨단기계설계학과", "스마트건설공학과", "전자제어공학부", 
#     "신소재공학부", "에너지시스템공학과", "컴퓨터학부", "데이터과학과"
# ]
# SCHOLARSHIP_TYPES = ["성적우수", "가계곤란", "근로장학", "리더십", "글로벌", "기업매칭", "산학협력"]
# NAMES_POOL = ["김민준", "이서연", "박도윤", "최서윤", "정주원", "강현우", "조예준", "임지우", "윤준서", "한소희"] # 중복 유도를 위한 이름 풀

# data = []

# # 2. 500명 데이터 생성 (학과당 50명씩)
# for dept in DEPARTMENTS:
#     for i in range(1, 51):
#         name = random.choice(NAMES_POOL)
#         # 학년은 골고루 분포
#         grade = f"{(i % 4) + 1}학년"
#         stype = random.choice(SCHOLARSHIP_TYPES)
#         amount = f"{random.randint(50, 400) * 10000:,}원"
        
#         data.append([dept, grade, name, stype, amount])

# df = pd.DataFrame(data, columns=["학과", "학년", "성명", "장학유형", "금액"])

# # 3. CSV 저장 (엑셀에서 열어서 HWP로 복사 가능)
# df.to_csv("실험용_장학명단_500명.csv", index=False, encoding="utf-8-sig")

# # 4. 골드 데이터셋(질문지) 생성
# # 실험의 변별력을 위해 '학과별 마지막 행'과 '동명이인' 위주로 질문 생성
# gold_dataset = []

# # 4-1. 각 학과의 맨 마지막 학생 질문 (병합 셀의 끝부분 문맥 확인용)
# for dept in DEPARTMENTS:
#     last_student = df[df["학과"] == dept].iloc[-1]
#     gold_dataset.append({
#         "question": f"{dept} 소속 {last_student['성명']} 학생의 장학금액은 얼마인가요?",
#         "answer": last_student["금액"],
#         "target_keyword": dept
#     })

# # 4-2. 동명이인 질문 (문맥 보정 확인용)
# # '김민준' 학생을 특정 학과와 결합하여 질문
# for dept in random.sample(DEPARTMENTS, 3):
#     student = df[(df["학과"] == dept) & (df["성명"] == "김민준")].iloc[0]
#     gold_dataset.append({
#         "question": f"{dept}의 김민준 학생이 받는 장학유형은 무엇인가요?",
#         "answer": student["장학유형"],
#         "target_keyword": dept
#     })

# with open("data/gold_dataset.json", "w", encoding="utf-8") as f:
#     json.dump(gold_dataset, f, indent=4, ensure_ascii=False)

# print("✅ '실험용_장학명단_500명.csv'와 'data/gold_dataset.json' 생성이 완료되었습니다.")

import pandas as pd
import json
import os

# 1. 기존 파일 로드
csv_path = "실험용_장학명단_500명.csv"
if not os.path.exists(csv_path):
    print(f"❌ '{csv_path}' 파일이 없습니다. 경로를 확인해주세요.")
    exit()

# 인코딩 에러 방지를 위해 utf-8-sig 사용
df = pd.read_csv(csv_path, encoding='utf-8-sig')

# 2. 질문 생성 로직 (학년 정보를 추가하여 모호성 제거)
gold_dataset = []

# 각 학과별로 골고루 질문을 생성 (학과당 5명씩, 총 50개 질문)
departments = df['학과'].unique()

for dept in departments:
    # 해당 학과의 학생들 중 5명을 무작위 샘플링
    subset = df[df['학과'] == dept]
    sample_size = min(len(subset), 5) # 데이터가 적을 경우 대비
    samples = subset.sample(n=sample_size)
    
    for _, row in samples.iterrows():
        # [핵심] 질문에 '학년'을 명시하여 동명이인 중 특정인을 지칭함
        question = f"{row['학과']} {row['학년']} 소속 {row['성명']} 학생의 장학금액은 얼마인가요?"
        
        gold_dataset.append({
            "question": question,
            "answer": str(row['금액']).strip(),
            "target_keyword": row['학과']
        })

# 3. JSON 저장
os.makedirs("data", exist_ok=True)
save_path = "data/gold_dataset.json"

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(gold_dataset, f, indent=4, ensure_ascii=False)

print(f"✅ 기존 CSV를 참고하여 {len(gold_dataset)}개의 새로운 질문을 생성했습니다.")
print(f"📍 저장 위치: {save_path}")
print(f"💡 이제 '학년' 정보가 포함되어 Case 4의 검색 정확도가 비약적으로 상승할 것입니다.")