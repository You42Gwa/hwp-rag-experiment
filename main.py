import os
from src.hwp_handler import get_table_from_hwp
from src.preprocessor import HWPPreprocessor

def main():
    # 1. 파일 경로 설정
    hwp_filename = "test.hwp" 
    hwp_path = os.path.join("data", "raw", hwp_filename)
    processed_dir = os.path.join("data", "processed")
    
    if not os.path.exists(hwp_path):
        print(f"❌ 파일을 찾을 수 없습니다: {hwp_path}")
        return

    print(f"🚀 '{hwp_filename}' 처리 및 3가지 실험군(Case 1, 2&3, 4) 생성 시작...")

    try:
        # 2. HWP에서 표 추출 (행 분리 로직 포함)
        df = get_table_from_hwp(hwp_path)
        
        # 3. 전처리기 실행
        prep = HWPPreprocessor(df)
        
        # 실험 설계에 따른 3가지 케이스 정의
        cases = {
            "Case 1 (Plain Text)": prep.get_case_1(),
            "Case 2 & 3 (Standard Markdown)": prep.get_case_2(),
            "Case 4 (Proposed - Context Padding)": prep.get_case_4()
        }

        # 4. 결과 출력 및 저장
        os.makedirs(processed_dir, exist_ok=True)
        
        # 파일명 매핑
        file_map = {
            "Case 1 (Plain Text)": "case1.md",
            "Case 2 & 3 (Standard Markdown)": "case2_3.md",
            "Case 4 (Proposed - Context Padding)": "case4.md"
        }

        for title, content in cases.items():
            # 파일 저장
            save_path = os.path.join(processed_dir, file_map[title])
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # 터미널 출력
            print("\n" + "="*50)
            print(f"[{title}]")
            print("-" * 50)
            print(content)
            print("="*50)

        print(f"\n✅ 모든 파일(case1.md, case2_3.md, case4.md)이 '{processed_dir}'에 저장되었습니다.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()