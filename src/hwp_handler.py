# import os
# import pandas as pd
# from pyhwpx import Hwp
# from bs4 import BeautifulSoup
# import io
# import re

# def get_table_from_hwp(file_path):
#     """HWP 파일을 열어 표를 추출하고, 한 셀에 뭉친 데이터를 여러 행으로 분리함"""
#     abs_path = os.path.abspath(file_path)
#     hwp = Hwp()
#     hwp.open(abs_path)
    
#     temp_html = "temp_table.html"
#     hwp.save_as(temp_html, "HTML")
#     hwp.quit()
    
#     html_content = ""
#     try:
#         # 1. 인코딩 대응하여 파일 읽기
#         try:
#             with open(temp_html, 'r', encoding='utf-8') as f:
#                 html_content = f.read()
#         except UnicodeDecodeError:
#             with open(temp_html, 'r', encoding='cp949') as f:
#                 html_content = f.read()
#     finally:
#         if os.path.exists(temp_html):
#             os.remove(temp_html)

#     try:
#         # 2. HTML 파싱 (html5lib 엔진 사용)
#         tables = pd.read_html(io.StringIO(html_content), flavor='html5lib')
#         if not tables:
#             raise Exception("문서 내에서 표를 찾을 수 없습니다.")
        
#         df = tables[0]

#         # 3. [핵심] 첫 번째 행을 컬럼명으로 설정 ('명단' 컬럼 인식을 위함)
#         df.columns = df.iloc[0]
#         df = df[1:].reset_index(drop=True)

#         # 4. 데이터 정제 (불필요한 공백 제거)
#         df = df.map(lambda x: str(x).strip() if pd.notnull(x) else x)

#         # 5. [중요] 한 셀 내의 여러 데이터를 리스트로 분리
#         # '1반', '2반' 등 반 정보를 기준으로 명단을 쪼개는 함수
#         def split_merged_cell(text):
#             if pd.isna(text) or text == 'nan': return [None]
#             # '숫자+반' 패턴 앞에 공백이 있으면 그 지점을 기준으로 쪼갬
#             parts = re.split(r'\s+(?=\d+반)', str(text))
#             return [p.strip() for p in parts if p.strip()]

#         if '명단' in df.columns and '생년월일' in df.columns:
#             # 명단 분리
#             df['명단'] = df['명단'].apply(split_merged_cell)
#             # 생년월일 분리 (공백 기준)
#             df['생년월일'] = df['생년월일'].apply(lambda x: str(x).split() if pd.notnull(x) else [None])
            
#             # 리스트를 각각의 행으로 확장 (Explode)
#             df = df.explode(['명단', '생년월일']).reset_index(drop=True)

#             # 6. 논문 실험을 위해 '학과' 정보가 첫 행에만 남도록 설정 (Case 2 vs 4 차이 생성)
#             # explode 후에는 학과가 반복되어 있으므로, 그룹별 첫 행만 남기고 나머지는 NaN 처리
#             # (학과명이 바뀔 때만 값을 남김)
#             df.loc[df['학과'].duplicated(), '학과'] = None

#         return df
        
#     except Exception as e:
#         raise Exception(f"표 파싱 실패: {e}")

import os
import pandas as pd
from pyhwpx import Hwp
from bs4 import BeautifulSoup
import io

def get_table_from_hwp(file_path):
    """행별 데이터 길이를 안전하게 맞추고 분리하는 안정화된 범용 추출기"""
    abs_path = os.path.abspath(file_path)
    hwp = Hwp()
    hwp.open(abs_path)
    
    temp_html = "temp_table.html"
    hwp.save_as(temp_html, "HTML")
    hwp.quit()
    
    try:
        # 1. 파일 읽기
        try:
            with open(temp_html, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, "html.parser")
        except UnicodeDecodeError:
            with open(temp_html, 'r', encoding='cp949') as f:
                soup = BeautifulSoup(f, "html.parser")

        # 2. 표 데이터 추출
        table = soup.find('table')
        if not table:
            raise Exception("표를 찾을 수 없습니다.")

        rows_data = []
        for tr in table.find_all('tr'):
            row = []
            for td in tr.find_all(['td', 'th']):
                # 셀 내부의 각 문단(<p>)을 리스트로 추출
                paragraphs = [p.get_text(strip=True) for p in td.find_all('p') if p.get_text(strip=True)]
                if not paragraphs:
                    paragraphs = [td.get_text(strip=True)]
                row.append(paragraphs)
            if any(row):
                rows_data.append(row)

        # 3. 임시 데이터프레임 생성 및 헤더 설정
        raw_df = pd.DataFrame(rows_data)
        headers = [str(c[0]).strip() if c else f"col_{i}" for i, c in enumerate(raw_df.iloc[0])]
        content_df = raw_df[1:].reset_index(drop=True)

        # 4. [오류 해결] 리스트 길이 동기화 및 새로운 데이터 구조 생성
        # broadcast 에러를 피하기 위해 리스트 컴프리헨션으로 데이터를 재구축합니다.
        synced_rows = []
        for _, row in content_df.iterrows():
            max_l = max(len(x) for x in row)
            # 각 셀의 리스트 길이를 행의 최대 길이에 맞춰 패딩(None 채우기)
            synced_row = [x + [None] * (max_l - len(x)) for x in row]
            synced_rows.append(synced_row)
        
        # 동기화된 데이터로 다시 데이터프레임 생성
        df = pd.DataFrame(synced_rows, columns=headers)

        # 5. 행 확장 (Explode)
        # 이제 모든 리스트의 길이가 동일하므로 에러 없이 확장됩니다.
        df = df.explode(list(df.columns)).reset_index(drop=True)

        # 6. 데이터 정제
        def clean_value(x):
            if x is None or str(x).lower() == 'nan' or str(x).strip() == '':
                return None
            return str(x).strip()
        
        df = df.map(clean_value)

        # 7. [논문 핵심] 상위 카테고리 중복 제거 (Case 2용 유실 상태 생성)
        first_col = df.columns[0]
        df.loc[df[first_col].duplicated(), first_col] = None

        return df
        
    finally:
        if os.path.exists(temp_html):
            os.remove(temp_html)