import os
import pandas as pd
from pyhwpx import Hwp
from bs4 import BeautifulSoup
import io

def get_table_from_hwp(file_path):
    """rowspan(셀 병합)으로 인한 열 밀림 현상을 완벽히 해결하는 파서"""
    abs_path = os.path.abspath(file_path)
    hwp = Hwp()
    hwp.open(abs_path)
    
    temp_html = "temp_table.html"
    hwp.save_as(temp_html, "HTML")
    hwp.quit()
    
    try:
        try:
            with open(temp_html, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, "html.parser")
        except UnicodeDecodeError:
            with open(temp_html, 'r', encoding='cp949') as f:
                soup = BeautifulSoup(f, "html.parser")

        table = soup.find('table')
        if not table:
            raise Exception("표를 찾을 수 없습니다.")

        # 1. 표의 전체 크기(행, 열) 파악 및 그리드 초기화
        rows = table.find_all('tr')
        num_rows = len(rows)
        num_cols = 0
        for cell in rows[0].find_all(['td', 'th']):
            num_cols += int(cell.get('colspan', 1))

        # 데이터가 밀리지 않도록 빈 그리드 공간을 만듦
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]

        # 2. 그리드에 데이터 채우기 (rowspan/colspan 고려)
        for r_idx, tr in enumerate(rows):
            c_idx = 0
            for td in tr.find_all(['td', 'th']):
                # 이미 rowspan으로 채워진 칸은 건너뜀
                while c_idx < num_cols and grid[r_idx][c_idx] is not None:
                    c_idx += 1
                
                if c_idx >= num_cols:
                    break

                rowspan = int(td.get('rowspan', 1))
                colspan = int(td.get('colspan', 1))
                
                # 셀 내부 텍스트 추출 (줄바꿈 보존)
                paragraphs = [p.get_text(strip=True) for p in td.find_all('p') if p.get_text(strip=True)]
                content = "\n".join(paragraphs) if paragraphs else td.get_text(strip=True)
                
                # 해당 셀이 차지하는 모든 칸에 데이터(또는 빈값 표시)를 채움
                for r in range(r_idx, min(r_idx + rowspan, num_rows)):
                    for c in range(c_idx, min(c_idx + colspan, num_cols)):
                        # 첫 칸에만 실제 데이터를 넣고 나머지는 None 유지 (나중에 ffill로 채움)
                        if r == r_idx and c == c_idx:
                            grid[r][c] = content if content else ""
                        else:
                            grid[r][c] = "" # 빈 칸임을 명시

                c_idx += colspan

        # 3. 데이터프레임 변환
        full_df = pd.DataFrame(grid)
        
        # 첫 줄을 컬럼명으로 설정
        full_df.columns = [str(c).strip() for c in full_df.iloc[0]]
        df = full_df[1:].reset_index(drop=True)

        # 4. 데이터 정제 (공백 등)
        def clean_value(x):
            if x is None or str(x).strip() == "" or str(x).lower() == 'nan':
                return None
            return str(x).strip()
        
        df = df.map(clean_value)

        # 5. [중요] 첫 번째 열(학과)의 중복을 제거하여 Case 2 상태를 만듦
        # (이미 preprocessor에서 ffill을 하므로, 여기서는 병합된 첫 칸만 남김)
        first_col = df.columns[0]
        # 진짜 비어있어야 할 칸(None)들을 확실히 처리
        df.loc[df[first_col].duplicated(keep='first'), first_col] = None

        return df
        
    finally:
        if os.path.exists(temp_html):
            os.remove(temp_html)