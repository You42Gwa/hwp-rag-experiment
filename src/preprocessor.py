import pandas as pd

class HWPPreprocessor:
    def __init__(self, df):
        # HWP에서 추출한 Pandas DataFrame을 받습니다.
        self.df = df

    def get_case_1(self):
        """Case 1: 단순 텍스트 추출 (Plain Text)"""
        return self.df.fillna("").to_string(header=False, index=False)

    def get_case_2(self):
        """Case 2 & 3: 일반 마크다운 (No Padding)"""
        return self.df.to_markdown(index=False)

    def get_case_4(self):
        """Case 4: 문맥 보정 마크다운 (Context Padding)"""
        df_padded = self.df.copy()
        # 첫 번째 열(보통 학과/카테고리)의 병합된 셀(NaN)을 위에서 아래로 채움
        df_padded.iloc[:, 0] = df_padded.iloc[:, 0].ffill()
        return df_padded.to_markdown(index=False)