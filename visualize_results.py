import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. 실험 데이터 입력 (유저님의 실제 결과값)
data = {
    "Metric": ["Hit@1", "Hit@3", "Hit@5", "MRR"],
    "Case 2 (Markdown)": [0.076923, 0.230769, 0.307692, 0.169231],
    "Case 3 (Hybrid)": [0.461538, 0.615385, 0.615385, 0.538462],
    "Case 4 (Proposed)": [0.615385, 0.615385, 0.615385, 0.615385]
}
# Case 1은 수치가 0이므로 시각적 대비를 위해 제외하거나 포함할 수 있습니다.

df = pd.DataFrame(data)
df_melted = df.melt(id_vars="Metric", var_name="Experiment Case", value_name="Score")

# 2. 그래프 스타일 설정
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")
palette = {"Case 2 (Markdown)": "#AAB8C2", "Case 3 (Hybrid)": "#657786", "Case 4 (Proposed)": "#1DA1F2"}

# 3. 막대그래프 그리기
ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Experiment Case", palette=palette)

# 4. 수치 표시 (막대 위에 점수 적기)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=11, fontweight='bold')

# 5. 그래프 디테일 설정
plt.title("RAG Performance Comparison: Context Padding Effectiveness", fontsize=16, fontweight='bold', pad=20)
plt.ylim(0, 0.8) # 여유 공간 확보
plt.ylabel("Performance Score (0-1)", fontsize=12)
plt.xlabel("Evaluation Metrics", fontsize=12)
plt.legend(title="Processing Method", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("data/processed/experiment_result_graph.png", dpi=300)
plt.show()