import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # 虽然此例中未使用Rectangle，但保留以备不时之需

# 读取 Excel 文件
try:
    df = pd.read_excel('219.xlsx')
except FileNotFoundError:
    print("文件未找到，请检查文件路径是否正确。")
    exit()
except Exception as e:
    print(f"读取Excel文件时发生错误: {e}")
    exit()

# 检查数据是否至少包含一列深度数据
if 'Depth' not in df.columns or df['Depth'].isnull().all():
    print("数据未包含有效的'Depth'列或'Depth'列全为空。")
    exit()

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 获取数据列数（除了'Depth'列）
num_columns = df.shape[1] - 1

# 创建一个包含适当数量子图的画布
# 这里我们假设每行只放置一个子图，根据列数调整画布宽度
fig, axes = plt.subplots(1, num_columns, figsize=(num_columns * 2.5, 12))  # 调整2.5以适应子图宽度

# 如果只有一列数据（除了'Depth'），则axes将是一个轴对象而不是数组，需要处理这种情况
if num_columns == 1:
    axes = [axes]

# 定义颜色（这里使用循环颜色，但可以根据需要调整）
colors = plt.cm.get_cmap('tab10', num_columns).colors  # 使用tab10颜色映射，但限制为num_columns种颜色

# 绘制每个数据列的测井曲线
for i, ax in enumerate(axes):
    try:
        # 使用iloc访问除了'Depth'之外的数据列
        data_column = df.columns[i + 1]
        ax.plot(df[data_column], df['Depth'], color=colors[i], label=data_column)
        ax.set_title(data_column)
        ax.set_xlabel('')
        if i == 0:
            ax.set_ylabel('Depth')
        else:
            ax.set_ylabel('')
        ax.invert_yaxis()

        # 隐藏数据标签和数据刻度值（根据需要进行调整）
        ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
        if i != 0:
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            # 将横向刻度放上边框朝外
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        # 如果需要显示图例，可以取消注释以下行（但可能需要调整位置以避免重叠）
        # ax.legend(loc='upper left', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

    except Exception as e:
        print(f"在绘制第{i + 1}列数据时发生错误: {e}")

    # 调整子图之间的间距（根据需要调整）
plt.tight_layout()

# 保存图形
plt.savefig('111.pdf', dpi=300)

# 显示图形
plt.show()