import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import os

# 创建输出目录
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 20

# 房间和墙壁参数
room_width = 8
room_height = 4
wall_x_start = 1.9
wall_x_end = 2.1
wall_y_range = [0, 4]
door_y_range = [1.5, 2.5]

# 定义所有AP布置方案
schemes = {
    'IG-MIC (Wall Optimization)': {
        'positions': [[0.0, 0.0], [3.0, 2.5], [2.5, 0.0], [7.5, 3.5]],
        'marker': '*',
        'markersize': 30,
        'color': 'red'  # 改为红色
    },
    'GenoPlacement': {
        'positions': [[6.5, 2.0], [7.0, 2.5], [8.0, 3.0], [5.5, 4.0]],
        'marker': 'o',
        'markersize': 20,
        'color': 'blue'  # 改为蓝色
    },
    'HOS': {
        'positions': [[8.0, 2.0], [0.0, 2.0], [7.0, 1.0], [0.0, 3.0]],
        'marker': '^',
        'markersize': 20,
        'color': 'green'  # 保持绿色
    },
    'MAAP': {
        'positions': [[3.99, 2.40], [5.00, 1.60], [2.44, 2.08], [6.40, 2.02]],
        'marker': 'X',
        'markersize': 24,
        'color': 'magenta'  # 改为洋红色
    }
}


def draw_wall_and_door(ax):
    """绘制墙壁和门"""
    # 绘制门下方的墙壁
    if door_y_range[0] > wall_y_range[0]:
        wall_bottom = Rectangle(
            (wall_x_start, wall_y_range[0]),
            wall_x_end - wall_x_start,
            door_y_range[0] - wall_y_range[0],
            facecolor='gray',
            edgecolor='black',
            linewidth=2,
            alpha=0.7,
            label='Wall'
        )
        ax.add_patch(wall_bottom)

    # 绘制门上方的墙壁
    if door_y_range[1] < wall_y_range[1]:
        wall_top = Rectangle(
            (wall_x_start, door_y_range[1]),
            wall_x_end - wall_x_start,
            wall_y_range[1] - door_y_range[1],
            facecolor='gray',
            edgecolor='black',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(wall_top)

    # 绘制门
    door = Rectangle(
        (wall_x_start, door_y_range[0]),
        wall_x_end - wall_x_start,
        door_y_range[1] - door_y_range[0],
        facecolor='lightgreen',
        edgecolor='green',
        linewidth=2,
        linestyle='--',
        alpha=0.5,
        label='Door'
    )
    ax.add_patch(door)


# 创建2×2子图布局
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('AP Placement Comparison in Wall Environment (4m × 8m)', fontsize=32, fontweight='bold')

# 为每个方案绘制独立子图
for idx, (name, scheme) in enumerate(schemes.items()):
    ax = axes[idx // 2, idx % 2]

    # 绘制房间边界（粗线）
    ax.plot([0, room_width], [0, 0], 'k-', linewidth=4)
    ax.plot([0, room_width], [room_height, room_height], 'k-', linewidth=4)
    ax.plot([0, 0], [0, room_height], 'k-', linewidth=4)
    ax.plot([room_width, room_width], [0, room_height], 'k-', linewidth=4)

    # 绘制墙壁和门
    draw_wall_and_door(ax)

    # 绘制AP位置
    positions = scheme['positions']
    for i, pos in enumerate(positions):
        if name == 'GenoPlacement':  # GenoPlacement使用空心圆
            ax.plot(pos[0], pos[1], scheme['marker'], color=scheme['color'],
                    markersize=scheme['markersize'], markerfacecolor='none', markeredgewidth=4)
        else:
            ax.plot(pos[0], pos[1], scheme['marker'], color=scheme['color'],
                    markersize=scheme['markersize'])

        # 添加AP标签
        ax.annotate(f'AP{i + 1}', xy=pos, xytext=(pos[0] + 0.2, pos[1] + 0.2),
                    fontsize=18, fontweight='bold')

    # 设置坐标轴属性
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel('Meters (m)', fontsize=22)
    ax.set_ylabel('Meters (m)', fontsize=22)
    ax.set_title(name, fontsize=24, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    # 设置刻度
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.tick_params(labelsize=18)

    # 只在第一个子图添加图例
    if idx == 0:
        ax.legend(loc='upper right', fontsize=14, framealpha=0.9)

plt.tight_layout()
plt.savefig(f'{output_dir}/ap_layouts_2x2.png', dpi=300, bbox_inches='tight')
print("已保存2×2子图布局")

# 创建综合对比图
fig2, ax2 = plt.subplots(figsize=(12, 7))

# 定义各算法的样式（用于综合对比图）
algorithm_styles = {
    'MIC-FD': {
        'positions': [[0.0, 0.0], [3.0, 2.5], [5.5, 1.0], [6.5, 3.0]],
        'color': 'red', 'marker': '*', 'size': 30  # 改为红色
    },
    'GenoPlacement': {
        'positions': [[6.5, 2.0], [7.0, 2.5], [8.0, 3.0], [5.5, 4.0]],
        'color': 'blue', 'marker': 'o', 'size': 20  # 改为蓝色
    },
    'HOS': {
        'positions': [[8.0, 2.0], [0.0, 2.0], [7.0, 1.0], [0.0, 3.0]],
        'color': 'green', 'marker': '^', 'size': 20  # 保持绿色
    },
    'MAAP': {
        'positions': [[3.99, 2.40], [5.00, 1.60], [2.44, 2.08], [6.40, 2.02]],
        'color': 'magenta', 'marker': 'X', 'size': 24  # 改为洋红色
    }
}

# 绘制所有算法的AP位置
for name, style in algorithm_styles.items():
    positions = style['positions']
    for i, pos in enumerate(positions):
        if name == 'GenoPlacement':  # GenoPlacement使用空心圆
            ax2.plot(pos[0], pos[1], style['marker'], color=style['color'],
                     markersize=style['size'], markerfacecolor='none',
                     markeredgewidth=4, label=name if i == 0 else '')
        else:
            ax2.plot(pos[0], pos[1], style['marker'], color=style['color'],
                     markersize=style['size'], label=name if i == 0 else '')

# 绘制房间边界（粗线）
ax2.plot([0, room_width], [0, 0], 'k-', linewidth=4)
ax2.plot([0, room_width], [room_height, room_height], 'k-', linewidth=4)
ax2.plot([0, 0], [0, room_height], 'k-', linewidth=4)
ax2.plot([room_width, room_width], [0, room_height], 'k-', linewidth=4)

# 绘制墙壁和门
draw_wall_and_door(ax2)

# 设置坐标轴属性
ax2.set_xlim(-0.5, 8.5)
ax2.set_ylim(-0.5, 4.5)
ax2.set_xlabel('Meters (m)', fontsize=28)
ax2.set_ylabel('Meters (m)', fontsize=28)
ax2.set_title('', fontsize=32, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_aspect('equal')

# 添加图例
ax2.legend(loc='upper left', fontsize=16, framealpha=0.9)

# 设置刻度
ax2.set_xticks(np.arange(0, 9, 1))
ax2.set_yticks(np.arange(0, 5, 1))
ax2.tick_params(labelsize=20)

# 添加墙壁位置标注
ax2.text(2.0, 4.3, 'Wall', fontsize=16, ha='center',
         bbox=dict(boxstyle='round', facecolor='gray', alpha=0.5))
ax2.text(2.0, 2.0, 'Door', fontsize=16, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/ap_layouts_combined.png', dpi=300, bbox_inches='tight')
print("已保存综合对比图")

# 打印AP位置信息
print("\n" + "=" * 70)
print("AP Position Details")
print("=" * 70)
for scheme_name, style in algorithm_styles.items():
    print(f"\n{scheme_name}:")
    for i, pos in enumerate(style['positions']):
        print(f"  AP{i + 1}: X={pos[0]:.2f}m, Y={pos[1]:.2f}m")

print(f"\nWall Position: X=[{wall_x_start}, {wall_x_end}]m, Y=[{wall_y_range[0]}, {wall_y_range[1]}]m")
print(f"Door Position: Y=[{door_y_range[0]}, {door_y_range[1]}]m")
print("\nCharts generated successfully!")

plt.show()