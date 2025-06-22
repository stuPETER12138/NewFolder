"""
最小二乘法拟合工具
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def read_data(file_path):
    """
    读取数据文件，文件格式为每行一个坐标点，以x,y形式存储
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        x_values: x坐标数组
        y_values: y坐标数组
    """
    x_values = []
    y_values = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        x, y = map(float, line.split(','))
                        x_values.append(x)
                        y_values.append(y)
                    except ValueError:
                        print(f"警告: 无法解析行: {line}，已跳过")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        sys.exit(1)
        
    return np.array(x_values), np.array(y_values)


def least_squares_fit(x, y):
    """
    使用最小二乘法计算线性拟合参数
    
    Args:
        x: x坐标数组
        y: y坐标数组
        
    Returns:
        slope: 斜率
        intercept: 截距
    """
    n = len(x)
    
    # 计算均值
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 计算斜率和截距
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        print("错误: x值没有变化，无法进行线性拟合")
        sys.exit(1)
        
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept


def calculate_sensitivity(x, y, slope):
    """
    计算灵敏度（y的变化量除以x的变化量）
    
    Args:
        x: x坐标数组
        y: y坐标数组
        slope: 斜率
        
    Returns:
        sensitivity: 灵敏度
    """
    # 灵敏度就是斜率，但为了保持一致性，我们返回其绝对值
    return abs(slope)


def calculate_nonlinearity_error(x, y, slope, intercept):
    """
    计算非线性误差（输出值与拟合直线的最大偏差除以x相对变化总量）
    
    Args:
        x: x坐标数组
        y: y坐标数组
        slope: 斜率
        intercept: 截距
        
    Returns:
        nonlinearity_error: 非线性误差百分比
    """
    # 计算拟合值
    y_fit = slope * x + intercept
    
    # 计算偏差
    deviations = np.abs(y - y_fit)
    
    # 最大偏差
    max_deviation = np.max(deviations)
    
    # x的相对变化总量（最大值减最小值）
    y_range = np.max(y) - np.min(y)
    
    if y_range == 0:
        print("警告: x值没有变化，非线性误差计算可能不准确")
        return 0
    
    # 非线性误差 = 最大偏差 / x相对变化总量 * 100%
    nonlinearity_error = (max_deviation / y_range) * 100
    
    return nonlinearity_error


def plot_data_and_fit(x, y, slope, intercept, sensitivity, nonlinearity_error, output_path=None):
    """
    绘制数据点和拟合直线
    
    Args:
        x: x坐标数组
        y: y坐标数组
        slope: 斜率
        intercept: 截距
        sensitivity: 灵敏度
        nonlinearity_error: 非线性误差
        output_path: 图像保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(x, y, color='blue', label='数据点')
    
    # 绘制拟合直线
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color='red', label=f'拟合直线: y = {slope:.4f}x + {intercept:.4f}')
    
    # 添加标题和标签
    plt.title(f'最小二乘法拟合结果 (灵敏度 = {sensitivity:.4f}, 非线性误差 = {nonlinearity_error:.2f}%)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    
    # 保存图像（如果指定了输出路径）
    if output_path:
        plt.savefig(output_path)
    
    plt.show()


def process_data_file(file_path, output_dir=None):
    """
    处理数据文件，计算最小二乘法拟合，并绘制结果
    
    Args:
        file_path: 数据文件路径
        output_dir: 输出目录（可选）
    """
    # 读取数据
    x, y = read_data(file_path)
    
    if len(x) < 2:
        print("错误: 数据点数量不足，至少需要2个点进行拟合")
        return
    
    # 计算最小二乘法拟合
    slope, intercept = least_squares_fit(x, y)
    
    # 计算灵敏度
    sensitivity = calculate_sensitivity(x, y, slope)
    
    # 计算非线性误差
    nonlinearity_error = calculate_nonlinearity_error(x, y, slope, intercept)
    
    # 打印结果
    print(f"\n最小二乘法拟合结果:")
    print(f"斜率 (k): {slope:.6f}")
    print(f"截距 (b): {intercept:.6f}")
    print(f"灵敏度: {sensitivity:.6f}")
    print(f"非线性误差: {nonlinearity_error:.2f}%")
    
    # 设置输出路径（如果指定了输出目录）
    output_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_name = Path(file_path).stem
        output_path = os.path.join(output_dir, f"{file_name}_fit.png")
    
    # 绘制结果
    plot_data_and_fit(x, y, slope, intercept, sensitivity, nonlinearity_error, output_path)
    
    return slope, intercept, sensitivity, nonlinearity_error


def main():
    """
    主函数
    """
    # 获取当前脚本所在目录的父目录（项目根目录）
    project_root = Path(__file__).parent.parent
    
    # 默认数据文件路径
    default_data_path = project_root / "data" / "example_data.txt"
    
    # 默认输出目录
    default_output_dir = project_root / "output"
    
    # 处理命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='最小二乘法拟合工具')
    parser.add_argument('-f', '--file', type=str, default=str(default_data_path),
                        help='数据文件路径 (默认: data/example_data.txt)')
    parser.add_argument('-o', '--output', type=str, default=str(default_output_dir),
                        help='输出目录路径 (默认: output/)')
    parser.add_argument('--no-plot', action='store_true',
                        help='不显示图像 (仅保存)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 处理数据文件
    try:
        result = process_data_file(args.file, args.output)
        if result and not args.no_plot:
            slope, intercept, sensitivity, nonlinearity_error = result
            print(f"\n图像已保存到: {args.output}")
    except Exception as e:
        print(f"处理数据时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()