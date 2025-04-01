import numpy as np

def lagrange_interpolation(x_values, y_values, x):
    """
    使用 Lagrange 插值法計算給定 x 的對應 y 值。
    :param x_values: 已知數據點的 x 值列表
    :param y_values: 已知數據點的 y 值列表
    :param x: 需要估算的 x 值
    :return: 對應的 y 值
    """
    n = len(x_values)
    result = 0
    
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    
    return result

def iterated_inverse_interpolation(x_values, y_values, y_target, iterations=5):
    """
    使用迭代逆插值法求解 x，使得 f(x) = y_target。
    :param x_values: 已知數據點的 x 值列表
    :param y_values: 已知數據點的 y 值列表
    :param y_target: 目標函數值
    :param iterations: 最大迭代次數
    :return: 近似的 x 值
    """
    x_guess = x_values[0]  # 初始猜測值
    for _ in range(iterations):
        x_guess = lagrange_interpolation(y_values, x_values, y_target)
    return x_guess

# 給定數據點
x_points = [0.3, 0.4, 0.5, 0.6]
y_points = [0.740818, 0.670320, 0.606531, 0.548812]
y_target = 0  # 要解的方程 0 = x - e^(-x)

# 使用迭代逆插值法求解
x_solution = iterated_inverse_interpolation(x_points, y_points, y_target)
print(f"Approximate solution to x - e^(-x) = 0: x ≈ {x_solution:.6f}")
