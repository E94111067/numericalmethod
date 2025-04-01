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

def error_bound(x_values, x_target):
    """
    計算 Lagrange 插值的誤差界限。
    由於 f^(n+1)(x) 對 cos(x) 來說最多為 1，誤差界限近似為：
    E_n(x) ≈ (max |f^(n+1)(ξ)| / (n+1)!) * Π (x - x_i)
    其中 max |f^(n+1)(ξ)| 取最大可能值 1。
    """
    n = len(x_values) - 1  # 插值多項式的次數
    factorial = np.math.factorial(n + 1)
    product_term = np.prod([x_target - x_i for x_i in x_values])
    return abs(product_term / factorial)

# 給定數據點
x_points = [0.698, 0.733, 0.768, 0.803]
y_points = [0.7661, 0.7432, 0.7193, 0.6946]
x_target = 0.750

# 計算各階 Lagrange 插值結果
for degree in range(1, 5):
    approx_value = lagrange_interpolation(x_points[:degree+1], y_points[:degree+1], x_target)
    error = error_bound(x_points[:degree+1], x_target)
    print(f"Degree {degree}: Lagrange approximation = {approx_value:.6f}, Error bound = {error:.6e}")
