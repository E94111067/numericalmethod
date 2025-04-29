import math
import numpy as np

# 定義微分方程：y' = 1 + (y/t) + (y/t)^2
def f(t, y):
    return 1 + (y/t) + (y/t)**2

# 計算 f'(t, y)，用於二階泰勒方法
def f_prime(t, y):
    # 偏導數 df/dt = -y/t^2 - 2y^2/t^3
    df_dt = -y/(t**2) - 2*(y**2)/(t**3)
    # 偏導數 df/dy = 1/t + 2y/t^2
    df_dy = 1/t + 2*y/(t**2)
    # f'(t, y) = df/dt + df/dy * f(t, y)
    return df_dt + df_dy * f(t, y)

# 精確解：y(t) = t * tan(ln(t))
def exact_solution(t):
    return t * math.tan(math.log(t))

# Euler 方法實現
def euler_method(t0, y0, t_end, h):
    # 初始化列表來儲存 t 和 y 的值
    t_values = [t0]
    y_values = [y0]
    
    # 當前的 t 和 y
    t = t0
    y = y0
    
    # 迭代直到達到 t_end
    while t < t_end:
        # 確保不會超過 t_end
        if t + h > t_end:
            h = t_end - t
        
        # Euler 方法公式：y_n+1 = y_n + h * f(t_n, y_n)
        y = y + h * f(t, y)
        t = t + h
        
        # 儲存新的 t 和 y 值
        t_values.append(t)
        y_values.append(y)
    
    return t_values, y_values

# 二階泰勒方法實現
def taylor_method_order2(t0, y0, t_end, h):
    # 初始化列表來儲存 t 和 y 的值
    t_values = [t0]
    y_values = [y0]
    
    # 當前的 t 和 y
    t = t0
    y = y0
    
    # 迭代直到達到 t_end
    while t < t_end:
        # 確保不會超過 t_end
        if t + h > t_end:
            h = t_end - t
        
        # 二階泰勒方法公式：y_n+1 = y_n + h * f(t_n, y_n) + (h^2)/2 * f'(t_n, y_n)
        y = y + h * f(t, y) + (h**2)/2 * f_prime(t, y)
        t = t + h
        
        # 儲存新的 t 和 y 值
        t_values.append(t)
        y_values.append(y)
    
    return t_values, y_values

# 參數設定
t0 = 1.0  # 初始 t
y0 = 0.0  # 初始 y
t_end = 2.0  # 終點 t
h = 0.1  # 步長

# 執行 Euler 方法
t_values_euler, y_euler = euler_method(t0, y0, t_end, h)

# 執行二階泰勒方法
t_values_taylor, y_taylor = taylor_method_order2(t0, y0, t_end, h)

# 計算每個 t 對應的精確解
y_exact = [exact_solution(t) for t in t_values_euler]

# 輸出 Euler 方法的結果表格
print("(a)Euler’s method")
print("t\tEuler y\t\tExact y\t\tError")
print("-" * 70)
for t, y_e, y_ex in zip(t_values_euler, y_euler, y_exact):
    error_euler = abs(y_ex - y_e)
    print(f"{t:.1f}\t{y_e:.6f}\t{y_ex:.6f}\t{error_euler:.6f}")

# 加入空行分隔
print("\n")

# 輸出二階泰勒方法的結果表格
print("(b)Taylor’s method of order 2")
print("t\tTaylor y\tExact y\t\tError")
print("-" * 70)
for t, y_t, y_ex in zip(t_values_taylor, y_taylor, y_exact):
    error_taylor = abs(y_ex - y_t)
    print(f"{t:.1f}\t{y_t:.6f}\t{y_ex:.6f}\t{error_taylor:.6f}")