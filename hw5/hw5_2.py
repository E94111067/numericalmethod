import math
import numpy as np

# 定義微分方程組：u' = f(t, u)
def f(t, u):
    u1, u2 = u
    # u1' = 9u1 + 24u2 + 5cos(t) - (1/3)sin(t)
    du1_dt = 9*u1 + 24*u2 + 5*math.cos(t) - (1/3)*math.sin(t)
    # u2' = -24u1 - 51u2 - 9cos(t) + (1/3)sin(t)
    du2_dt = -24*u1 - 51*u2 - 9*math.cos(t) + (1/3)*math.sin(t)
    return np.array([du1_dt, du2_dt])

# 精確解：u1 = 2e^(-3t) - e^(-39t) + (1/3)cos(t), u2 = -e^(-3t) + 2e^(-39t) - (1/3)cos(t)
def exact_solution(t):
    u1 = 2*math.exp(-3*t) - math.exp(-39*t) + (1/3)*math.cos(t)
    u2 = -math.exp(-3*t) + 2*math.exp(-39*t) - (1/3)*math.cos(t)
    return np.array([u1, u2])

# 四階 Runge-Kutta 方法實現
def runge_kutta_4(t0, u0, t_end, h):
    # 初始化列表來儲存 t 和 u 的值
    t_values = [t0]
    u_values = [u0]
    
    # 當前的 t 和 u
    t = t0
    u = u0
    
    # 迭代直到達到 t_end
    while t < t_end - 1e-10:  # 加上小偏移避免浮點誤差
        # 確保不會超過 t_end
        if t + h > t_end:
            h = t_end - t
        
        # 四階 Runge-Kutta 公式
        k1 = f(t, u)
        k2 = f(t + h/2, u + (h/2)*k1)
        k3 = f(t + h/2, u + (h/2)*k2)
        k4 = f(t + h, u + h*k3)
        
        # 更新 u
        u = u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        
        # 儲存新的 t 和 u 值
        t_values.append(t)
        u_values.append(u)
    
    return t_values, u_values

# 參數設定
t0 = 0.0  # 初始 t
u0 = np.array([4/3, 2/3])  # 初始條件 u1(0) = 4/3, u2(0) = 2/3
t_end = 1.0  # 終點 t（假設）
h_values = [0.05, 0.1]  # 步長 h = 0.05 和 0.1

# 針對每個 h 值進行計算並顯示結果
for h in h_values:
    # 執行 Runge-Kutta 方法
    t_values, u_values = runge_kutta_4(t0, u0, t_end, h)
    
    # 計算每個 t 對應的精確解
    u_exact = [exact_solution(t) for t in t_values]
    
    # 輸出結果表格
    print(f"\nRunge-Kutta 方法結果 (h = {h})")
    print("t\tRK u1\t\tRK u2\t\tExact u1\tExact u2\tError u1\tError u2")
    print("-" * 100)
    for t, u_rk, u_ex in zip(t_values, u_values, u_exact):
        error_u1 = abs(u_ex[0] - u_rk[0])
        error_u2 = abs(u_ex[1] - u_rk[1])
        print(f"{t:.2f}\t{u_rk[0]:.6f}\t{u_rk[1]:.6f}\t{u_ex[0]:.6f}\t{u_ex[1]:.6f}\t{error_u1:.6f}\t{error_u2:.6f}")
        