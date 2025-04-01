import numpy as np
import scipy.interpolate as spi

# 觀測數據
T = np.array([0, 3, 5, 8, 13])  # 時間 (秒)
D = np.array([0, 200, 375, 620, 990])  # 距離 (英尺)
V = np.array([75, 77, 80, 74, 72])  # 速度 (英尺/秒)

# 進行 Hermite 插值
hermite_poly = spi.CubicHermiteSpline(T, D, V)

# (a) 預測 t = 10 秒時的位置與速度
position_at_10 = hermite_poly(10)
speed_at_10 = hermite_poly.derivative()(10)
speed_at_10_mph = speed_at_10 * 3600 / 5280  # 轉換為 mi/h
print(f"(a)：t = 10 s 時，位置 = {position_at_10:.2f} 英尺，速度 = {speed_at_10_mph:.2f} mi/h")

# (b) 找出超過 55 mi/h (55 * 5280 / 3600) = 80.67 英尺/秒的時間點
speed_limit = 55 * 5280 / 3600  # 80.67 英尺/秒
roots = hermite_poly.derivative().roots()
exceed_times = [t for t in roots if hermite_poly.derivative()(t) > speed_limit and 0 <= t <= 13]
if exceed_times:
    print(f"(b)：首次超過 55 mi/h 的時間為 t = {exceed_times[0]:.2f} 秒")
else:
    print("(b)：汽車未曾超過 55 mi/h ")

# (c) 預測最大速度
from scipy.optimize import minimize_scalar
result = minimize_scalar(lambda t: -hermite_poly.derivative()(t), bounds=(0, 13), method='bounded')
max_speed = -result.fun
max_speed_time = result.x
print(f"(c)：t = {max_speed_time:.2f} 秒時，最大速度為 {max_speed:.2f} 英尺/秒")