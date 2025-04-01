import numpy as np
import scipy.interpolate as spi
from scipy.optimize import minimize_scalar

def hermite_interpolation(T, D, V, t_eval):
    hermite_poly = spi.CubicHermiteSpline(T, D, V)
    D_pred = hermite_poly(t_eval)
    V_pred = hermite_poly.derivative()(t_eval)
    return D_pred, V_pred

# 已知數據點
time_points = np.array([0, 3, 5, 8, 13])
distance_points = np.array([0, 200, 375, 620, 990])
speed_points = np.array([75, 77, 80, 74, 72])

target_time = 10  # 預測 t = 10 秒時的位置與速度
D_10, V_10 = hermite_interpolation(time_points, distance_points, speed_points, target_time)
print(f"當 t = {target_time} 秒時，預測距離為 {D_10:.2f} 英尺，預測速度為 {V_10:.2f} 英尺/秒")

# (b) 找出超過 55 mi/h (80.67 ft/s) 的時間點
speed_limit = 80.67
hermite_poly = spi.CubicHermiteSpline(time_points, distance_points, speed_points)

# 生成細化的時間點來評估速度
t_values = np.linspace(0, 13, 500)
v_values = hermite_poly.derivative()(t_values)

# 找出速度大於 80.67 ft/s 的時間區間
exceed_indices = np.where(v_values > speed_limit)[0]
if len(exceed_indices) > 0:
    first_exceed_time = t_values[exceed_indices[0]]
    print(f"汽車首次超過限速 55 mi/h 的時間為 t ≈ {first_exceed_time:.2f} 秒")
else:
    print("汽車未曾超過限速 55 mi/h")

# (c) 預測最大速度
result = minimize_scalar(lambda t: -hermite_poly.derivative()(t), bounds=(0, 13), method='bounded')
max_speed = -result.fun
max_time = result.x
print(f"預測最大速度為 {max_speed:.2f} 英尺/秒，發生於 t ≈ {max_time:.2f} 秒")
