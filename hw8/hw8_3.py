import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 設定 m 個取樣點
m = 16
x = np.linspace(0, 1, m, endpoint=False)
f = x**2 * np.sin(x)

# 初始化 Fourier 係數
a0 = np.sum(f) / m
ak = np.zeros(4)
bk = np.zeros(4)

for k in range(1, 5):
    ak[k-1] = 2 * np.sum(f * np.cos(2 * np.pi * k * x)) / m
    bk[k-1] = 2 * np.sum(f * np.sin(2 * np.pi * k * x)) / m

# 定義 S4(x)
def S4(x_val):
    result = a0
    for k in range(1, 5):
        result += ak[k-1] * np.cos(2 * np.pi * k * x_val) + bk[k-1] * np.sin(2 * np.pi * k * x_val)
    return result

# (b) 計算 ∫₀¹ S₄(x) dx
S4_integral, _ = quad(S4, 0, 1)
print(f"(b) ∫₀¹ S₄(x) dx ≈ {S4_integral:.8f}")

# (c) 計算 ∫₀¹ x² sin(x) dx 做比較
true_integral, _ = quad(lambda x: x**2 * np.sin(x), 0, 1)
print(f"(c) ∫₀¹ x² sin(x) dx = {true_integral:.8f}")
print(f"Difference = {abs(S4_integral - true_integral):.8f}")

# (d) 計算誤差 E(S4) = Σ (f(x_j) - S₄(x_j))²
x_test = np.linspace(0, 1, 1000)
f_test = x_test**2 * np.sin(x_test)
S4_test = S4(x_test)
error = np.sqrt(np.mean((f_test - S4_test)**2))
print(f"(d) RMS Error E(S₄) ≈ {error:.8f}")

# 畫圖比較
plt.plot(x_test, f_test, label="f(x) = x² sin(x)")
plt.plot(x_test, S4_test, label="S₄(x)", linestyle="--")
plt.title("Least Squares Trigonometric Polynomial S₄(x)")
plt.legend()
plt.grid(True)
plt.show()
