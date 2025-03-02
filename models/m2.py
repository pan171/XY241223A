import numpy as np


def model_2(w, delta_p, Q_loss, mu, r_w, v_m):
    """定义 Verga 模型方程"""
    term1 = (6 * Q_loss * mu) / (np.pi * w**3)
    term2 = np.log((v_m / (np.pi * w) + r_w**2) / r_w)
    return term1 * term2 - delta_p


def df_model_2(w, Q_loss, mu, r_w, v_m):
    """计算 Verga 模型方程的导数"""
    term1 = (-18 * Q_loss * mu) / (np.pi * w**4)
    term2 = np.log((v_m / (np.pi * w) + r_w**2) / r_w)

    term3_numerator = -v_m / (np.pi * w**2)
    term3_denominator = (v_m / (np.pi * w) + r_w**2) * r_w
    term3 = (6 * Q_loss * mu) / (np.pi * w**3) * (term3_numerator / term3_denominator)

    return term1 * term2 + term3


def newton_method_Verga(w0, delta_P, Q_loss, mu, r_w, V_m, tol=1e-6, max_iter=100):
    """使用牛顿法求解 w"""
    w = w0
    for _ in range(max_iter):
        f_w = model_2(w, delta_P, Q_loss, mu, r_w, V_m)
        df_w = df_model_2(w, Q_loss, mu, r_w, V_m)

        if abs(df_w) < 1e-8:
            print("导数接近零，牛顿法可能失败。")
            return None

        w_new = w - f_w / df_w

        if abs(w_new - w) < tol:
            return w_new

        w = w_new

    print("牛顿法未能在最大迭代次数内收敛")
    return None


# 示例输入
delta_P = 10.0  # 例如 10 Pa
Q_loss = 0.01  # 例如 0.01 m³/s
mu = 0.001  # 例如 0.001 Pa·s (水的粘度)
r_w = 0.05  # 例如 0.05 m
V_m = 0.02  # 例如 0.02 m³
w0 = 0.01  # 初始猜测值

# 计算 w 的根
w_root = newton_method_Verga(w0, delta_P, Q_loss, mu, r_w, V_m)
print("求解出的 w =", w_root)
