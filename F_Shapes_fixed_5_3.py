import os
os.environ["PYCHARM_MATPLOTLIB_BACKEND"] = "off"
import matplotlib
matplotlib.use('TkAgg')  # 用系统原生画图，不使用PyCharm插件
import warnings
warnings.filterwarnings('default')  # 保持默认告警行为
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
# 如果您仍然看到与 matplotlib 数学文本相关的 invalid escape SyntaxWarning，
# 可取消下一行注释进行屏蔽（不推荐长期使用）：
# warnings.filterwarnings('ignore', category=SyntaxWarning, message=r'.*invalid escape sequence.*')

# -*- coding: utf-8 -*-
"""
F_Shapes.py
=================
将 Jupyter Notebook `F_Shapes.ipynb` 转换为可直接运行的 Python 脚本。

使用说明（Usage）
-----------------
1) 直接运行（复现 Notebook 的顺序执行逻辑）：
   python F_Shapes.py

2) 如需在交互式 IDE 中按单元调试（VS Code / PyCharm / Spyder 支持）：
   本脚本使用了 `# %%` 单元标记，可在 IDE 中逐段运行。

依赖（Dependencies）
--------------------
- 推荐使用与 Notebook 相同的 Python 环境。
- 脚本中 `import` 的库即为所需依赖；若缺失请先通过 pip/conda 安装。

输入 / 输出（Inputs / Outputs）
-------------------------------
- 本脚本是对 Notebook 的线性等价转换：
  * 若原 Notebook 中读取/写出文件（如 .npy/.mat/.csv/.png 等），本脚本将保持相同行为；
  * 如需修改输入/输出路径，请在相应代码段中调整变量或函数参数。
- 下方“自动检测的 I/O 提示”给出基于代码静态扫描的**可能**输入/输出位置（仅供参考）。

注意（Notes）
-------------
- 已自动注释/删除 Jupyter 魔法命令（如 %matplotlib inline 等）。
- 如 Notebook 中包含交互小部件/内联显示，本脚本将尽量兼容，但在纯终端下表现可能不同。
"""
# 必要依赖导入（由脚本自动添加）
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- 中文字体与负号显示设置（尽力匹配系统已安装字体） ----
try:
    import matplotlib as _mpl
    # 优先匹配常见中文字体；若系统无该字体，Matplotlib 会静默回退到默认字体
    _mpl.rcParams['font.sans-serif'] = [
        'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', 'Microsoft YaHei',
        'SimHei', 'WenQuanYi Zen Hei', 'Source Han Sans SC', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    _mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
except Exception:
    pass
# --------------------------------------------------

# ====== 全局数值稳定性与积分辅助函数（自动添加） ======
import numpy as _np
# 全局容差，避免 1/0 或 k**2 等分母为零导致的 NaN/Inf；根据需要可调小/调大
EPS = 1e-12

# 尽量减少不必要的 RuntimeWarning，但保留溢出（over）告警
_np.seterr(divide='ignore', invalid='ignore', over='warn', under='ignore')

def nan_to_num_strict(a):#将数组中的 NaN、+inf、-inf 替换为 0.0，防止积分时出错
    return _np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def trapz_safe(y, x=None, axis=-1):#梯形积分函数
    y = nan_to_num_strict(y)
    if x is None:
        if hasattr(_np, "trapezoid"):
            return _np.trapezoid(y, axis=axis)
        else:
            return _np.trapz(y, axis=axis)
    else:
        x = _np.asarray(x)
        if hasattr(_np, "trapezoid"):
            return _np.trapezoid(y, x, axis=axis)
        else:
            return _np.trapz(y, x, axis=axis)

# Pillow（可选），用于读取位图；若不可用则置为 None 并在后续逻辑中跳过
try:
    from PIL import Image as _PIL_Image
except Exception:
    _PIL_Image = None
# =======================================================



# %% [代码单元 1]
def current_max_sc(x, d, L):#输入：x 空间坐标（数组），d 间隙半宽？实际为馈电区长度的一半，L 臂长。输出：归一化电流分布，两端有增强（最大超导电流模型）。

    inv_F = np.sqrt(np.pi / 2.) / L * \
    ((d + L - x) * np.sign(d + L - x) \
     + (d - x) * np.sign(-d + x) \
     - (d + x) * np.sign(d + x) \
     + (d + L + x) * np.sign(d + L + x))
    
    return inv_F / np.nanmax(inv_F)

def current_no_gap(x, d, L):#三角分布 (1 - |x|/L)，在 |x| <= L 内线性下降，外部为零。参数 d 未使用
    
    inv_F = (1 - np.abs(x) / L) * np.heaviside(L - x, 0) * np.heaviside(L + x, 0)
    
    return inv_F

def current_AWAS(x, d, L, f, SC_current = True):#AWAS 解法的电流分布，包含馈电区斜坡 (inv_d) 和外部线性下降 (inv_F)。
    
    inv_d = ((np.abs(x) / d) * f + (1 - f)) * np.heaviside(d - x, 0) * np.heaviside(d + x, 0)
    
    inv_F = (1 - (np.abs(x) - d) / L) * np.heaviside(L + d - x, 0) * np.heaviside(L + d + x, 0)
    
    inv_F[(x < d) & (x > -d)] = 0
    
    if SC_current:
        return inv_F + inv_d
    else:
        return inv_F


d = 1.5
L = 2.
f = 1./3.
x = np.linspace(-L-d, L+d, 500)


mpl.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(2, 2, figsize = (12, 11), sharex = 'col', sharey = False, \
                       gridspec_kw=dict(hspace=0.25, wspace=0.25))


for axes in ax.flatten():
    axes.grid()
    axes.set_xlabel('x [m]')

ax[0,0].set_ylabel('j(x)')
ax[1,0].set_ylabel('j(x)')
ax[1,1].set_ylabel('j(x)')
ax[0,0].set_title('no SC current')
ax[0,1].set_title('max SC current')
ax[1,0].set_title('AWAS solution (PSP)')
ax[1,1].set_title('AWAS solution (SolO)')

ax[0,0].plot(x, current_AWAS(x, d, L, f, SC_current = False))
ax[1,0].plot(x, current_AWAS(x, d, L, f))
d = 1.979
L = 6.5
f = 1./25
x = np.linspace(-L-d, L+d, 500)
ax[0,1].plot(x, current_max_sc(x, d, L))
ax[1,1].plot(x, current_AWAS(x, d, L, f=f))

#plt.savefig('Figures/Currents_Merged.png', bbox_inches='tight')

# %% [代码单元 2]
# （空代码单元）

# %% [代码单元 3]
def BM_Sine_Low(x): #计算 |x|<=4  时的正弦积分 Si(x) = ∫₀ˣ sin(t)/t dt。使用 Boersma 提出的有理逼近公式
        C = [-4.54393409816329991e-2, 1.15457225751016682e-3, -1.41018536821330254e-5, 9.43280809438713025e-8, -3.53201978997168357e-10, 7.08240282274875911e-13, -6.05338212010422477e-16, 1.01162145739225565e-2, 4.99175116169755106e-5, 1.55654986308745614e-7, 3.28067571055789734e-10, 4.5049097575386581e-13, 3.21107051193712168e-16]
        a  = np.power(x,2.0)     


        Si = x*(1.0 + a*(C[0] + a*(C[1] + a*(C[2] + a*(C[3] + a*(C[4] + a*(C[5] + a*(C[6])))))))) / (1.0 + a*(C[7] + a*(C[8] + a*(C[9] + a*(C[10] + a*(C[11] + a*(C[12])))))))
    

        return Si   

def BM_Sine_High(x):#同上，计算|x|>4 时的正弦积分 Si(x) = ∫₀ˣ sin(t)/t dt。
        
        a = [7.44437068161936700618e2, 1.96396372895146869801e5, 2.37750310125431834034e7, 1.43073403821274636888e9, 4.33736238870432522765e10, 6.40533830574022022911e11, 4.20968180571076940208e12, 1.00795182980368574617e13, 4.94816688199951963482e12, -4.94701168645415959931e11, 7.46437068161927678031e2, 1.97865247031583951450e5, 2.41535670165126845144e7, 1.47478952192985464958e9, 4.58595115847765779830e10, 7.08501308149515401563e11, 5.06084464593475076774e12, 1.43468549171581016479e13, 1.11535493509914254097e13]


        b = [8.1359520115168615e2, 2.35239181626478200e5, 3.12557570795778731e7, 2.06297595146763354e9, 6.83052205423625007e10, 1.09049528450362786e12, 7.57664583257834349e12, 1.81004487464664575e13, 6.43291613143049485e12, -1.36517137670871689e12, 8.19595201151451564e2, 2.40036752835578777e5, 3.26026661647090822e7, 2.23355543278099360e9, 7.87465017341829930e10, 1.39866710696414565e12, 1.17164723371736605e13, 4.01839087307656620e13, 3.99653257887490811e13]
        y  =  np.power(x,-2.0)

    
        f = (1.0 +  y*(a[0] + y*(a[1] + y*(a[2] + y*(a[3] + y*(a[4] +y*(a[5] +y*(a[6] +y*(a[7] +y*(a[8] + y*(a[9])))))))))))/ (x*(1.0 + y*(a[10] + y*(a[11] + y*(a[12] + y*(a[13] + y*(a[14] + y*(a[15] + y*(a[16] + y*(a[17] + y*(a[18])))))))))))

        g = (y*(1.0 +  y*(b[0] + y*(b[1] + y*(b[2] + y*(b[3] + y*(b[4] +y*(b[5] +y*(b[6] +y*(b[7] +y*(b[8] + y*(b[9]))))))))))))/ (1.0 + y*(b[10] + y*(b[11] + y*(b[12] + y*(b[13] + y*(b[14] + y*(b[15] + y*(b[16] + y*(b[17] + y*(b[18]))))))))))

        Si = np.pi/2.0 - f*np.cos(x) - g*np.sin(x)
        
        return Si

def BM_Sine_Integral_arr(x):#对数组 x 每个元素自动选择低阶或高阶逼近，返回 Si(x)。
      
    result = np.where(x<=4,BM_Sine_Low(x),BM_Sine_High(x))
    
    return result

def BM_AntennaF_arr(x): #计算无间隙连续偶极子的响应函数 F(x)，其中 x = k·L_ant

    result = (1.0/x)*(BM_Sine_Integral_arr(x) - 0.5*BM_Sine_Integral_arr(2.0*x)-(2.0/x)*(np.power(np.sin(x/2),4.0)))

    return result

# %% [代码单元 4]
def F_angled(x, phi):
    
    arr = np.linspace(-1, 1, 1000)
    
    x = x * arr
    
    jk = (2 * np.sin(x/2)**2) / x + 1j * (np.sin(x) - x ) / x + \
    (2. * np.sin((x / (2.*np.cos(phi))))**2. / (x / np.cos(phi)) - \
    1j * ((np.sin(x / np.cos(phi)) - (x / np.cos(phi)) ) / (x / np.cos(phi))) ) * \
    (np.cos(phi) + 1j * np.sin(phi))
    
    F = trapz_safe(np.abs(jk)**2., x = arr) / 16.
    
    return F

# %% [代码单元 5]
from scipy import integrate

def F_angled(x, Psi):#通过 scipy.integrate.nquad 在球面上数值积分，得到精确的响应 F(x)
    def func(theta, phi, x, Psi):
        
        xx = x* np.sin(theta) * np.cos(phi)
        xa = x* np.sin(theta) * np.cos(phi - Psi)
        
        kj = 2. * np.sin(xx/2.)**2. / xx + 1j * (np.sin(xx) / xx - 1.) \
            + 2. * np.sin(xa/2.)**2. / xa - 1j * (np.sin(xa) / xa - 1.)
        
        integrand = np.abs(kj)**2. * np.sin(theta) / (32. * np.pi)
        
        return integrand

    return integrate.nquad(func, [[0, np.pi], [0, 2. * np.pi]], args=(x, Psi,))[0]

# %% [代码单元 6]
F_angled(1.1, 4. * np.pi / 36.)

# %% [代码单元 7]
x_arr = np.linspace(-0.5, 10, 70)
phi_arr = np.linspace(0.1, 0.9, 9)
F_ang = np.full(x_arr.size, np.nan)

for phi in phi_arr:
    label = str(phi)
    phi = phi * np.pi
    
    for x in np.arange(x_arr.size):
        F_ang[x] = F_angled(x_arr[x], phi)
    
    plt.plot(x_arr, F_ang, label = "%.1f" % round(phi / np.pi, 1))
    low_arr = 1/96. * x_arr**2 * (3 + 1/np.cos(phi)**2.)
    high_arr = (1 - np.cos(phi)) / 4.
    
plt.plot(x_arr, BM_AntennaF_arr(x_arr), label  = '0')
plt.text(1.75, 0.03, r'$\psi / \pi = $')
#4.2, 0.025
#plt.legend(ncol=3)
plt.xlabel('x'); plt.ylabel('F (x)')
plt.xlim(0, 10); plt.ylim(0, None)
#plt.savefig('Figures/Responses/F_angled_Dipole_NoGap.png', bbox_inches='tight') 

# %% [代码单元 8]
# （空代码单元）

# %% [代码单元 9]

#这些函数输入波数 k、臂长 L、间隙参数 d（馈电区半长），输出响应 F。
def Si(x):
    
    return x * BM_Sine_Integral_arr(x)

def F_gapped_NMV(k, L, d):

    F = 1 / (2. * k**2 * L**2) * \
    (Si(k*(L + 2*d)) + Si(k*L) - (Si(2*k*(L + d))/2) - (Si(2*k*d)/2) - ((np.cos(k*(L + d)) - np.cos(k*d))**2) )
    
    return F

def F_gapped(k, L, d):

    F = - 1 / (8 * d * k**2 * L**2) * \
    (2. * d * (2. - k**2 * L**2 + 2. * np.pi * k * (d + L) + \
    np.cos(2.*d*k) - 2. * np.cos(k*L) + np.cos(2.*k*(d + L)) - 2. * np.cos(k*(2.*d + L))) + k * L**2 * np.sin(2*d*k) + \
    4. * d * k * (d + L) * (BM_Sine_Integral_arr(2*d*k) + BM_Sine_Integral_arr(2*k*(d + L)) - \
          2. * BM_Sine_Integral_arr(k*(2*d + L)) - np.pi))
    
    return F

def F_sphere(k, L, d):

    F = (1 - np.sin(k*L) / (k*L)) / 4# * np.sin(k*d)**2 / (k**2 * d**2)
    
    return F

def F_AWAS(k, L, d, f):#AWAS 电流分布，对应的响应，通过对辅助变量 u 一维积分得到。
    
    #print(k, L, d, f)
    
    u = np.linspace(-1, 1, 500)
    
    jk2 = 0.25 * (f * L - (d + f * L) * np.cos(d*k*u) + d * np.cos(k*u*(d+L)))**2. / (d**2. * L**2. * k**2. * u**2.)

    F = trapz_safe(jk2, x = u)
    
    return F

# %% [代码单元 10]
k = np.linspace(0, 10/2., 500)
L = 2
#L = 6.5
d = 2.975 / 2.
#d = 2.2
f = 1./3.

F_A = np.zeros(k.size)
for x in np.arange(k.size):
    F_A[x] = F_AWAS(k[x], L, d, f)

plt.plot(k*L, F_A, label = 'AWAS solution')
plt.plot(k*L, F_sphere(k, L, d), label = 'double sphere')
plt.plot(k*L, F_gapped(k, L, d), label = 'no SC current')
plt.plot(k*L, F_gapped_NMV(k, L, d), label = 'max SC current')
plt.plot(k*L, BM_AntennaF_arr(k * L), label = 'no gap')
#plt.legend()

plt.xlabel(r'$\mathrm{kL_{ant}}$'); plt.ylabel('F')
plt.title(r'$\mathrm{L_{ant} = }$' + "%.1f" % L + ' m' + ', d = ' + "%.2f" % d + ' m' + ', f = ' + "%.2f" % f)
#plt.savefig('Figures/Responses/F_linear_Dipole_Gap.png', bbox_inches='tight') 

# %% [代码单元 11]
# （空代码单元）

# %% [代码单元 12]
def distFunc_int(theta, phi, psi = 0, f = 1/3., L_ant = 6.5, d = 2.623, x = 1.):#构建球坐标下的被积函数 |jk(θ,φ)|² sinθ
    
    k = x / L_ant
    
    kx = k * np.sin(theta) * np.cos(phi)
    ka = k * np.sin(theta) * np.cos(phi - psi)

    jk = (np.exp(-1j*kx*(d + L_ant)) * (-1 + np.exp(1j*kx*L_ant) * (1 - 1j*kx*L_ant)))/(kx*L_ant) + (np.exp(-1j*d*kx)*(f + 1j*d*kx + 1j*np.exp(1j*d*kx)*(1j*f + d*(-1 + f)*kx)))/(d*kx) + \
    (np.exp(1j*d*ka) * (1 - np.exp(1j*ka*L_ant) + 1j*ka*L_ant)) / (ka*L_ant) + (-f - 1j*d*(-1 + f)*ka + np.exp(1j*d*ka) * (f - 1j*d*ka))/(d*ka)

    jk2 = np.abs(jk)**2.
    
    return jk2 * np.sin(theta)

def F_AWAS_SolO_case(x, psi = 0, f = 1/3., L_ant = 6.5, d = 2.623):#对单个 x 值，在 θ 和 φ 网格上双重梯形积分，得到 F(x)。

    theta_arr = np.linspace(0.001, np.pi, 200)
    phi_arr = np.linspace(0.001, 2.*np.pi, 200)
    X, Y = np.meshgrid(theta_arr, phi_arr)

    list1=distFunc_int(X, Y, psi = psi, f = f, L_ant = L_ant, d = d, x = x)
    int_exp_2d = trapz_safe(trapz_safe(list1, theta_arr, axis=0), phi_arr, axis=0)
    
    return 1 / (32. * np.pi) * int_exp_2d 

def F_SolO_AWAS(x, psi = 0, f = 1/3., L_ant = 6.5, d = 2.623):#向量化版本，若 x 是标量则调用单点函数；若为数组则循环计算。
    
    if np.isscalar(x):
        return F_AWAS_SolO_case(x, psi = psi, f = f, L_ant = L_ant, d = d)
    
    F_A = np.zeros(x.size)
    for x_0 in np.arange(x.size):
        F_A[x_0] = F_AWAS_SolO_case(x[x_0], psi = psi, f = f, L_ant = L_ant, d = d)
            
    return F_A



# %% [代码单元 13]
# %% 新图1：张角响应函数 + 参考模型曲线

import os
os.makedirs("Figures", exist_ok=True)

mpl.rcParams.update({'font.size': 18})

# -----------------------------
# 1. 基本参数：保持原图风格
# -----------------------------
nop = 120
x_arr = np.linspace(0., 10, nop) + 0.001
psi_arr = np.linspace(0.1, 0.9, 9)

L_ant = 6.5
d = 1.979
f = 1. / 25.

fig, ax = plt.subplots(figsize=(7.0, 5.3))

# -----------------------------
# 2. 不同张角下的 V 形天线响应函数
# -----------------------------
colors = plt.cm.viridis(np.linspace(0.2, 0.8, 9))

for i, psi in enumerate(psi_arr):
    psi_rad = psi * np.pi

    F_AWAS_ang = F_SolO_AWAS(
        x_arr,
        psi=psi_rad,
        f=f,
        L_ant=L_ant,
        d=d
    )

    # 重点突出 0.1, 0.5, 0.9，其他曲线稍细
    if i in [0, 4, 8]:
        ax.plot(
            x_arr,
            F_AWAS_ang,
            color=colors[i],
            linewidth=2.2,
            label=f'{psi:.1f}'
        )
    else:
        ax.plot(
            x_arr,
            F_AWAS_ang,
            color=colors[i],
            linewidth=1.4,
            alpha=0.75,
            label=f'{psi:.1f}'
        )

# -----------------------------
# 3. 加入原来那些参考模型曲线
# -----------------------------
k = x_arr / L_ant

F_awas = np.zeros(k.size)
for i in np.arange(k.size):
    F_awas[i] = F_AWAS(k[i], L_ant, d, f)

F_no_sc = F_gapped(k, L_ant, d)
F_max_sc = F_gapped_NMV(k, L_ant, d)
F_no_gap = BM_AntennaF_arr(x_arr)

ax.plot(
    x_arr,
    F_awas,
    color='black',
    linewidth=2.3,
    label='AWAS solution'
)

ax.plot(
    x_arr,
    F_no_sc,
    color='red',
    linestyle='--',
    linewidth=1.8,
    label='no SC current'
)

ax.plot(
    x_arr,
    F_max_sc,
    color='purple',
    linestyle='-.',
    linewidth=1.8,
    label='max SC current'
)

ax.plot(
    x_arr,
    F_no_gap,
    color='blue',
    linestyle=':',
    linewidth=2.0,
    label='no gap'
)


# -----------------------------
# 4. 图像格式
# -----------------------------
ax.set_xlabel(r'$x=kL_{\mathrm{ant}}$')
ax.set_ylabel('F (x)')
ax.set_xlim(0, 10)
ax.set_ylim(0, None)
ax.grid(True)

ax.set_title(
    r'Response function versus $x=kL_{\mathrm{ant}}$'
)

ax.legend(
    title=r'$\psi/\pi$',
    loc='upper right',
    frameon=True,
    ncol=2,
    fontsize=9,
    title_fontsize=13
)

fig.tight_layout()
fig.savefig(
    "Figures/Figure_1_angle_with_reference_models.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# %% [代码单元 14]
# （空代码单元）

# %% 图2：三个中心区电流模型横向并列，尽量保持原图参数风格

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.makedirs("Figures", exist_ok=True)
mpl.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)

# =============================
# 统一采用 SolO 参数
# =============================
L = 6.5            # 单臂长度 (m)
d = 1.979          # 中心区半宽 / gap 参数 (m)

# 横坐标范围统一
x = np.linspace(-(L + d), L + d, 500)

# =============================
# (a) no SC current
# 对应 f = 0
# 说明：在 current_AWAS(..., SC_current=False) 中，
# 中心区电流直接被置零，因此 f 实际不参与结果；
# 这里写 f_no = 0 只是为了物理含义清楚。
# =============================
f_no = 0.0
j_no_sc = current_AWAS(x, d, L, f_no, SC_current=False)

ax[0].plot(x, j_no_sc, linewidth=2)
ax[0].set_title('(a) no SC current')
ax[0].set_xlabel('x [m]')
ax[0].set_ylabel(r'j(x) / $I_a$')
ax[0].grid(True)
ax[0].set_xlim(-(L + d), L + d)

# =============================
# (b) max SC current
# 对应 f = 1
# 说明：current_max_sc() 本身不含 f 参数，
# 这里写 f_max = 1 只是为了表明其物理含义。
# =============================
f_max = 1.0
j_max_sc = current_max_sc(x, d, L)

ax[1].plot(x, j_max_sc, linewidth=2)
ax[1].set_title('(b) max SC current')
ax[1].set_xlabel('x [m]')
ax[1].grid(True)
ax[1].set_xlim(-(L + d), L + d)

# =============================
# (c) AWAS solution (SolO)
# 对应 f = 0.04 = 1/25
# =============================
f_awas = 1.0 / 25.0
j_awas = current_AWAS(x, d, L, f_awas, SC_current=True)

ax[2].plot(x, j_awas, linewidth=2)
ax[2].set_title('(c) AWAS solution (SolO)')
ax[2].set_xlabel('x [m]')
ax[2].grid(True)
ax[2].set_xlim(-(L + d), L + d)

# 统一纵轴范围
for a in ax:
    a.set_ylim(-0.05, 1.08)

# 可选：在图上方统一写参数
fig.suptitle(
    r'$L_{\mathrm{ant}} = 6.5\,\mathrm{m},\quad d = 1.979\,\mathrm{m}$',
    fontsize=16,
    y=1.05
)

fig.tight_layout()
fig.savefig("Figures/Figure_4A_current_models_SolO.png",
            dpi=300, bbox_inches='tight')

plt.show()

'''
# %% [代码单元 15]
F_03 = F_SolO_AWAS(x_arr, psi = 0.3 * np.pi, f = 0.04, L_ant = L_ant, d = d)

# %% [代码单元 16]
fig, ax = plt.subplots()
ax.plot(x_arr, F_03, label = r'F (x); $\mathrm{\psi=0.3 \pi}$', color = 'green')
ax.plot(x_arr*1.07, F_A*0.97, label = r'0.97 $\cdot$ F (1.07x)', color = 'red')
ax.legend()
ax.set_title('max current F shift')
ax.set_xlabel('x'); ax.set_ylabel('F (x)')
''

# %% [代码单元 17]
# （空代码单元）

# %% [代码单元 18]
F_03 = F_SolO_AWAS(x_arr, psi = 0.3 * np.pi, f = 0.04, L_ant = L_ant, d = d)

# %% [代码单元 19]
fig, ax = plt.subplots()
ax.plot(x_arr, F_03, label = r'F (x); $\mathrm{\psi=0.3 \pi}$', color = 'green')
#ax.plot(x_arr*1.07, F_A*0.97, label = r'0.97 $\cdot$ F (1.07x)', color = 'red')
ax.legend()
ax.set_xlabel('x'); ax.set_ylabel('F (x)')
ax.plot(x_arr, F_03)
ax.plot(x_arr*1.1, F_A*0.97, label = r'0.97 $\cdot$ F (1.1x)', color = 'red')
ax.legend()
ax.set_title('F(x) shift')
'''

# %% [代码单元 20]
# （空代码单元）

# %% [代码单元 21 - 优化版]
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 增大宽度，减小高度，使比例更协调 (宽, 高)
fig, ax = plt.subplots(2, 1, figsize=(10, 10),
                       gridspec_kw={'height_ratios': [1,  1], 'hspace': 0.3})

# # ---- 第1个子图：PSP 示意图（保持原有逻辑） ----
# # 优先在当前目录或脚本同目录寻找 PSP_v2.png
# from pathlib import Path as _Path
# _img_candidates = [
#     _Path("PSP_v2.png"),
#     _Path(__file__).with_name("PSP_v2.png") if '__file__' in dir() else _Path("PSP_v2.png"),
#     _Path("assets") / "PSP_v2.png",
# ]
# _img_path = None
# for _p in _img_candidates:
#     if _p.exists():
#         _img_path = str(_p)
#         break
#
# img = None
# if _img_path is not None:
#     try:
#         img = plt.imread(_img_path)
#     except Exception as _e:
#         print("[WARN] 无法读取图片：", _img_path)
#
# ax[0].axis('off')
# if img is not None:
#     ax[0].imshow(img, aspect='auto')
# else:
#     ax[0].text(0.5, 0.5, "PSP_v2.png 未找到", ha="center", va="center", transform=ax[0].transAxes)
#
# # 图例移到左下角，避免遮挡示意图
# legend_elements = [
#     plt.Line2D([0], [0], marker='o', color='w', label='nodes', markerfacecolor='red', markersize=10),
#     plt.Line2D([0], [0], marker='o', color='w', label='ports', markerfacecolor='g', markersize=10),
#     plt.Line2D([0], [0], color='b', lw=3, label='segments')
# ]
# ax[0].legend(handles=legend_elements, loc='lower left', framealpha=0.8, fontsize=10)


# ---- 第2个子图：电流分布 j(x) ----
d = 1.5
L = 2.
f = 1./3.
x = np.linspace(-L-d, L+d, 500)
x_ex = np.linspace(-d, d, 500)

ax[0].plot(x, current_AWAS(x, d, L, f), color='black', label='AWAS solution', linewidth=2)
ax[0].plot(x_ex, x_ex * 0 + 1, '--', color='magenta', label='max SC current', linewidth=1.5)
ax[0].plot(x_ex, x_ex * 0, '--', color='grey', label='no SC current', linewidth=1.5)

ax[0].grid(True, alpha=0.3)
ax[0].set_xlabel('x [m]', fontsize=12)
ax[0].set_ylabel(r'j (x) / $I_a$', fontsize=12)

# 图例移到右下角，避让中间的箭头标注
ax[0].legend(loc='lower right', fontsize=10, framealpha=0.9)

# 保留原有的箭头标注
ax[0].annotate(text='', xy=(0, 1), xytext=(0, 2/3),
               arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax[0].annotate(r'$\rm{\zeta}$ = 0.33', xy=(0, 0.59), color='blue',
               xytext=(0, 0.59), va="bottom", ha="center", fontsize=11)


# ---- 第3个子图：F(x) 响应曲线（重点优化图例） ----
k = np.linspace(0, 10/2., 500)
L = 2
d = 2.975 / 2.
f = 1./3.

F_A = np.zeros(k.size)
for x_idx in np.arange(k.size):
    F_A[x_idx] = F_AWAS(k[x_idx], L, d, f)

# 使用不同线型和颜色，增加可区分度
ax[1].plot(k*L, F_A, color='black', label='AWAS solution', linewidth=2)
ax[1].plot(k*L, F_sphere(k, L, d), label='double sphere', color='orange', linestyle='--', linewidth=1.5)
ax[1].plot(k*L, F_gapped(k, L, d), color='grey', label='no SC current', linestyle='-.', linewidth=1.5)
ax[1].plot(k*L, F_gapped_NMV(k, L, d), color='magenta', label='max SC current', linestyle=':', linewidth=2)
ax[1].plot(k*L, BM_AntennaF_arr(k * L), label='no gap', color='blue', linewidth=1.5)

ax[1].set_xlabel(r'$\mathrm{kL_{ant}}$', fontsize=12)
ax[1].set_ylabel('F', fontsize=12)
ax[1].grid(True, alpha=0.3)

# 关键优化：图例放在图右侧外部，并分成2列
ax[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
             ncol=1, fontsize=10, frameon=True, fancybox=True)

# 自动调整布局，确保右侧图例不被截断
plt.tight_layout()

# 保存（如需）
# fig.savefig(r"C:\Users\ZXM\Desktop\新建文件夹\AWAS_05_clean.png", bbox_inches='tight', dpi=150)
plt.show()
# %% [代码单元 22]
# （空代码单元）

# %% [代码单元 23]
# （空代码单元）

# %% [代码单元 24]
BM_Sine_Integral_arr(x)

def g_uneq(x_):
    
    return (np.cos(x_) - 1.) / x_ + BM_Sine_Integral_arr(x_)

def F_uneq(k_, L1_, L2_):
    
    return 1. / (4.*k_) * (L1_ + L2_) / (L1_ * L2_) * ( g_uneq(k_*L1_) + g_uneq(k_*L2_) - g_uneq(k_*(L1_+L2_)) )

# %% [代码单元 25]
L1 = 50
L2 = 30
L2_arr = np.arange(0.3, 1, 0.1) * L1

k_arr = np.linspace(-0.5, 20, 70) / L1
F_ang = np.full(k_arr.size, np.nan)

for L2 in L2_arr:
    plt.plot(k_arr*L1, F_uneq(k_arr, L1, L2), label = '$L_2$ = ' + str(L2 / L1) + ' $L_1$')
#4.2, 0.025
plt.legend(ncol=2)
plt.xlabel('x'); plt.ylabel('F (x)')
plt.xlim(0, 20); #plt.ylim(0, None)

plt.plot(k_arr*L1, BM_AntennaF_arr(k_arr*L1), label  = 'equal')
#plt.savefig('Figures/Responses/F_angled_Dipole_NoGap.png', bbox_inches='tight') 

# %% [代码单元 26]
# （空代码单元）

# %% [代码单元 27]
psi = 12*np.pi/36;
f = 1/3;
k = 0.2;
L_ant = 6.5;
d = 4.12;
x = k*L_ant;
theta = 2. * np.pi/7;
phi = np.pi/3;

kx = k * np.sin(theta) * np.cos(phi)
ka = k * np.sin(theta) * np.cos(phi - psi)

jk = (np.exp(-1j*kx*(d + L_ant)) * (-1 + np.exp(1j*kx*L_ant) * (1 - 1j*kx*L_ant)))/(kx*L_ant) + (np.exp(-1j*d*kx)*(f + 1j*d*kx + 1j*np.exp(1j*d*kx)*(1j*f + d*(-1 + f)*kx)))/(d*kx) + \
(np.exp(1j*d*ka) * (1 - np.exp(1j*ka*L_ant) + 1j*ka*L_ant)) / (ka*L_ant) + (-f - 1j*d*(-1 + f)*ka + np.exp(1j*d*ka) * (f - 1j*d*ka))/(d*ka)

jk2 = np.abs(jk)**2.

jk

x_test = np.logspace(-2, 0, 20)  # 0.01 到 1
F_test = [F_SolO_AWAS(x, psi=np.pi/2, f=0.04, L_ant=1.0, d=0.3) for x in x_test]

plt.loglog(x_test, F_test, 'o', label='Numerical')
plt.loglog(x_test, x_test**2/48, '--', label='Theory: $x^2/48$')
plt.xlabel('$kL$'); plt.ylabel('$F(k)$')
plt.legend(); plt.title('Short antenna limit verification')
plt.savefig('short_limit_verify.png')
#---------------------------------------------------------------------------------------------------

L_ant = 6.5
d = 1.979
f = 1. / 25.
psi_list = [0.1, 0.3, 0.5, 0.7]



def find_peak_for_psi(psi_val):
    psi_rad = psi_val * np.pi

    # 目标函数：因为 minimize_scalar 是求极小值，所以我们在 F 前面加负号来求极大值
    # 限制搜索区间 bounds 为 x 轴的 [1.0, 7.0]，避免受两端边界或噪点干扰
    objective = lambda x: -F_SolO_AWAS(x, psi=psi_rad, f=f, L_ant=L_ant, d=d)

    res = minimize_scalar(objective, bounds=(1.0, 7.0), method='bounded')

    if res.success:
        return res.x, -res.fun
    else:
        return None, None


# 循环计算并打印输出
print("==================================================")
print("  张角 ψ/π      |   峰值坐标 x     |   峰值大小 Peak F(x)")
print("==================================================")
for psi in psi_list:
    x_opt, f_max = find_peak_for_psi(psi)
    if x_opt is not None:
        print(f"  ψ/π = {psi:.1f}       |   x = {x_opt:.4f}      |   F(x) = {f_max:.4f}")
    else:
        print(f"  ψ/π = {psi:.1f}       |   计算失败")
print("==================================================")

# ==================================================
# 固定张角 ψ = 125°，扫描天线长度 L_ant 对响应函数的影响
# 电流模型：AWAS/SolO (F_SolO_AWAS)
# ==================================================


# 固定参数
psi_deg = 125.0  # 张角 125°
psi_rad = psi_deg * np.pi / 180.0
f_param = 0.04  # AWAS 下凹参数
d = 1.979  # 中心间隙半宽 (m)

# 待扫描的天线长度列表 (m) — 可自行增减
L_list = [3.25, 4.5, 6.5, 8.0, 10.0]

# ==================================================
# Figure 5a 修改版：
# 固定 ψ=125°，扫描 L_ant
# 横坐标改为真实波数 k
# ==================================================

psi_deg = 125.0
psi_rad = psi_deg * np.pi / 180.0

f_param = 0.04
d = 1.979


# 天线长度列表
L_list = [3.25, 4.5, 6.5, 8.0, 10.0]


# 真实波数范围
# 单位 m^-1
k_arr = np.linspace(0.001, 2.0, 200)


results = []

plt.figure(figsize=(8,5))


for L in L_list:

    # 注意：
    # 函数内部需要 x=kL
    x_arr = k_arr * L


    F_arr = F_SolO_AWAS(
        x_arr,
        psi=psi_rad,
        f=f_param,
        L_ant=L,
        d=d
    )


    idx_peak = np.argmax(F_arr)

    k_peak = k_arr[idx_peak]
    F_max = F_arr[idx_peak]

    results.append(
        (L, k_peak, F_max)
    )


    plt.plot(
        k_arr,
        F_arr,
        linewidth=1.8,
        label=fr'$L_{{ant}}={L:.1f}\ \mathrm{{m}}$'
    )


plt.xlabel(
    r'$k\ (\mathrm{m}^{-1})$',
    fontsize=14
)

plt.ylabel(
    r'$F_{\psi}(k)$',
    fontsize=14
)

plt.title(
    f'V-dipole response versus physical wave number '
    f'($\\psi={psi_deg:.0f}^\\circ$, AWAS)'
)

plt.legend(fontsize=10)

plt.grid(True,alpha=0.3)

plt.legend(fontsize=10)

plt.tight_layout()


plt.savefig(
    'Response_vs_L_ant_k_axis.png',
    dpi=300
)

plt.show()



print("\n天线长度扫描结果（横坐标为真实波数 k）")
print(" L_ant(m) | k_peak(m^-1) | F_max")
print("--------------------------------")

for L,kp,Fm in results:
    print(
        f" {L:6.2f} | {kp:10.4f} | {Fm:7.4f}"
    )

# ==================================================
# 固定张角 ψ=125°
# 扫描中心间隙长度 d 对响应函数的影响
# 横坐标改为真实波数 k
# ==================================================

psi_deg = 125.0
psi_rad = psi_deg * np.pi / 180.0

L_ant = 6.5
f_param = 0.04


# 扫描中心间隙长度
d_list = [0.5, 1.0, 1.5, 1.98, 2.5, 3.0]


# 真实波数范围
k_arr = np.linspace(0.001, 2.0, 200)


results = []

plt.figure(figsize=(8,5))


for d_val in d_list:

    # 内部仍然使用 x=kL
    x_arr = k_arr * L_ant


    F_arr = F_SolO_AWAS(
        x_arr,
        psi=psi_rad,
        f=f_param,
        L_ant=L_ant,
        d=d_val
    )


    # 找峰值
    idx_peak = np.argmax(F_arr)

    k_peak = k_arr[idx_peak]
    F_max = F_arr[idx_peak]

    results.append(
        (d_val, k_peak, F_max)
    )


    plt.plot(
        k_arr,
        F_arr,
        linewidth=1.8,
        label=f'd = {d_val:.2f} m'
    )


plt.xlabel(
    r'$k\ (\mathrm{m}^{-1})$',
    fontsize=14
)

plt.ylabel(
    r'$F_{\psi}(k)$',
    fontsize=14
)


plt.title(
    r'V-dipole response function '
    r'($\psi=125^\circ$, $L_{\mathrm{ant}}=6.5m$, $f=0.04$)'
    '\nfor different gap length $d$'
)


plt.grid(True, alpha=0.3)

plt.legend(fontsize=9)

plt.tight_layout()


plt.savefig(
    'Response_vs_gap_k_axis.png',
    dpi=300,
    bbox_inches='tight'
)


plt.show()



print("\n中心间隙长度扫描结果")
print("(横坐标为真实波数 k)")
print("--------------------------------")
print(" d(m) | k_peak(m^-1) | F_max")
print("--------------------------------")

for d_val,kp,Fm in results:
    print(
        f"{d_val:4.2f} | {kp:10.4f} | {Fm:7.4f}"
    )
# %%
if __name__ == "__main__":
    # 说明：当直接执行本脚本时，将按上方从上到下的顺序运行所有代码单元。
    # 若需要传入自定义参数，可在对应代码段中自行添加 argparse 解析逻辑。
    pass  # 大多数 Notebook 在导入时已执行主要逻辑；若需要主动调用函数，请在此处添加。
