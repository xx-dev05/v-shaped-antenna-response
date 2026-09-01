import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 14
})

try:
    from scipy.special import dawsn
except ImportError:
    raise ImportError(
        "This code needs scipy for the Maxwellian dielectric function. "
        "Please install it by running: pip install scipy"
    )

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13

OUTDIR = Path("vshape_step12_results")
OUTDIR.mkdir(exist_ok=True)

EPS = 1e-12
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def make_sphere_grid(n_theta=160, n_phi=220):
    theta = np.linspace(1e-4, np.pi - 1e-4, n_theta)
    phi = np.linspace(1e-4, 2.0 * np.pi - 1e-4, n_phi)

    T, P = np.meshgrid(theta, phi, indexing="ij")

    khat_x = np.sin(T) * np.cos(P)
    khat_y = np.sin(T) * np.sin(P)
    khat_z = np.cos(T)

    khat = np.stack([khat_x, khat_y, khat_z], axis=-1)
    weight = np.sin(T)

    return theta, phi, khat, weight


THETA_GRID, PHI_GRID, KHAT, WEIGHT = make_sphere_grid()


def sphere_integral(values):
    integrand = values * WEIGHT
    int_phi = trapz(integrand, PHI_GRID, axis=1)
    int_theta = trapz(int_phi, THETA_GRID, axis=0)
    return int_theta

def vshape_unit_vectors(theta_deg):
    half = np.deg2rad(theta_deg) / 2.0

    u1 = np.array([np.cos(half),  np.sin(half), 0.0])
    u2 = np.array([np.cos(half), -np.sin(half), 0.0])

    return u1, u2


def projected_arm_factor(z):
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape, dtype=complex)

    small = np.abs(z) < EPS
    large = ~small

    zz = z[large]

    out[large] = (
        2.0 * np.sin(zz / 2.0) ** 2 / zz
        + 1j * (np.sin(zz) / zz - 1.0)
    )

    zs = z[small]
    out[small] = zs / 2.0 + 1j * (-zs**2 / 6.0)

    return out

def F_vshape_no_gap_single(x, theta_deg):
    u1, u2 = vshape_unit_vectors(theta_deg)

    mu1 = np.tensordot(KHAT, u1, axes=([-1], [0]))
    mu2 = np.tensordot(KHAT, u2, axes=([-1], [0]))

    z1 = x * mu1
    z2 = x * mu2

    j1 = projected_arm_factor(z1)
    j2 = projected_arm_factor(z2)

    jk = j1 - j2

    F = sphere_integral(np.abs(jk) ** 2) / (32.0 * np.pi)

    return float(np.real(F))


def F_vshape_no_gap_curve(x_array, theta_deg):
    return np.array([F_vshape_no_gap_single(x, theta_deg) for x in x_array])


def gap_correction_smooth(x, d_over_L, x_c=5.0, p=2.0):
    a = d_over_L
    r_low = (1.0 + 2.0 * a) ** 2

    correction = 1.0 + (r_low - 1.0) * np.exp(- (x / x_c) ** p)

    return correction


def F_vshape_gap_curve(x_array, theta_deg, d_over_L, x_c=5.0, p=2.0):
    F0 = F_vshape_no_gap_curve(x_array, theta_deg)
    R_gap = gap_correction_smooth(x_array, d_over_L, x_c=x_c, p=p)

    return F0 * R_gap


def plot_step1_angle_effect():
    x = np.linspace(0.05, 30.0, 220)
    angles = [30, 60, 90, 120]

    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    for theta in angles:
        F = F_vshape_no_gap_curve(x, theta)
        ax.plot(x, F, linewidth=2.2, label=fr"$\theta={theta}^\circ$")

    ax.set_xlabel(r"$K^{*}L$", fontsize=16)
    ax.set_ylabel(r"$F$", fontsize=16)
    ax.set_title("Step 1: V-shaped antenna response for different angles", fontsize=14)

    ax.set_xlim(0, 30)
    ax.set_ylim(bottom=0)

    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)

    fig.tight_layout()

    save_path = OUTDIR / "step1_vshape_angles.png"
    fig.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")

def plot_step2_gap_effect():
    x = np.linspace(0.05, 30.0, 220)

    theta_deg = 125.0
    d_over_L_list = [0.00, 0.05, 0.10, 0.20,0.40]

    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    for d_over_L in d_over_L_list:
        F = F_vshape_gap_curve(
            x_array=x,
            theta_deg=theta_deg,
            d_over_L=d_over_L,
            x_c=5.0,
            p=2.0,
        )

        ax.plot(
            x,
            F,
            linewidth=2.2,
            label=fr"$d/L={d_over_L:.2f}$"
        )

    ax.set_xlabel(r"$K^{*}L$", fontsize=16)
    ax.set_ylabel(r"$F$", fontsize=16)
    ax.set_title(
        r"Step 2: gap effect for V-shaped antenna, $\theta=125^\circ$",
        fontsize=14,
    )

    ax.set_xlim(0, 30)
    ax.set_ylim(bottom=0)

    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)

    fig.tight_layout()

    save_path = OUTDIR / "step2_vshape_gap_smooth.png"
    fig.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")



def save_model_note():
    note = """
V-shaped antenna response calculation: Step 1, Step 2 and Step 3

Step 1:
    The no-gap V-shaped antenna response is calculated for opening angles
    30°, 60°, 90°, and 120°. The response is obtained from the differential
    contribution of the two antenna arms.

Step 2:
    The gap effect is treated as a smooth asymptotic correction to the no-gap
    V-shaped antenna response. This correction is based on the known behavior
    of a gapped linear dipole:
        - small K*L: gap increases the effective antenna length;
        - large K*L: gap effect becomes weak and approaches the no-gap case.

    This Step 2 result is an approximate trend model, not a complete
    spacecraft-body model.

Step 3:
    The current-decay effect is introduced by comparing the no-gap
    V-shaped response with finite-gap current-distribution models.

    The no-gap curve is calculated from the no-gap V-shaped antenna model.

    The finite-gap curves use:
        L_ant = 6.5 m
        d = 1.979 m

    Three current assumptions are compared:
        f = 0       : no SC current
        f = 1/25    : AWAS / Solar Orbiter current
        f = 1       : max SC current

    The Step 3 figure still shows the antenna response function F(K*L_ant),
    not the complete QTN spectrum.
"""
    save_path = OUTDIR / "model_note.txt"
    save_path.write_text(note.strip(), encoding="utf-8")
    print(f"Saved: {save_path}")


def safe_complex_divide(num, den):
    den_safe = np.where(np.abs(den) < EPS, EPS + 0j, den)
    return num / den_safe


def jk_vshape_current_decay(x, theta_deg=125.0, L_ant=6.5, d=0.0, f=1.0):
    k = x / L_ant

    u1, u2 = vshape_unit_vectors(theta_deg)

    k1 = k * np.tensordot(KHAT, u1, axes=([-1], [0]))
    k2 = k * np.tensordot(KHAT, u2, axes=([-1], [0]))

    if abs(d) < EPS:
        term1 = safe_complex_divide(
            np.exp(-1j * k1 * L_ant)
            * (-1.0 + np.exp(1j * k1 * L_ant) * (1.0 - 1j * k1 * L_ant)),
            k1 * L_ant,
        )

        term2 = safe_complex_divide(
            1.0 - np.exp(1j * k2 * L_ant) + 1j * k2 * L_ant,
            k2 * L_ant,
        )

        jk = term1 + term2
        return np.nan_to_num(jk, nan=0.0, posinf=0.0, neginf=0.0)

    term1 = safe_complex_divide(
        np.exp(-1j * k1 * (d + L_ant))
        * (-1.0 + np.exp(1j * k1 * L_ant) * (1.0 - 1j * k1 * L_ant)),
        k1 * L_ant,
    )

    term2 = safe_complex_divide(
        np.exp(-1j * d * k1)
        * (
            f
            + 1j * d * k1
            + 1j * np.exp(1j * d * k1)
            * (1j * f + d * (-1.0 + f) * k1)
        ),
        d * k1,
    )

    term3 = safe_complex_divide(
        np.exp(1j * d * k2)
        * (1.0 - np.exp(1j * k2 * L_ant) + 1j * k2 * L_ant),
        k2 * L_ant,
    )

    term4 = safe_complex_divide(
        -f
        - 1j * d * (-1.0 + f) * k2
        + np.exp(1j * d * k2) * (f - 1j * d * k2),
        d * k2,
    )

    jk = term1 + term2 + term3 + term4

    return np.nan_to_num(jk, nan=0.0, posinf=0.0, neginf=0.0)


def F_vshape_current_decay_single(
    x,
    theta_deg=125.0,
    L_ant=6.5,
    d=1.979,
    f=1.0 / 25.0,
):
    jk = jk_vshape_current_decay(
        x=x,
        theta_deg=theta_deg,
        L_ant=L_ant,
        d=d,
        f=f,
    )

    F = sphere_integral(np.abs(jk) ** 2) / (32.0 * np.pi)

    return float(np.real(F))


def F_vshape_current_decay_curve(
    x_array,
    theta_deg=125.0,
    L_ant=6.5,
    d=1.979,
    f=1.0 / 25.0,
):
    return np.array(
        [
            F_vshape_current_decay_single(
                x=x,
                theta_deg=theta_deg,
                L_ant=L_ant,
                d=d,
                f=f,
            )
            for x in x_array
        ]
    )


def plot_step3_current_decay_effect():
    """
    Step 3:
        Add current decay and compare from no-gap to finite-gap cases.

    This figure compares:
        1. no gap baseline
        2. finite gap + no SC current
        3. finite gap + AWAS / Solar Orbiter current
        4. finite gap + max SC current

    This is still antenna response F(K*L_ant), not the final QTN spectrum.
    """
    x = np.linspace(0.05, 30.0, 220)

    theta_deg = 125.0
    L_ant = 6.5

    # Solar Orbiter / AWAS gap scale
    d = 1.979

    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    # --------------------------------------------------------
    # 1. no gap baseline
    # --------------------------------------------------------
    F_no_gap = F_vshape_no_gap_curve(
        x_array=x,
        theta_deg=theta_deg,
    )

    ax.plot(
        x,
        F_no_gap,
        linewidth=2.4,
        color="black",
        label="no gap",
    )

    # --------------------------------------------------------
    # 2. finite gap + current decay models
    # --------------------------------------------------------
    current_cases = [
        ("no SC current  $f=0$", 0.0),
        ("AWAS / Solar Orbiter  $f=1/25$", 1.0 / 25.0),
        ("max SC current  $f=1$", 1.0),
    ]

    for label, f in current_cases:
        F = F_vshape_current_decay_curve(
            x_array=x,
            theta_deg=theta_deg,
            L_ant=L_ant,
            d=d,
            f=f,
        )

        ax.plot(
            x,
            F,
            linewidth=2.2,
            label=label,
        )

    ax.set_xlabel(r"$K^{*}L_{\rm ant}$", fontsize=16)
    ax.set_ylabel(r"$F$", fontsize=16)

    ax.set_title(
        r"Step 3: current decay effect from no-gap to finite-gap, "
        + fr"$\theta={theta_deg}^\circ$",
        fontsize=14,
    )

    ax.set_xlim(0, 30)
    ax.set_ylim(bottom=0)

    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()

    save_path = OUTDIR / "step3_current_decay_from_no_gap_to_gap.png"
    fig.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")

# ============================================================
# 9. Step 4: complete QTN spectrum with V-shaped response
# ============================================================

def epsilon_L_maxwellian(kLD, z):
    """
    Maxwellian longitudinal dielectric function in dimensionless form.

    Parameters
    ----------
    kLD:
        k * L_D

    z:
        z = omega / (k * v_the)

    Formula idea:
        epsilon_L = 1 + 1/(k^2 L_D^2) * [1 + z Z(z)]

    For real z:
        Re[1 + zZ(z)] = 1 - 2 z D(z)
        Im[1 + zZ(z)] = sqrt(pi) z exp(-z^2)

    D(z) is Dawson's integral.
    """
    kLD_safe = np.where(np.abs(kLD) < EPS, EPS, kLD)

    factor = 1.0 / (kLD_safe ** 2)

    eps_re = 1.0 + factor * (1.0 - 2.0 * z * dawsn(z))
    eps_im = factor * (np.sqrt(np.pi) * z * np.exp(-z ** 2))

    return eps_re + 1j * eps_im


def build_response_interpolator(
    case_name,
    theta_deg=125.0,
    L_ant=6.5,
    d_gap=1.979,
    f_value=1.0 / 25.0,
):
    """
    Precompute antenna response F(x), then build interpolation function.

    x = k * L_ant

    This avoids recalculating the expensive spherical integral inside
    the QTN frequency loop.
    """
    # Use log grid because QTN integral samples a wide range of kL.
    x_grid = np.logspace(-3, 2.5, 180)

    if case_name == "no_gap":
        F_grid = F_vshape_no_gap_curve(
            x_array=x_grid,
            theta_deg=theta_deg,
        )
    else:
        F_grid = F_vshape_current_decay_curve(
            x_array=x_grid,
            theta_deg=theta_deg,
            L_ant=L_ant,
            d=d_gap,
            f=f_value,
        )

    F_grid = np.nan_to_num(F_grid, nan=0.0, posinf=0.0, neginf=0.0)
    F_grid = np.maximum(F_grid, 0.0)

    log_x_grid = np.log(x_grid)

    def F_interp(x):
        x = np.asarray(x, dtype=float)
        x_clip = np.clip(x, x_grid[0], x_grid[-1])
        return np.interp(np.log(x_clip), log_x_grid, F_grid)

    return F_interp


def qtn_spectrum_single(
    freq_ratio,
    F_interp,
    L_over_LD=8.0,
    z_min=1e-3,
    z_max=8.0,
    n_z=900,
):
    """
    Calculate one point of normalized QTN spectrum.

    Parameters
    ----------
    freq_ratio:
        f / f_p = omega / omega_p

    F_interp:
        interpolation function for antenna response F(k L_ant)

    L_over_LD:
        L_ant / L_D.
        The thesis examples often use normalized antenna length such as
        L_ant / L_D = 8.

    z:
        z = omega / (k v_the)

    Relation:
        L_D = v_the / (sqrt(2) omega_p)

    Therefore:
        k L_D = (omega / omega_p) / (sqrt(2) z)
        k L_ant = (k L_D) * (L_ant / L_D)

    Output:
        normalized QTN spectral value.
        This is a complete Maxwellian QTN integral shape, but the absolute
        calibration constant is not included.
    """
    z = np.logspace(np.log10(z_min), np.log10(z_max), n_z)

    kLD = freq_ratio / (np.sqrt(2.0) * z)
    x = kLD * L_over_LD

    eps = epsilon_L_maxwellian(kLD, z)

    F_val = F_interp(x)

    # QTN integrand shape based on the dielectric response.
    integrand = (
        F_val
        * np.imag(eps)
        / (np.abs(eps) ** 2)
        / (z ** 2)
    )

    integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

    return float(np.real(trapz(integrand, z)))


def qtn_spectrum_curve(
    freq_ratio_array,
    F_interp,
    L_over_LD=8.0,
):
    """
    Calculate normalized QTN spectrum curve.
    """
    return np.array(
        [
            qtn_spectrum_single(
                freq_ratio=f_ratio,
                F_interp=F_interp,
                L_over_LD=L_over_LD,
            )
            for f_ratio in freq_ratio_array
        ]
    )


def plot_step4_qtn_spectrum():
    """
    Step 4:
        Complete QTN spectrum using the V-shaped antenna response.

    This version only changes the figure style for paper use.
    The calculation part is unchanged.
    """
    theta_deg = 125.0
    L_ant = 6.5
    d_gap = 1.979

    # normalized antenna length used for QTN integral
    L_over_LD = 8.0

    freq_ratio = np.linspace(0.1, 3.5, 180)

    cases = [
        ("no gap", "no_gap", None, "black", "-"),
        (r"finite gap, no SC current ", "finite_gap", 0.0, "tab:blue", "-"),
        (r"finite gap, AWAS / SolO ", "finite_gap", 1.0 / 25.0, "tab:orange", "--"),
        (r"finite gap, max SC current ", "finite_gap", 1.0, "tab:green", "-."),
    ]

    spectra = []

    # ========================================================
    # 1. Calculate QTN spectra
    # ========================================================
    for label, case_name, f_value, color, linestyle in cases:
        if case_name == "no_gap":
            F_interp = build_response_interpolator(
                case_name="no_gap",
                theta_deg=theta_deg,
                L_ant=L_ant,
                d_gap=d_gap,
                f_value=1.0,
            )
        else:
            F_interp = build_response_interpolator(
                case_name="finite_gap",
                theta_deg=theta_deg,
                L_ant=L_ant,
                d_gap=d_gap,
                f_value=f_value,
            )

        qtn = qtn_spectrum_curve(
            freq_ratio_array=freq_ratio,
            F_interp=F_interp,
            L_over_LD=L_over_LD,
        )

        spectra.append(qtn)

    # Normalize all curves by the maximum of the no-gap QTN.
    norm = np.max(spectra[0])
    if norm < EPS:
        norm = 1.0

    # Visual scaling only, not absolute calibration.
    scale = 1e-16 / norm

    # ========================================================
    # 2. Plot: paper-style figure
    # ========================================================
    fig, ax = plt.subplots(figsize=(10.2, 7.0))

    for (label, _, _, color, linestyle), qtn in zip(cases, spectra):
        if label == "no gap":
            linewidth = 3.0
        else:
            linewidth = 2.6

        ax.semilogy(
            freq_ratio,
            qtn * scale,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    # Axes labels
    ax.set_xlabel(r"$ω/ω_p$", fontsize=20)
    ax.set_ylabel(
        r"$V^2/T_e^{1/2}\ "
        r"(\mathrm{V}^2\,\mathrm{Hz}^{-1}\,\mathrm{K}^{-1/2})$",
        fontsize=18,
    )

    # Short paper-style title
    ax.set_title(
        r"QTN spectrum for a V-shaped antenna ($\psi=125^\circ$)",
        fontsize=18,
        pad=12,
    )

    # Axis range
    ax.set_xlim(0.1, 3.5)
    ax.set_ylim(1e-18, 5e-16)

    # Ticks
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)

    # Grid
    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)

    # Legend
    legend = ax.legend(
        fontsize=12,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        borderpad=0.8,
    )
    legend.get_frame().set_edgecolor("0.75")

    # Frame
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()

    save_path = OUTDIR / "step4_vshape_qtn_spectrum_paper_style.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")

def plot_step4b_qtn_angle_effect():
    """
    Step 4b:
        QTN spectra for different V-shaped antenna opening angles.

    This figure keeps the current model fixed and changes only the
    opening angle theta. It is used to show how the V-shaped angle
    affects the final QTN spectrum.
    """

    # 固定天线与电流模型参数
    L_ant = 6.5
    d_gap = 1.979
    f_value = 1.0 / 25.0      # AWAS / SolO current model

    # 固定归一化天线长度
    L_over_LD = 8.0

    # 频率范围
    freq_ratio = np.linspace(0.1, 3.5, 180)


    # 只改变夹角：对应 theta/pi = 0.1, 0.3, 0.5, 0.7, 0.9
    angle_ratio_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    angle_list = [18.0, 54.0, 90.0, 126.0, 162.0]

    spectra = []

    # ========================================================
    # 1. Calculate QTN spectra for different angles
    # ========================================================
    for theta_deg in angle_list:

        F_interp = build_response_interpolator(
            case_name="finite_gap",
            theta_deg=theta_deg,
            L_ant=L_ant,
            d_gap=d_gap,
            f_value=f_value,
        )

        qtn = qtn_spectrum_curve(
            freq_ratio_array=freq_ratio,
            F_interp=F_interp,
            L_over_LD=L_over_LD,
        )

        spectra.append(qtn)

    # 用 theta/pi = 0.7，即 126° 作为归一化参考
    ref_index = angle_list.index(126.0)
    norm = np.max(spectra[ref_index])
    if norm < EPS:
        norm = 1.0

    scale = 1e-16 / norm

    # ========================================================
    # 2. Plot
    # ========================================================
    fig, ax = plt.subplots(figsize=(10.2, 7.0))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(angle_list)))
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for theta_ratio, theta_deg, qtn, color, linestyle in zip(
            angle_ratio_list, angle_list, spectra, colors, linestyles
    ):
        ax.semilogy(
            freq_ratio,
            qtn * scale,
            color=color,
            linestyle=linestyle,
            linewidth=2.8,
            label=fr"$\psi/\pi={theta_ratio:.1f}$",
        )

    ax.set_xlabel(r"$f/f_p$", fontsize=20)
    ax.set_ylabel(
        r"$V^2/T_e^{1/2}\ "
        r"(\mathrm{V}^2\,\mathrm{Hz}^{-1}\,\mathrm{K}^{-1/2})$",
        fontsize=18,
    )

    ax.set_title(
        r"QTN spectra for different V-shaped antenna angles",
        fontsize=18,
        pad=12,
    )

    ax.set_xlim(0.1, 3.5)
    ax.set_ylim(1e-18, 5e-16)

    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)

    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)

    legend = ax.legend(
        fontsize=12,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        borderpad=0.8,
    )
    legend.get_frame().set_edgecolor("0.75")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()

    save_path = OUTDIR / "step4b_qtn_angle_effect.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")

def main():
    plot_step1_angle_effect()
    plot_step2_gap_effect()
    plot_step3_current_decay_effect()

    # 原来的 QTN 图：固定 125°，比较不同中心区电流模型
    plot_step4_qtn_spectrum()

    # 新增 QTN 图：固定 AWAS/SolO 电流模型，比较不同夹角
    plot_step4b_qtn_angle_effect()

    save_model_note()

    print("\nFinished.")
    print("Generated files:")
    print(f"  1. {OUTDIR / 'step1_vshape_angles.png'}")
    print(f"  2. {OUTDIR / 'step2_vshape_gap_smooth.png'}")
    print(f"  3. {OUTDIR / 'step3_vshape_current_decay_from_no_gap_to_gap.png'}")
    print(f"  4. {OUTDIR / 'step4_vshape_qtn_spectrum.png'}")
    print(f"  5. {OUTDIR / 'step4b_qtn_angle_effect.png'}")
    print(f"  6. {OUTDIR / 'model_note.txt'}")


if __name__ == "__main__":
    main()