
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import dawsn

import matplotlib
matplotlib.use('TkAgg')  # 或 'Agg'（无弹窗，仅保存图片）

# import modules


from meyer_vernet_impedance import (calculate_impedance,calculate_QTN_voltage,eps0,e,me, kb)
from F_Shapes_fixed_5_3 import (F_SolO_AWAS)

# constants and parameters
ne_cm3 = 30.2
ne = ne_cm3 * 1e6
Te_eV = 19.5
Te = Te_eV * e / kb
psi_deg = 125.0
psi = np.deg2rad(psi_deg)
L_ant = 6.5
gap = 1.979
f_AWAS = 1.0 / 25.0

print("==============================")
print("Solar Orbiter parameters")
print("==============================")

print("ne = %.3e m^-3" %ne)
print("Te = %.3e K"%Te)
print("psi = %.3f rad"%psi)

# plasma parameters

omega_p = np.sqrt(ne*e**2/(me*eps0))
fp = omega_p/(2*np.pi)
LD = np.sqrt(eps0*kb*Te/(ne*e**2))
vte = np.sqrt( 2*kb*Te/me)

print("\n==============================")
print("Derived plasma parameters")
print("==============================")
print("fp = %.3f kHz"%(fp/1000))
print("Debye length = %.3f m"%LD)

# dielectric function

EPS = 1e-12

def epsilon_L_maxwellian(kLD,z):

    kLD_safe = np.where( np.abs(kLD)<EPS,EPS, kLD)
    factor = 1/(kLD_safe**2)

    eps_real = (1+factor*(1-2*z*dawsn(z)))

    eps_imag = (factor*np.sqrt(np.pi) *z* np.exp(-z*z))

    return eps_real + 1j*eps_imag

# build real antenna response



print("\nCalculating antenna response...")

# x = kL
x_grid = np.logspace( -3,2,250)
F_grid = F_SolO_AWAS(x_grid,psi=psi,f=f_AWAS,L_ant=L_ant,d=gap)
F_grid=np.real(F_grid)
# avoid negative numerical values
F_grid=np.maximum(
    F_grid,
    0
)

def F_interp(x):

    x=np.asarray(x)
    x_safe=np.maximum( x,1e-10)
    return np.interp( np.log(x_safe),np.log(x_grid), F_grid)

# calculate impedance

freq=np.linspace(0.1*fp,3.0*fp,180)
# k integration range
k_array=np.logspace( -5, 1,1000)
Za=[]
QTN=[]

print("\nCalculating impedance...")

for f in freq:

    omega=2*np.pi*f

    za=calculate_impedance(
        k_array=k_array,
        F_interp=F_interp,
        omega=omega,
        epsilon_func=epsilon_L_maxwellian,
        ne=ne,
        Te=Te,
        L_ant=L_ant
    )

    Za.append(za)
    qtn=calculate_QTN_voltage(za,Te)

    QTN.append(qtn)

Za=np.array(Za)
QTN=np.array(QTN)

# output check

peak_index=np.argmax(QTN)

print("\n==============================")
print("Result")
print("==============================")

print("QTN peak frequency = %.3f kHz" %(freq[peak_index]/1000))
print( "Plasma frequency = %.3f kHz" %(fp/1000))
print("Maximum Re(Za) = %.5e Ohm"%np.max(Za.real))
print( "Maximum QTN = %.5e V^2/Hz" %np.max(QTN))


# plot impedance

plt.figure(figsize=(8,5))
plt.plot(freq/1000,Za.real)

plt.xlabel( "Frequency (kHz)")
plt.ylabel("Re(Za) (Ohm)")
plt.title("Solar Orbiter V antenna impedance")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.semilogy(freq/1000,QTN)
plt.xlabel("Frequency (kHz)")
plt.ylabel(r"$V_{QTN}^{2}$ (V$^2$/Hz)")
plt.title("Solar Orbiter V-shaped antenna absolute QTN spectrum")
plt.grid()
plt.tight_layout()
plt.show()