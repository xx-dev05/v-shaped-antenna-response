import numpy as np

eps0 = 8.8541878128e-12
e = 1.602176634e-19
me = 9.1093837e-31
kb = 1.380649e-23
# Calculate impedance

def calculate_impedance(k_array,F_interp,omega,epsilon_func,ne,Te,L_ant):

    # plasma parameters
    LD = np.sqrt(
        eps0*kb*Te /
        (ne*e**2)
    )
    vte = np.sqrt(
        2*kb*Te/me
    )
    F_values = []
    eps_values = []
    # k integration


    for k in k_array:
        # F is calculated by x=kL_ant
        x = k*L_ant
        F = F_interp(x)
        F_values.append(F)
        # dielectric function
        kLD = k*LD
        z = omega/(k*vte)
        eps = epsilon_func(
            kLD,
            z
        )
        eps_values.append(eps)

    F_values = np.asarray(F_values)

    eps_values = np.asarray(eps_values)

    # Meyer-Vernet integral

    integrand = (F_values / eps_values)

    integral = np.trapezoid(integrand,k_array)

    # antenna impedance
    Za = (4j /(np.pi**2*omega*eps0)) * integral

    return Za
# QTN spectrum
def calculate_QTN_voltage(
        Za,
        Te
):

    return (4*kb*Te*np.real(Za))
