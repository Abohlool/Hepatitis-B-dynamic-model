import numpy as np
import matplotlib.pyplot as plt

#$ Models--------------------------------------------------------------------------------------------------------------

def deterministic_model(t, Y, Theta):
    S, A, B, R = Y
    scale = 100
    
    Lambda, eta, mu_0, nu, alpha, gamma, gamma_1, beta, gamma_2, mu_1 = Theta
    
    dS = (1 - eta * B / scale) * Lambda - (nu + mu_0) * S - (A + gamma * B) * alpha * S / scale
    dA = alpha * S * A / scale + gamma * alpha * S * B / scale - (gamma_1 + beta + mu_0) * A
    dB = beta * A - ( mu_1 + gamma_2 + mu_0) * B + eta * Lambda * B / scale
    dR = gamma_2 * B - mu_0 * R + gamma_1 * A + nu * S
    
    return np.array([dS, dA, dB, dR])


def adjoint_system(Y, L, Theta, weights, u1, u2):
    S, A, B, _ = Y
    l1, l2, l3, l4 = L
    scale = 100
    
    Lambda, eta, mu_0, _, alpha, gamma, gamma_1, beta, gamma_2, mu_1 = Theta
    w1, w2, _, _ = weights
    
    dl1 = (mu_0 + u1)*l1 + alpha*(A + gamma*B)/scale*(l1 - l2) - l4*u1
    dl2 = w1 + alpha*S/scale*(l1 - l2) + (mu_0 + gamma_1 + beta + u2) * l2 - beta * l3 - (gamma_1 + u2)*l4
    dl3 = w2 - gamma*alpha*S/scale*(l1 - l2) + (mu_0 + mu_1- eta*Lambda + gamma_2 + u2)*l3 - (gamma_2 + u2)*l4
    dl4 = mu_0*l4
    
    return np.array([dl1, dl2, dl3, dl4])


def optimal_controls(Y, L, weights):
    S, A, B, _ = Y
    l1, l2, l3, l4 = L
    
    _, _, w3, w4 = weights
    
    u1 = np.max([0, np.min([1, (S * (l4 - l1)) / w3])])
    u2 = np.max([0, np.min([1, ((l4 - l3)*B/w4 - (l2 - l4)*A)])])
    
    return u1, u2


def controlled_model(Y, Theta, u1, u2):
    S, A, B, R = Y
    scale = 100
    
    Lambda, eta, mu_0, _, alpha, gamma, gamma_1, beta, gamma_2, mu_1 = Theta
    
    dS = (1 - eta * B / scale) * Lambda - alpha * S * A / scale - gamma * alpha * S * B / scale - (mu_0 + u1) * S
    dA = alpha * S * A / scale + gamma * alpha * S * B / scale - (mu_0 + gamma_1 + beta + u2) * A
    dB = beta * A - (mu_0 - Lambda * eta + gamma_2 + mu_1 + u2) * B
    dR = gamma_1 * A + gamma_2 * B + u1 * S + (A + B) * u2 - mu_0 * R
    
    return np.array([dS, dA, dB, dR])



def R0_controlled(u1, u2, Theta):
    Lambda, eta, mu_0, _, alpha, gamma, gamma_1, beta, gamma_2, mu_1 = Theta
    
    numerator = alpha * Lambda * (gamma * (mu_0 + gamma_1 + beta + u2) + beta)
    denominator = ((mu_0 + u1) * (mu_0 + gamma_1 + beta + u2) * (mu_0 + mu_1 + gamma_2 + u2 - eta * Lambda))
    return numerator / denominator


#$ -------------------------------------Numeric-Methods------------------------------------- 

def rk4(f, t, Theta, Y0, dt):
    k1 = dt * f(t, Y0, Theta)
    k2 = dt * f(t + dt/2, Y0 + k1/2, Theta)
    k3 = dt * f(t + dt/2, Y0 + k2/2, Theta)
    k4 = dt * f(t + dt, Y0 + k3, Theta)
    
    return Y0 + (k1 + 2*k2 + 2*k3 + k4) / 6


def solve(Y0, Theta, t0=0, tf=20, dt=0.01):
    ts = np.arange(t0, tf + dt, dt)
    n = len(ts)
    Y = np.zeros((n, 4))
    Y[0] = Y0
    for i in range(n - 1):
        Y[i+1] = rk4(lambda t, y, _: deterministic_model(t, y, Theta), ts, Theta, Y[i], dt)
    return ts, Y


def solve_optimal_control(Y0, Theta, weights, t0=0, tf=20, dt=0.01, iter=40):
    t = np.arange(t0, tf+dt, dt)
    n = len(t)
    
    u1 = np.zeros(n)
    u2 = np.zeros(n)
    
    for _ in range(iter):
        Y = np.zeros((n, 4))
        Y[0] = Y0
        for i in range(n - 1):
            Y[i+1] = rk4(lambda t, y, _: controlled_model(y, Theta, u1[i], u2[i]), t[i], Theta, Y[i], dt)
        
        L = np.zeros((n, 4))
        for i in reversed(range(n - 1)):
            L[i] = L[i + 1] - adjoint_system(Y[i], L[i+1], Theta, weights, u1[i], u2[i])
        
        for i in range(n):
            u1[i], u2[i] = optimal_controls(Y[i], L[i], weights)
        
    return t, Y, u1, u2

#$ -------------------------------------plots-------------------------------------

def plot_deterministic(Y0, Theta, t0=0, tf=20, dt=0.05):
    S0, A0, B0, R0 = Y0
    
    fig = plt.figure(figsize=(14, 10))
    
    t, Y = solve(Y0, Theta, t0, tf, dt)
    S, A, B, R = Y[:,0], Y[:,1], Y[:,2], Y[:,3]
    
    plt.plot(t, S, ls="--", lw=1.7, label=f"S(0)={S0}")
    plt.plot(t, A, ls="--", lw=1.7, label=f"A(0)={A0}")
    plt.plot(t, B, ls="--", lw=1.7, label=f"B(0)={B0}")
    plt.plot(t, R, ls="--", lw=1.7, label=f"R(0)={R0}")
    
    plt.xlabel("Time (Months)")
    plt.ylabel("Population")
    plt.xlim(t0, tf)
    plt.ylim(min(*S, *A, *B, *R), max(*S, *A, *B, *R))
    plt.grid(True, ls="--")
    plt.legend()
    
    fig.suptitle("Deterministic system")
    plt.show()


def plot_controlled(Y0, Theta, weights, t0=0, tf=20, dt=0.05):
    S0, A0, B0, R0 = Y0
    
    fig = plt.figure(figsize=(14, 10))
    
    t, Y, _, _ = solve_optimal_control(Y0, Theta, weights, t0=t0, tf=tf, dt=dt)
    S, A, B, R = Y[:,0], Y[:,1], Y[:,2], Y[:,3]
    
    plt.plot(t, S, ls="--", lw=1.7, label=f"S(0)={S0}")
    plt.plot(t, A, ls="--", lw=1.7, label=f"A(0)={A0}")
    plt.plot(t, B, ls="--", lw=1.7, label=f"B(0)={B0}")
    plt.plot(t, R, ls="--", lw=1.7, label=f"R(0)={R0}")
    
    plt.xlabel("Time (Months)")
    plt.ylabel("Population")
    plt.xlim(t0, tf)
    plt.ylim(min(*S, *A, *B, *R), max(*S, *A, *B, *R))
    plt.grid(True, ls="--")
    plt.legend()
    
    fig.suptitle("Controlled system")
    plt.show()


def plot_comparison(Y0, Theta, weights, t0=0, tf=50, dt=0.05):
    tu, Yu = solve(Y0, Theta, t0, tf, dt)
    tc, Yc, _, _ = solve_optimal_control(Y0, Theta, weights, t0, tf, dt)
    
    Su, Au, Bu, Ru = Yu[:,0], Yu[:,1], Yu[:,2], Yu[:,3]
    Sc, Ac, Bc, Rc = Yc[:,0], Yc[:,1], Yc[:,2], Yc[:,3]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    axs[0].plot(tu, Su, '--', label="Uncontrolled")
    axs[0].plot(tc, Sc, label="Controlled")
    axs[0].set_title("Susceptible")
    
    axs[1].plot(tu, Au, '--', label="Uncontrolled")
    axs[1].plot(tc, Ac, label="Controlled")
    axs[1].set_title("Acute")
    
    axs[2].plot(tu, Bu, '--', label="Uncontrolled")
    axs[2].plot(tc, Bc, label="Controlled")
    axs[2].set_title("Chronic")
    
    axs[3].plot(tu, Ru, '--', label="Uncontrolled")
    axs[3].plot(tc, Rc, label="Controlled")
    axs[3].set_title("Recovered")
    
    for ax in axs:
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Population")
        ax.set_xlim(t0, tf)
        ax.grid(True)
        ax.legend()
    
    axs[0].set_ylim(min(Su.min(), Sc.min()), max(Su.max(), Sc.max()))
    axs[1].set_ylim(min(Au.min(), Ac.min()), max(Au.max(), Ac.max()))
    axs[2].set_ylim(min(Bu.min(), Bc.min()), max(Bu.max(), Bc.max()))
    axs[3].set_ylim(min(Ru.min(), Rc.min()), max(Ru.max(), Rc.max()))
    
    fig.suptitle("Deterministic vs. Controlled")
    plt.show()


def plot_controls(Y0, Theta, weights, t0=0, tf=50, dt=0.05):
    S0, A0, B0, R0 = Y0
    t, _, u1, u2 = solve_optimal_control([S0, A0, B0, R0], Theta, weights, t0, tf, dt)
    
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    
    axs[0].plot(t, u1, "r--")
    axs[0].set_xlabel("Time (Months)")
    axs[0].set_ylabel(r"$Vaccination Control - u_1(t)$")
    
    axs[1].plot(t, u2, "r--")
    axs[1].set_xlabel("Time (Months)")
    axs[1].set_ylabel(r"$Treatment Control - u_2(t)$")
    
    for ax in axs:
        ax.set_xlim(t0, tf)
        ax.set_ylim(0, 1.01)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True)
        
    fig.suptitle("Control functions")
    plt.show()


def plot_R0(Theta):
    u1_vals = np.linspace(0, 1, 100)
    u2_vals = np.linspace(0, 1, 100)
    U2, U1 = np.meshgrid(u2_vals, u1_vals)
    
    R0_vals = R0_controlled(U1, U2, Theta)
    
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(U1, U2, R0_vals, cmap='jet', edgecolor='none')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(np.arange(0, 1.1, .2))
    ax1.set_yticks(np.arange(0, 1.1, .2))
    ax1.view_init(azim=225)
    ax1.set_xlabel("$u_1$")
    ax1.set_ylabel("$u_2$")
    ax1.set_zlabel("$Reproductive number - R_0$")
    ax1.set_title("Effect of controls on $R_0$")
    
    ax2 = fig.add_subplot(122)
    contourPlot = ax2.contourf(U1, U2, R0_vals, levels=30, cmap='jet')
    
    ax2.set_xlabel("$u_1(t)$", size=12)
    ax2.set_ylabel("$u_2(t)$", size=12)
    ax2.set_title("Contour plot of $R_0$")
    
    fig.colorbar(contourPlot, ax=ax2, label="$R_0$")
    
    fig.suptitle(r"$R_0 over control functions$")
    plt.tight_layout()
    plt.show()
