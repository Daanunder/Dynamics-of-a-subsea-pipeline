import matplotlib.pyplot as plt
import numpy as np

L = 5000
EI = 10**9
rho = 7850
A = .2

def wn(n):
    return (np.pi * n / L)**2 * np.sqrt(EI/rho/A)

def b(n):
    return np.sqrt(wn(n))*(rho*A/EI)**0.25

def plot_anal_sol():
    fig, ax = plt.subplots(2,3, figsize=(12,6))
    x = np.linspace(0, L, 1000)
    plt.tight_layout(pad=2.0)
    n = 1
    for a in ax.flatten():
        a.plot(x, np.sin(b(n)*x))
        a.set_title(f"Mode {n}")
        n+=1
    
    plt.savefig("analytical_modal_response.png")

