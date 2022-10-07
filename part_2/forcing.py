from irregular_waves import *

def get_forcing(Hm0, Tp, tspan, f_min=0.001, f_max=0.2, num_of_comp=100, resolution=100):
    f = np.linspace(f_min, f_max, resolution)
    E_js = jonswap(f, Hm0, Tp)
    df = abs(f_min - f_max)/num_of_comp
    sample_index = np.random.randint(0, 100, size=num_of_comp)
    sample_freq = f[sample_index]
    sample_amplitude = 4*np.sqrt(df*E_js[sample_index]) # 4 sqrt(m0)
    phases = np.random.uniform(0, np.pi*2, num_of_comp)
    forcing = np.sum([ sample_amplitude[i] * np.cos(sample_freq[i]*tspan + phases[i]) for i in range(len(sample_index))], axis=0)
    return forcing, E_js, f
