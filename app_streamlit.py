import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="Simulador de Procesos IPN")

st.title("📊 Simulador de Capacidad de Proceso")
st.markdown("Compara dos máquinas frente a un modelo teórico ideal (Cp=1.0)")

# Sidebar para controles
with st.sidebar:
    st.header("Configuración")
    tolerancia = st.slider("Tolerancia ±", 1.0, 5.0, 3.0, 0.5)
    st.subheader("Máquina 1")
    media_m1 = st.slider("Media M1", 196.0, 204.0, 200.0, 0.1)
    desv_m1 = st.slider("Desv. Est. M1", 0.1, 2.5, 0.8, 0.05)
    n_m1 = st.number_input("n M1", 10, 500, 55)
    
    st.subheader("Máquina 2")
    media_m2 = st.slider("Media M2", 196.0, 204.0, 201.5, 0.1)
    desv_m2 = st.slider("Desv. Est. M2", 0.1, 2.5, 0.4, 0.05)
    n_m2 = st.number_input("n M2", 10, 500, 55)

# Lógica de cálculo
objetivo = 200
LSL, USL = objetivo - tolerancia, objetivo + tolerancia

np.random.seed(42)
datos_m1 = np.random.normal(media_m1, desv_m1, n_m1)
datos_m2 = np.random.normal(media_m2, desv_m2, n_m2)

sigma_ref = tolerancia / 3
q1_t, q3_t = objetivo - 0.67449 * sigma_ref, objetivo + 0.67449 * sigma_ref
datos_ref_teorico = [LSL, q1_t, objetivo, q3_t, USL]

# Gráficos
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax_box = fig.add_subplot(gs[1, :])

x_range = np.linspace(190, 210, 400)
y_ideal = norm.pdf(x_range, objetivo, sigma_ref)

datasets = [datos_m1, datos_m2]
n_inputs = [n_m1, n_m2]
axes = [ax1, ax2]
colores = ['#3498db', '#e67e22']

for i in range(2):
    data = datasets[i]
    mu_calc, sigma_calc = np.mean(data), np.std(data, ddof=1)
    cp = (USL - LSL) / (6 * sigma_calc)
    cpk = min((USL - mu_calc) / (3 * sigma_calc), (mu_calc - LSL) / (3 * sigma_calc))
    
    axes[i].hist(data, bins=15, density=True, alpha=0.3, color='gray')
    axes[i].plot(x_range, norm.pdf(x_range, mu_calc, sigma_calc), lw=3, color=colores[i])
    axes[i].plot(x_range, y_ideal, color='green', ls=':', label='Ideal')
    axes[i].axvline(LSL, color='red', ls='--')
    axes[i].axvline(USL, color='red', ls='--')
    axes[i].set_title(f"Máquina {i+1} | Cp: {cp:.2f} | Cpk: {cpk:.2f}")
    
    color_bg = '#eaffea' if cpk >= 1.33 else ('#ffffea' if cpk >= 1.0 else '#ffeaea')
    axes[i].set_facecolor(color_bg)

# Boxplots
bp = ax_box.boxplot([datos_m1, datos_m2, datos_ref_teorico], vert=False, patch_artist=True, 
                    labels=['Máq 1', 'Máq 2', 'Ideal (Cp=1.0)'], widths=0.6)

for patch, color in zip(bp['boxes'], ['#3498db', '#e67e22', '#2ecc71']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax_box.vlines([LSL, USL], 2.7, 3.3, color='green', lw=2)
ax_box.axvline(LSL, color='red', ls='--')
ax_box.axvline(USL, color='red', ls='--')

st.pyplot(fig)
