import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def decaimiento_exponencial(x, V0, V, B, W, C):
    return V0 + V * np.exp(-B * x) * np.cos(W * x + C)

# Ocultar la ventana principal de tkinter
root = tk.Tk()
root.withdraw()

print("Por favor selecciona el archivo CSV en la ventana que se abrirá...")
archivo = filedialog.askopenfilename(
    title="Selecciona el archivo Act4_Amortiguado_mas_cerca.csv",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

if archivo:
    print(f"Archivo seleccionado: {archivo}")
    try:
        df = pd.read_csv(archivo, skiprows=17, encoding='iso-8859-1')
        
        # Usar iloc para seleccionar por posición (índices 3 y 4)
        columna3 = df.iloc[:, 3]  # Todas las filas, columna 3
        columna4 = df.iloc[:, 4]  # Todas las filas, columna 4
        
        # Crear un nuevo DataFrame limpio
        df_limpio = pd.DataFrame({
            'Tiempo (s)': columna3.values,
            'Voltaje Pico-Pico (V)': columna4.values
        })
        
        # CONFIGURACIÓN PARA MOSTRAR MÁS DECIMALES
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        print("\nDataFrame limpio (con más decimales):")
        print(df_limpio)
        
        # ======== FILTRAR DATOS (8.7s a 30s) ========
        x_data = df_limpio['Tiempo (s)']
        y_data = df_limpio['Voltaje Pico-Pico (V)']

        mask = (x_data >= 8.7) & (x_data <= 30.0)
        x = x_data[mask].to_numpy()
        y = y_data[mask].to_numpy()

        # Eliminar NaN/Inf
        mask_finite = np.isfinite(x) & np.isfinite(y)
        x = x[mask_finite]
        y = y[mask_finite]

        print(f"Datos usados: {len(x)} puntos")

        # ======== FUNCIÓN DE AJUSTE ========
        def modelo(x, V0, V, B, W, C):
            return V0 + V * np.exp(-B * x) * np.cos(W * x + C)

        # ======== VALORES INICIALES ========
        V0_est = np.mean(y[-20:])        # offset con últimos puntos
        V_est = (np.max(y) - np.min(y))  # amplitud aprox
        B_est = 0.3                      # amortiguamiento típico
        W_est = 5.4                      # rad/s, con base en tu caso anterior
        C_est = 0                        # fase inicial

        p0 = [V0_est, V_est, B_est, W_est, C_est]

        # Bounds razonables
        bounds = (
            [-np.inf, 0,     0.01, 1,      -2*np.pi],   # min
            [ np.inf, np.inf, 3.0, 20,      2*np.pi]    # max
        )

        popt, pcov = curve_fit(modelo, x, y, p0=p0, bounds=bounds, maxfev=20000)

        V0, V, B, W, C = popt
        perr = np.sqrt(np.diag(pcov))

        print("\nParámetros del ajuste:")
        print(f"V0 = {V0:.5f} ± {perr[0]:.5f}")
        print(f"V  = {V:.5f} ± {perr[1]:.5f}")
        print(f"B  = {B:.5f} ± {perr[2]:.5f}")
        print(f"W  = {W:.5f} ± {perr[3]:.5f}")
        print(f"C  = {C:.5f} ± {perr[4]:.5f}")

        # ======== GRAFICAR ========
        y_fit = modelo(x, *popt)
        resid = y - y_fit
        # Residuales
        std_residuales = np.std(resid)
        print(f"\nDesviación estándar de residuales: {std_residuales:.6f} V")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.set_facecolor("white")

        ax1.set_xlim(8.5, 30)
        ax1.scatter(x, y, color = 'black', s=30, alpha=0.7, label='Datos experimentales', edgecolors='black', linewidth=0.5)
        ax1.plot(x, y_fit, color = '#EFA5FF', linewidth=2.5, label=f'Ajuste: $V_0 (V) + V e^{{-{B:.4f}(s⁻¹)t}} \cos({W:.4f}(rad/s)t + {C:.4f} (rad)$')
        ax1.set_ylabel("Voltaje Pico a Pico (V)", fontsize = 12)
        ax1.set_title("Amortiguamiento magnético intermedio", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        textstr = f'$V_0$ = {V0:.4f} V\n$V$ = {V:.4f} V\n$B$ = {B:.4f} Hz\n$W$ = {W:.4f} rad/s\n$C$ = {C:.4f} rad'
        props = dict(boxstyle='round', facecolor='#F6CFFF', alpha=0.8)
        ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

        ax2.set_xlim(8.5, 30)
        ax2.axhline(0, color='#F6CFFF', linestyle='--', linewidth=2)
        ax2.scatter(x, resid, color = 'black', s=20, alpha=0.7)
        ax2.set_ylabel("Residuales Normalizados (V)", fontsize=12)
        ax2.set_xlabel("Tiempo (s)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ajuste_amortiguamiento_intermedio.png', dpi=300, bbox_inches='tight')
        plt.show()

        # ======== ANÁLISIS DE RESIDUALES ========
        print(f"\n--- ANÁLISIS DE RESIDUALES ---")
        print(f"Media de residuales: {np.mean(resid):.6f} V")
        print(f"Desviación estándar: {std_residuales:.6f} V")
        print(f"Residual máximo: {np.max(resid):.6f} V")
        print(f"Residual mínimo: {np.min(resid):.6f} V")
        print(f"68% de residuales entre: ±{std_residuales:.6f} V")
        print(f"95% de residuales entre: ±{2*std_residuales:.6f} V")

        # Datos Calculados

        w_o = np.sqrt(W**2 + B**2)
        Q = w_o / (2*B)

        #incertidumbres

        #Omega
        d_wo_w = W/w_o
        d_wo_b = B/w_o

        incertidumbre_wo = np.sqrt( (d_wo_w*perr[3]) **2 + (d_wo_b*perr[2])**2)

        #Q

        d_Q_wo = 1/(2*B)
        d_Q_b = - w_o/(2*B**2)
        inecrtidumbre_Q = np.sqrt((d_Q_wo *incertidumbre_wo)**2 + (d_Q_b*perr[2])**2) 

        print(f"\n -- CÁLCULOS ADICIONALES --")
        print(f"Frecuencia natural w_o: {w_o:.6f} ± {incertidumbre_wo:.6f} rad/s")
        print(f"Factor de calidad Q: {Q:.6f} ± {inecrtidumbre_Q:.6f}")

    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
else:
    print("No se seleccionó ningún archivo.")
