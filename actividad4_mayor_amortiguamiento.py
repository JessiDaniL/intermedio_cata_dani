import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def decaimiento_exponencial(x,V0, V, B, W, C):
    return V0 +  V * np.exp(-B * x) * np.cos(W * x + C)

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
        
        # print("Datos extraídos correctamente:")
        # print(f"Columna 3 (primeros 5): {columna3.values[:5]}")
        # print(f"Columna 4 (primeros 5): {columna4.values[:5]}")
        
        # Crear un nuevo DataFrame limpio
        df_limpio = pd.DataFrame({
            'Tiempo (s)': columna3.values,
            'Voltaje Pico-Pico (V)': columna4.values
        })
        
        # CONFIGURACIÓN PARA MOSTRAR MÁS DECIMALES
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        print("\nDataFrame limpio (con más decimales):")
        print(df_limpio)
        
        # ======== FILTRAR DATOS (6s a 20s) ========
        x_data = df_limpio['Tiempo (s)']
        y_data = df_limpio['Voltaje Pico-Pico (V)']
        
        # Crear máscara para el rango 6-20 segundos
        mask_tiempo = (x_data >= 6.0) & (x_data <= 18.0)
        x_filtrado = x_data[mask_tiempo]
        y_filtrado = y_data[mask_tiempo]
        
        print(f"\nDatos originales: {len(x_data)} puntos")
        print(f"Datos filtrados (6-18s): {len(x_filtrado)} puntos")
        
        # Limpiar datos (eliminar NaN/inf) dentro del rango filtrado
        mask_finite = np.isfinite(x_filtrado) & np.isfinite(y_filtrado)
        x_clean = x_filtrado[mask_finite]
        y_clean = y_filtrado[mask_finite]
        
        print(f"Datos limpios para ajuste: {len(x_clean)} puntos")
        
        # Realizar el ajuste SOLO en el rango 6-20s
        try:
            popt, pcov = curve_fit(
                decaimiento_exponencial,
                x_clean,
                y_clean,
                p0=[0, max(y_clean), 0.1, 1.0, 0],  # Valores iniciales ajustados
                maxfev=5000
            )
            
            V0, V, B, W, C = popt
            perr = np.sqrt(np.diag(pcov))
            
            print("\nParámetros del ajuste (6-18s):")
            print(f"V0  = {V0:.6f} ± {perr[0]:.6f}")
            print(f"V   = {V:.6f} ± {perr[1]:.6f}") 
            print(f"B   = {B:.6f} ± {perr[2]:.6f}")
            print(f"W   = {W:.6f} ± {perr[3]:.6f}")
            print(f"C   = {C:.6f} ± {perr[4]:.6f}")
            
            # Calcular valores ajustados
            y_fit = decaimiento_exponencial(x_clean, *popt)
            
            # ======== CORREGIR CÁLCULO DE RESIDUALES ========
            # Residuales simples (diferencia entre dato y ajuste)
            residuales = y_clean - y_fit
            
            # Calcular error estándar de los residuales
            std_residuales = np.std(residuales)
            print(f"\nDesviación estándar de residuales: {std_residuales:.6f} V")
            
            # ======== GRÁFICAS MEJORADAS ========
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            fig.set_facecolor("white")
            
            # Gráfica 1: Ajuste en el rango 6-18s
            ax1.scatter(x_clean[::2], y_clean[::2], color='black', s=30, 
                       alpha=0.7, label='Datos experimentales', edgecolors='black', linewidth=0.5)
            ax1.plot(x_clean, y_fit, color='pink', linewidth=2.5, 
                    label=f'Ajuste: $V_0 (V) + V e^{{-{B:.4f}(s⁻¹)t}} \cos({W:.4f}(rad/s)t + {C:.4f} (rad)$')
            
            ax1.set_xlim(6, 18)  # Forzar límites de 6 a 18 segundos
            ax1.set_title("Mayor amortiguamiento magnético", fontsize=14, fontweight='bold')
            ax1.set_ylabel("Voltaje Pico-Pico (V)", fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Agregar texto con parámetros
            textstr = f'$V_0$ = {V0:.4f} V\n$V$ = {V:.4f} V\n$B$ = {B:.4f} Hz\n$W$ = {W:.4f} rad/s\n$C$ = {C:.4f} rad'
            props = dict(boxstyle='round', facecolor='#FBC6BB', alpha=0.8)
            ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)
            
            # Gráfica 2: Residuales corregidos
            ax2.scatter(x_clean, residuales, color='black', s=20, alpha=0.7)
            ax2.axhline(0, color='pink', linestyle='--', linewidth=2)


            
            # # Agregar bandas de ±1σ y ±2σ
            # ax2.axhline(std_residuales, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'±1σ (±{std_residuales:.4f} V)')
            # ax2.axhline(-std_residuales, color='orange', linestyle=':', linewidth=1, alpha=0.7)
            # ax2.axhline(2*std_residuales, color='purple', linestyle=':', linewidth=1, alpha=0.5, label=f'±2σ (±{2*std_residuales:.4f} V)')
            # ax2.axhline(-2*std_residuales, color='purple', linestyle=':', linewidth=1, alpha=0.5)
            
            ax2.set_xlim(6, 18)  # Mismo rango temporal
            ax2.set_ylabel("Residuales Normalizados (V)", fontsize=12)
            ax2.set_xlabel("Tiempo (s)", fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ajuste_amortiguamiento_mas_cerca.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # ======== ANÁLISIS DE RESIDUALES ========
            print(f"\n--- ANÁLISIS DE RESIDUALES ---")
            print(f"Media de residuales: {np.mean(residuales):.6f} V")
            print(f"Desviación estándar: {std_residuales:.6f} V")
            print(f"Residual máximo: {np.max(residuales):.6f} V")
            print(f"Residual mínimo: {np.min(residuales):.6f} V")
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

            
            # Guardar datos filtrados
            df_filtrado = pd.DataFrame({
                'Tiempo (s)': x_clean,
                'Voltaje Pico-Pico (V)': y_clean,
                'Ajuste (V)': y_fit,
                'Residuales (V)': residuales
            })
            
            df_filtrado.to_csv('datos_ajuste_6-18s.csv', index=False, float_format='%.10f')
            print("\nDatos del ajuste (6-18s) guardados en 'datos_ajuste_6-18s.csv'")
            
        except Exception as e:
            print(f"Error en el ajuste: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No se seleccionó ningún archivo")