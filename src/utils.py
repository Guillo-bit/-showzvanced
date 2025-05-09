# utils.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = '..reports//figures/final'
os.makedirs(FIG_DIR, exist_ok=True)

def simulate_romi_from_real_values(df, ltv_col='LTV_180', cac_col='CAC_source_30', source_col='marketing_channel'):
    try:
        # Usamos directamente las columnas reales de LTV y CAC
        df = df.dropna(subset=[ltv_col, cac_col])  # filtramos para evitar NaNs
        df['ltv_pred'] = df[ltv_col]
        df['cac_pred'] = df[cac_col]
        df['romi'] = (df['ltv_pred'] - df['cac_pred']) / df['cac_pred']

        # Agrupación por canal de marketing
        source_romi = df.groupby(source_col)['romi'].mean()
        source_cost = df.groupby(source_col)['cac_pred'].sum()

        # Escenario base
        base_romi = (df['ltv_pred'].sum() - df['cac_pred'].sum()) / df['cac_pred'].sum()
        print(f"[BASELINE ROMI]: {base_romi:.4f}")

        # Escenario A: +10% en mejor fuente
        best_source = source_romi.idxmax()
        boost_costs = source_cost.copy()
        boost_costs[best_source] *= 1.1
        total_boost = boost_costs.sum()
        weights_a = boost_costs / total_boost
        simulated_ltv_a = (weights_a * df['ltv_pred'].sum()).sum()
        simulated_cac_a = total_boost
        romi_a = (simulated_ltv_a - simulated_cac_a) / simulated_cac_a

        # Escenario B: redistribución proporcional según ROMI histórico
        weights_b = source_romi / source_romi.sum()
        total_budget = source_cost.sum()
        simulated_ltv_b = (weights_b * df['ltv_pred'].sum()).sum()
        romi_b = (simulated_ltv_b - total_budget) / total_budget

        # Serie resumen
        romis = pd.Series({
            'Baseline': base_romi,
            f'+10% en fuente {best_source}': romi_a,
            'Redistribución proporcional': romi_b
        })

        # Gráfico
        plt.figure(figsize=(8, 5))
        romis.plot(kind='barh', color='teal')
        plt.title('Simulación de ROMI')
        plt.xlabel('ROMI')
        plt.tight_layout()
        plt.savefig(f'{FIG_DIR}/romi_simulation.png')
        plt.close()

        best_strategy = romis.idxmax()
        print(f'[RECOMENDACIÓN] Mejor estrategia: {best_strategy} (ROMI esperado: {romis.max():.4f})')

    except Exception as e:
        print(f"[ROMI SIMULATION ERROR]: {e}")
