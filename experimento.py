import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
from scipy import stats

warnings.filterwarnings("ignore")

from genetico import AlgoritmoGenetico
from gwo import OptimizadorLoboGris, HibridoAGGWO
from fitness import calcular_fitness, simular_distribucion_dosis
from entorno import LIMITES_CLINICOS


# ── Configuración del experimento ─────────────────────────────────────────────
CONFIG = {
    "n_pacientes":    20,     # pacientes del dataset a evaluar
    "n_corridas":     15,     # repeticiones por algoritmo (para estadística)
    "n_individuos":   50,     # tamaño de población/manada
    "n_generaciones": 100,    # generaciones AG / iteraciones GWO
    "n_gen_ag_hibrido": 50,   # generaciones de AG en el híbrido
    "n_iter_gwo_hibrido": 50, # iteraciones de GWO en el híbrido
}

# Colores para las gráficas
COLORES = {
    "AG":      "#3A5FC8",   # azul
    "GWO":     "#F59E0B",   # ámbar
    "Híbrido": "#10B981",   # verde
}


def correr_algoritmo(nombre_algo, paciente, corrida, config):

    semilla = corrida * 100  # semilla diferente por corrida
    
    if nombre_algo == "AG":
        algo = AlgoritmoGenetico(
            n_individuos=config["n_individuos"],
            n_generaciones=config["n_generaciones"],
            semilla=semilla
        )
        pesos, fitness = algo.optimizar(paciente, verbose=False)
        historial = algo.historial_mejor
    
    elif nombre_algo == "GWO":
        algo = OptimizadorLoboGris(
            n_lobos=config["n_individuos"],
            n_iteraciones=config["n_generaciones"],
            semilla=semilla
        )
        pesos, fitness = algo.optimizar(paciente, verbose=False)
        historial = algo.historial_mejor
    
    elif nombre_algo == "Híbrido":
        algo = HibridoAGGWO(
            n_individuos=config["n_individuos"],
            n_gen_ag=config["n_gen_ag_hibrido"],
            n_iter_gwo=config["n_iter_gwo_hibrido"],
            semilla=semilla
        )
        pesos, fitness = algo.optimizar(paciente, verbose=False)
        historial = algo.historial_mejor
    
    return historial, fitness, pesos


def calcular_metricas_clinicas(pesos, paciente):
   
    dosis = simular_distribucion_dosis(pesos, paciente)
    
    metricas = {"D95_tumor": paciente.calcular_D95(dosis)}
    
    for oar, config in LIMITES_CLINICOS.items():
        if config["tipo"] == "Dmean":
            metricas[f"Dmean_{oar}"] = paciente.calcular_Dmean_oar(oar, dosis)
        else:
            metricas[f"Dmax_{oar}"] = paciente.calcular_Dmax_oar(oar, dosis)
    
    return metricas


def graficar_curvas_convergencia(historiales, ruta_salida):
   
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for nombre, listas in historiales.items():
        if not listas:
            continue
        # Rellenar historiales de diferente longitud
        max_len = max(len(h) for h in listas)
        arr = np.array([
            h + [h[-1]] * (max_len - len(h))
            for h in listas
        ])
        
        promedio = arr.mean(axis=0)
        std      = arr.std(axis=0)
        gens     = np.arange(1, max_len + 1)
        
        color = COLORES[nombre]
        ax.plot(gens, promedio, color=color, linewidth=2.5, label=nombre)
        ax.fill_between(gens, promedio - std, promedio + std,
                        alpha=0.15, color=color)
    
    ax.set_xlabel("Generación / Iteración", fontsize=12)
    ax.set_ylabel("Fitness (mejor encontrado)", fontsize=12)
    ax.set_title("Curvas de convergencia — AG vs GWO vs Híbrido", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardada: {ruta_salida}")


def graficar_boxplot(fitness_por_algo, ruta_salida):
    """
    Boxplot que muestra la distribución de fitness final de las 10 corridas.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    nombres = list(fitness_por_algo.keys())
    datos   = [fitness_por_algo[n] for n in nombres]
    colores = [COLORES[n] for n in nombres]
    
    bp = ax.boxplot(datos, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", linewidth=2))
    
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Puntos individuales encima del boxplot
    for i, (d, color) in enumerate(zip(datos, colores)):
        jitter = np.random.uniform(-0.1, 0.1, len(d))
        ax.scatter([i+1+j for j in jitter], d,
                   color=color, alpha=0.6, s=30, zorder=3)
    
    ax.set_xticks(range(1, len(nombres)+1))
    ax.set_xticklabels(nombres, fontsize=12)
    ax.set_ylabel("Fitness final", fontsize=12)
    ax.set_title("Distribución del fitness final — 10 corridas", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardada: {ruta_salida}")


def graficar_dvh(planes, paciente, ruta_salida, dosis_real=True):

    from fitness import simular_distribucion_dosis

    # Estructuras que graficamos (con sus colores). PTV en línea sólida gruesa.
    estructuras = [
        ("PTV70",        "Tumor (PTV)",      "#D62728"),
        ("SpinalCord",   "Médula espinal",   "#1F77B4"),
        ("Brainstem",    "Tronco encefálico","#9467BD"),
        ("LeftParotid",  "Parótida izq.",    "#2CA02C"),
        ("RightParotid", "Parótida der.",    "#17BECF"),
    ]

    # Niveles de dosis (eje X) — 0 a 90 Gy en pasos de 0.5 Gy
    niveles = np.linspace(0, 90, 181)

    # Una columna por algoritmo
    n_algos = len(planes)
    fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 5), sharey=True)
    if n_algos == 1:
        axes = [axes]

    for ax, (nombre, pesos) in zip(axes, planes.items()):
        dosis = simular_distribucion_dosis(pesos, paciente)

        for clave, etiqueta, color in estructuras:
            mask = paciente.mascaras.get(clave)
            if mask is None or mask.sum() == 0:
                continue
            d_struct = dosis[mask]
            # % volumen que recibe ≥ cada nivel de dosis
            vol_pct = [(d_struct >= lv).mean() * 100 for lv in niveles]
            lw = 3 if clave == "PTV70" else 1.8
            ax.plot(niveles, vol_pct, color=color, linewidth=lw, label=etiqueta)

        # Línea de la dosis real del oncólogo (solo en el primer subplot)
        if dosis_real and ax is axes[0] and getattr(paciente, "dosis_real", None) is not None:
            d_real_tumor = paciente.dosis_real[paciente.mascaras["PTV70"]]
            if len(d_real_tumor) > 0:
                vol_pct = [(d_real_tumor >= lv).mean() * 100 for lv in niveles]
                ax.plot(niveles, vol_pct, color="#D62728", linewidth=2,
                        linestyle="--", alpha=0.7, label="PTV (plan real)")

        ax.axvline(70, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(70.5, 5, "70 Gy", fontsize=8, color="gray")

        ax.set_xlabel("Dosis (Gy)", fontsize=11)
        ax.set_title(nombre, fontsize=12, fontweight="bold", color=COLORES.get(nombre, "black"))
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Volumen ≥ dosis (%)", fontsize=11)
    axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.suptitle(f"Histograma Dosis-Volumen — {paciente.id}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardada: {ruta_salida}")


def graficar_metricas_clinicas(metricas_df, ruta_salida):
    """
    Barras comparando D95 del tumor para los 3 algoritmos.
    Incluye línea del límite clínico.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ── Subplot 1: D95 tumor ──────────────────────────────────────────────
    ax = axes[0]
    nombres = metricas_df["Algoritmo"].unique()
    d95_vals = [metricas_df[metricas_df["Algoritmo"]==n]["D95_tumor"].mean() for n in nombres]
    d95_stds = [metricas_df[metricas_df["Algoritmo"]==n]["D95_tumor"].std() for n in nombres]
    
    bars = ax.bar(nombres, d95_vals, color=[COLORES[n] for n in nombres],
                  alpha=0.8, width=0.5)
    ax.errorbar(nombres, d95_vals, yerr=d95_stds, fmt="none",
                color="black", capsize=5, linewidth=1.5)
    ax.axhline(70, color="red", linestyle="--", linewidth=1.5,
               label="Meta clínica (70 Gy)")
    ax.set_ylabel("D95 Tumor (Gy)", fontsize=11)
    ax.set_title("Cobertura del tumor", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(d95_vals)*1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # ── Subplot 2: Dmean médula espinal ───────────────────────────────────
    ax = axes[1]
    col = "Dmax_SpinalCord"
    if col in metricas_df.columns:
        vals = [metricas_df[metricas_df["Algoritmo"]==n][col].mean() for n in nombres]
        stds = [metricas_df[metricas_df["Algoritmo"]==n][col].std() for n in nombres]
        
        ax.bar(nombres, vals, color=[COLORES[n] for n in nombres], alpha=0.8, width=0.5)
        ax.errorbar(nombres, vals, yerr=stds, fmt="none",
                    color="black", capsize=5, linewidth=1.5)
        ax.axhline(45, color="red", linestyle="--", linewidth=1.5,
                   label="Límite clínico (45 Gy)")
        ax.set_ylabel("Dmax Médula Espinal (Gy)", fontsize=11)
        ax.set_title("Protección de médula espinal", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    plt.suptitle("Métricas clínicas por algoritmo", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Guardada: {ruta_salida}")


def prueba_wilcoxon(datos_a, datos_b, nombre_a, nombre_b):
   
    try:
        a = np.asarray(datos_a, dtype=float)
        b = np.asarray(datos_b, dtype=float)
        # Quitar pares donde la diferencia es exactamente 0 (no aportan rango)
        dif = a - b
        if np.all(dif == 0):
            return {
                "comparacion": f"{nombre_a} vs {nombre_b}",
                "estadistico": 0.0, "p_value": 1.0,
                "significativo (p<0.05)": "❌ NO",
                "media_a": round(a.mean(), 4), "media_b": round(b.mean(), 4),
                "ganador": "empate",
            }
        stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        significativo = "✅ SÍ" if p < 0.05 else "❌ NO"
        return {
            "comparacion": f"{nombre_a} vs {nombre_b}",
            "estadistico": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "significativo (p<0.05)": significativo,
            "media_a": round(float(a.mean()), 4),
            "media_b": round(float(b.mean()), 4),
            "ganador": nombre_a if a.mean() > b.mean() else nombre_b,
            "n_pares": int(len(a)),
        }
    except Exception as e:
        return {"comparacion": f"{nombre_a} vs {nombre_b}", "error": str(e)}


def correr_experimento_completo(pacientes, config, ruta_salida="resultados"):

    # Crear carpetas de salida
    os.makedirs(f"{ruta_salida}/tablas", exist_ok=True)
    os.makedirs(f"{ruta_salida}/graficas", exist_ok=True)
    os.makedirs(f"{ruta_salida}/estadistica", exist_ok=True)
    
    algoritmos      = ["AG", "GWO", "Híbrido"]
    fitness_final   = {a: [] for a in algoritmos}   # aplanado: pac×corrida
    historiales     = {a: [] for a in algoritmos}   # una curva por paciente (corrida 0)
    metricas_rows   = []
    # Para Wilcoxon pareado: fitness por (paciente, corrida) — orden idéntico entre algos
    fitness_pareado = {a: [] for a in algoritmos}
    # Mejor plan global por algoritmo (para DVH)
    mejor_plan = {a: {"fitness": -np.inf, "pesos": None, "paciente": None}
                  for a in algoritmos}
    
    n_pac  = min(len(pacientes), config["n_pacientes"])
    n_cor  = config["n_corridas"]
    total  = n_pac * len(algoritmos) * n_cor
    cuenta = 0
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO COMPLETO")
    print(f"{n_pac} pacientes × {len(algoritmos)} algoritmos × {n_cor} corridas")
    print(f"Total evaluaciones: {total}")
    print(f"{'='*60}\n")
    
    for i, paciente in enumerate(pacientes[:n_pac]):
        print(f"\nPaciente {i+1}/{n_pac}: {paciente.id}")
        
        for algo in algoritmos:   # Comparamos los 3 algoritmos en el MISMO paciente para que sea apples-to-apples
            fits_pac = []
            
            for corrida in range(n_cor):
                cuenta += 1
                pct = 100 * cuenta / total
                print(f"  [{pct:5.1f}%] {algo} corrida {corrida+1}/{n_cor}...", end=" ") #no  salta de línea para mostrar el fitness
                
                historial, fitness, pesos = correr_algoritmo(algo, paciente, corrida, config)

                fits_pac.append(fitness) #imprime el promedio del fitness por paciente para cada algoritmo
                fitness_final[algo].append(fitness) #fitness final de cada corrida para boxplot tabla de resumen
                fitness_pareado[algo].append(fitness)   # pareado por (paciente, corrida)

                # Guardar historial de solo una corrida por paciente para la gráfica
                if corrida == 0:
                    historiales[algo].append(historial)

                # Actualizar el mejor plan global de este algoritmo (para DVH)
                if fitness > mejor_plan[algo]["fitness"]: #si este plan es mejor del que tenemos guardado, lo actualizamos
                    mejor_plan[algo] = {
                        "fitness": float(fitness), #guardamos el fitness como float para evitar problemas de serialización
                        "pesos": np.asarray(pesos).copy(), #copiamos los pesos para evitar problemas de referencia mutable
                        "paciente": paciente, #guardamos el paciente para luego graficar el DVH del mejor plan sobre ese mismo paciente
                    }

                # Métricas clínicas del mejor plan de esta corrida
                metricas = calcular_metricas_clinicas(pesos, paciente)
                metricas["Paciente"]   = paciente.id
                metricas["Algoritmo"]  = algo
                metricas["Corrida"]    = corrida
                metricas["fitness"]    = fitness
                metricas_rows.append(metricas)

                print(f"fitness={fitness:.4f}")
            
            print(f"  → {algo}: mean={np.mean(fits_pac):.4f} ± {np.std(fits_pac):.4f}") #la desviación estándar, si es alta, indica que el algoritmo es inestable (depende mucho de la semilla) 
                                                                                        #y si es baja, indica que es consistente (siempre encuentra soluciones similares)
    
    # ── Guardar tabla de fitness resumen ──────────────────────────────────────
    print("\n\nGenerando tablas y gráficas...")
    
    filas_resumen = []
    for algo in algoritmos:
        vals = fitness_final[algo]
        filas_resumen.append({
            "Algoritmo": algo,
            "Media":     round(np.mean(vals), 4),
            "Std":       round(np.std(vals),  4),
            "Mediana":   round(np.median(vals),4),
            "Min":       round(np.min(vals),  4),
            "Max":       round(np.max(vals),  4),
            "N":         len(vals)
        })
    
    df_resumen = pd.DataFrame(filas_resumen)
    df_resumen.to_csv(f"{ruta_salida}/tablas/fitness_resumen.csv", index=False)
    print(f"  ✓ Guardada: {ruta_salida}/tablas/fitness_resumen.csv")
    print(df_resumen.to_string(index=False))
    
    # ── Guardar métricas clínicas ──────────────────────────────────────────────
    df_metricas = pd.DataFrame(metricas_rows)
    df_metricas.to_csv(f"{ruta_salida}/tablas/metricas_clinicas.csv", index=False)
    print(f"  ✓ Guardada: {ruta_salida}/tablas/metricas_clinicas.csv")
    
    # ── Generar gráficas ──────────────────────────────────────────────────────
    graficar_curvas_convergencia(historiales,
        f"{ruta_salida}/graficas/curvas_convergencia.png")
    graficar_boxplot(fitness_final,
        f"{ruta_salida}/graficas/boxplot_fitness.png")
    graficar_metricas_clinicas(df_metricas,
        f"{ruta_salida}/graficas/barras_metricas.png")
    
    # ── DVH del mejor plan global por algoritmo ──────────────────────────────
    # Tomamos el paciente del mejor plan del Híbrido (o cualquiera disponible)
    # y comparamos los 3 planes sobre el MISMO paciente para que sea apples-to-apples.
    pac_dvh = (mejor_plan["Híbrido"]["paciente"]
               or mejor_plan["AG"]["paciente"]
               or mejor_plan["GWO"]["paciente"])
    if pac_dvh is not None:
        planes_para_dvh = {}
        for a in algoritmos:
            if mejor_plan[a]["pesos"] is not None:
                planes_para_dvh[a] = mejor_plan[a]["pesos"]
        if planes_para_dvh:
            graficar_dvh(planes_para_dvh, pac_dvh,
                         f"{ruta_salida}/graficas/dvh_mejor_plan.png")

    # ── Pruebas estadísticas Wilcoxon (PAREADAS por paciente×corrida) ────────
    # Importante: usar fitness_pareado (mismo orden entre algoritmos) para que
    # cada par compare al mismo (paciente, corrida) y aísle el efecto del algoritmo.
    comparaciones = [
        ("AG",  "GWO",     fitness_pareado["AG"],  fitness_pareado["GWO"]),
        ("AG",  "Híbrido", fitness_pareado["AG"],  fitness_pareado["Híbrido"]),
        ("GWO", "Híbrido", fitness_pareado["GWO"], fitness_pareado["Híbrido"]),
    ]

    resultados_wilcoxon = []
    for a, b, da, db in comparaciones:
        if len(da) == len(db) and len(da) > 0:
            res = prueba_wilcoxon(da, db, a, b)
            resultados_wilcoxon.append(res)
    
    df_wilcoxon = pd.DataFrame(resultados_wilcoxon)
    df_wilcoxon.to_csv(f"{ruta_salida}/estadistica/wilcoxon.csv", index=False)
    
    with open(f"{ruta_salida}/estadistica/wilcoxon.txt", "w") as f:
        f.write("PRUEBAS DE WILCOXON — COMPARACIÓN ESTADÍSTICA\n")
        f.write("="*60 + "\n\n")
        f.write("Un p-value < 0.05 indica que la diferencia entre\n")
        f.write("los algoritmos es estadísticamente significativa.\n\n")
        f.write(df_wilcoxon.to_string(index=False))
    
    print(f"\n  Pruebas de Wilcoxon:")
    for r in resultados_wilcoxon:
        if "error" not in r:
            print(f"    {r['comparacion']}: p={r['p_value']} {r['significativo (p<0.05)']}")
    
    print(f"\n{'='*60}")
    print(f"✓ EXPERIMENTO COMPLETO")
    print(f"  Resultados en: {ruta_salida}/")
    print(f"{'='*60}")
    
    return df_resumen, df_metricas, df_wilcoxon


# ── EJECUCIÓN PRINCIPAL ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("experimento.py — Optimización de Radioterapia con OpenKBP")
    print("="*60)
    print("Cargando pacientes reales del dataset OpenKBP...")
    print("Asegúrate de tener clonado: github.com/ababier/open-kbp")
    print()

    # Siempre usamos los datos reales del dataset OpenKBP — los pacientes
    # simulados solo servían para pruebas internas y se han eliminado.
    from entorno import cargar_pacientes

    # ── Configuración del experimento ────────────────────────────────────────
    # MEDIA (recomendada para la propuesta): ~40 s en máquina estándar.
    # Para escalar al experimento completo cambia los valores comentados ↓.
    config_experimento = {
    "n_pacientes":        50,
    "n_corridas":         10,
    "n_individuos":       30,
    "n_generaciones":     100,
    "n_gen_ag_hibrido":   60,
    "n_iter_gwo_hibrido": 60,
}

    # ── ¿QUÉ PACIENTES USAR? ──────────────────────────────────────────────────
    # Cambia SELECCION para probar con distintos datos y ver resultados distintos.
    # Opciones disponibles:
    #   "primeros"   → primeros N de test-pats (pt_241...) — comportamiento original
    #   "otros"      → N distintos del mismo conjunto, saltando los primeros 50
    #   "train"      → N pacientes del conjunto de entrenamiento (pt_1...)
    #   "validation" → N pacientes del conjunto de validación (pt_201...)
    #   "aleatorio"  → N pacientes al azar de test-pats (reproducible con semilla)
    #   "manual"     → una lista de IDs específicos que tú elijas
    SELECCION = "primeros"

    n = config_experimento["n_pacientes"]
    if SELECCION == "primeros":
        pacientes = cargar_pacientes("open-kbp", n_pacientes=n, subset="test-pats")
    elif SELECCION == "otros":
        pacientes = cargar_pacientes("open-kbp", n_pacientes=n, subset="test-pats",
                                     offset=50)
    elif SELECCION == "train":
        pacientes = cargar_pacientes("open-kbp", n_pacientes=n, subset="train-pats")
    elif SELECCION == "validation":
        pacientes = cargar_pacientes("open-kbp", n_pacientes=n, subset="validation-pats")
    elif SELECCION == "aleatorio":
        pacientes = cargar_pacientes("open-kbp", n_pacientes=n, subset="test-pats",
                                     aleatorio=True, semilla=2026)
    elif SELECCION == "manual":
        # Edita esta lista con los pacientes que quieras
        pacientes = cargar_pacientes("open-kbp", subset="test-pats",
                                     ids=["pt_241", "pt_270", "pt_300", "pt_320", "pt_340"])
    else:
        raise ValueError(f"SELECCION desconocida: {SELECCION}")

    # Ajustar n_pacientes al número realmente cargado (por si 'manual' difiere)
    config_experimento["n_pacientes"] = len(pacientes)

    df_resumen, df_metricas, df_wilcoxon = correr_experimento_completo(
        pacientes,
        config_experimento,
        ruta_salida="resultados"
    )

    print("\nPara regenerar el reporte HTML con los nuevos resultados:")
    print("    python generar_reporte.py")
