
import os
import sys
import numpy as np
import pandas as pd

from entorno import cargar_pacientes, LIMITES_CLINICOS
from fitness import simular_distribucion_dosis


def verificar_dosis(paciente, dosis, etiqueta=""):
    """Imprime una tabla clínica con ✅/❌ por cada estructura."""
    print(f"\n  {'─'*65}")
    print(f"  {etiqueta}")
    print(f"  {'─'*65}")
    print(f"  {'Estructura':<22} {'Métrica':<8} {'Valor':>9}  {'Límite':>9}  Estado")
    print(f"  {'─'*65}")

    # Tumor — D95 ≥ 70 Gy
    d95 = paciente.calcular_D95(dosis)
    ok = "✅ PASA" if d95 >= 70 else "❌ FALLA"
    print(f"  {'PTV (tumor)':<22} {'D95':<8} {d95:>7.1f} Gy  {'≥70.0 Gy':>9}  {ok}")

    n_pasa, n_falla = (1 if d95 >= 70 else 0), (0 if d95 >= 70 else 1)

    # OARs
    for oar, cfg in LIMITES_CLINICOS.items():
        if paciente.mascaras.get(oar) is None or paciente.mascaras[oar].sum() == 0:
            continue
        if cfg["tipo"] == "Dmean":
            val = paciente.calcular_Dmean_oar(oar, dosis)
        else:
            val = paciente.calcular_Dmax_oar(oar, dosis)
        cumple = val <= cfg["limite"]
        ok = "✅ PASA" if cumple else "❌ FALLA"
        n_pasa += int(cumple); n_falla += int(not cumple)
        lim_str = f"≤{cfg['limite']:.0f}.0 Gy"
        print(f"  {oar:<22} {cfg['tipo']:<8} {val:>7.1f} Gy  {lim_str:>9}  {ok}")

    print(f"  {'─'*65}")
    print(f"  Resumen: {n_pasa} criterios cumplidos, {n_falla} fallidos")
    return n_pasa, n_falla


def encontrar_mejores_planes(ruta_csv):

    if not os.path.exists(ruta_csv):
        return None
    return pd.read_csv(ruta_csv)


def main():
    print("=" * 70)
    print("  VERIFICACIÓN CLÍNICA DE LOS PLANES")
    print("  Compara los planes contra los límites QUANTEC/DAHANCA")
    print("=" * 70)

    # Filtrar paciente si se pidió uno específico
    pac_filtro = sys.argv[1] if len(sys.argv) > 1 else None

    n_pacientes = 1 if pac_filtro else 5  # demo: primeros 5 si no se filtra
    pacientes = cargar_pacientes("open-kbp", n_pacientes=n_pacientes)

    if pac_filtro:
        pacientes = [p for p in pacientes if p.id == pac_filtro]
        if not pacientes:
            print(f"\n⚠ No se encontró {pac_filtro}")
            return

    # Cargar las métricas del experimento (si existen)
    df_metricas = encontrar_mejores_planes("resultados/tablas/metricas_clinicas.csv")

    for paciente in pacientes:
        print(f"\n{'═'*70}")
        print(f"  PACIENTE: {paciente.id}")
        print(f"  Tumor PTV70: {int(paciente.mascaras['PTV70'].sum()):,} vóxeles")
        print(f"{'═'*70}")

        # ── 1. Verificar el PLAN REAL del oncólogo (referencia humana) ─────
        verificar_dosis(paciente, paciente.dosis_real,
                        "REFERENCIA HUMANA — Plan del oncólogo (de OpenKBP)")

        # ── 2. Verificar el plan de pesos uniformes = 1.0 (baseline ingenuo) ─
        pesos_baseline = np.ones(9)
        dosis_baseline = simular_distribucion_dosis(pesos_baseline, paciente)
        verificar_dosis(paciente, dosis_baseline,
                        "BASELINE — Plan ingenuo (todos los haces a intensidad 1)")

        # ── 3. Verificar el "mejor plan" según los resultados experimentales ─
        if df_metricas is not None:
            df_pac = df_metricas[df_metricas["Paciente"] == paciente.id]
            if len(df_pac) > 0:
                print(f"\n  {'─'*65}")
                print(f"  RESULTADOS EXPERIMENTALES (de metricas_clinicas.csv)")
                print(f"  {'─'*65}")
                for algo in ["AG", "GWO", "Híbrido"]:
                    sub = df_pac[df_pac["Algoritmo"] == algo]
                    if len(sub) == 0:
                        continue
                    idx_mejor = sub["fitness"].idxmax()
                    row = sub.loc[idx_mejor]
                    d95 = row["D95_tumor"]
                    dmax_sp = row.get("Dmax_SpinalCord", 0)
                    dmean_lp = row.get("Dmean_LeftParotid", 0)
                    dmean_rp = row.get("Dmean_RightParotid", 0)

                    ok_d95 = "✅" if d95 >= 70 else "❌"
                    ok_sp  = "✅" if dmax_sp <= 45 else "❌"
                    ok_lp  = "✅" if dmean_lp <= 26 else "❌"
                    ok_rp  = "✅" if dmean_rp <= 26 else "❌"

                    print(f"  {algo:<10}  D95={d95:5.1f}{ok_d95}  "
                          f"Dmax_SC={dmax_sp:5.1f}{ok_sp}  "
                          f"Dmean_LP={dmean_lp:5.1f}{ok_lp}  "
                          f"Dmean_RP={dmean_rp:5.1f}{ok_rp}  "
                          f"fitness={row['fitness']:.4f}")

    print(f"\n{'═'*70}")
    print("  GUÍA DE LECTURA")
    print(f"{'═'*70}")
    print("""
  ✅ PASA  → la métrica cumple el límite clínico de QUANTEC/DAHANCA
  ❌ FALLA → la métrica excede el límite (no clínicamente aceptable)

  Notas:
  • Que el PLAN REAL del oncólogo también muestre algunos ❌ es NORMAL en
    pacientes complejos — el oncólogo a veces acepta exceder un límite si
    es la única forma de cubrir un tumor grande. Es un compromiso clínico.
  • Que el BASELINE muestre muchos ❌ es esperado — es un plan ingenuo.
  • Los resultados de AG/GWO/Híbrido deben estar entre el oncólogo y el
    baseline: mejores que ingenuo, comparables al oncólogo.
""")


if __name__ == "__main__":
    main()
