"""
fitness.py
----------
La función que califica qué tan bueno es un plan de radioterapia.
Es el corazón de todo el proyecto — AG y GWO la llaman miles de veces.

Fórmula:
    fitness = α·D95_tumor  −  β·Dmean_OARs  −  γ·penalización_sobredosis

Cuanto mayor sea el fitness, mejor es el plan.
"""

import numpy as np


# ── Pesos de la función de fitness ────────────────────────────────────────────
# Puedes ajustarlos para cambiar la prioridad entre cubrir el tumor vs proteger órganos
ALPHA = 1.0   # peso de cobertura del tumor (D95) — más alto = más importante cubrir tumor
BETA  = 0.5   # peso de daño a órganos — más alto = más penaliza daño
GAMMA = 2.0   # peso de penalización por sobredosis — muy alto para evitar daños críticos

# Dosis objetivo para el tumor
DOSIS_OBJETIVO_TUMOR = 70.0  # Gy — meta clínica estándar para cabeza y cuello


def _construir_kernels_haces(paciente, n_haces=9, sigma_xy=8.0, sigma_z=12.0,
                             radio=15.0):

    if hasattr(paciente, "_kernels") and paciente._kernels is not None:
        return paciente._kernels, paciente._centro_tumor

    sz = paciente.mascaras["PTV70"].shape
    coords_tumor = np.argwhere(paciente.mascaras["PTV70"])
    if len(coords_tumor) == 0:
        paciente._kernels = None
        paciente._centro_tumor = None
        return None, None

    centro = coords_tumor.mean(axis=0)
    angulos = np.linspace(0, 2 * np.pi, n_haces, endpoint=False)

    x = np.arange(sz[0]); y = np.arange(sz[1]); z = np.arange(sz[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    kernels = []
    for ang in angulos:
        cx = centro[0] + np.cos(ang) * radio
        cy = centro[1] + np.sin(ang) * radio
        cz = centro[2]
        g = np.exp(
            -((xx - cx) ** 2 / (2 * sigma_xy ** 2)
              + (yy - cy) ** 2 / (2 * sigma_xy ** 2)
              + (zz - cz) ** 2 / (2 * sigma_z  ** 2))
        )
        kernels.append(g.astype(np.float32))

    paciente._kernels = kernels
    paciente._centro_tumor = centro
    return kernels, centro


def simular_distribucion_dosis(pesos, paciente, modo="anclado"):

    sz = paciente.mascaras["PTV70"].shape
    pesos = np.clip(np.asarray(pesos, dtype=float), 0, 1)

    # Si no hay tumor, no hay nada que simular
    if paciente.mascaras["PTV70"].sum() == 0:
        return np.zeros(sz, dtype=np.float32)

    # ── Componente: suma ponderada de kernels gaussianos (haces) ────────────
    kernels, _ = _construir_kernels_haces(paciente)
    suma_kernels = np.zeros(sz, dtype=np.float32)
    for w, k in zip(pesos, kernels):
        if w >= 0.01:
            suma_kernels += w * k

    # Normalización: que peso medio = 1 entregue ~DOSIS_OBJETIVO_TUMOR en el tumor.
    # Tomamos el percentil 95 del kernel-sum dentro del tumor como referencia.
    en_tumor = suma_kernels[paciente.mascaras["PTV70"]]
    if len(en_tumor) > 0 and en_tumor.max() > 1e-6:
        # Escala para que el percentil 95 del tumor reciba la dosis objetivo con peso medio=1
        peso_medio = max(pesos.mean(), 0.01)
        ref = np.percentile(en_tumor, 95)
        escala = (DOSIS_OBJETIVO_TUMOR / ref) if ref > 1e-6 else 0.0
        kern_dosis = suma_kernels * escala
    else:
        kern_dosis = suma_kernels * 0.0

    # ── Modo anclado: mezclar con la dosis real del oncólogo si existe ──────
    dosis_real = getattr(paciente, "dosis_real", None)
    if modo == "anclado" and dosis_real is not None and np.any(dosis_real > 0):
        # El peso medio modula cuánta dosis "global" se entrega (plan más agresivo
        # o más conservador). Cuando todos los pesos son altos, la dosis se acerca
        # a la del oncólogo; cuando son bajos, baja proporcionalmente.
        alpha_mix = 0.6   # peso del componente anatómico real
        peso_medio = pesos.mean()
        dosis = (alpha_mix * dosis_real.astype(np.float32) * peso_medio
                 + (1 - alpha_mix) * kern_dosis)
        return dosis

    return kern_dosis


def _construir_cache_roi(paciente):
 
    if getattr(paciente, "_roi_cache", None) is not None:
        return paciente._roi_cache

    # Unión: tumor + todos los OAR conocidos
    estruct = ["PTV70", "SpinalCord", "RightParotid", "LeftParotid",
               "Mandible", "Brainstem", "Larynx"]
    roi = np.zeros_like(paciente.mascaras["PTV70"], dtype=bool)
    for e in estruct:
        m = paciente.mascaras.get(e)
        if m is not None:
            roi |= m

    # Construir kernels gaussianos completos (una sola vez)
    kernels_3d, _ = _construir_kernels_haces(paciente)
    if kernels_3d is None:
        paciente._roi_cache = None
        return None

    # Extraer kernels solo en la ROI → (9, n_roi)
    kernels_at_roi = np.stack([k[roi] for k in kernels_3d]).astype(np.float32)

    # Submáscaras: bool (n_roi,) indicando qué vóxeles de la ROI pertenecen a cada estructura
    submasks = {e: paciente.mascaras[e][roi]
                for e in estruct if paciente.mascaras.get(e) is not None
                and paciente.mascaras[e].sum() > 0}

    # Dosis real recortada a ROI (si existe)
    dr = getattr(paciente, "dosis_real", None)
    dosis_real_at_roi = dr[roi].astype(np.float32) if dr is not None else None

    # Escala: con pesos=1, el percentil 95 del tumor debe dar la dosis objetivo
    suma_kern_tumor = kernels_at_roi.sum(axis=0)[submasks["PTV70"]]
    if len(suma_kern_tumor) > 0 and suma_kern_tumor.max() > 1e-6:
        ref_p95 = float(np.percentile(suma_kern_tumor, 95))
    else:
        ref_p95 = 1.0

    paciente._roi_cache = {
        "kernels_at_roi": kernels_at_roi,
        "submasks": submasks,
        "dosis_real_at_roi": dosis_real_at_roi,
        "ref_p95": ref_p95,
    }
    return paciente._roi_cache


def calcular_fitness(pesos, paciente, alpha=ALPHA, beta=BETA, gamma=GAMMA,
                     verbose=False):
    # Asegurar que los pesos estén en [0,1]
    pesos = np.clip(np.asarray(pesos, dtype=float), 0, 1)

    # ── Paso 1: Simular dosis usando el cache de ROI (rápido) ──────────────
    cache = _construir_cache_roi(paciente)
    if cache is None:
        return -1e6  # paciente sin tumor → penalización fuerte

    # Componente kernel: combinación lineal de los 9 haces, solo en ROI
    # (pesos @ kernels_at_roi) tiene forma (n_roi,) — ~6 500 vóxeles, no 2 M
    suma_kern = pesos.astype(np.float32) @ cache["kernels_at_roi"]
    escala = (DOSIS_OBJETIVO_TUMOR / cache["ref_p95"]) if cache["ref_p95"] > 1e-6 else 0.0
    kern_dosis = suma_kern * escala

    # Anclar a la dosis real del oncólogo (si existe en este paciente)
    if cache["dosis_real_at_roi"] is not None:
        alpha_mix = 0.6
        peso_medio = float(pesos.mean())
        dosis_roi = (alpha_mix * cache["dosis_real_at_roi"] * peso_medio
                     + (1 - alpha_mix) * kern_dosis)
    else:
        dosis_roi = kern_dosis

    submasks = cache["submasks"]

    # ── Paso 2: D95 del tumor ─────────────────────────────────────────────
    d_tumor = dosis_roi[submasks["PTV70"]]
    D95 = float(np.percentile(d_tumor, 5)) if len(d_tumor) else 0.0
    D95_norm = D95 / DOSIS_OBJETIVO_TUMOR

    # ── Paso 3: Daño promedio en órganos en riesgo ────────────────────────
    from entorno import LIMITES_CLINICOS
    danio_total, n_oars = 0.0, 0
    for oar, config in LIMITES_CLINICOS.items():
        m = submasks.get(oar)
        if m is None or m.sum() == 0:
            continue
        d_oar = dosis_roi[m]
        val = float(d_oar.mean()) if config["tipo"] == "Dmean" else float(d_oar.max())
        danio_total += val / config["limite"]
        n_oars += 1
    danio_promedio_norm = danio_total / n_oars if n_oars > 0 else 0.0

    # ── Paso 4: Penalización por sobredosis crítica (médula) ──────────────
    penalizacion = 0.0
    m_sp = submasks.get("SpinalCord")
    if m_sp is not None and m_sp.sum() > 0:
        dmax_medula = float(dosis_roi[m_sp].max())
        lim = LIMITES_CLINICOS["SpinalCord"]["limite"]
        if dmax_medula > lim:
            penalizacion = (dmax_medula - lim) / lim

    # ── Paso 5: Fitness final ─────────────────────────────────────────────
    fitness = (alpha * D95_norm) - (beta * danio_promedio_norm) - (gamma * penalizacion)
    
    if verbose:
        print(f"\n  Desglose del fitness:")
        print(f"    D95 tumor:        {D95:.1f} Gy  → término = {alpha * D95_norm:.4f}")
        print(f"    Daño OARs (norm): {danio_promedio_norm:.4f} → término = {beta * danio_promedio_norm:.4f}")
        print(f"    Penalización:     {penalizacion:.4f} → término = {gamma * penalizacion:.4f}")
        print(f"    FITNESS TOTAL:    {fitness:.4f}")
    
    return float(fitness)


# ── Prueba rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from entorno import LIMITES_CLINICOS
    
    print("Probando fitness.py con paciente simulado")
    print("=" * 50)
    
    # Crear paciente simulado
    class PacienteSimulado:
        def __init__(self):
            self.id = "pt_simulado"
            sz = (32, 32, 32)
            mask = np.zeros(sz, dtype=bool)
            mask[12:20, 12:20, 12:20] = True
            self.mascaras = {"PTV70": mask}
            for oar in ["SpinalCord","RightParotid","LeftParotid","Mandible","Brainstem","Larynx"]:
                m = np.zeros(sz, dtype=bool)
                m[2:5, 2:5, :] = True
                self.mascaras[oar] = m
        
        def calcular_D95(self, dosis):
            d = dosis[self.mascaras["PTV70"]]
            return float(np.percentile(d, 5)) if len(d) else 0.0
        def calcular_Dmean_oar(self, oar, dosis):
            d = dosis[self.mascaras[oar]]
            return float(np.mean(d)) if len(d) else 0.0
        def calcular_Dmax_oar(self, oar, dosis):
            d = dosis[self.mascaras[oar]]
            return float(np.max(d)) if len(d) else 0.0
    
    paciente = PacienteSimulado()
    
    # Plan malo: todos los pesos iguales
    pesos_malos = np.ones(9) * 0.3
    f1 = calcular_fitness(pesos_malos, paciente, verbose=True)
    print(f"\nPlan malo (pesos uniformes bajos): fitness = {f1:.4f}")
    
    # Plan bueno: pesos optimizados manualmente
    pesos_buenos = np.array([0.9, 0.8, 0.7, 0.9, 0.8, 0.7, 0.9, 0.8, 0.7])
    f2 = calcular_fitness(pesos_buenos, paciente, verbose=True)
    print(f"\nPlan bueno (pesos altos): fitness = {f2:.4f}")
    
    print(f"\n✓ El plan bueno {'supera' if f2>f1 else 'no supera'} al malo ({f2:.3f} vs {f1:.3f})")
