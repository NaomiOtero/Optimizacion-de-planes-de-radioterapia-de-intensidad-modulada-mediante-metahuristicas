

import numpy as np
import pandas as pd
import os

GRID_SIZE = (128, 128, 128)

# Nombre del PTV principal (mayor dosis → el tumor más agresivo)
PTV_PRINCIPAL = "PTV70"

# Todos los archivos de estructura que puede tener un paciente
ESTRUCTURAS = [
    "PTV70",
    "PTV63",
    "PTV56",
    "SpinalCord",
    "RightParotid",
    "LeftParotid",
    "Mandible",
    "Brainstem",
    "Larynx",
]

# Límites clínicos internacionales (DAHANCA/QUANTEC)
LIMITES_CLINICOS = {
    "SpinalCord":   {"tipo": "Dmax",  "limite": 45.0},
    "RightParotid": {"tipo": "Dmean", "limite": 26.0},
    "LeftParotid":  {"tipo": "Dmean", "limite": 26.0},
    "Mandible":     {"tipo": "Dmax",  "limite": 70.0},
    "Brainstem":    {"tipo": "Dmax",  "limite": 54.0},
    "Larynx":       {"tipo": "Dmean", "limite": 45.0},
}


def cargar_csv_a_matriz(ruta_csv, dtype=float):

    if not os.path.exists(ruta_csv):
        return np.zeros(GRID_SIZE, dtype=dtype)

    if os.path.getsize(ruta_csv) < 10:
        return np.zeros(GRID_SIZE, dtype=dtype)

    try:
        df = pd.read_csv(ruta_csv, index_col=0)

        if df.empty:
            return np.zeros(GRID_SIZE, dtype=dtype)

        matriz = np.zeros(GRID_SIZE, dtype=dtype)

        # Vectorizado — mucho más rápido que iterar fila por fila
        ids = df.index.to_numpy(dtype=np.int64)

        # Decodificar posición 3D desde el voxel_id (orden C)
        r = ids // (GRID_SIZE[1] * GRID_SIZE[2])        # row
        c = (ids // GRID_SIZE[2]) % GRID_SIZE[1]        # col
        s = ids % GRID_SIZE[2]                          # slice

        # Filtrar índices fuera del rango válido
        validos = (r >= 0) & (r < GRID_SIZE[0]) & \
                  (c >= 0) & (c < GRID_SIZE[1]) & \
                  (s >= 0) & (s < GRID_SIZE[2])
        r, c, s = r[validos], c[validos], s[validos]

        # ── Detectar máscara vs. volumen con valores ─────────────────────
        # Convención OpenKBP: si la columna 'data' tiene NaN → es máscara.
        # Aquí lo hacemos robusto: si TODA la columna es NaN, es máscara.
        col_data = df.iloc[:, 0]
        es_mascara = col_data.isna().all()

        if es_mascara or dtype == bool:
            # Las máscaras se marcan con 1 / True por la mera presencia del índice
            matriz[r, c, s] = 1
        else:
            vals = col_data.to_numpy(dtype=float)[validos]
            vals = np.nan_to_num(vals, nan=0.0)  # seguridad
            matriz[r, c, s] = vals

        return matriz

    except Exception as e:
        print(f"    ⚠ Error leyendo {os.path.basename(ruta_csv)}: {e}")
        return np.zeros(GRID_SIZE, dtype=dtype)


class Paciente:
    

    def __init__(self, ruta_paciente):
        self.id   = os.path.basename(ruta_paciente)
        self.ruta = ruta_paciente

        # ── Dosis real del oncólogo ────────────────────────────────────────
        self.dosis_real = cargar_csv_a_matriz(
            os.path.join(ruta_paciente, "dose.csv"), dtype=float
        )

        # ── CT del paciente ────────────────────────────────────────────────
        self.ct = cargar_csv_a_matriz(
            os.path.join(ruta_paciente, "ct.csv"), dtype=float
        )

        # ── Máscaras de estructuras ────────────────────────────────────────
        # Los archivos están directamente en ruta_paciente/ (no en subcarpeta)
        self.mascaras = {}
        for estructura in ESTRUCTURAS:
            ruta_csv = os.path.join(ruta_paciente, f"{estructura}.csv")
            self.mascaras[estructura] = cargar_csv_a_matriz(ruta_csv, dtype=bool)

        # PTV combinado: unión de PTV70 + PTV63 + PTV56
        # Usamos el de mayor dosis como tumor principal
        # Si PTV70 tiene vóxeles, usarlo; si no, usar PTV63; si no, PTV56
        if self.mascaras["PTV70"].sum() > 0:
            self.mascaras[PTV_PRINCIPAL] = self.mascaras["PTV70"]
        elif self.mascaras["PTV63"].sum() > 0:
            self.mascaras[PTV_PRINCIPAL] = self.mascaras["PTV63"]
            print(f"    ℹ {self.id}: usando PTV63 como tumor principal")
        elif self.mascaras["PTV56"].sum() > 0:
            self.mascaras[PTV_PRINCIPAL] = self.mascaras["PTV56"]
            print(f"    ℹ {self.id}: usando PTV56 como tumor principal")

        n_tumor = int(self.mascaras[PTV_PRINCIPAL].sum())
        n_dosis  = int((self.dosis_real > 0).sum())
        print(f"  ✓ {self.id} — tumor: {n_tumor:,} vóx | dosis real: {n_dosis:,} vóx")

    # ── Métricas clínicas ──────────────────────────────────────────────────

    def calcular_D95(self, dosis_3d):
        """D95 = dosis que recibe el 95% del volumen tumoral. Meta: ≥70 Gy."""
        dosis_tumor = dosis_3d[self.mascaras[PTV_PRINCIPAL]]
        if len(dosis_tumor) == 0:
            return 0.0
        return float(np.percentile(dosis_tumor, 5))

    def calcular_Dmean_oar(self, nombre_oar, dosis_3d):
        """Dosis promedio en un órgano en riesgo."""
        mask = self.mascaras.get(nombre_oar)
        if mask is None or mask.sum() == 0:
            return 0.0
        return float(np.mean(dosis_3d[mask]))

    def calcular_Dmax_oar(self, nombre_oar, dosis_3d):
        """Dosis máxima en un órgano en riesgo."""
        mask = self.mascaras.get(nombre_oar)
        if mask is None or mask.sum() == 0:
            return 0.0
        return float(np.max(dosis_3d[mask]))

    def resumen(self, dosis_3d):
        """Imprime tabla clínica completa para un plan de dosis dado."""
        print(f"\n  {'─'*55}")
        print(f"  Resumen clínico — {self.id}")
        print(f"  {'─'*55}")
        print(f"  {'Estructura':<22} {'Métrica':<8} {'Valor':>7}  {'Límite':>8}  Estado")
        print(f"  {'─'*55}")

        d95 = self.calcular_D95(dosis_3d)
        ok  = "✅" if d95 >= 70 else "❌"
        print(f"  {'PTV (tumor)':<22} {'D95':<8} {d95:>7.1f}  {'≥70 Gy':>8}  {ok}")

        for oar, cfg in LIMITES_CLINICOS.items():
            if cfg["tipo"] == "Dmean":
                val = self.calcular_Dmean_oar(oar, dosis_3d)
            else:
                val = self.calcular_Dmax_oar(oar, dosis_3d)
            ok = "✅" if val <= cfg["limite"] else "❌"
            lim_str = f"≤{cfg['limite']} Gy"
            print(f"  {oar:<22} {cfg['tipo']:<8} {val:>7.1f}  {lim_str:>8}  {ok}")

        print(f"  {'─'*55}")


def cargar_pacientes(ruta_base, n_pacientes=30, subset="test-pats",
                     offset=0, aleatorio=False, semilla=None, ids=None):
 
    ruta_subset = os.path.join(ruta_base, "provided-data", subset)

    if not os.path.exists(ruta_subset):
        raise FileNotFoundError(
            f"No encontrado: {ruta_subset}\n"
            f"Clona el dataset con: git clone https://github.com/ababier/open-kbp.git"
        )

    todas = sorted(
        [d for d in os.listdir(ruta_subset)
         if os.path.isdir(os.path.join(ruta_subset, d)) and d.startswith("pt_")],
        key=lambda s: int(s.split("_")[1])  # ordenar numéricamente (pt_2 antes que pt_10)
    )

    # Seleccionar qué carpetas cargar
    if ids is not None:
        carpetas = [d for d in todas if d in set(ids)]
    elif aleatorio:
        rng = np.random.default_rng(semilla)
        carpetas = list(rng.choice(todas, size=min(n_pacientes, len(todas)),
                                   replace=False))
        carpetas = sorted(carpetas, key=lambda s: int(s.split("_")[1]))
    else:
        carpetas = todas[offset: offset + n_pacientes]

    print(f"\nCargando {len(carpetas)} pacientes de '{subset}'"
          f"{f' (offset={offset})' if offset else ''}"
          f"{' [aleatorio]' if aleatorio else ''}...")

    pacientes = []
    for carpeta in carpetas:
        try:
            pac = Paciente(os.path.join(ruta_subset, carpeta))
            pacientes.append(pac)
        except Exception as e:
            print(f"  ⚠ Error en {carpeta}: {e}")

    n_ok = sum(1 for p in pacientes if p.mascaras[PTV_PRINCIPAL].sum() > 0)
    print(f"\n✓ {len(pacientes)} pacientes cargados — {n_ok} con tumor detectado\n")
    return pacientes


# ── Prueba directa ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Probando entorno.py con datos REALES de OpenKBP")
    print("=" * 55)

    ruta_pac = "open-kbp/provided-data/test-pats/pt_241"

    if not os.path.exists(ruta_pac):
        print("No se encontró la carpeta. Ejecuta desde la raíz del proyecto.")
    else:
        pac = Paciente(ruta_pac)

        print(f"\nArchivos cargados:")
        print(f"  CT:         {(pac.ct > 0).sum():,} vóxeles con valor")
        print(f"  Dosis real: {(pac.dosis_real > 0).sum():,} vóxeles con valor")
        for e in ESTRUCTURAS:
            n = int(pac.mascaras[e].sum())
            if n > 0:
                print(f"  {e}: {n:,} vóxeles")

        print(f"\nMétricas de la dosis REAL del oncólogo:")
        pac.resumen(pac.dosis_real)
        print("\n✓ entorno.py funcionando con datos reales.")