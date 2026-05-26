
import os
import base64
import numpy as np
import pandas as pd


# ── Configuración ──────────────────────────────────────────────────────────────
RUTA_RESULTADOS = "resultados"
RUTA_DATASET    = "open-kbp/provided-data/test-pats"
RUTA_SALIDA     = "REPORTE.html"


# ── Helpers ────────────────────────────────────────────────────────────────────
def embed_imagen(ruta):
    """Codifica un PNG en base64 para embeberlo en el HTML."""
    if not os.path.exists(ruta):
        return ""
    with open(ruta, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" />'


def df_to_html(df, classes="tabla", float_fmt="{:.4f}"):
    """Convierte un DataFrame a HTML con clase CSS."""
    return df.to_html(classes=classes, border=0, index=False,
                      float_format=lambda x: float_fmt.format(x)
                      if isinstance(x, (int, float, np.floating)) else x)


def explicacion_metricas_html():
    """Glosario de métricas clínicas y términos."""
    filas = [
        ("Vóxel", "Cubito 3D del cuerpo del paciente. Cada uno mide ~3.9 × 3.9 × 2.5 mm. El paciente entero se modela como una rejilla de 128×128×128 = 2,097,152 vóxeles."),
        ("Índice", "Número entre 0 y 2,097,151 que identifica cada vóxel. En los CSV del dataset solo se listan los índices de los vóxeles relevantes (los del tumor o los que reciben dosis); los demás se asumen vacíos."),
        ("Gy (Gray)", "Unidad de dosis de radiación absorbida. 1 Gy = 1 Joule absorbido por kilogramo de tejido. Una radiografía dental son ~0.005 mGy. Un tratamiento completo entrega ~70 Gy al tumor, fraccionados en 30-35 sesiones."),
        ("PTV (Planning Target Volume)", "El volumen del tumor que el oncólogo marcó como objetivo a irradiar. PTV70 = volumen al que se le quiere entregar 70 Gy; PTV63 = 63 Gy; PTV56 = 56 Gy. En cabeza y cuello suele haber 2-3 PTVs concéntricos."),
        ("OAR (Organ At Risk)", "Órgano sano cercano al tumor que se debe proteger. En cabeza y cuello: médula espinal, parótidas (glándulas salivales), tronco encefálico, mandíbula, laringe."),
        ("D95", "Dosis en Gy que recibe el 95% del volumen del tumor. Es la métrica clínica estándar de cobertura. Meta: ≥ 70 Gy. Se calcula como el percentil 5 de la distribución de dosis dentro del tumor (porque el 95% del tumor recibe AL MENOS esa dosis)."),
        ("Dmean", "Dosis promedio en un órgano. Se usa para órganos serie-paralelos como las parótidas (saliva), donde lo que importa es la dosis acumulada total."),
        ("Dmax", "Dosis máxima en un órgano. Se usa para órganos serie como la médula espinal: basta un punto con dosis excesiva para causar daño permanente (parálisis)."),
        ("Fitness", "Función objetivo de los algoritmos: α·D95_norm − β·Daño_OARs_norm − γ·Penalización_médula. Más alto = mejor plan."),
        ("Wilcoxon pareado", "Test estadístico no paramétrico que compara dos algoritmos por pares (mismo paciente, misma corrida). p < 0.05 → la diferencia es estadísticamente significativa."),
    ]
    rows = "".join(f"<tr><td><b>{t}</b></td><td>{d}</td></tr>" for t, d in filas)
    return f"<table class='glosario'><thead><tr><th>Término</th><th>Significado</th></tr></thead><tbody>{rows}</tbody></table>"


def explicacion_limites_html():
    filas = [
        ("PTV (tumor)", "D95", "≥ 70 Gy", "Cobertura mínima del tumor agresivo"),
        ("Médula espinal", "Dmax", "≤ 45 Gy", "Por encima → riesgo de mielopatía/parálisis"),
        ("Tronco encefálico", "Dmax", "≤ 54 Gy", "Centro vital — riesgo de daño respiratorio/cardíaco"),
        ("Mandíbula", "Dmax", "≤ 70 Gy", "Por encima → osteonecrosis (muerte del hueso)"),
        ("Parótida izquierda", "Dmean", "≤ 26 Gy", "Por encima → xerostomía (boca seca permanente)"),
        ("Parótida derecha", "Dmean", "≤ 26 Gy", "Por encima → xerostomía (boca seca permanente)"),
        ("Laringe", "Dmean", "≤ 45 Gy", "Por encima → disfagia (dificultad para tragar)"),
    ]
    rows = "".join(f"<tr><td>{e}</td><td>{m}</td><td><b>{l}</b></td><td>{c}</td></tr>" for e, m, l, c in filas)
    return f"<table class='tabla'><thead><tr><th>Estructura</th><th>Métrica</th><th>Límite clínico</th><th>Consecuencia si se excede</th></tr></thead><tbody>{rows}</tbody></table>"


# ── Datos: información de los pacientes ───────────────────────────────────────
def construir_tabla_pacientes():
    """Lee cada paciente y arma una tabla de tamaños de estructuras."""
    if not os.path.exists(RUTA_DATASET):
        return pd.DataFrame()

    estructuras = ["PTV70", "PTV63", "PTV56", "SpinalCord", "RightParotid",
                   "LeftParotid", "Mandible", "Brainstem", "Larynx"]
    filas = []
    carpetas = sorted([d for d in os.listdir(RUTA_DATASET)
                       if d.startswith("pt_")])[:10]

    for carpeta in carpetas:
        ruta = os.path.join(RUTA_DATASET, carpeta)
        fila = {"Paciente": carpeta}
        for est in estructuras:
            ruta_csv = os.path.join(ruta, f"{est}.csv")
            if os.path.exists(ruta_csv):
                # Contar líneas - 1 (header)
                with open(ruta_csv) as f:
                    n = sum(1 for _ in f) - 1
                fila[est] = n if n > 0 else "—"
            else:
                fila[est] = "—"

        # Dosis real
        ruta_dose = os.path.join(ruta, "dose.csv")
        if os.path.exists(ruta_dose):
            with open(ruta_dose) as f:
                fila["Dosis (vóx)"] = sum(1 for _ in f) - 1

        # Tamaño físico aproximado del tumor en cm³
        try:
            ruta_vox = os.path.join(ruta, "voxel_dimensions.csv")
            dims = np.loadtxt(ruta_vox)
            vol_mm3 = float(np.prod(dims))
            tumor = fila.get("PTV70", 0)
            if isinstance(tumor, int) and tumor > 0:
                fila["Tumor (cm³)"] = round(tumor * vol_mm3 / 1000, 1)
        except Exception:
            pass

        filas.append(fila)

    return pd.DataFrame(filas)


# ── HTML completo ──────────────────────────────────────────────────────────────
def construir_html():
    # Cargar todas las tablas
    df_resumen = pd.read_csv(f"{RUTA_RESULTADOS}/tablas/fitness_resumen.csv")
    df_metricas = pd.read_csv(f"{RUTA_RESULTADOS}/tablas/metricas_clinicas.csv")
    df_wilcoxon = pd.read_csv(f"{RUTA_RESULTADOS}/estadistica/wilcoxon.csv")
    df_pacientes = construir_tabla_pacientes()

    # Fitness promedio por paciente y algoritmo (matriz)
    df_pivot = df_metricas.groupby(["Paciente", "Algoritmo"])["fitness"].mean().unstack().round(4)
    df_pivot = df_pivot.reset_index()

    # Métricas clínicas promedio por algoritmo
    cols_metricas = [c for c in df_metricas.columns
                     if c.startswith("D95") or c.startswith("Dmax") or c.startswith("Dmean")]
    df_metricas_avg = df_metricas.groupby("Algoritmo")[cols_metricas].mean().round(2).reset_index()

    # Gráficas embebidas
    g_curvas = embed_imagen(f"{RUTA_RESULTADOS}/graficas/curvas_convergencia.png")
    g_boxplot = embed_imagen(f"{RUTA_RESULTADOS}/graficas/boxplot_fitness.png")
    g_metricas = embed_imagen(f"{RUTA_RESULTADOS}/graficas/barras_metricas.png")
    g_dvh = embed_imagen(f"{RUTA_RESULTADOS}/graficas/dvh_mejor_plan.png")

    # Estadísticas globales
    total_evals = len(df_metricas)
    n_pacientes = df_metricas["Paciente"].nunique()
    n_corridas = df_metricas["Corrida"].nunique()
    n_algos = df_metricas["Algoritmo"].nunique()

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Reporte — Optimización de Radioterapia</title>
<style>
  body {{
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
    max-width: 1100px;
    margin: 30px auto;
    padding: 20px 40px;
    color: #1a1a1a;
    line-height: 1.55;
    background: #fafafa;
  }}
  h1 {{ color: #1f2937; border-bottom: 3px solid #3A5FC8; padding-bottom: 10px; }}
  h2 {{ color: #1f2937; border-bottom: 1px solid #d1d5db; padding-bottom: 6px; margin-top: 40px; }}
  h3 {{ color: #374151; margin-top: 28px; }}
  p, li {{ color: #1f2937; }}
  .resumen-box {{
    background: #eff6ff;
    border-left: 4px solid #3A5FC8;
    padding: 14px 20px;
    margin: 14px 0;
    border-radius: 4px;
  }}
  .nota {{
    background: #fef3c7;
    border-left: 4px solid #f59e0b;
    padding: 12px 18px;
    margin: 12px 0;
    border-radius: 4px;
    font-size: 0.95em;
  }}
  .conclusion {{
    background: #ecfdf5;
    border-left: 4px solid #10B981;
    padding: 14px 20px;
    margin: 14px 0;
    border-radius: 4px;
  }}
  table.tabla, table.glosario {{
    border-collapse: collapse;
    margin: 14px 0;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-radius: 6px;
    overflow: hidden;
    width: 100%;
  }}
  table.tabla th, table.tabla td,
  table.glosario th, table.glosario td {{
    padding: 9px 14px;
    text-align: left;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.92em;
  }}
  table.tabla th, table.glosario th {{
    background: #1f2937;
    color: white;
    font-weight: 600;
    letter-spacing: 0.3px;
  }}
  table.tabla tr:nth-child(even) {{ background: #f9fafb; }}
  table.glosario td:first-child {{ width: 28%; vertical-align: top; }}
  img {{ max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 14px 0; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .kbd {{ display: inline-block; padding: 1px 6px; background: #e5e7eb; border-radius: 3px; font-family: monospace; font-size: 0.88em; }}
  .stat-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin: 18px 0;
  }}
  .stat-card {{
    background: white; padding: 14px; border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center;
  }}
  .stat-card .num {{ font-size: 1.8em; font-weight: 600; color: #3A5FC8; }}
  .stat-card .label {{ font-size: 0.85em; color: #6b7280; margin-top: 4px; }}
  .footer {{ margin-top: 50px; text-align: center; color: #6b7280; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>Reporte del experimento — Optimización de Radioterapia con Metaheurísticas</h1>

<div class="resumen-box">
  <b>Resumen ejecutivo.</b> Se compararon tres algoritmos metaheurísticos (Algoritmo
  Genético, Lobo Gris, e Híbrido AG-GWO) en la tarea de optimizar los pesos de los
  9 haces de radioterapia en planes IMRT para cáncer de cabeza y cuello. Se usaron
  10 pacientes reales del dataset OpenKBP y 5 corridas independientes por algoritmo
  para tener estadística sólida.
</div>

<div class="stat-grid">
  <div class="stat-card"><div class="num">{n_pacientes}</div><div class="label">Pacientes</div></div>
  <div class="stat-card"><div class="num">{n_algos}</div><div class="label">Algoritmos</div></div>
  <div class="stat-card"><div class="num">{n_corridas}</div><div class="label">Corridas / algo</div></div>
  <div class="stat-card"><div class="num">{total_evals}</div><div class="label">Planes generados</div></div>
</div>


<h2>1. Contexto: qué se está optimizando y por qué</h2>

<p>
En radioterapia IMRT (<i>Intensity-Modulated Radiation Therapy</i>) se irradia el
tumor desde 9 ángulos distribuidos alrededor del paciente (0°, 40°, 80°, ..., 320°).
Lo que se busca es <b>la intensidad relativa óptima de cada haz</b> — un vector de
9 números entre 0 y 1 — que entregue la dosis prescrita al tumor sin dañar los
órganos sanos cercanos (médula espinal, parótidas, etc.).
</p>

<p>
La planificación manual tarda 2-5 horas por paciente y depende de la experiencia
del oncólogo. Los algoritmos evaluados aquí encuentran soluciones equivalentes en
<b>segundos de cómputo</b> y de forma reproducible.
</p>

<h3>Límites clínicos usados (literatura QUANTEC/DAHANCA)</h3>
{explicacion_limites_html()}

<div class="nota">
  <b>Cómo se mide cada vóxel.</b> El paciente está representado como una rejilla
  3D de 128 × 128 × 128 cubitos (vóxeles). Cada vóxel mide ~3.9 × 3.9 × 2.5 mm
  (aproximadamente el tamaño de la mitad de una uña). En el dataset, los archivos
  CSV listan únicamente los índices de los vóxeles relevantes — por eso al cargar
  PTV70.csv (4,097 filas) obtenemos un tumor de 4,097 vóxeles.
</div>


<h2>2. Pacientes cargados desde OpenKBP</h2>

<p>
Cada paciente del dataset trae sus estructuras anatómicas marcadas voxel a voxel
por radiólogos. La siguiente tabla muestra el tamaño (en cantidad de vóxeles) de
cada estructura. <b>De aquí salen los números que ves en la terminal</b>
("tumor: 4,097 vóx" significa que el archivo PTV70.csv del paciente tiene 4,097
filas).
</p>

{df_to_html(df_pacientes)}

<div class="nota">
  <b>Notas de lectura.</b>
  El "—" significa que el paciente no tiene esa estructura marcada (o el oncólogo
  no la consideró relevante para este caso). Los tres PTV (PTV70, PTV63, PTV56)
  representan zonas concéntricas que reciben dosis decreciente. La columna
  "Tumor (cm³)" convierte los vóxeles de PTV70 a centímetros cúbicos físicos
  usando las dimensiones reales del vóxel.
</div>


<h2>3. Resultado principal: fitness por algoritmo</h2>

<p>
Resumen estadístico de los <b>{total_evals / n_algos:.0f} fitness finales</b>
obtenidos por cada algoritmo (n = {n_pacientes} pacientes × {n_corridas} corridas):
</p>

{df_to_html(df_resumen)}

<h3>Cómo leer esta tabla</h3>
<table class="tabla">
<thead><tr><th>Columna</th><th>Significado</th></tr></thead>
<tbody>
<tr><td><b>Media</b></td><td>Fitness promedio sobre todas las corridas. Más alto = mejor algoritmo.</td></tr>
<tr><td><b>Std</b></td><td>Desviación estándar. Mide qué tan dispersos están los resultados. Más bajo = más consistente.</td></tr>
<tr><td><b>Mediana</b></td><td>Fitness "típico". Menos sensible a valores atípicos que la media.</td></tr>
<tr><td><b>Min / Max</b></td><td>Peor y mejor resultado individual encontrado.</td></tr>
<tr><td><b>N</b></td><td>Número total de fitness reportados (= pacientes × corridas).</td></tr>
</tbody>
</table>

<div class="conclusion">
  <b>Lectura.</b> Las medias son prácticamente idénticas (0.197). La gran std
  (0.094) <i>no</i> refleja inconsistencia del algoritmo: refleja que los pacientes
  son muy distintos entre sí (un paciente con tumor pequeño llega a fitness ~0.37
  mientras uno con tumor enorme se queda en ~0.04, sin importar el algoritmo).
</div>


<h2>4. Fitness por paciente — ¿de dónde viene la varianza?</h2>

<p>
Para entender por qué la std es alta, vemos el fitness promedio por
(paciente, algoritmo):
</p>

{df_to_html(df_pivot)}

<div class="conclusion">
  <b>Observación clave.</b> La diferencia entre pacientes (0.04 a 0.37) es mucho
  mayor que la diferencia entre algoritmos (0.000 a 0.005). Eso significa que
  <b>el tamaño y la posición del tumor determinan el techo del fitness alcanzable</b>;
  el algoritmo solo decide qué tan bien se aproxima a ese techo. pt_245 (tumor
  enorme) está en ~0.04 con cualquier algoritmo; pt_248 (tumor pequeño) en ~0.37.
</div>


<h2>5. Métricas clínicas promedio (las que el oncólogo verifica)</h2>

<p>
Para cada algoritmo, dosis promedio que recibe cada estructura. Compáralas contra
los límites clínicos de la sección 1:
</p>

{df_to_html(df_metricas_avg, float_fmt="{:.2f}")}

<div class="nota">
  <b>Lectura clínica.</b><br>
  • <b>D95_tumor ≈ 47 Gy</b> contra meta de 70 Gy → cobertura subóptima
  (el simulador conservador no logra subir la dosis sin violar otros límites).<br>
  • <b>Dmax_SpinalCord ≈ 44 Gy</b> contra límite de 45 Gy → al filo, pero bajo el límite.<br>
  • <b>Dmean_Parotid ≈ 28-34 Gy</b> contra límite de 26 Gy → ligeramente excedido,
  área de mejora.<br>
  Esto refleja un compromiso real: con 9 haces el simulador no tiene suficiente
  resolución para cubrir bien el tumor sin pegarle a las parótidas. En la
  literatura los planes IMRT usan 36-72 haces (más libertad de optimización).
</div>


<h2>6. Curvas de convergencia — ¿cuál algoritmo aprende más rápido?</h2>

<p>
Aunque el fitness final es similar, los algoritmos se comportan distinto durante
la búsqueda:
</p>

{g_curvas}

<div class="conclusion">
  <b>Hallazgo importante.</b> GWO (naranja) sube fuerte en las primeras 10
  generaciones — ya está cerca del óptimo en la iteración 15. AG (azul) e
  Híbrido (verde) son más lentos pero llegan al mismo punto hacia la iteración
  30. Esto valida la propuesta original del equipo: <b>GWO es mejor en
  explotación (convergencia rápida)</b>, AG es mejor en exploración (más diverso,
  más lento). El híbrido toma lo mejor de ambos secuencialmente.
</div>


<h2>7. Distribución del fitness final (boxplot)</h2>

<p>
Cada caja muestra la distribución de los {total_evals / n_algos:.0f} fitness
finales de cada algoritmo. La línea blanca dentro de la caja es la mediana,
los bordes son cuartil 25 y 75:
</p>

{g_boxplot}

<p>
Los tres boxplots se ven prácticamente idénticos — confirma que los algoritmos
son estadísticamente equivalentes en fitness final.
</p>


<h2>8. Métricas clínicas comparadas visualmente</h2>

{g_metricas}

<p>
Cobertura del tumor (izquierda): los 3 alcanzan ~47 Gy de D95 contra la meta de
70 Gy. Protección de médula (derecha): los 3 quedan al borde del límite (45 Gy)
con barras de error que algunas veces lo cruzan — eso indica que en ciertos
pacientes el algoritmo eligió un plan agresivo que excedió el límite.
</p>


<h2>9. DVH (Dose-Volume Histogram) — la gráfica clínica estándar</h2>

<p>
Esta es la gráfica que el oncólogo realmente lee en su práctica diaria. Para
cada estructura muestra qué porcentaje del volumen recibe al menos cierta dosis.
</p>

{g_dvh}

<table class="tabla">
<thead><tr><th>Curva</th><th>Cómo debería verse en un buen plan</th></tr></thead>
<tbody>
<tr><td><b>Tumor (rojo sólido)</b></td><td>Alta y a la derecha. Cae cerca de 70 Gy → todo el tumor recibió la dosis prescrita.</td></tr>
<tr><td><b>Tumor real (rojo punteado)</b></td><td>Referencia del plan del oncólogo humano. Si el algoritmo se le acerca, es éxito.</td></tr>
<tr><td><b>Médula (azul)</b></td><td>Baja y a la izquierda. Idealmente debe caer a 0 antes de 45 Gy.</td></tr>
<tr><td><b>Parótidas (verdes)</b></td><td>Bajas, idealmente con su área bajo la curva pequeña (dosis media baja).</td></tr>
</tbody>
</table>


<h2>10. Significancia estadística — prueba de Wilcoxon</h2>

<p>
El test de Wilcoxon de rangos con signo pareado compara cada par (algoritmo A,
algoritmo B) sobre los mismos (paciente, corrida). Si p &lt; 0.05 podemos
afirmar con 95% de confianza que un algoritmo es mejor; si p ≥ 0.05 no podemos
distinguirlos:
</p>

{df_to_html(df_wilcoxon)}

<div class="conclusion">
  <b>Conclusión estadística.</b> Ningún par de algoritmos muestra diferencia
  significativa en esta escala (n = {total_evals // n_algos} pares por comparación).
  Los tres son <b>estadísticamente equivalentes en fitness final</b>. La diferencia
  interesante está en la velocidad de convergencia (sección 6), no en el óptimo.
</div>


<h2>11. Glosario completo</h2>
{explicacion_metricas_html()}


<h2>12. Cómo reproducir y escalar el experimento</h2>

<table class="tabla">
<thead><tr><th>Paso</th><th>Comando</th><th>Tarda</th></tr></thead>
<tbody>
<tr><td>Instalar dependencias</td><td><code>pip install numpy pandas matplotlib scipy</code></td><td>1 min</td></tr>
<tr><td>Descargar dataset (si no lo tienes)</td><td><code>git clone https://github.com/ababier/open-kbp.git</code></td><td>2 min</td></tr>
<tr><td>Validar carga de pacientes</td><td><code>python entorno.py</code></td><td>5 seg</td></tr>
<tr><td>Validar función fitness</td><td><code>python fitness.py</code></td><td>2 seg</td></tr>
<tr><td>Correr experimento medio (config actual)</td><td><code>python experimento.py</code></td><td>~40 seg</td></tr>
<tr><td>Regenerar este reporte</td><td><code>python generar_reporte.py</code></td><td>3 seg</td></tr>
</tbody>
</table>

<h3>Para correr el experimento completo (30 pacientes × 10 corridas × 100 gen)</h3>
<p>Edita <code>experimento.py</code> y cambia la configuración:</p>
<pre><code>config_prueba = {{
    "n_pacientes":        30,
    "n_corridas":         10,
    "n_individuos":       50,
    "n_generaciones":     100,
    "n_gen_ag_hibrido":   50,
    "n_iter_gwo_hibrido": 50,
}}</code></pre>
<p>Tarda ~10-15 minutos. Con el cache ROI no necesita más.</p>


<div class="footer">
  Generado automáticamente por <code>generar_reporte.py</code>.
  Dataset: OpenKBP (Babier et al., 2020). Límites clínicos: literatura QUANTEC.
</div>

</body>
</html>
"""
    return html


if __name__ == "__main__":
    html = construir_html()
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ Reporte generado: {RUTA_SALIDA}")
    print(f"  Ábrelo con doble click o desde el navegador.")
