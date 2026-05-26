"""
gwo.py

Jerarquía:
    α (alpha) — el mejor lobo = el mejor plan encontrado, lidera
    β (beta)  — segundo mejor, apoya al alpha
    δ (delta) — tercer mejor, refuerza la dirección
    ω (omega) — el resto de lobos, se mueven según α, β y δ

Actualización de posición:
    X(t+1) = promedio de X1, X2, X3
    donde X1 = Xα - A1 · |C1·Xα - X|
          X2 = Xβ - A2 · |C2·Xβ - X|
          X3 = Xδ - A3 · |C3·Xδ - X|

El parámetro 'a' disminuye linealmente de 2 a 0 con las iteraciones,
lo que hace que el algoritmo pase de explorar (buscar amplio) a
explotar (afinar la mejor solución encontrada).
"""

import numpy as np
from fitness import calcular_fitness


class OptimizadorLoboGris:
    
    def __init__(self,
                 n_lobos=50,          # tamaño de la manada
                 n_iteraciones=100,   # cuántas actualizaciones de posición
                 n_haces=9,           # dimensiones del problema
                 semilla=None):
        
        self.n_lobos       = n_lobos
        self.n_iteraciones = n_iteraciones
        self.n_haces       = n_haces
        
        if semilla is not None:
            np.random.seed(semilla)
        
        # Historial para graficar
        self.historial_mejor    = []
        self.historial_promedio = []
        
        # Los tres mejores lobos
        self.alpha_pos     = None  # posición del lobo alpha
        self.alpha_fitness = -np.inf
        self.beta_pos      = None
        self.beta_fitness  = -np.inf
        self.delta_pos     = None
        self.delta_fitness = -np.inf
    
    def inicializar_manada(self):
       
        return np.random.uniform(0, 1, (self.n_lobos, self.n_haces))
    
    def actualizar_lideres(self, manada, fitness_array):
    
        # Ordenar índices de mayor a menor fitness
        indices_ordenados = np.argsort(fitness_array)[::-1]
        
        self.alpha_pos     = manada[indices_ordenados[0]].copy()
        self.alpha_fitness = fitness_array[indices_ordenados[0]]
        
        self.beta_pos      = manada[indices_ordenados[1]].copy()
        self.beta_fitness  = fitness_array[indices_ordenados[1]]
        
        self.delta_pos     = manada[indices_ordenados[2]].copy()
        self.delta_fitness = fitness_array[indices_ordenados[2]]
    
    def actualizar_posiciones(self, manada, iteracion):
        # a decrece linealmente de 2 a 0
        a = 2.0 - 2.0 * (iteracion / self.n_iteraciones)
        
        nueva_manada = np.empty_like(manada)
        
        for i in range(self.n_lobos):
            # Saltar α, β y δ (no se mueven solos, son los líderes)
            # (en la práctica sí se recalculan, pero no se actualizan con esta fórmula)
            
            # Vectores aleatorios — uno por cada líder y por cada dimensión
            r1 = np.random.uniform(0, 1, self.n_haces)
            r2 = np.random.uniform(0, 1, self.n_haces)
            A1 = 2 * a * r1 - a   # puede ser negativo → permite alejarse del líder
            C1 = 2 * r2
            
            r1 = np.random.uniform(0, 1, self.n_haces)
            r2 = np.random.uniform(0, 1, self.n_haces)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            
            r1 = np.random.uniform(0, 1, self.n_haces)
            r2 = np.random.uniform(0, 1, self.n_haces)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            
            # Posición respecto a cada líder
            X1 = self.alpha_pos - A1 * np.abs(C1 * self.alpha_pos - manada[i])
            X2 = self.beta_pos  - A2 * np.abs(C2 * self.beta_pos  - manada[i])
            X3 = self.delta_pos - A3 * np.abs(C3 * self.delta_pos - manada[i])
            
            # Nueva posición = promedio de los tres movimientos
            nueva_pos = (X1 + X2 + X3) / 3.0
            
            # Mantener en [0,1]
            nueva_manada[i] = np.clip(nueva_pos, 0, 1)
        
        return nueva_manada
    
    def optimizar(self, paciente, verbose=True, manada_inicial=None):
       
        self.historial_mejor    = []
        self.historial_promedio = []
        self.alpha_fitness      = -np.inf

        if verbose:
            print(f"\nGWO iniciando | {self.n_lobos} lobos × "
                  f"{self.n_iteraciones} iteraciones")

        # ── Inicialización de la manada ────────────────────────────────────
        if manada_inicial is not None:
            # El híbrido entrega aquí la población final evolucionada por el AG
            manada = np.clip(manada_inicial.astype(float).copy(), 0, 1)
            # Ajustar el tamaño si difiere
            if manada.shape[0] != self.n_lobos:
                self.n_lobos = manada.shape[0]
            if verbose:
                print(f"  (manada inicializada con población evolucionada del AG, N={self.n_lobos})")
        else:
            manada = self.inicializar_manada()
        
        for it in range(self.n_iteraciones):
            
            # ── Evaluar toda la manada ─────────────────────────────────────
            fitness_array = np.array([
                calcular_fitness(lobo, paciente)
                for lobo in manada
            ])
            
            # ── Actualizar α, β, δ ─────────────────────────────────────────
            self.actualizar_lideres(manada, fitness_array)
            
            # Guardar historial
            self.historial_mejor.append(self.alpha_fitness)
            self.historial_promedio.append(fitness_array.mean())
            
            if verbose and (it % 10 == 0 or it == self.n_iteraciones - 1):
                a_actual = 2.0 - 2.0 * (it / self.n_iteraciones)
                print(f"  Iter {it+1:3d}/{self.n_iteraciones} | "
                      f"Alpha: {self.alpha_fitness:.4f} | "
                      f"Promedio: {fitness_array.mean():.4f} | "
                      f"a={a_actual:.2f}")
            
            # ── Mover a todos los lobos ω ──────────────────────────────────
            manada = self.actualizar_posiciones(manada, it)
        
        if verbose:
            print(f"\n✓ GWO terminado. Mejor fitness (alpha): {self.alpha_fitness:.4f}")
            print(f"  Posición alpha: {np.round(self.alpha_pos, 3)}")
        
        return self.alpha_pos, self.alpha_fitness


class HibridoAGGWO:
   
    
    def __init__(self,
                 n_individuos=50,
                 n_gen_ag=50,        # generaciones de AG (fase 1)
                 n_iter_gwo=50,      # iteraciones de GWO (fase 2)
                 n_haces=9,
                 semilla=None):
        
        from genetico import AlgoritmoGenetico
        
        self.ag = AlgoritmoGenetico(
            n_individuos=n_individuos,
            n_generaciones=n_gen_ag,
            n_haces=n_haces,
            semilla=semilla
        )
        self.gwo = OptimizadorLoboGris(
            n_lobos=n_individuos,
            n_iteraciones=n_iter_gwo,
            n_haces=n_haces,
            semilla=semilla
        )
        
        self.historial_mejor    = []
        self.historial_promedio = []
    
    def optimizar(self, paciente, verbose=True):
        """
        Fase 1: AG explora. Fase 2: GWO afina.
        """
        if verbose:
            print("\n" + "="*50)
            print("HÍBRIDO AG-GWO")
            print("="*50)
            print("\n--- FASE 1: Algoritmo Genético (exploración) ---")
        
        # ── FASE 1: AG ────────────────────────────────────────────────────
        mejor_pesos_ag, mejor_fitness_ag = self.ag.optimizar(paciente, verbose)

        # Historial de la fase AG
        self.historial_mejor    = self.ag.historial_mejor.copy()
        self.historial_promedio = self.ag.historial_promedio.copy()

        if verbose:
            print(f"\n  Fin fase AG — mejor fitness: {mejor_fitness_ag:.4f}")
            print("\n--- FASE 2: Lobo Gris (explotación) ---")
            print("  Inicializando manada con la población evolucionada del AG...")

    
        manada_inicial = self.ag.poblacion_final  # guardada al terminar el AG

        mejor_pesos_gwo, mejor_fitness_gwo = self.gwo.optimizar(
            paciente, verbose, manada_inicial=manada_inicial
        )
        
        # Agregar historial de GWO
        self.historial_mejor    += self.gwo.historial_mejor
        self.historial_promedio += self.gwo.historial_promedio
        
        # El mejor final es el mejor entre AG y GWO
        if mejor_fitness_ag >= mejor_fitness_gwo:
            mejor_final = mejor_pesos_ag
            fitness_final = mejor_fitness_ag
        else:
            mejor_final = mejor_pesos_gwo
            fitness_final = mejor_fitness_gwo
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"HÍBRIDO TERMINADO")
            print(f"  Fase AG  → fitness: {mejor_fitness_ag:.4f}")
            print(f"  Fase GWO → fitness: {mejor_fitness_gwo:.4f}")
            print(f"  Mejor final: {fitness_final:.4f}")
        
        return mejor_final, fitness_final


# ── Prueba rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    print("Probando gwo.py con paciente simulado")
    print("=" * 50)
    
    class PacienteSimulado:
        def __init__(self):
            sz = (32,32,32)
            mask = np.zeros(sz, dtype=bool)
            mask[12:20,12:20,12:20] = True
            self.mascaras = {"PTV70": mask}
            for oar in ["SpinalCord","RightParotid","LeftParotid","Mandible","Brainstem","Larynx"]:
                m = np.zeros(sz, dtype=bool)
                m[2:5,2:5,:] = True
                self.mascaras[oar] = m
        def calcular_D95(self,d): return float(np.percentile(d[self.mascaras["PTV70"]],5))
        def calcular_Dmean_oar(self,oar,d): return float(np.mean(d[self.mascaras[oar]]))
        def calcular_Dmax_oar(self,oar,d): return float(np.max(d[self.mascaras[oar]]))
    
    paciente = PacienteSimulado()
    
    gwo = OptimizadorLoboGris(n_lobos=20, n_iteraciones=30, semilla=42)
    pos, fit = gwo.optimizar(paciente, verbose=True)
    
    print(f"\nMejora: {gwo.historial_mejor[0]:.4f} → {gwo.historial_mejor[-1]:.4f}")
    print("\n✓ gwo.py funciona correctamente.")
