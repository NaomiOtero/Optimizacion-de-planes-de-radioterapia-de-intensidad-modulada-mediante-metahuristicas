

import numpy as np
from fitness import calcular_fitness


class AlgoritmoGenetico:
    """
    Algoritmo Genético para optimizar pesos de haces de radioterapia.
    
    Cada individuo = vector de 9 pesos en [0,1]
    Cada peso = intensidad de uno de los 9 haces de radiación
    """
    
    def __init__(self,
                 n_individuos=50,    # tamaño de la población
                 n_generaciones=100, # cuántos ciclos evolutivos
                 tasa_mutacion=0.05, # probabilidad de mutar un gen (5%)
                 sigma_mutacion=0.1, # cuánto cambia el gen al mutar
                 n_haces=9,          # haces de radiación (fijo en IMRT estándar)
                 semilla=None):      # para reproducibilidad
        
        self.n_individuos    = n_individuos
        self.n_generaciones  = n_generaciones
        self.tasa_mutacion   = tasa_mutacion
        self.sigma_mutacion  = sigma_mutacion
        self.n_haces         = n_haces
        
        if semilla is not None:
            np.random.seed(semilla)
        
        # Historial para graficar después
        self.historial_mejor   = []   # mejor fitness por generación
        self.historial_promedio = []  # fitness promedio por generación
        self.mejor_individuo   = None
        self.mejor_fitness     = -np.inf
    
    def inicializar_poblacion(self):
   
        return np.random.uniform(0, 1, (self.n_individuos, self.n_haces))
    
    def evaluar_poblacion(self, poblacion, paciente):
   
        fitness_array = np.array([
            calcular_fitness(individuo, paciente)
            for individuo in poblacion
        ])
        return fitness_array
    
    def seleccion_torneo(self, poblacion, fitness_array, k=3):
  
        seleccionados = np.empty_like(poblacion)
        
        for i in range(self.n_individuos):
            # Elegir k índices al azar
            indices = np.random.choice(self.n_individuos, k, replace=False)
            # El ganador del torneo es el de mayor fitness
            ganador = indices[np.argmax(fitness_array[indices])]
            seleccionados[i] = poblacion[ganador]
        
        return seleccionados
    
    def cruce_un_punto(self, padre1, padre2):
        """
        Cruce en un punto: combina dos padres para crear dos hijos.
        
        Ejemplo con 9 genes:
            padre1 = [0.8, 0.3, 0.6, 0.1, 0.9, 0.4, 0.7, 0.2, 0.5]
            padre2 = [0.2, 0.7, 0.1, 0.8, 0.4, 0.9, 0.3, 0.6, 0.1]
            punto  = 4
            hijo1  = [0.8, 0.3, 0.6, 0.1, | 0.4, 0.9, 0.3, 0.6, 0.1]
            hijo2  = [0.2, 0.7, 0.1, 0.8, | 0.9, 0.4, 0.7, 0.2, 0.5]
        
        Retorna:
            hijo1, hijo2: dos nuevos individuos
        """
        punto = np.random.randint(1, self.n_haces)  # punto de corte entre 1 y 8
        
        hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
        hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
        
        return hijo1, hijo2
    
    def mutar(self, individuo):
        mutado = individuo.copy()
        
        for i in range(self.n_haces):
            if np.random.random() < self.tasa_mutacion:
                ruido = np.random.normal(0, self.sigma_mutacion)
                mutado[i] = np.clip(mutado[i] + ruido, 0, 1)
        
        return mutado
    
    def crear_nueva_generacion(self, seleccionados):
     
        nueva_poblacion = np.empty_like(seleccionados)
        
        # Elitismo: guardar al mejor individuo sin cambios
        # (lo asignaremos al final del array)
        
        # Crear N-1 nuevos individuos por cruce + mutación
        i = 0
        while i < self.n_individuos - 1:
            # Elegir dos padres al azar de los seleccionados
            idx1, idx2 = np.random.choice(self.n_individuos, 2, replace=False)
            padre1 = seleccionados[idx1]
            padre2 = seleccionados[idx2]
            
            # Cruce
            hijo1, hijo2 = self.cruce_un_punto(padre1, padre2)
            
            # Mutación
            hijo1 = self.mutar(hijo1)
            hijo2 = self.mutar(hijo2)
            
            nueva_poblacion[i] = hijo1
            if i + 1 < self.n_individuos - 1:
                nueva_poblacion[i + 1] = hijo2
            i += 2
        
        return nueva_poblacion
    
    def optimizar(self, paciente, verbose=True):
        
        # Reiniciar historial
        self.historial_mejor    = []
        self.historial_promedio = []
        self.mejor_individuo    = None
        self.mejor_fitness      = -np.inf
        self.poblacion_final    = None   # ← se llena al terminar; el híbrido la usa
        self.fitness_final      = None

        if verbose:
            print(f"\nAG iniciando | {self.n_individuos} individuos × "
                  f"{self.n_generaciones} generaciones")

        # ── Paso 1: Inicialización ─────────────────────────────────────────
        poblacion = self.inicializar_poblacion()
        
        for gen in range(self.n_generaciones):
            
            # ── Paso 2: Evaluación ─────────────────────────────────────────
            fitness_array = self.evaluar_poblacion(poblacion, paciente)
            
            # Estadísticas de esta generación
            idx_mejor     = np.argmax(fitness_array)
            mejor_fitness = fitness_array[idx_mejor]
            fitness_prom  = fitness_array.mean()
            
            # Actualizar el mejor global encontrado hasta ahora
            if mejor_fitness > self.mejor_fitness:
                self.mejor_fitness    = mejor_fitness
                self.mejor_individuo  = poblacion[idx_mejor].copy()
            
            # Guardar historial
            self.historial_mejor.append(self.mejor_fitness)
            self.historial_promedio.append(fitness_prom)
            
            if verbose and (gen % 10 == 0 or gen == self.n_generaciones - 1):
                print(f"  Gen {gen+1:3d}/{self.n_generaciones} | "
                      f"Mejor: {self.mejor_fitness:.4f} | "
                      f"Promedio: {fitness_prom:.4f}")
            
            # ── Paso 3: Selección ──────────────────────────────────────────
            seleccionados = self.seleccion_torneo(poblacion, fitness_array)
            
            # ── Paso 4 y 5: Cruce + Mutación ──────────────────────────────
            nueva_poblacion = self.crear_nueva_generacion(seleccionados)
            
            # Elitismo: preservar al mejor individuo
            nueva_poblacion[-1] = self.mejor_individuo.copy()
            
            poblacion = nueva_poblacion

        # Guardar población final (el híbrido la usará como manada inicial del GWO)
        self.poblacion_final = poblacion.copy()
        self.fitness_final   = self.evaluar_poblacion(poblacion, paciente)

        if verbose:
            print(f"\n✓ AG terminado. Mejor fitness: {self.mejor_fitness:.4f}")
            print(f"  Pesos óptimos: {np.round(self.mejor_individuo, 3)}")

        return self.mejor_individuo, self.mejor_fitness


# ── Prueba rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    
    print("Probando genetico.py con paciente simulado")
    print("=" * 50)
    
    # Paciente simulado pequeño para prueba rápida
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
        def calcular_D95(self, d):
            return float(np.percentile(d[self.mascaras["PTV70"]], 5))
        def calcular_Dmean_oar(self, oar, d):
            return float(np.mean(d[self.mascaras[oar]]))
        def calcular_Dmax_oar(self, oar, d):
            return float(np.max(d[self.mascaras[oar]]))
    
    paciente = PacienteSimulado()
    
    ag = AlgoritmoGenetico(
        n_individuos=20,    # pequeño para prueba rápida
        n_generaciones=30,
        semilla=42
    )
    
    mejor_pesos, mejor_fitness = ag.optimizar(paciente, verbose=True)
    print(f"\nLongitud historial: {len(ag.historial_mejor)} generaciones")
    print(f"Fitness inicial: {ag.historial_mejor[0]:.4f}")
    print(f"Fitness final:   {ag.historial_mejor[-1]:.4f}")
    mejora = ag.historial_mejor[-1] - ag.historial_mejor[0]
    print(f"Mejora total:    {mejora:+.4f}")
    print("\n✓ genetico.py funciona correctamente.")
