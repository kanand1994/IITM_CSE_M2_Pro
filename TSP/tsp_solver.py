import random
import math
import numpy as np
from typing import List, Tuple
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

def greedy_tsp(distance_matrix: List[List[float]], start: int) -> List[int]:
    """Greedy algorithm starting from specified index"""
    n = len(distance_matrix)
    visited = [False]*n
    tour = [start]
    visited[start] = True
    
    for _ in range(n-1):
        nearest = min((j for j in range(n) if not visited[j]), 
                    key=lambda j: distance_matrix[tour[-1]][j])
        tour.append(nearest)
        visited[nearest] = True
    return tour

def two_opt_swap(tour: List[int], dist_matrix: List[List[float]]) -> List[int]:
    """2-opt optimization with progress tracking"""
    best_tour = tour.copy()
    improved = True
    
    with tqdm(desc="Optimizing route") as pbar:
        while improved:
            improved = False
            for i in range(1, len(tour)-2):
                for j in range(i+1, len(tour)):
                    if j-i == 1: continue
                    a, b, c, d = best_tour[i-1], best_tour[i], best_tour[j-1], best_tour[j%len(best_tour)]
                    delta = (dist_matrix[a][c] + dist_matrix[b][d]) - \
                            (dist_matrix[a][b] + dist_matrix[c][d])
                    if delta < 0:
                        best_tour[i:j] = best_tour[i:j][::-1]
                        improved = True
                        pbar.update()
                        break
                if improved: break
    return best_tour

def simulated_annealing(dist_matrix: List[List[float]], 
                       temp=10000, cooling=0.995) -> List[int]:
    """Simulated annealing implementation"""
    n = len(dist_matrix)
    current = random.sample(range(n), n)
    current_cost = sum(dist_matrix[current[i]][current[i+1]] for i in range(n-1)) + dist_matrix[current[-1]][current[0]]
    
    with tqdm(total=1000, desc="Simulated Annealing") as pbar:
        for _ in range(1000):
            i, j = sorted(random.sample(range(n), 2))
            new = current[:i] + current[i:j][::-1] + current[j:]
            new_cost = sum(dist_matrix[new[i]][new[i+1]] for i in range(n-1)) + dist_matrix[new[-1]][new[0]]
            
            if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost)/temp):
                current, current_cost = new, new_cost
            
            temp *= cooling
            pbar.update()
    
    return current

def calculate_total_distance(tour: List[int], dist_matrix: List[List[float]]) -> float:
    return sum(dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1]][tour[0]]

def christofides_tsp(dist_matrix: List[List[float]]) -> List[int]:
    """Christofides algorithm for metric TSP"""
    n = len(dist_matrix)
    
    # Step 1: Find Minimum Spanning Tree (MST)
    mst = prim_mst(dist_matrix)
    
    # Step 2: Find odd-degree vertices
    degrees = defaultdict(int)
    for u, v in mst:
        degrees[u] += 1
        degrees[v] += 1
    odd_vertices = [v for v, d in degrees.items() if d % 2 != 0]
    
    # Step 3: Minimum weight matching
    matching = greedy_matching(odd_vertices, dist_matrix)
    
    # Step 4: Combine MST and matching
    multigraph = mst + matching
    
    # Step 5: Find Eulerian circuit
    eulerian = hierholzer(multigraph)
    
    # Step 6: Make Hamiltonian
    return make_hamiltonian(eulerian)

def prim_mst(dist_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """Prim's algorithm for MST"""
    n = len(dist_matrix)
    in_mst = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n
    key[0] = 0

    for _ in range(n):
        u = min((v for v in range(n) if not in_mst[v]), key=lambda v: key[v])
        in_mst[u] = True
        for v in range(n):
            if dist_matrix[u][v] < key[v] and not in_mst[v]:
                key[v] = dist_matrix[u][v]
                parent[v] = u
    return [(parent[v], v) for v in range(1, n) if parent[v] != -1]

def greedy_matching(vertices: List[int], dist_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """Greedy matching approximation"""
    matching = []
    unpaired = set(vertices)
    
    while unpaired:
        u = unpaired.pop()
        if not unpaired:
            break
        v = min(unpaired, key=lambda x: dist_matrix[u][x])
        unpaired.remove(v)
        matching.append((u, v))
    return matching

def hierholzer(edges: List[Tuple[int, int]]) -> List[int]:
    """Hierholzer's algorithm for Eulerian circuit"""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    stack = [next(iter(adj))]
    circuit = []
    
    while stack:
        current = stack[-1]
        if adj[current]:
            next_node = adj[current].pop()
            adj[next_node].remove(current)
            stack.append(next_node)
        else:
            circuit.append(stack.pop())
    return circuit[::-1]

def make_hamiltonian(circuit: List[int]) -> List[int]:
    """Convert Eulerian circuit to Hamiltonian path"""
    visited = set()
    tour = []
    for node in circuit:
        if node not in visited:
            visited.add(node)
            tour.append(node)
    tour.append(tour[0])
    return tour

class AntColonyOptimizer:
    """Ant Colony Optimization implementation"""
    def __init__(self, dist_matrix: List[List[float]], 
                 n_ants=10, iterations=50, decay=0.1,
                 alpha=1, beta=2):
        self.dist_matrix = dist_matrix
        self.n = len(dist_matrix)
        self.n_ants = n_ants
        self.iterations = iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones((self.n, self.n)) / self.n

    def run(self) -> List[int]:
        best_tour = None
        best_length = float('inf')
        
        for _ in tqdm(range(self.iterations), desc="Ant Colony Optimization"):
            all_tours = []
            for _ in range(self.n_ants):
                tour = self._gen_tour()
                length = calculate_total_distance(tour, self.dist_matrix)
                all_tours.append((tour, length))
                if length < best_length:
                    best_tour, best_length = tour, length
            
            self._update_pheromone(all_tours)
        
        return best_tour

    def _gen_tour(self) -> List[int]:
        tour = [random.randint(0, self.n-1)]
        visited = set(tour)
        
        for _ in range(self.n-1):
            probs = self._calc_probs(tour[-1], visited)
            next_city = np.random.choice(range(self.n), p=probs)
            tour.append(next_city)
            visited.add(next_city)
        return tour

    def _calc_probs(self, current: int, visited: set) -> List[float]:
        pheromone = np.copy(self.pheromone[current])
        pheromone[list(visited)] = 0
        distances = np.array(self.dist_matrix[current])
        distances[distances == 0] = 1e-10
        scores = (pheromone ** self.alpha) * ((1 / distances) ** self.beta)
        scores /= scores.sum()
        return scores

    def _update_pheromone(self, tours: List[Tuple[List[int], float]]):
        self.pheromone *= (1 - self.decay)
        for tour, length in tours:
            for i in range(len(tour)-1):
                u, v = tour[i], tour[i+1]
                self.pheromone[u][v] += 1 / length
                self.pheromone[v][u] += 1 / length

class GeneticAlgorithm:
    """Genetic Algorithm implementation"""
    def __init__(self, dist_matrix: List[List[float]],
                 population_size=50, generations=100,
                 mutation_rate=0.02):
        self.dist_matrix = dist_matrix
        self.n = len(dist_matrix)
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def run(self) -> List[int]:
        population = self._init_population()
        best_tour = min(population, key=lambda x: self._fitness(x))
        
        with tqdm(total=self.generations, desc="Genetic Algorithm") as pbar:
            for _ in range(self.generations):
                population = self._evolve(population)
                current_best = min(population, key=lambda x: self._fitness(x))
                if self._fitness(current_best) < self._fitness(best_tour):
                    best_tour = current_best
                pbar.update()
        
        return best_tour

    def _init_population(self) -> List[List[int]]:
        return [random.sample(range(self.n), self.n) for _ in range(self.pop_size)]

    def _fitness(self, tour: List[int]) -> float:
        return calculate_total_distance(tour, self.dist_matrix)

    def _evolve(self, population: List[List[int]]) -> List[List[int]]:
        new_pop = []
        for _ in range(self.pop_size):
            parent1, parent2 = self._select_parents(population)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_pop.append(child)
        return new_pop

    def _select_parents(self, population: List[List[int]]) -> Tuple[List[int], List[int]]:
        tournament = random.sample(population, k=5)
        tournament.sort(key=lambda x: self._fitness(x))
        return tournament[0], tournament[1]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        start, end = sorted(random.sample(range(self.n), 2))
        child = [None] * self.n
        child[start:end] = parent1[start:end]
        
        ptr = 0
        for gene in parent2:
            if gene not in child:
                while child[ptr] is not None:
                    ptr += 1
                child[ptr] = gene
        return child

    def _mutate(self, tour: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour