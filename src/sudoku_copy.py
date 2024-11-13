""" Store old version of SudokuGA class """
import random
import numpy as np
from typing import List, Tuple

class SudokuGA:
    def __init__(self, initial_board: List[List[int]], population_size: int = 100, mutation_rate: float = 0.1):
        self.initial_board = np.array(initial_board)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.best_solution = None
        self.best_fitness = float('-inf')

    def create_individual(self) -> np.ndarray:
        """Create a valid individual (a Sudoku solution)"""
        board = self.initial_board.copy()
        # Fill the missing numbers in each 3x3 block
        for block_row in range(3):
            for block_col in range(3):
                self._fill_block(board, block_row, block_col)
        return board

    def _fill_block(self, board: np.ndarray, block_row: int, block_col: int):
        """Fill the missing numbers in a 3x3 block"""
        start_row = block_row * 3
        start_col = block_col * 3
        
        # Find the numbers already in the block
        used_numbers = set()
        for i in range(3):
            for j in range(3):
                val = board[start_row + i][start_col + j]
                if val != 0:
                    used_numbers.add(val)
        
        # Create a list of missing numbers
        missing_numbers = list(set(range(1, 10)) - used_numbers)
        random.shuffle(missing_numbers)
        
        # Fill the missing numbers
        idx = 0
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == 0:
                    board[start_row + i][start_col + j] = missing_numbers[idx]
                    idx += 1

    def fitness(self, board: np.ndarray) -> float:
        """Calculate the fitness of a solution"""
        penalty = 0
        
        # Check rows
        for i in range(9):
            row = board[i]
            penalty += len(set(row)) - len(row)
        
        # Check columns
        for j in range(9):
            col = board[:, j]
            penalty += len(set(col)) - len(col)
            
        return penalty

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Crossover two parent individuals to create a child"""
        child = parent1.copy()
        
        # Randomly select a block to exchange
        block_row = random.randint(0, 2)
        block_col = random.randint(0, 2)
        
        # Exchange blocks between two parents
        start_row = block_row * 3
        start_col = block_col * 3
        
        for i in range(3):
            for j in range(3):
                if self.initial_board[start_row + i][start_col + j] == 0:
                    child[start_row + i][start_col + j] = parent2[start_row + i][start_col + j]
                    
        return child

    def mutate(self, board: np.ndarray) -> np.ndarray:
        """Mutate an individual by swapping two numbers in a block"""
        if random.random() > self.mutation_rate:
            return board
            
        mutated = board.copy()
        block_row = random.randint(0, 2)
        block_col = random.randint(0, 2)
        
        start_row = block_row * 3
        start_col = block_col * 3
        
        # Randomly select two positions in the block to swap
        pos1 = (random.randint(0, 2), random.randint(0, 2))
        pos2 = (random.randint(0, 2), random.randint(0, 2))
        
        # Only swap if both positions are not fixed numbers initially
        if (self.initial_board[start_row + pos1[0]][start_col + pos1[1]] == 0 and 
            self.initial_board[start_row + pos2[0]][start_col + pos2[1]] == 0):
            mutated[start_row + pos1[0]][start_col + pos1[1]], mutated[start_row + pos2[0]][start_col + pos2[1]] = \
            mutated[start_row + pos2[0]][start_col + pos2[1]], mutated[start_row + pos1[0]][start_col + pos1[1]]
            
        return mutated

    def solve(self, max_generations: int = 1000) -> Tuple[np.ndarray, float]:
        """Solve Sudoku using genetic algorithm"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(max_generations):
            # print(f"Generation {generation + 1}/{max_generations}")
            # Evaluate fitness
            fitness_scores = [self.fitness(individual) for individual in population]
            # print(f"Best fitness: {max(fitness_scores)} and Worst fitness: {min(fitness_scores)}")

            # Update the best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_solution = population[best_idx].copy()
                
                # If a perfect solution is found
                if self.best_fitness == 0:
                    break
            
            # Selection
            new_population = []

            # Elitism: keep top 10% of the population
            elite_size = int(0.1 * self.population_size)
            elite_indices = np.argsort(fitness_scores)[:elite_size]
            new_population.extend([population[i] for i in elite_indices])

            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 5
                tournament = random.sample(list(enumerate(population)), tournament_size)
                parent1_idx = max(tournament, key=lambda x: fitness_scores[x[0]])[0]
                tournament = random.sample(list(enumerate(population)), tournament_size)
                parent2_idx = max(tournament, key=lambda x: fitness_scores[x[0]])[0]
                
                # Crossover and mutation
                child = self.crossover(population[parent1_idx], population[parent2_idx])
                child = self.mutate(child)
                new_population.append(child)
                
            population = new_population
            
        return self.best_solution, self.best_fitness

if __name__ == "__main__":
    initial_board = [
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9]
    ]

    initial_board_2 = [
        [8, 0, 2, 0, 0, 3, 5, 1, 0],
        [0, 6, 0, 0, 9, 1, 0, 0, 3],
        [7, 0, 1, 0, 0, 0, 8, 9, 4],
        [6, 0, 8, 0, 0, 4, 0, 2, 1],
        [0, 0, 0, 2, 5, 8, 0, 6, 0],
        [9, 2, 0, 3, 1, 0, 4, 0, 0],
        [0, 0, 0, 4, 0, 2, 7, 8, 0],
        [0, 0, 5, 0, 8, 9, 0, 0, 0],
        [2, 0, 0, 0, 0, 7, 1, 0, 0]
    ]

    zero_count = sum(row.count(0) for row in initial_board)
    print(f"Initial board has {zero_count} positions to fill")    
    solver = SudokuGA(initial_board)
    solution, fitness = solver.solve()
    
    print("Init board:")
    print(np.array(initial_board))
    print("\nSolution:")
    print(solution)
    print(f"\nFitness: {fitness}")