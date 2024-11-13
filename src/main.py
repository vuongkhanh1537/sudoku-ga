import time
import pandas as pd
from typing import List, Dict
import concurrent.futures
from tabulate import tabulate
from sudoku_generator import SudokuGenerator
from sudoku import SudokuGA
import random


def benchmark_sudoku_ga(num_boards: int, difficulties: List[str] = None, 
                       max_generations: int = 1000, timeout: int = 300) -> pd.DataFrame:
    """
    Run GA benchmark on multiple Sudoku boards and record results
    Args:
        num_boards: Number of boards to test
        difficulties: List of difficulties to test. If None, 'medium' will be used
        max_generations: Maximum number of generations for each solve
        timeout: Maximum allowed time (seconds) for each board
    Returns:
        DataFrame containing benchmark results
    """
    if difficulties is None:
        difficulties = ['medium']
    
    results = []
    generator = SudokuGenerator()
    
    def solve_single_board(board_idx: int, difficulty: str) -> Dict:
        """
        Solve a single Sudoku board and record the results
        """
        # Generate a new board
        initial_board = generator.generate_board_with_difficulty(difficulty)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Create and run solver
            solver = SudokuGA(initial_board)
            solution, generations = solver.solve(generations=max_generations)
            
            # Calculate time and check result
            solve_time = time.time() - start_time
            solved = (solver.calculate_fitness(solution) == 0)
            
            return {
                'Board': board_idx + 1,
                'Difficulty': difficulty,
                'Solved': solved,
                'Time (s)': round(solve_time, 2),
                'Generations': generations,
                'Initial Board': initial_board.copy(),
                'Solution': solution.copy() if solved else None
            }
            
        except Exception as e:
            return {
                'Board': board_idx + 1,
                'Difficulty': difficulty,
                'Solved': False,
                'Time (s)': timeout,
                'Generations': max_generations,
                'Error': str(e),
                'Initial Board': initial_board.copy(),
                'Solution': None
            }
    
    # Create list of tasks to perform
    tasks = []
    for i in range(num_boards):
        difficulty = random.choice(difficulties)
        tasks.append((i, difficulty))
    
    # Use ThreadPoolExecutor to run in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_board = {
            executor.submit(solve_single_board, idx, diff): (idx, diff) 
            for idx, diff in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_board):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                idx, diff = future_to_board[future]
                results.append({
                    'Board': idx + 1,
                    'Difficulty': diff,
                    'Solved': False,
                    'Time (s)': timeout,
                    'Generations': max_generations,
                    'Error': 'Timeout',
                    'Initial Board': None,
                    'Solution': None
                })
    
    # Convert results to DataFrame and sort by board number
    df = pd.DataFrame(results)
    df = df.sort_values('Board')
    
    return df

def print_benchmark_results(df: pd.DataFrame, show_boards: bool = False):
    """
    Print benchmark results in a readable format
    Args:
        df: DataFrame containing benchmark results
        show_boards: If True, print input boards and solutions
    """
    # Calculate statistics
    total_boards = len(df)
    solved_boards = df['Solved'].sum()
    success_rate = solved_boards / total_boards * 100
    avg_time = df['Time (s)'].mean()
    avg_generations = df['Generations'].mean()
    
    # Print overall statistics
    print("\n=== SUDOKU GA BENCHMARK RESULTS ===")
    print(f"Total boards tested: {total_boards}")
    print(f"Successfully solved: {solved_boards} ({success_rate:.1f}%)")
    print(f"Average time per board: {avg_time:.2f} seconds")
    print(f"Average generations per board: {int(avg_generations)}")
    
    # Print detailed results table
    results_table = df[['Board', 'Difficulty', 'Solved', 'Time (s)', 'Generations']]
    print("\n=== DETAILED RESULTS ===")
    print(tabulate(results_table, headers='keys', tablefmt='grid'))
    
    # Print statistics by difficulty
    print("\n=== RESULTS BY DIFFICULTY ===")
    difficulty_stats = df.groupby('Difficulty').agg({
        'Solved': ['count', 'sum', lambda x: (sum(x)/len(x)*100)],
        'Time (s)': 'mean',
        'Generations': 'mean'
    }).round(2)
    difficulty_stats.columns = ['Total', 'Solved', 'Success Rate %', 'Avg Time (s)', 'Avg Generations']
    print(tabulate(difficulty_stats, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    # Test with different difficulties
    difficulties = ['easy', 'medium', 'hard']
    results_df = benchmark_sudoku_ga(
        num_boards=10,
        difficulties=difficulties,
        max_generations=1000,
        timeout=300
    )
    
    # Print results
    print_benchmark_results(results_df, show_boards=False)
    
    # # Print details of some successful and failed boards
    # success_board = results_df[results_df['Solved']].iloc[0]
    # failed_board = results_df[~results_df['Solved']].iloc[0] if len(results_df[~results_df['Solved']]) > 0 else None
    
    # print("\nExample of a successfully solved board:")
    # print(f"Board {success_board['Board']} (Time: {success_board['Time (s)']}s)")
    # print("Initial board:")
    # print_board(success_board['Initial Board'])
    # print("\nSolution:")
    # print_board(success_board['Solution'])
    
    # if failed_board is not None:
    #     print("\nExample of an unsolved board:")
    #     print(f"Board {failed_board['Board']} (Time: {failed_board['Time (s)']}s)")
    #     print("Initial board:")
    #     print_board(failed_board['Initial Board'])