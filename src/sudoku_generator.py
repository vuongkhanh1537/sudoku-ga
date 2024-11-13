import numpy as np
import random
from typing import List, Tuple, Set

class SudokuGenerator:
    def __init__(self) -> None:
        self.board = np.zeros((9, 9), dtype=int)

    def is_valid(self, num: int, pos: Tuple[int, int]) -> bool:
        """
        Kiểm tra xem số num có thể đặt vào vị trí pos không
        Args:
            num: Số cần kiểm tra (1-9)
            pos: Tuple (row, col) chỉ vị trí cần kiểm tra
        Returns:
            bool: True nếu hợp lệ, False nếu không
        """
        row, col = pos

        # Kiểm tra hàng
        if num in self.board[row]:
            return False

        # Kiểm tra cột
        if num in self.board[:, col]:
            return False

        # Kiểm tra block 3x3
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if self.board[i, j] == num:
                    return False

        return True

    def find_empty(self) -> Tuple[int, int]:
        """
        Tìm một ô trống trong bảng
        Returns:
            Tuple (row, col) của ô trống đầu tiên tìm thấy
            None nếu không có ô trống
        """
        for i in range(9):
            for j in range(9):
                if self.board[i, j] == 0:
                    return (i, j)
        return None

    def solve_board(self) -> bool:
        """
        Giải bảng Sudoku hiện tại sử dụng backtracking
        Returns:
            bool: True nếu có lời giải, False nếu không
        """
        empty = self.find_empty()
        if not empty:
            return True

        row, col = empty
        numbers = list(range(1, 10))
        random.shuffle(numbers)  # Xáo trộn để tạo bảng ngẫu nhiên

        for num in numbers:
            if self.is_valid(num, (row, col)):
                self.board[row, col] = num

                if self.solve_board():
                    return True

                self.board[row, col] = 0

        return False

    def generate_board(self, num_clues: int) -> np.ndarray:
        """
        Tạo bảng Sudoku với số ô điền sẵn cho trước
        Args:
            num_clues: Số ô điền sẵn mong muốn (17-30 là phổ biến)
        Returns:
            np.ndarray: Bảng Sudoku với các ô điền sẵn
        """
        # Đảm bảo số ô điền sẵn hợp lệ
        num_clues = max(17, min(num_clues, 81))  # Ít nhất 17 ô để có lời giải duy nhất

        # Tạo một bảng Sudoku hoàn chỉnh
        self.board = np.zeros((9, 9), dtype=int)
        self.solve_board()
        complete_board = self.board.copy()

        # Danh sách các vị trí có thể xóa
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)

        # Xóa các ô cho đến khi đạt được số ô điền sẵn mong muốn
        num_to_remove = 81 - num_clues
        removed_positions = []

        for pos in positions[:num_to_remove]:
            row, col = pos
            temp = self.board[row, col]
            self.board[row, col] = 0
            removed_positions.append((row, col, temp))

        return self.board
    
    def generate_board_with_difficulty(self, difficulty: str = 'medium') -> np.ndarray:
        """
        Tạo bảng Sudoku với độ khó cho trước
        Args:
            difficulty: 'easy', 'medium', 'hard', hoặc 'expert'
        Returns:
            np.ndarray: Bảng Sudoku với độ khó tương ứng
        """
        difficulty_clues = {
            'easy': (35, 45),      # Nhiều ô điền sẵn
            'medium': (28, 34),    # Số ô điền sẵn trung bình
            'hard': (22, 27),      # Ít ô điền sẵn
            'expert': (17, 21)     # Rất ít ô điền sẵn
        }

        if difficulty not in difficulty_clues:
            difficulty = 'medium'

        min_clues, max_clues = difficulty_clues[difficulty]
        num_clues = random.randint(min_clues, max_clues)
        return self.generate_board(num_clues)
