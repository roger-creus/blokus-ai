import numpy as np
from blokus_env.constants import PIECES

class GameLogic:
    """
    Class containing the core game logic for the Blokus environment.
    """
    def __init__(self, options):
        """
        Initialize the GameLogic class.
        """
        
        self.options = options
        self.initial_positions = [(0, 0), (0, self.options["board_size"]-1), (self.options["board_size"]-1, 0), (self.options["board_size"]-1, self.options["board_size"]-1)]

    def place_piece(self, board, pieces, player, piece_index, x, y, rotation, horizontal_flip=0, vertical_flip=0):
        """
        Place a piece on the board if the move is valid.

        Parameters:
        - board (np.ndarray): The current board state.
        - pieces (np.ndarray): Array indicating which pieces are available for each player.
        - player (int): The current player (1-indexed).
        - piece_index (int): Index of the piece to be placed.
        - x (int): X coordinate for the placement.
        - y (int): Y coordinate for the placement.
        - rotation (int): Rotation of the piece (0, 1, 2, 3).
        - horizontal_flip (int): Whether the piece is horizontally flipped (0 or 1).
        - vertical_flip (int): Whether the piece is vertically flipped (0 or 1).

        Returns:
        - bool: True if the piece is placed successfully, False otherwise.
        """
        piece_shape = self.get_piece_shape(piece_index, rotation, horizontal_flip, vertical_flip)
        if self.is_valid_move(board, pieces, player, piece_shape, x, y):
            for dx, dy in piece_shape:
                px, py = x + dx, y + dy
                board[px, py] = player
            pieces[player - 1, piece_index] = 0  # Mark the piece as used
            return True
        return False

    def get_piece_shape(self, piece_index, rotation, horizontal_flip, vertical_flip):
        """
        Get the shape of the piece after applying rotation and flips.

        Parameters:
        - piece_index (int): Index of the piece.
        - rotation (int): Rotation of the piece (0, 1, 2, 3).
        - horizontal_flip (int): Whether the piece is horizontally flipped (0 or 1).
        - vertical_flip (int): Whether the piece is vertically flipped (0 or 1).

        Returns:
        - list: List of coordinates representing the shape of the piece.
        """
        piece_name = list(PIECES.keys())[piece_index]
        piece_shape = PIECES[piece_name]
        piece_shape = self.rotate_piece(piece_shape, rotation)

        if horizontal_flip:
            piece_shape = self.flip_horizontally(piece_shape)
        if vertical_flip:
            piece_shape = self.flip_vertically(piece_shape)

        return piece_shape

    def rotate_piece(self, piece, rotation):
        """
        Rotate the piece.

        Parameters:
        - piece (list): List of coordinates representing the piece.
        - rotation (int): Rotation of the piece (0, 1, 2, 3).

        Returns:
        - list: Rotated piece.
        """
        if rotation == 0:
            return piece
        elif rotation == 1:
            return [(y, -x) for x, y in piece]
        elif rotation == 2:
            return [(-x, -y) for x, y in piece]
        elif rotation == 3:
            return [(-y, x) for x, y in piece]

    def flip_horizontally(self, piece):
        """
        Flip the piece horizontally.

        Parameters:
        - piece (list): List of coordinates representing the piece.

        Returns:
        - list: Horizontally flipped piece.
        """
        return [(-x, y) for x, y in piece]

    def flip_vertically(self, piece):
        """
        Flip the piece vertically.

        Parameters:
        - piece (list): List of coordinates representing the piece.

        Returns:
        - list: Vertically flipped piece.
        """
        return [(x, -y) for x, y in piece]

    def is_first_move(self, player, board):
        """
        Check if it is the first move for the player.

        Parameters:
        - player (int): The current player (1-indexed).
        - board (np.ndarray): The current board state.

        Returns:
        - bool: True if it is the first move, False otherwise.
        """
        initial_position = self.initial_positions[player - 1]
        return board[initial_position] == 0

    def is_valid_move(self, board, pieces, player, piece_shape, x, y):
        """
        Check if the move is valid.

        Parameters:
        - board (np.ndarray): The current board state.
        - pieces (np.ndarray): Array indicating which pieces are available for each player.
        - player (int): The current player (1-indexed).
        - piece_shape (list): List of coordinates representing the shape of the piece.
        - x (int): X coordinate for the placement.
        - y (int): Y coordinate for the placement.

        Returns:
        - bool: True if the move is valid, False otherwise.
        """
        # Ensure the piece is within the board boundaries and does not overlap existing pieces
        for dx, dy in piece_shape:
            px, py = x + dx, y + dy
            if not (0 <= px < self.options["board_size"] and 0 <= py < self.options["board_size"]):
                return False
            if board[px, py] != 0:
                return False

        # For the first move, the piece must cover the initial position
        if self.is_first_move(player, board):
            initial_position = self.initial_positions[player - 1]
            if not any((x + dx, y + dy) == initial_position for dx, dy in piece_shape):
                return False
        else:
            # For subsequent moves, the piece must touch another piece of the same player by corner
            if not self.touches_corner(board, player, piece_shape, x, y):
                return False
            # The piece must not touch another piece of the same player by side
            if self.touches_side(board, player, piece_shape, x, y):
                return False

        return True

    def get_valid_actions(self, board, pieces, player):
        """
        Get 1 valid action for the current player.

        Parameters:
        - board (np.ndarray): The current board state.
        - pieces (np.ndarray): Array indicating which pieces are available for each player.
        - player (int): The current player (1-indexed).

        Returns:
        - list: List of valid actions.
        """
        valid_actions = []
        piece_indices = np.random.permutation(len(PIECES))
        for piece_index in piece_indices:
            if len(valid_actions) > 0:
                break
            if pieces[player - 1, piece_index] == 1:  # Piece is available
                for rotation in range(4):
                    for horizontal_flip in range(2):  # Horizontal flip (0 or 1)
                        for vertical_flip in range(2):  # Vertical flip (0 or 1)
                            piece_shape = self.get_piece_shape(piece_index, rotation, horizontal_flip, vertical_flip)
                            for x in range(self.options["board_size"]):
                                for y in range(self.options["board_size"]):
                                    if self.is_valid_move(board, pieces, player, piece_shape, x, y):
                                        valid_actions.append((piece_index, x, y, rotation, horizontal_flip, vertical_flip))
        return valid_actions

    def get_invalid_action_masks(self, board, pieces, player):
        """
        Get masks for invalid actions for the current player.

        Parameters:
        - board (np.ndarray): The current board state.
        - pieces (np.ndarray): Array indicating which pieces are available for each player.
        - player (int): The current player (1-indexed).

        Returns:
        - np.ndarray: Masks for invalid actions.
        """
        invalid_action_masks = np.zeros((len(PIECES), self.options["board_size"], self.options["board_size"], 4, 2, 2))
        for piece_index in range(len(PIECES)):
            if pieces[player - 1, piece_index] == 1:
                for rotation in range(4):
                    for horizontal_flip in range(2):
                        for vertical_flip in range(2):
                            piece_shape = self.get_piece_shape(piece_index, rotation, horizontal_flip, vertical_flip)
                            for x in range(self.options["board_size"]):
                                for y in range(self.options["board_size"]):
                                    if self.is_valid_move(board, pieces, player, piece_shape, x, y):
                                        invalid_action_masks[piece_index, x, y, rotation, horizontal_flip, vertical_flip] = 1
        return invalid_action_masks

    def is_game_over(self, board, pieces):
        """
        Check if the game is over (no valid moves left for any player).

        Parameters:
        - board (np.ndarray): The current board state.
        - pieces (np.ndarray): Array indicating which pieces are available for each player.

        Returns:
        - bool: True if the game is over, False otherwise.
        """
        info = {}
        for player in range(1, self.options["players"] + 1):
            for piece_index, available in enumerate(pieces[player - 1]):
                if available:
                    for x in range(self.options["board_size"]):
                        for y in range(self.options["board_size"]):
                            for rotation in range(4):
                                for horizontal_flip in range(2):  # Horizontal flip (0 or 1)
                                    for vertical_flip in range(2):  # Vertical flip (0 or 1)
                                        if self.is_valid_move(
                                            board, pieces, player,
                                            self.get_piece_shape(piece_index, rotation, horizontal_flip, vertical_flip),
                                            x, y
                                        ):
                                            return False
        return True

    def touches_corner(self, board, player, piece_shape, x, y):
        """
        Check if the piece touches another piece of the same player by the corner.

        Parameters:
        - board (np.ndarray): The current board state.
        - player (int): The current player (1-indexed).
        - piece_shape (list): List of coordinates representing the shape of the piece.
        - x (int): X coordinate for the placement.
        - y (int): Y coordinate for the placement.

        Returns:
        - bool: True if the piece touches another piece of the same player by the corner, False otherwise.
        """
        adjacent_corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in piece_shape:
            px, py = x + dx, y + dy
            for ax, ay in adjacent_corners:
                cx, cy = px + ax, py + ay
                if 0 <= cx < self.options["board_size"] and 0 <= cy < self.options["board_size"]:
                    if board[cx, cy] == player:
                        return True
        return False

    def touches_side(self, board, player, piece_shape, x, y):
        """
        Check if the piece touches another piece of the same player by the side.

        Parameters:
        - board (np.ndarray): The current board state.
        - player (int): The current player (1-indexed).
        - piece_shape (list): List of coordinates representing the shape of the piece.
        - x (int): X coordinate for the placement.
        - y (int): Y coordinate for the placement.

        Returns:
        - bool: True if the piece touches another piece of the same player by the side, False otherwise.
        """
        adjacent_sides = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in piece_shape:
            px, py = x + dx, y + dy
            for ax, ay in adjacent_sides:
                sx, sy = px + ax, py + ay
                if 0 <= sx < self.options["board_size"] and 0 <= sy < self.options["board_size"]:
                    if board[sx, sy] == player:
                        return True
        return False