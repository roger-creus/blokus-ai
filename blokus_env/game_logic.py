# blokus_env/game_logic.py

import numpy as np
from blokus_env.constants import BOARD_SIZE, PIECES, NUM_PLAYERS, INITIAL_POSITIONS
from IPython import embed

class GameLogic:
    def __init__(self):
        pass

    def place_piece(self, board, pieces, player, piece_index, x, y, rotation, horizontal_flip=0, vertical_flip=0):
        piece_shape = self.get_piece_shape(piece_index, rotation, horizontal_flip, vertical_flip)
        if self.is_valid_move(board, pieces, player, piece_shape, x, y):
            for dx, dy in piece_shape:
                px, py = x + dx, y + dy
                board[px, py] = player
            pieces[player - 1, piece_index] = 0  # Mark the piece as used
            return True
        return False

    def get_piece_shape(self, piece_index, rotation, horizontal_flip, vertical_flip):
        piece_name = list(PIECES.keys())[piece_index]
        piece_shape = PIECES[piece_name]
        piece_shape = self.rotate_piece(piece_shape, rotation)

        if horizontal_flip:
            piece_shape = self.flip_horizontally(piece_shape)
        if vertical_flip:
            piece_shape = self.flip_vertically(piece_shape)

        return piece_shape

    def rotate_piece(self, piece, rotation):
        if rotation == 0:
            return piece
        elif rotation == 1:
            return [(y, -x) for x, y in piece]
        elif rotation == 2:
            return [(-x, -y) for x, y in piece]
        elif rotation == 3:
            return [(-y, x) for x, y in piece]

    def flip_horizontally(self, piece):
        return [(-x, y) for x, y in piece]

    def flip_vertically(self, piece):
        return [(x, -y) for x, y in piece]
    
    def is_first_move(self, player, board):
        initial_position = INITIAL_POSITIONS[player - 1]
        return board[initial_position] == 0

    def is_valid_move(self, board, pieces, player, piece_shape, x, y):
        # Ensure the piece is within the board boundaries and does not overlap existing pieces
        for dx, dy in piece_shape:
            px, py = x + dx, y + dy
            if not (0 <= px < BOARD_SIZE and 0 <= py < BOARD_SIZE):
                return False
            if board[px, py] != 0:
                return False

        # For the first move, the piece must cover the initial position
        if self.is_first_move(player, board):
            initial_position = INITIAL_POSITIONS[player - 1]
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
                            for x in range(BOARD_SIZE):
                                for y in range(BOARD_SIZE):
                                    if self.is_valid_move(board, pieces, player, piece_shape, x, y):
                                        valid_actions.append((piece_index, x, y, rotation, horizontal_flip, vertical_flip))
        return valid_actions
    
    def get_invalid_action_masks(self, board, pieces, player):
        invalid_action_masks = np.zeros((len(PIECES), BOARD_SIZE, BOARD_SIZE, 4, 2, 2))
        for piece_index in range(len(PIECES)):
            if pieces[player - 1, piece_index] == 1:
                for rotation in range(4):
                    for horizontal_flip in range(2):
                        for vertical_flip in range(2):
                            piece_shape = self.get_piece_shape(piece_index, rotation, horizontal_flip, vertical_flip)
                            for x in range(BOARD_SIZE):
                                for y in range(BOARD_SIZE):
                                    if self.is_valid_move(board, pieces, player, piece_shape, x, y):
                                        invalid_action_masks[piece_index, x, y, rotation, horizontal_flip, vertical_flip] = 1
        return invalid_action_masks

    def is_game_over(self, board, pieces):
        # Check if the game is over (no valid moves left for any player)
        info = {}
        for player in range(1, NUM_PLAYERS + 1):
            for piece_index, available in enumerate(pieces[player - 1]):
                if available:
                    for x in range(BOARD_SIZE):
                        for y in range(BOARD_SIZE):
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
        adjacent_corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in piece_shape:
            px, py = x + dx, y + dy
            for ax, ay in adjacent_corners:
                cx, cy = px + ax, py + ay
                if 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
                    if board[cx, cy] == player:
                        return True
        return False

    def touches_side(self, board, player, piece_shape, x, y):
        adjacent_sides = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in piece_shape:
            px, py = x + dx, y + dy
            for ax, ay in adjacent_sides:
                sx, sy = px + ax, py + ay
                if 0 <= sx < BOARD_SIZE and 0 <= sy < BOARD_SIZE:
                    if board[sx, sy] == player:
                        return True
        return False
