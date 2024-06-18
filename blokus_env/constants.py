BOARD_SIZE = 20
NUM_PLAYERS = 4
INITIAL_POSITIONS = [(0, 0), (0, BOARD_SIZE-1), (BOARD_SIZE-1, 0), (BOARD_SIZE-1, BOARD_SIZE-1)]
PLAYER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
COLORS = ["red", "green", "blue", "yellow"]
PIECES = {
    # Monomino
    'monomino': [(0, 0)],

    # Domino
    'domino': [(0, 0), (0, 1)],

    # Trominoes
    'tromino_line': [(0, 0), (0, 1), (0, 2)],
    'tromino_l': [(0, 0), (0, 1), (1, 0)],

    # Tetrominoes
    'tetromino_square': [(0, 0), (0, 1), (1, 0), (1, 1)],
    'tetromino_line': [(0, 0), (0, 1), (0, 2), (0, 3)],
    'tetromino_t': [(0, 0), (1, 0), (1, 1), (1, 2)],
    'tetromino_l': [(0, 0), (1, 0), (2, 0), (2, 1)],
    'tetromino_z': [(0, 0), (0, 1), (1, 1), (1, 2)],

    # Pentominoes
    'pentomino_line': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    'pentomino_l': [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    'pentomino_v': [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    'pentomino_t': [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)],
    'pentomino_z': [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    'pentomino_w': [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    'pentomino_x': [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    'pentomino_p': [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    'pentomino_q': [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
    'pentomino_u': [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)],
    'pentomino_y': [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)],
    'pentomino_n': [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)]
}
