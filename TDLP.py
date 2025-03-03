# Agent Play
import pygame
import numpy as np
import random
import pickle
from pathlib import Path
import time
from collections import namedtuple
from copy import copy

SIZE = 4
TILE_SIZE = 100
GAP_SIZE = 10
MARGIN = 20
SCREEN_SIZE = SIZE * TILE_SIZE + (SIZE + 1) * GAP_SIZE + 2 * MARGIN
BACKGROUND_COLOR = (250, 248, 239)
EMPTY_TILE_COLOR = (205, 193, 180)
SHADOW_COLOR = (187, 173, 160)
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}
FONT_COLORS = {
    "dark": (119, 110, 101),
    "light": (249, 246, 242)
}

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class IllegalAction(Exception):
    pass

class GameOver(Exception):
    pass

def compress(row):
    "remove all 0 in list"
    return [x for x in row if x != 0]

def merge(row):
    row = compress(row)
    reward = 0
    r = []
    hold = -1
    while len(row) > 0:
        v = row.pop(0)
        if hold != -1:
            if hold == v:
                reward = reward + (2 ** (hold + 1))
                r.append(hold + 1)
                hold = -1
            else:
                r.append(hold)
                hold = v
        else:
            hold = v
    if hold != -1:
        r.append(hold)
        hold = -1
    while len(r) < 4:
        r.append(0)
    return reward, r

class Board:
    def __init__(self, board=None):
        """board is a list of 16 integers"""
        if board is not None:
            self.board = board
        else:
            self.reset()

    def reset(self):
        self.clear()
        self.spawn_tile()
        self.spawn_tile()

    def clear(self):
        self.board = [0] * 16

    def spawn_tile(self, random_tile=True):
        empty_tiles = self.empty_tiles()
        if len(empty_tiles) == 0:
            raise GameOver("Board is full. Cannot spawn any tile.")
        if random_tile:
            k = 2 if random.random() < 0.1 else 1
            self.board[random.choice(empty_tiles)] = k
        else:
            self.board[empty_tiles[0]] = 1

    def empty_tiles(self):
        return [i for (i, v) in enumerate(self.board) if v == 0]

    def get_display_board(self):
        return [2 ** v if v > 0 else 0 for v in self.board]

    def act(self, a):
        original = self.board.copy()
        if a == LEFT:
            r = self.merge_to_left()
        if a == RIGHT:
            r = self.rotate().rotate().merge_to_left()
            self.rotate().rotate()
        if a == UP:
            r = self.rotate().rotate().rotate().merge_to_left()
            self.rotate()
        if a == DOWN:
            r = self.rotate().merge_to_left()
            self.rotate().rotate().rotate()
        if original == self.board:
            raise IllegalAction("Action did not move any tile.")
        return r

    def rotate(self):
        "Rotate the board inplace 90 degrees clockwise."
        size = 4
        b = []
        for i in range(size):
            b.extend(self.board[i::4][::-1])
        self.board = b
        return self

    def merge_to_left(self):
        "merge board to the left, returns the reward for merging tiles"
        r = []
        board_reward = 0
        for nrow in range(4):
            idx = nrow * 4
            row = self.board[idx : idx + 4]
            row_reward, row = merge(row)
            board_reward = board_reward + row_reward
            r.extend(row)
        self.board = r
        return board_reward

    def copyboard(self):
        return copy(self.board)

class nTupleNewrok:
    def __init__(self, tuples):
        self.TUPLES = tuples
        self.TARGET_PO2 = 15
        self.LUTS = self.initialize_LUTS(self.TUPLES)

    def initialize_LUTS(self, tuples):
        LUTS = []
        for tp in tuples:
            LUTS.append(np.zeros((self.TARGET_PO2 + 1) ** len(tp)))
        return LUTS

    def tuple_id(self, values):
        values = values[::-1]
        k = 1
        n = 0
        for v in values:
            if v >= self.TARGET_PO2:
                raise ValueError(
                    f"digit {v} should be smaller than the base {self.TARGET_PO2}"
                )
            n += v * k
            k *= self.TARGET_PO2
        return n

    def V(self, board, delta=None):
        vals = []
        for i, (tp, LUT) in enumerate(zip(self.TUPLES, self.LUTS)):
            tiles = [board[i] for i in tp]
            tpid = self.tuple_id(tiles)
            if delta is not None:
                LUT[tpid] += delta
            v = LUT[tpid]
            vals.append(v)
        return np.mean(vals)

    def evaluate(self, s, a):
        b = Board(s.copy())
        try:
            r = b.act(a)
            return r + self.V(b.board)
        except IllegalAction:
            return -float('inf')

    def best_action(self, s):
        a_best = None
        r_best = -float('inf')
        for a in [UP, RIGHT, DOWN, LEFT]:
            r = self.evaluate(s, a)
            if r > r_best:
                r_best = r
                a_best = a
        return a_best

def draw_tile(screen, value, x, y):
    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
    if value == 0:
        pygame.draw.rect(screen, EMPTY_TILE_COLOR, rect, border_radius=5)
    else:
        shadow_rect = pygame.Rect(x+4, y+4, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, SHADOW_COLOR, shadow_rect, border_radius=5)
        color = TILE_COLORS.get(value, (60, 58, 50))
        pygame.draw.rect(screen, color, rect, border_radius=5)
        font_color = FONT_COLORS["dark"] if value in (2, 4) else FONT_COLORS["light"]
        font = pygame.font.SysFont('Arial', 40 if value < 100 else 32 if value < 1000 else 24, bold=True)
        text = font.render(str(value), True, font_color)
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)
        
def has_legal_moves(board):
    if len(board.empty_tiles()) > 0:
        return True

    for i in range(0, 16, 4):
        for j in range(i, i + 3):
            if board.board[j] == board.board[j + 1] and board.board[j] != 0:
                return True

    for i in range(0, 4):
        for j in range(i, i + 12, 4):
            if board.board[j] == board.board[j + 4] and board.board[j] != 0:
                return True
    
    return False

def get_game_mode():
    print("\nSelect game mode:")
    print("1 - Play (Arrow/WASD to move)")
    print("2 - TDL Agent")
    while True:
        try:
            mode = int(input("1/2: "))
            if mode in [1, 2]:
                return mode
            print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number")

def setup_game():
    mode = get_game_mode()
    agent = None
    move_delay = 0
    
    if mode == 2:
        path = Path("agents")
        saves = list(path.glob("*.pkl"))
        if len(saves) == 0:
            print("no agents found")
            return None, None, None
            
        print("saved agents:")
        for i, f in enumerate(saves):
            print(f"{i} - {f}")
        
        try:
            k = 0#int(input("Enter the number of the agent to load: "))
            n_games, agent = pickle.load(saves[k].open("rb"))
            print(f"Loaded agent that played {n_games} games")
            move_delay = 0
        except Exception as e:
            print(f"Error loading agent: {e}")
            return None, None, None
            
    return mode, agent, move_delay

def main():
    mode, agent, move_delay = setup_game()
    if mode is None:
        return

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption('2048 Game')
    
    board = Board()
    score = 0
    running = True
    max_tile = 0
    moves = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif mode == 1:
                    action = None
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        action = LEFT
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        action = RIGHT
                    elif event.key == pygame.K_UP or event.key == pygame.K_w:
                        action = UP
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        action = DOWN
                        
                    if action is not None:
                        try:
                            reward = board.act(action)
                            score += reward
                            moves += 1
                            board.spawn_tile(random_tile=True)
                        except IllegalAction:
                            pass
                        
        if mode == 2 and agent: 
            action = agent.best_action(board.board)
            if action is None or not has_legal_moves(board):
                print(f"Game Over! Final Score: {score}, Max Tile: {max_tile}, Moves: {moves}")
                break
                
            try:
                reward = board.act(action)
                score += reward
                moves += 1
                board.spawn_tile(random_tile=True)
                time.sleep(move_delay)
            except (IllegalAction, GameOver):
                print(f"Game Over! Final Score: {score}, Max Tile: {max_tile}, Moves: {moves}")
                break

        if mode == 1 and not has_legal_moves(board):
            print(f"Game Over! Final Score: {score}, Max Tile: {max_tile}, Moves: {moves}")
            time.sleep(2)
            break

        current_max = max(2 ** v if v > 0 else 0 for v in board.board)
        if current_max > max_tile:
            max_tile = current_max
            print(f"New max tile achieved: {max_tile}")

        screen.fill(BACKGROUND_COLOR)
        display_board = board.get_display_board()
        for i in range(SIZE):
            for j in range(SIZE):
                x = MARGIN + GAP_SIZE + j*(TILE_SIZE + GAP_SIZE)
                y = MARGIN + GAP_SIZE + i*(TILE_SIZE + GAP_SIZE)
                draw_tile(screen, display_board[i*SIZE + j], x, y)

        font = pygame.font.SysFont('Arial', 32, bold=True)
        score_text = font.render(f"Score: {score}", True, FONT_COLORS["dark"])
        max_tile_text = font.render(f"Max Tile: {max_tile}", True, FONT_COLORS["dark"])
        moves_text = font.render(f"Moves: {moves}", True, FONT_COLORS["dark"])
        
        screen.blit(score_text, (MARGIN, 5))
        screen.blit(max_tile_text, (SCREEN_SIZE//2 - max_tile_text.get_width()//2, 5))
        screen.blit(moves_text, (SCREEN_SIZE - MARGIN - moves_text.get_width(), 5))
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()