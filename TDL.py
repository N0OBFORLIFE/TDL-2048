import numpy as np
import random
import pickle
from copy import copy
from collections import namedtuple
from pathlib import Path
from collections import defaultdict
import pandas as pd
import json
from datetime import datetime
import uuid

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def action_name(a):
    return "UP RIGHT DOWN LEFT".split()[a]

class IllegalAction(Exception):
    pass

class GameOver(Exception):
    pass

def compress(row):
    "remove 0s in list"
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
        if board is not None:
            self.board = board
        else:
            self.reset()

    def reset(self):
        self.clear()
        self.board[random.choice(self.empty_tiles())] = 1
        self.board[random.choice(self.empty_tiles())] = 2

    def spawn_tile(self, random_tile=False):
        empty_tiles = self.empty_tiles()
        if len(empty_tiles) == 0:
            raise GameOver("Board full. Cant spawn tile.")
        if random_tile:
            k = 2 if np.random.rand() <= 0.1 else 1
            self.board[random.choice(empty_tiles)] = k
        else:
            self.board[empty_tiles[0]] = 1

    def clear(self):
        self.board = [0] * 16

    def empty_tiles(self):
        return [i for (i, v) in enumerate(self.board) if v == 0]

    def display(self):
        def format_row(lst):
            s = ""
            for l in lst:
                s += " {:3d}".format(l)
            return s
        for row in range(4):
            idx = row * 4
            print(format_row(self.base10_board[idx : idx + 4]))

    @property
    def base10_board(self):
        return [2 ** v if v > 0 else 0 for v in self.board]

    def act(self, a):
        original = self.board
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
        size = 4
        b = []
        for i in range(size):
            b.extend(self.board[i::4][::-1])
        self.board = b
        return self

    def merge_to_left(self):
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
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MetricsTracker:
    def __init__(self, metrics_dir="metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.session_metrics = defaultdict(list)
        self.agent_session_id = None
        
        self.sessions_file = self.metrics_dir / "session_mapping.json"
        self.session_mapping = self._load_session_mapping()

        Path("agents").mkdir(exist_ok=True)

    def _load_session_mapping(self):
        try:
            with self.sessions_file.open('r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_session_mapping(self):
        with self.sessions_file.open('w') as f:
            json.dump(self.session_mapping, f, indent=2)

    def _generate_session_id(self, agent):
        if self.agent_session_id:
            return self.agent_session_id
            
        unique_id = str(uuid.uuid4())[:8]
        agent_name = agent.__class__.__name__.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.agent_session_id = f"{agent_name}_{timestamp}_{unique_id}"
        
        self.session_mapping[self.agent_session_id] = {
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'agent_type': agent.__class__.__name__,
            'uuid': unique_id
        }
        self._save_session_mapping()
        
        return self.agent_session_id

    def print_progress(self, metrics, n_games, n_episode):
        print(f"\nGames Played: {n_games}")
        print(f"Avg Score: {metrics['score']:.1f}")
        print(f"Max Tile Achieved: {metrics['highest_tile']}")
        print(f"2048 Rate: {metrics['2048_rate']*100:.2f}%")
        print(f"Moves/Game: {metrics['moves_per_game']:.1f}")
        print(f"Merge Efficiency: {metrics['merge_efficiency']:.2f}")
    
    def _calculate_merge_efficiency(self, gameplays):
        total_score = sum(gp.game_reward for gp in gameplays)
        total_moves = sum(len(gp.transition_history) for gp in gameplays)
        return total_score / total_moves if total_moves > 0 else 0
    
    def _count_tiles_achieved(self, gameplays):
        tile_counts = defaultdict(int)
        for gp in gameplays:
            max_tile = gp.max_tile
            for tile_value in [128, 256, 512, 1024, 2048]:
                if max_tile >= tile_value:
                    tile_counts[tile_value] += 1
        return tile_counts

    def _cleanup_old_metrics(self, uuid):
        try:
            for metrics_file in self.metrics_dir.glob(f"*_{uuid}_metrics.json"):
                try:
                    metrics_file.unlink()
                    print(f"Removed old metrics file: {metrics_file}")
                except Exception as e:
                    print(f"Warning: Could not remove old metrics file {metrics_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error during metrics cleanup: {str(e)}")

    def _save_detailed_metrics(self):
        if self.agent_session_id is None:
            return

        metrics_file = self.metrics_dir / f"{self.agent_session_id}_metrics.json"

        metrics_data = {
            'session_id': self.agent_session_id,
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics_history': {
                k: [float(v) if isinstance(v, (np.floating, float)) else int(v) 
                   for v in vals]
                for k, vals in self.session_metrics.items()
            }
        }

        session_info = self.session_mapping.get(self.agent_session_id, {})
        uuid = session_info.get('uuid')

        if uuid:
            self._cleanup_old_metrics(uuid)

        with metrics_file.open('w') as f:
            json.dump(metrics_data, f, indent=2, cls=NumpyEncoder)

    def _load_previous_metrics(self, previous_session_id):
        try:
            metrics_file = self.metrics_dir / f"{previous_session_id}_metrics.json"
            if metrics_file.exists():
                with metrics_file.open('r') as f:
                    previous_metrics = json.load(f)
                    for key, values in previous_metrics['metrics_history'].items():
                        self.session_metrics[key].extend(values)
                return True
            return False
        except Exception as e:
            print(f"Warning: Could not load previous metrics: {str(e)}")
            return False

    def continue_training(self, previous_session_id, agent):
        previous_uuid = None
        if previous_session_id in self.session_mapping:
            previous_uuid = self.session_mapping[previous_session_id].get('uuid')
        
        parts = previous_session_id.split('_')
        if len(parts) >= 3 and previous_uuid:
            self.agent_session_id = f"{agent.__class__.__name__.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{previous_uuid}"
        else:
            self.agent_session_id = self._generate_session_id(agent)
            
        if self._load_previous_metrics(previous_session_id):
            print(f"Successfully loaded metrics from previous session: {previous_session_id}")
            print(f"Continuing with new session ID: {self.agent_session_id}")
            
            if previous_session_id in self.session_mapping:
                self.session_mapping[self.agent_session_id] = {
                    **self.session_mapping[previous_session_id],
                    'continued_from': previous_session_id,
                    'continued_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self._save_session_mapping()

            old_agent_files = list(Path("agents").glob(f"{previous_session_id}*.pkl"))
            for old_file in old_agent_files:
                try:
                    old_file.unlink()
                    print(f"Removed old agent file: {old_file}")
                except Exception as e:
                    print(f"Warning: Could not remove old agent file {old_file}: {str(e)}")
        else:
            print("No previous metrics found or could not load them. Starting fresh.")

    @property
    def score(self):
        """Average score across all games in the session"""
        return np.mean(self.session_metrics['score']) if self.session_metrics['score'] else 0
    
    @property
    def max_tile(self):
        return max(self.session_metrics['highest_tile']) if self.session_metrics['highest_tile'] else 0
    
    @property
    def win_rate(self):
        return np.mean(self.session_metrics['2048_rate']) if self.session_metrics['2048_rate'] else 0
    
    @property
    def moves_per_game(self):
        return np.mean(self.session_metrics['moves_per_game']) if self.session_metrics['moves_per_game'] else 0
    
    @property
    def merge_efficiency(self):
        """Average merge efficiency in the session"""
        return np.mean(self.session_metrics['merge_efficiency']) if self.session_metrics['merge_efficiency'] else 0
        
    def update(self, gameplays, agent=None):
        if self.agent_session_id is None and agent is not None:
            self.agent_session_id = self._generate_session_id(agent)
            
        metrics = {
            'score': np.mean([gp.game_reward for gp in gameplays]),
            'max_score': max([gp.game_reward for gp in gameplays]),
            'mean_max_tile': np.mean([gp.max_tile for gp in gameplays]),
            'highest_tile': max([gp.max_tile for gp in gameplays]),
            '2048_rate': np.mean([1 if gp.max_tile >= 2048 else 0 for gp in gameplays]),
            'moves_per_game': np.mean([len(gp.transition_history) for gp in gameplays]),
            'merge_efficiency': self._calculate_merge_efficiency(gameplays)
        }
        
        tile_counts = self._count_tiles_achieved(gameplays)
        for tile_value in [128, 256, 512, 1024, 2048]:
            metrics[f'{tile_value}_rate'] = tile_counts[tile_value] / len(gameplays)
        
        for key, value in metrics.items():
            self.session_metrics[key].append(value)
            
        self._save_detailed_metrics()
        return metrics

    def save_session(self, n_games, agent):
        if self.agent_session_id is None:
            self.agent_session_id = self._generate_session_id(agent)
            
        self._save_detailed_metrics()
            
        summaries_file = self.metrics_dir / "session_summaries.json"
        try:
            with summaries_file.open('r') as f:
                try:
                    summaries = json.load(f)
                except json.JSONDecodeError:
                    summaries = []
        except FileNotFoundError:
            summaries = []
        
        summary = {
            'session_id': self.agent_session_id,
            'agent_type': agent.__class__.__name__,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'n_games': int(n_games),
            'score': float(self.score),
            'max_tile': int(self.max_tile),
            'win_rate': float(self.win_rate),
            'moves_per_game': float(self.moves_per_game),
            'merge_efficiency': float(self.merge_efficiency)
        }
        
        updated = False
        for i, existing_summary in enumerate(summaries):
            if existing_summary.get('session_id') == self.agent_session_id:
                summaries[i] = summary
                updated = True
                break
                
        if not updated:
            summaries.append(summary)
        
        with summaries_file.open('w') as f:
            json.dump(summaries, f, indent=2, cls=NumpyEncoder)
            
        try:
            old_agent_files = list(Path("agents").glob(f"{self.agent_session_id}*.pkl"))
            for old_file in old_agent_files:
                try:
                    old_file.unlink()
                    print(f"Removed old agent file: {old_file}")
                except Exception as e:
                    print(f"Warning: Could not remove old agent file {old_file}: {str(e)}")
        except Exception as e:
            print(f"Warning: Error during agent file cleanup: {str(e)}")
                
        save_path = f"agents/{self.agent_session_id}_{n_games}games.pkl"
        pickle.dump((n_games, agent), open(save_path, "wb"))
        print(f"\nAgent saved to {save_path}")
        print(f"Metrics saved to {self.metrics_dir}")

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
                    "digit %d should be smaller than the base %d" % (v, self.TARGET_PO2)
                )
            n += v * k
            k *= self.TARGET_PO2
        return n

    def V(self, board, delta=None, debug=False):
        """Return the expected total future rewards of the board.
        Updates the LUTs if a delta is given and return the updated value.
        """
        if debug:
            print(f"V({board})")
        vals = []
        for i, (tp, LUT) in enumerate(zip(self.TUPLES, self.LUTS)):
            tiles = [board[i] for i in tp]
            tpid = self.tuple_id(tiles)
            if delta is not None:
                LUT[tpid] += delta
            v = LUT[tpid]
            if debug:
                print(f"LUTS[{i}][{tiles}]={v}")
            vals.append(v)
        return np.mean(vals)

    def evaluate(self, s, a):
        "Return expected total rewards of performing action (a) on the given board state (s)"
        b = Board(s)
        try:
            r = b.act(a)
            s_after = b.copyboard()
        except IllegalAction:
            return 0
        return r + self.V(s_after)

    def best_action(self, s):
        "returns the action with the highest expected total rewards on the state (s)"
        a_best = None
        r_best = -1
        for a in [UP, RIGHT, DOWN, LEFT]:
            r = self.evaluate(s, a)
            if r > r_best:
                r_best = r
                a_best = a
        return a_best

    def learn(self, s, a, r, s_after, s_next, alpha=0.01, debug=False):
        """Learn from a transition experience by updating the belief
        on the after state (s_after) towards the sum of the next transition rewards (r_next) and
        the belief on the next after state (s_after_next).
        """
        a_next = self.best_action(s_next)
        b = Board(s_next)
        try:
            r_next = b.act(a_next)
            s_after_next = b.copyboard()
            v_after_next = self.V(s_after_next)
        except IllegalAction:
            r_next = 0
            v_after_next = 0
        delta = r_next + v_after_next - self.V(s_after)
        if debug:
            print("s_next")
            Board(s_next).display()
            print("a_next", action_name(a_next), "r_next", r_next)
            print("s_after_next")
            Board(s_after_next).display()
            self.V(s_after_next, debug=True)
            print(
                f"delta ({delta:.2f}) = r_next ({r_next:.2f}) + v_after_next ({v_after_next:.2f}) - V(s_after) ({self.V(s_after):.2f})"
            )
            print(
                f"V(s_after) <- V(s_after) ({self.V(s_after):.2f}) + alpha * delta ({alpha} * {delta:.1f})"
            )
        self.V(s_after, alpha * delta)

Transition = namedtuple("Transition", "s, a, r, s_after, s_next")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile")

def play(agent, board, spawn_random_tile=False):
    "Return a gameplay of playing the given (board) until terminal states."
    b = Board(board)
    r_game = 0
    a_cnt = 0
    transition_history = []
    while True:
        a_best = agent.best_action(b.board)
        s = b.copyboard()
        try:
            r = b.act(a_best)
            r_game += r
            s_after = b.copyboard()
            b.spawn_tile(random_tile=spawn_random_tile)
            s_next = b.copyboard()
        except (IllegalAction, GameOver) as e:
            r = None
            s_after = None
            s_next = None
            break
        finally:
            a_cnt += 1
            transition_history.append(
                Transition(s=s, a=a_best, r=r, s_after=s_after, s_next=s_next)
            )
    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=2 ** max(b.board),
    )
    learn_from_gameplay(agent, gp)
    return gp

def learn_from_gameplay(agent, gp, alpha=0.1):
    "Learn transitions in reverse order except the terminal transition"
    for tr in gp.transition_history[:-1][::-1]:
        agent.learn(tr.s, tr.a, tr.r, tr.s_after, tr.s_next, alpha=alpha)

def load_agent(path):
    try:
        n_games, agent = pickle.load(path.open("rb"))
        
        filename = path.stem
        previous_session_id = '_'.join(filename.split('_')[:-1]) 
        
        return n_games, agent, previous_session_id
    except Exception as e:
        print(f"Error loading agent: {e}")
        return None, None, None

TUPLES = [
    # horizontal 4-tuples
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    # vertical 4-tuples
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    # all 4-tile squares
    [0, 1, 4, 5],
    [4, 5, 8, 9],
    [8, 9, 12, 13],
    [1, 2, 5, 6],
    [5, 6, 9, 10],
    [9, 10, 13, 14],
    [2, 3, 6, 7],
    [6, 7, 10, 11],
    [10, 11, 14, 15],
]

def modify_main_loop(agent, n_session=500, n_episode=100, spawn_random_tile=True, initial_games=0, previous_session_id=None):
    tracker = MetricsTracker()
    
    if previous_session_id:
        tracker.agent_session_id = previous_session_id
        tracker.continue_training(previous_session_id, agent)
    
    n_games = initial_games
    games_in_current_session = n_games % n_episode
    games_to_complete_session = n_episode - games_in_current_session if games_in_current_session != 0 else 0
    
    try:
        if games_to_complete_session > 0:
            print(f"Completing current session with {games_to_complete_session} games...")
            gameplays = []
            for i_ep in range(games_to_complete_session):
                gp = play(agent, None, spawn_random_tile=spawn_random_tile)
                gameplays.append(gp)
                n_games += 1
            
            metrics = tracker.update(gameplays, agent)
            tracker.print_progress(metrics, n_games, n_episode)
        
        remaining_sessions = n_session - ((n_games + n_episode - 1) // n_episode)
        
        for i_se in range(remaining_sessions):
            gameplays = []
            for i_ep in range(n_episode):
                gp = play(agent, None, spawn_random_tile=spawn_random_tile)
                gameplays.append(gp)
                n_games += 1
                
            metrics = tracker.update(gameplays, agent)
            tracker.print_progress(metrics, n_games, n_episode)
            
            if (i_se + 1) % 100 == 0:
                tracker.save_session(n_games, agent)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    finally:
        tracker.save_session(n_games, agent)
        return n_games, agent, tracker.agent_session_id

if __name__ == "__main__":
    agent = None
    n_games = 0
    previous_session_id = None
    
    path = Path("agents")
    saves = list(path.glob("*.pkl"))
    if len(saves) > 0:
        print("Found %d saved agents:" % len(saves))
        for i, f in enumerate(saves):
            print("{:2d} - {}".format(i, str(f)))
        k = input("input the id to load an agent, input nothing to create a fresh agent: ")
        if k.strip() != "":
            k = int(k)
            try:
                n_games, agent, previous_session_id = load_agent(saves[k])
                if agent is not None:
                    print(f"load agent {saves[k].stem}, {n_games} games played")
                else:
                    print("Failed to load agent, creating fresh one")
                    agent = nTupleNewrok(TUPLES)
                    n_games = 0
                    previous_session_id = None
            except Exception as e:
                print(f"Error loading agent: {e}")
                print("Creating fresh agent instead")
                agent = nTupleNewrok(TUPLES)
                n_games = 0
                previous_session_id = None
    
    if agent is None:
        print("initialize agent")
        agent = nTupleNewrok(TUPLES)
    
    n_games, agent, session_id = modify_main_loop(
        agent, 
        initial_games=n_games,
        previous_session_id=previous_session_id
    )