from game import GameEngine, Player
from rl_player import RLPlayer
import random

debug = False

def play_test_game():
    # 1. Initialize the agent and set epsilon to 0 (no random exploration)
    trained_agent = RLPlayer("TrainedBot", epsilon=0.0, epsilon_min=0.0)
    
    # 2. Load the trained weights
    trained_agent.load_model("model/6nimmt_dqn_final.pth")
    
    # 3. Setup opponents
    bots = [Player("RandomBot_1"), Player("RandomBot_2"), Player("RandomBot_3")]
    players = [trained_agent] + bots
    
    # 4. Run a standard game
    engine = GameEngine(players, debug=debug)
    
    print("\nStarting evaluation game...")
    while not engine.is_game_over(threshold=66):
        engine.start_new_round()
        
        for turn in range(1, 11):
            turn_actions = {}
            for p in engine.players:
                if isinstance(p, RLPlayer):
                    # The agent will use the loaded Neural Network to pick
                    chosen_card_number = p.act(engine.board) 
                else:
                    chosen_card_number = random.choice(p.hand).number 
                turn_actions[p] = chosen_card_number
            
            engine.play_turn(turn_actions)
            
    # Show Results
    # print("\n--- Final Results ---")
    # engine.players.sort(key=lambda p: p.score)
    # for i, player in enumerate(engine.players):
    #     print(f"{i + 1}. {player.name}: {player.score} points")

    return engine.get_winner()

if __name__ == "__main__":
    rl_wins = 0
    random_wins = 0
    trials = 10_000
    for _ in range(trials):
        winner = play_test_game()
        if isinstance(winner, RLPlayer):
            rl_wins += 1
        else:
            random_wins += 1
        print(f"Trial {_+1}/{trials} - Winner: {winner.name} (RL Wins: {rl_wins}, Random Wins: {random_wins})", end="\r")

    print(f"\nAfter {trials} trials:")
    print(f"RL Agent wins: {rl_wins}")
    print(f"Random Bots wins: {random_wins}")
