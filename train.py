from game import GameEngine, Player
from rl_player import RLPlayer
import random

def train_agent(episodes=1000):
    # Setup 1 RL Agent and 3 Random Bots (using standard Player class)
    agent = RLPlayer("DeepMindBot")
    bots = [Player(f"Bot_{i}") for i in range(1, 4)]
    players = [agent] + bots
    
    engine = GameEngine(players)
    
    for e in range(episodes):
        engine.setup_game()
        done = False
        
        print(f"\n--- start new game ---")
        while not engine.is_game_over(threshold=66):
            # A round consists of exactly 10 turns
            for turn in range(1, 11):
                turn_actions = {}
                
                # 1. Collect actions
                for p in engine.players:
                    if isinstance(p, RLPlayer):
                        chosen_card_number = p.act(engine.board)
                    else:
                        chosen_card_number = random.choice(p.hand).number 
                    turn_actions[p] = chosen_card_number
                
                # Note the agent's score before the turn resolves
                agent_score_before = agent.score
                
                # 2. Process the turn (engine updates the board and player scores)
                engine.play_turn(turn_actions)
                
                # 3. Calculate Reward & Learn
                penalty_taken = agent.score - agent_score_before
                reward = -penalty_taken  # Negative reward for taking bullheads
                
                # Check if this was the last turn of the game
                done = engine.is_game_over(threshold=66)
                
                # Store experience
                agent.remember(reward, engine.board, done)
                
                # Train the network on a batch of memories
                agent.replay(batch_size=32)
                
                if done:
                    break
                    
            if not done:
                print(f"\n--- start new round ---")
                engine.start_new_round()
                
        print(f"Episode {e+1}/{episodes} - Agent Score: {agent.score} - Epsilon: {agent.epsilon:.5f}")
            
        # Optional: Save a checkpoint every 100 episodes
        if (e + 1) % 10 == 0:
            agent.save_model(f"model/6nimmt_dqn_checkpoint_{e+1}.pth")

    # Save the final model when training is completely done
    agent.save_model("model/6nimmt_dqn_final.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_agent(episodes=200)
