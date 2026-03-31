from game import Player
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from dqn import DQN

class RLPlayer(Player):
    def __init__(self, name, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.gamma = 0.95  
        self.epsilon = epsilon  
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=10000) 
        
        # Buffer to hold all actions taken in a single turn 
        # (could be just 1 card play, or 1 card play + 1 row selection)
        self.turn_transitions = []

    def get_state_vector(self, rows):
        """Converts the board rows and player's hand into a 116-dim tensor."""
        state = []
        for row in rows:
            if row:
                state.append(row[-1].number / 104.0) 
                state.append(len(row) / 5.0)         
                state.append(sum(c.bullheads for c in row) / 30.0) 
            else:
                state.extend([0.0, 0.0, 0.0])
                
        hand_flags = [0.0] * 104
        for card in self.hand:
            hand_flags[card.number - 1] = 1.0
        state.extend(hand_flags)
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def act(self, board):
        """Action 1: Pick a card to play (Actions 0-103)"""
        state = self.get_state_vector(board.rows)
        
        # Map card numbers (1-104) to action indices (0-103)
        valid_action_indices = [card.number - 1 for card in self.hand]
        
        if np.random.rand() <= self.epsilon:
            chosen_action = random.choice(valid_action_indices)
        else:
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state)[0]
                
            # Mask out invalid cards AND row-selection actions (104-107)
            masked_q_values = torch.full_like(q_values, -float('inf'))
            for idx in valid_action_indices:
                masked_q_values[idx] = q_values[idx]
                
            chosen_action = torch.argmax(masked_q_values).item()
            
        # Record the state and action index taken
        self.turn_transitions.append((state, chosen_action))
        
        # Convert action index back to card number
        return chosen_action + 1 

    def choose_row_to_take(self, board_rows):
        """Action 2: Pick a row to take (Actions 104-107)"""
        state = self.get_state_vector(board_rows)
        
        # Action indices for rows 0, 1, 2, 3
        valid_action_indices = [104, 105, 106, 107]
        
        if np.random.rand() <= self.epsilon:
            chosen_action = random.choice(valid_action_indices)
        else:
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state)[0]
                
            # Mask out card-selection actions (0-103)
            masked_q_values = torch.full_like(q_values, -float('inf'))
            for idx in valid_action_indices:
                masked_q_values[idx] = q_values[idx]
                
            chosen_action = torch.argmax(masked_q_values).item()
            
        # Record this secondary action
        self.turn_transitions.append((state, chosen_action))
        
        # Convert action index back to row index (0-3)
        return chosen_action - 104

    def remember(self, reward, next_board, done):
        """Stores ALL transitions from the turn into the replay buffer."""
        next_state = self.get_state_vector(next_board.rows)
        
        # Apply the final turn penalty (reward) to every action taken this turn
        for state, action in self.turn_transitions:
            self.memory.append((state, action, reward, next_state, done))
            
        # Clear the temporary buffer for the next turn
        self.turn_transitions = []

    def replay(self, batch_size=32):
        """Trains the neural network on past experiences."""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        self.model.train()
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
                
            target_f = self.model(state)
            target_f[0][action] = target # Update the specific action's Q-value
            
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filepath="6nimmt_dqn.pth"):
        """Saves the neural network's weights to a file."""
        torch.save(self.model.state_dict(), filepath)
        # print(f"Model saved to {filepath}")

    def load_model(self, filepath="6nimmt_dqn.pth"):
        """Loads the neural network's weights from a file."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval() # Set the model to evaluation mode
        # print(f"Model loaded from {filepath}")
