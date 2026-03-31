import random

class Card:
    """Represents a single card in 6 nimmt!"""
    def __init__(self, number):
        self.number = number
        self.bullheads = self._calculate_bullheads(number)

    def _calculate_bullheads(self, n):
        """Calculates penalty points based on the card number."""
        if n == 55:
            return 7
        elif n % 11 == 0:
            return 5
        elif n % 10 == 0:
            return 3
        elif n % 5 == 0:
            return 2
        else:
            return 1

    def __repr__(self):
        return f"[{self.number} ({self.bullheads}*)]"


class Player:
    """Represents a player with a hand of cards and a score."""
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.score = 0

    def reset_for_new_round(self):
        """Clears the player's hand for a new round."""
        self.hand = []
        self.score = 0

    def receive_cards(self, cards):
        self.hand.extend(cards)
        self.hand.sort(key=lambda c: c.number)

    def play_card(self, card_number):
        """Plays a card from the hand by its number. No strategy, just selection."""
        for i, card in enumerate(self.hand):
            if card.number == card_number:
                return self.hand.pop(i)
        raise ValueError(f"Player {self.name} does not have card {card_number}")

    def choose_row_to_take(self, board_rows):
        """
        If a played card is lower than all rows, the player must pick a row to take.
        For simplicity in this basic player class, it automatically picks the row 
        with the fewest penalty points.
        """
        min_bullheads = float('inf')
        chosen_row_idx = 0
        for idx, row in enumerate(board_rows):
            bullheads = sum(c.bullheads for c in row)
            if bullheads < min_bullheads:
                min_bullheads = bullheads
                chosen_row_idx = idx
        return chosen_row_idx

    def __repr__(self):
        return f"{self.name} (Score: {self.score})"


class Board:
    """Manages the cards currently on the floor (the 4 rows)."""
    def __init__(self):
        self.rows = [[], [], [], []]

    def setup_board(self, cards):
        """Initializes the 4 rows with 1 card each."""
        for i in range(4):
            self.rows[i].append(cards[i])

    def place_card(self, player, card):
        """
        Places a card in the correct row and returns any penalty points incurred.
        """
        target_row_idx = -1
        min_diff = float('inf')

        # Find the correct row: lowest difference where card > last card in row
        for i, row in enumerate(self.rows):
            last_card = row[-1]
            if card.number > last_card.number:
                diff = card.number - last_card.number
                if diff < min_diff:
                    min_diff = diff
                    target_row_idx = i

        # Condition 1: Card is lower than all end cards -> Player must take a row
        if target_row_idx == -1:
            row_to_take = player.choose_row_to_take(self.rows)
            penalty = sum(c.bullheads for c in self.rows[row_to_take])
            self.rows[row_to_take] = [card] # Replace row with the new card
            return penalty

        # Condition 2: Card fits into a row. Is it the 6th card? (6 nimmt!)
        self.rows[target_row_idx].append(card)
        if len(self.rows[target_row_idx]) == 6:
            # Take the first 5 cards as penalty, keep the 6th as the new start
            penalty_cards = self.rows[target_row_idx][:5]
            penalty = sum(c.bullheads for c in penalty_cards)
            self.rows[target_row_idx] = [self.rows[target_row_idx][5]]
            return penalty

        # Condition 3: Card placed successfully, no penalties
        return 0

    def print_board(self):
        print("\n--- Current Board ---")
        for i, row in enumerate(self.rows):
            print(f"Row {i + 1}: {row}")
        print("---------------------\n")

class GameEngine:
    """Controls the deck, state, rounds, and turn resolution."""
    def __init__(self, players, debug=False):
        self.players = players
        self.board = Board()
        self.deck = []
        self.debug = debug

    def setup_game(self):
        """Shuffles and deals 10 cards to each player, 4 to the board."""
        self.deck = [Card(i) for i in range(1, 105)]
        random.shuffle(self.deck)
        
        # Deal 10 cards to each player
        for player in self.players:
            player.reset_for_new_round()
            player.receive_cards([self.deck.pop() for _ in range(10)])
            
        # Deal 4 cards to the board
        self.board.setup_board([self.deck.pop() for _ in range(4)])
        
    def start_new_round(self):
        """
        Gathers all 104 cards, reshuffles, and resets the board and hands.
        This is called at the start of the game and whenever players run out of cards.
        """
        if self.debug:
            print("\n" + "="*30)
            print("Starting a New Round!")
            print("="*30)
        
        for player in self.players:
            player.receive_cards([self.deck.pop() for _ in range(10)])

    def is_game_over(self, threshold=66):
        """
        Checks if the game is over.
        The game ends if any player has reached or exceeded the penalty threshold.
        """
        if len(self.deck) < len(self.players) * 10 and all(len(player.hand) == 0 for player in self.players):
            return True
        for player in self.players:
            if player.score >= threshold:
                return True
        return False

    def get_winner(self):
        """Returns the player with the lowest score."""
        # The min() function can search through the players list 
        # and compare them based on their score attribute.
        return min(self.players, key=lambda p: p.score)

    def play_turn(self, player_actions):
        """Resolves a single turn based on chosen cards."""
        played_cards = []
        
        for player, card_number in player_actions.items():
            card = player.play_card(card_number)
            played_cards.append((player, card))

        played_cards.sort(key=lambda x: x[1].number)

        for player, card in played_cards:
            if self.debug:
                print(f"{player.name} plays {card}")
            penalty = self.board.place_card(player, card)
            if penalty > 0:
                if self.debug:
                    print(f"  -> {player.name} takes a penalty of {penalty} bullheads!")
                player.score += penalty
