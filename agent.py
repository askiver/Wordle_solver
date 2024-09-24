# Create an agent that plays wordle and collect it's guesses and results which are then used to train a neural network.
from random import random

import torch


class Agent:
    def __init__(self, num_games: int):
        self.all_solutions = create_all_solutions()
        self.num_games = num_games  # How many games the agent will play before the model is retrained

    def play_game(self, model):
        # Play a game of wordle
        # Set a random word as the correct solution
        target = random.choice(list(self.all_solutions))
        guesses = []
        num_guesses = 0
        while num_guesses < 6:
            # Get legal guess from the model
            guess = self.get_legal_guess(model)


    # Function for forcing a legal guess from the model
    def get_legal_guess(self, model):
        num_tries = 0
        while True:
            num_tries += 1
            if num_tries % 101 == 100:
                print(num_tries)
            guess = transform_tensor_target(model.get_guess())
            if guess in self.all_solutions:
                return guess


# Function for creating tensors from all possible guesses
def create_all_solutions():
    word_solutions = set()
    # load solution from file
    with open("answers.txt", "r") as file:
        for line in file:
            word_solutions.add(line.strip().upper())

    return word_solutions


# Function that transforms a word into a tensor format
def transform_word_target(word: str):
    # Words are always of length 5 in wordle
    # Value of A char is 65, there are 26 letters in the alphabet
    tensor = torch.zeros(26, 5)
    for i, letter in enumerate(word):
        tensor[i][(ord(letter) - 65)] = 1
    return tensor.flatten()


# Function that transforms a target tensor into a word
def transform_tensor_target(tensor: torch.Tensor):
    word = ""
    for i in range(5):
        word += chr(torch.argmax(tensor[i * 26:(i + 1) * 26]).item() + 65)
    return word
