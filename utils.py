# File that contains utility functions for the project

import torch


# Function that transform a guess word into a tensor format
def transform_word_guess(word: str, info):
    # There are 3 different values for information
    tensor = torch.zeros(26, 5, 3)
    for i, letter in enumerate(word):
        tensor[i][(ord(letter) - 65)][info[i]] = 1
    return tensor


# Function that combines info from guesses into a single tensor
def combine_guess_tensor(guesses):
    stacked_guesses = torch.stack(guesses)

    # Do element-wise max for all guesses
    guess_tensor = torch.max(stacked_guesses, dim=0).values
    return guess_tensor


# Reward function based on information from the guess
def reward_function(guess: torch.tensor, target: torch.tensor, info: torch.tensor, num_guesses: int):
    # Check if the guess is correct
    if torch.equal(guess, target):
        return 100 * (6 - num_guesses)
    # Check how much information the model gathered
    # Check number of incorrect letter identified
    incorrect = torch.sum(torch.max(info[:, :, 0], dim=0).values)

    # check number of correct letter in wrong positions
    partially_correct = torch.sum(torch.max(info[:, :, 1], dim=0).values)

    # check number of correct letter in correct positions
    correct = torch.sum(info[:, :, 2])

    # Calculate reward based on the information
    reward = 5 * correct + 3 * partially_correct + 1 * incorrect

    return reward


if __name__ == "__main__":
    print(transform_word_target("HELLO"))
    print(transform_tensor_target(transform_word_target("HELLO")))
# print(create_all_solutions())
