import os

import numpy as np
from math import ceil

ROUNDING_FACTOR: int = 2


def generate_dataset(time_interval: int, users: int, scenario: int):
    dataset = generate_user_similarity_matrix(users)
    mask = inc_and_dec_scenario_mask(users)
    save_dataset(dataset, scenario, 1)
    print("Original dataset:\n", dataset, "\n")

    for day in range(2, time_interval + 1):  # The loop begins at day 2, as day 1 is the random dataset
        temp = generate_dateset_change(dataset, day)
        # print(f"Change on iteration {day}:\n", temp, "\n")

        if scenario == 1:
            dataset += temp
        elif scenario == 2:
            dataset -= temp
        elif scenario == 3:
            dataset += temp * mask
        else:
            print("Scenario entry must be between 1-3")
            return

        dataset = dataset.clip(0, 1)  # Value must be between 0 and 1
        save_dataset(dataset, scenario, day)
        print(f"New dataset on iteration {day}:\n", dataset, "\n")


def generate_user_similarity_matrix(users: int) -> np.ndarray:
    dataset = symmetrize(np.random.uniform(0.0, 1.0, size=(users, users)))
    np.fill_diagonal(dataset, 1)  # A User is similar to themselves
    return np.round(dataset, ROUNDING_FACTOR)


def symmetrize(np_array: np.ndarray)-> np.ndarray:
    return (np_array + np_array.transpose()) / 2


def generate_dateset_change(np_array: np.ndarray, day: int) -> np.ndarray:
    return np.round((1 - np_array[:]) / day, ROUNDING_FACTOR)


def inc_and_dec_scenario_mask(users: int) -> np.ndarray:
    size = users * users
    temp = np.ones(size)
    temp[ceil(size / 2):] *= -1
    return temp.reshape([users, users])


def save_dataset(dataset: np.ndarray, scenario: int, day: int):
    dest = f'../data.toy/scenario.{scenario}'
    if os.path.exists(dest):
        np.save(dest + f'/day.{day:03d}', dataset)
    else:
        os.makedirs(dest)
        np.save(dest + f'/day.{day:03d}', dataset)


if __name__ == '__main__':
    generate_dataset(3, 3, 3)
