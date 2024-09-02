# Defenses for Large Language Models

This directory contains various methods for defending large language models against different types of attacks. All defense methods inherit from the base class, `DefenseBase`, which provides a standardized framework for implementing different types of defenses.

## DefenseBase Class

The `DefenseBase` class serves as the base class for all defense mechanisms in this library. It initializes with four parameters:

### Parameters:

- **model**: The language model object to be defended. This can be any model object that implements a standardized query interface.

- **data**: The data that will be used to apply the defense. This could be a list of text strings, a file path, or any other dataset format that is compatible with your specific model.

- **prompt**: The prompt that will be used in the defense mechanism. This is usually a text string that the model uses to generate a response.

- **params**: A dictionary of hyperparameters specific to the defense mechanism. These could be any additional settings required to properly execute the defense.

### Functionalities:

- **execute**: This is an abstract method that should be overridden by specific defense implementations. When executed, it returns an updated model object after applying the specific defense mechanism.

## Directory Structure

- `README.md`: This document.
- `DefenseBase.py`: Contains the base class for defenses.
- `DP/`: Contains implementations for defenses using Differential Privacy.
- `Unlearning/`: Contains methods for machine unlearning.

## How to Extend

To implement a new defense, create a new Python file in the relevant subdirectory. Your defense class should inherit from `DefenseBase` and override the `execute` function as per the logic of the specific defense mechanism. You can add more functions as needed.

## Examples

For usage examples, please refer to the main project README.

