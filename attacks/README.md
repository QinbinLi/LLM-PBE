# Attacks on Large Language Models

This directory contains code for various types of attacks on large language models. All attack methods inherit from a base class, `AttackBase`, which provides a standardized interface for implementing different types of attacks.

## AttackBase Class

The `AttackBase` class is designed to be a base class for all specific attack types. It takes four parameters in its constructor:

### Parameters:

- **model**: The language model object to be attacked. This can be any model object that implements a standardized query interface.
  
- **data**: The data that was used to train/fine-tune the model. This could be a list of text strings, a file path, or any other dataset format that is compatible with your specific model.
  
- **prompt**: The prompt that was injected into the model. This is generally a piece of text that the model uses to generate a response.
  
- **metric**: The metric used to evaluate the success of the attack. This can either be a string indicating a predefined metric, or a custom function that takes the attack results as input and returns a numerical value.

### Functionalities:

- **execute**: This is an abstract method that should be overridden by specific attack implementations. It executes the attack and returns metrics that evaluate the success or failure of the attack.

- **evaluate**: This method evaluates the attack based on the metric specified during the initialization. It takes the raw results of the `execute` method as an argument and returns an evaluation score.

## Directory Structure

- `README.md`: This document.
- `AttackBase.py`: Contains the base class for attacks.
- `DataExtraction/`: Contains implementations of data extraction attacks.
- `MIA/`: Contains implementations of membership inference attacks.
- `PromptLeakage/`: Contains implementations of prompt-based attacks.
- `Jailbreak/`: Contains implementations of jailbreak attacks.

## How to Extend

To implement a new attack, create a new Python file in the corresponding subdirectory and make sure your attack class inherits from `AttackBase`. Then, implement the `execute` and optionally the `evaluate` methods according to your attack logic.

## Examples

For usage examples, please refer to the main project README.

