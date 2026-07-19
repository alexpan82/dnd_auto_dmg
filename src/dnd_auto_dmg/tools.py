import random
import re
import numpy as np
import json
import math
from rapidfuzz import process, fuzz
from langchain_core.tools import tool


# --------- TOOL: Dice Roller --------- #
@tool
def roll_dice(num_dice: int, num_sides: int, modifier: int) -> int:
    """Rolls random set of dice and then adds modifier

    Args:
        num_dice: number of dice to roll
        num_sides: number of sides on dice
        modifier: add to the result after die roll
    """
    dice_rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return sum(dice_rolls) + modifier

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    result = math.floor(a / b)
    result = result if result > 0 else 1
    return result

# Adding json cleanup
def extract_json(text:str) -> dict:
    """
    Extract JSON from a string by removing leading and trailing non-JSON characters.
    """
    # Find the first opening brace or bracket
    start_pattern = r'[{\[]'
    start_match = re.search(start_pattern, text)

    if not start_match:
        return None

    start_pos = start_match.start()
    start_char = text[start_pos]
    end_char = '}' if start_char == '{' else ']'

    # Count nested braces/brackets to find the matching closing one
    count = 0
    for i, char in enumerate(text[start_pos:], start_pos):
        if char == start_char:
            count += 1
        elif char == end_char:
            count -= 1
            if count == 0:
                cleaned_str = text[start_pos:i+1]
                try:
                    return json.loads(cleaned_str)
                except json.decoder.JSONDecodeError:
                    return None

    return None


def fuzzy_match(query:str, possible_matches:list[str], threshold: int = 40) -> tuple[str, float]:
    """
    Justification: players often truncate phrases / nouns or over-complicate the description
    Therefore we take the max b/t the partial_ratio and token_set_ratio
    """
    partial_ratios = [fuzz.partial_ratio(query, q) for q in possible_matches]
    token_set_ratio = [fuzz.token_set_ratio(query, q) for q in possible_matches]

    result_list = [(q, max(a, b)) for a, b, q in zip(partial_ratios, token_set_ratio, possible_matches)]
    result_list = sorted(result_list, key=lambda x: x[1], reverse=True)

    best_match = result_list[0]

    if best_match[1] <= threshold:
        return((None, None))
    else:
        return best_match
