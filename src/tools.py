import random
import re

# --------- TOOL: Dice Roller --------- #
def roll(dice_expr: str) -> int:
    # Basic dice parser: 2d6+3 => [2,6,+3]
    match = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", dice_expr.replace(" ", ""))
    if not match:
        raise ValueError(f"Invalid dice expression: {dice_expr}")
    num = int(match.group(1)) if match.group(1) else 1
    die = int(match.group(2))
    mod = int(match.group(3)) if match.group(3) else 0
    return sum(random.randint(1, die) for _ in range(num)) + mod
