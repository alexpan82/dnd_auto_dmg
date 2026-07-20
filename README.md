# Making DnD combat easy with an Auto-Damage Calculator

Managing damage calculations and combat logistics in DnD is infamously complex and slows down the game. Using popular 3rd party tools like DnD Beyond / Roll20 still burdens the player with manual damage roll clicks and requires players to have perfect knowledge of character stats / feats. Additionally, these systems are not flexible to homebrew mechanics / items. They also take away from the player experience by unintentionally promoting online distractions which makes it difficult for players to be "present" during the session.

We need an automated system that seamlessly integrates into the role-playing experience, does not allow for other distractions, and takes administrative burden away from DMs. Additionally, the system must be flexible to an ever-evolving campaign and generalizable to any (homebrew) campaign.

We present DnD-Auto-Damage (DaD), an LLM-driven agentic workflow that addresses these limitations in managing combat logistics. Notable DaD features include:
- Full multi-combatant combat tracker — PCs, NPCs, and monsters are all tracked as combatants with their own HP pools
- Damage is automatically applied to the resolved targets, with death/unconscious tracking as HP hits zero
- Statuses and buffs with round-based durations (e.g. Tenser's Transformation lasting 10 rounds)
- Ad-hoc enemies spun up on the fly from plain narration (e.g. "a group of 3 kobolds") — no need to pre-register every monster
- Multi-action messages ("Avantor attacks the goblin, then casts fireball at the kobolds") are parsed into individual resolved actions
- Tool usage so that players can feel assured that a real RNG is rolling hit die
- State persistence so that combat can pick up where it left off
- Provider-agnostic LLM layer — defaults to `gpt-4o`, swap providers/models with a single environment variable
- Optional turn-by-turn narration summarizing what just happened in the fight

![Agent Graph](public/graph.png "Agent Graph")

## Set up

### Create venv and install dependencies using uv

```sh
# First install python 3.10
uv venv --python 3.10
source .venv/bin/activate
```

Then install dependencies with one of the following:

```sh
# Recommended: editable install of the dnd_auto_dmg package + pytest for the test suite
uv pip install -e ".[dev]" --python .venv/bin/python
```

```sh
# Alternative: install only the runtime dependencies from requirements.txt
uv pip install -r requirements.txt
```

### Set up API Keys

DaD uses `gpt-4o` as the default model. API calls to OpenAI generally require payment on-file and can be configured on OpenAI's API [website](https://openai.com/api/).

```sh
export OPENAI_API_KEY="ENTER_YOUR_KEY"
```

To use a different provider/model, set `DND_MODEL` to an `init_chat_model`-style `"provider:model"` string (LangChain's provider-agnostic chat model constructor). This requires that provider's LangChain integration package to be installed (e.g. `langchain-anthropic` for Anthropic):

```sh
export DND_MODEL="anthropic:claude-sonnet-4-5"
```

Optionally, point the app at a different data directory (see [Customization](#customization-data) below) instead of the repo's `data/`:

```sh
export DND_DATA_DIR="/path/to/your/data"
```

We also provide a chat interface using Chainlit, which requires a local key generation:

```sh
chainlit create-secret
# Copy and paste key
export CHAINLIT_AUTH_SECRET="ENTER_YOUR_KEY"
```

## Usage

### Front-end UI

The chatbot can be accessed via any web browser at `http://localhost:8000` after running the following command. If prompted for a username and password, type in `admin` for both.

```sh
chainlit run src/app.py
```

### CLI demo

A scripted five-turn combat (casting a buff, attacking, fighting a group of ad-hoc kobolds, etc.) that prints the combatant roster's HP after every turn. Requires `OPENAI_API_KEY` — this makes real LLM calls.

```sh
.venv/bin/python src/demo.py
```

### Tests

The full pytest suite runs against a scripted fake LLM, so no API key is required:

```sh
.venv/bin/python -m pytest
```

## Customization (`data/`)

DaD has robust 5e knowledge including the stats of character races/classes and common weapons, spells, and items. If you have homebrewed features, spells, weapons, or monsters you'd like to include, or if DaD is incorrectly remembering common weapon/spell/item/feature stats, the system reads pydantic-validated JSON files under `data/` that can be easily modified.

Every file is a JSON object keyed by a lowercase, snake_case **slug id** (e.g. `flametongue_greatsword`) whose entry also carries a human-readable `name` (e.g. "Flametongue Greatsword") — both the id and the name participate in fuzzy matching (see below). Entries are validated against pydantic models at load time; an invalid entry raises a clear error naming the file and the offending entry id, e.g. `weapons.json entry 'flametongue_greatsword': ...`.

Dice strings (damage dice, healing dice, scaling dice) must match the regex `^\d+d\d+([+-]\d+)?$` — e.g. `"2d6"`, `"1d8+2"`, `"2d6-1"` are valid; `"d6"`, `"2d6+"`, `"2x6"` are not.

### `data/characters.json`

Player-character roster entries. Fields: `name`, `kind` (`"pc"`), `class` (list), `subclass`, `race`, `level`, `max_hp`, `ac`, `attributes` (all six abilities), `proficiency_bonus`, `features` (list of feature ids), `inventory` (list of weapon ids), `spells` (list of spell ids). Shipped example (`avantor`):

```json
{
    "avantor": {
        "name": "Avantor",
        "kind": "pc",
        "class": ["Wizard"],
        "subclass": "Bladesinging",
        "race": "Half-Elf",
        "level": 10,
        "max_hp": 62,
        "ac": 15,
        "attributes": {
            "strength": 18,
            "dexterity": 16,
            "constitution": 14,
            "intelligence": 18,
            "wisdom": 12,
            "charisma": 10
        },
        "proficiency_bonus": 4,
        "features": ["savage_attacks", "great_weapon_master"],
        "inventory": ["flametongue_greatsword"],
        "spells": ["fireball", "tensers_transformation"]
    }
}
```

A minimal `generic` PC (level 1 Fighter, no features/inventory/spells) also ships as a starting template.

### `data/weapons.json`

Weapon definitions. Fields: `name`, `damage` (dict of damage type -> dice string), `magic_bonus`, `ability`, `properties`, `special_effects`. Shipped example (`flametongue_greatsword`):

```json
{
    "flametongue_greatsword": {
        "name": "Flametongue Greatsword",
        "damage": {"slashing": "2d6", "fire": "2d6"},
        "magic_bonus": 1,
        "ability": "strength",
        "properties": ["two-handed", "heavy"],
        "special_effects": "While the sword is ablaze, it deals an extra 2d6 fire damage on a hit."
    }
}
```

### `data/spells.json`

Spell definitions. Fields: `name`, `level`, `damage` (dict or `null`), `healing` (dice string or `null`), `save` (`{ability, half_on_success}` or `null`), `attack_roll`, `scaling_per_higher_level` (dict or `null`), `targeting` (`"single"` | `"area"` | `"self"`), `applies_status` (`{name, duration_rounds, effect}` or `null`), `description`. Shipped examples:

```json
{
    "fireball": {
        "name": "Fireball",
        "level": 3,
        "damage": {"fire": "8d6"},
        "healing": null,
        "save": {"ability": "dexterity", "half_on_success": true},
        "attack_roll": false,
        "scaling_per_higher_level": {"fire": "1d6"},
        "targeting": "area",
        "applies_status": null,
        "description": "A bright streak flashes to a point you choose then blossoms into an explosion of flame in a 20-foot radius."
    },
    "tensers_transformation": {
        "name": "Tenser's Transformation",
        "level": 6,
        "damage": null,
        "healing": null,
        "save": null,
        "attack_roll": false,
        "scaling_per_higher_level": null,
        "targeting": "self",
        "applies_status": {
            "name": "tensers_transformation",
            "duration_rounds": 10,
            "effect": "Your weapon attacks deal an extra 2d12 force damage on a hit."
        },
        "description": "You endow yourself with endurance and martial prowess fueled by magic."
    }
}
```

### `data/features.json`

Class/racial feature definitions. Fields: `name`, `description`, `effect` (the mechanical text consumed by the damage-calculation LLM). Shipped examples:

```json
{
    "savage_attacks": {
        "name": "Savage Attacks",
        "description": "When you score a critical hit with a melee weapon attack, you can roll one of the weapon's damage dice one additional time and add it to the extra damage of the critical hit.",
        "effect": "On a critical hit with a melee weapon: roll one additional weapon damage die and add it to the crit damage."
    },
    "great_weapon_master": {
        "name": "Great Weapon Master",
        "description": "You've learned to put the weight of a weapon to your advantage. Before you make a melee attack with a heavy weapon you are proficient with, you can choose to take a -5 penalty to the attack roll to gain +10 damage.",
        "effect": "Optional on melee attacks with heavy weapons: -5 to hit, +10 to damage."
    }
}
```

### `data/monsters.json`

Monster/statblock definitions used to spin up NPC combatants (rostered or ad-hoc). Fields: `name`, `kind` (`"monster"`), `max_hp`, `ac`, `attributes`. Shipped example (`kobold`; `goblin`, `skeleton`, `zombie`, `bandit`, and `wolf` also ship):

```json
{
    "kobold": {
        "name": "Kobold",
        "kind": "monster",
        "max_hp": 5,
        "ac": 12,
        "attributes": {
            "strength": 7,
            "dexterity": 15,
            "constitution": 9,
            "intelligence": 8,
            "wisdom": 7,
            "charisma": 8
        }
    }
}
```

### Adding homebrew content

1. Add a new character: open `data/characters.json` and add a new entry keyed by a unique slug id.
2. Add custom weapons/spells/features/monsters: open the relevant `data/*.json` file and define a new entry with a unique id, following the field shapes above.
3. Restart the application: changes take effect on the next app launch (the registry loads and validates all of `data/` once at startup).

## Fuzzy matching

DaD uses intelligent fuzzy matching (via `rapidfuzz`, taking the max of partial-ratio and token-set-ratio) to handle natural-language variations in player input, matching against **both** an entry's slug id and its human-readable `name`. There are two thresholds:

- **Lookup threshold (40)** — used when resolving weapons, spells, features, monsters, and characters against `data/`. It's deliberately lenient since players often truncate or paraphrase names (e.g. "flame sword" matching "Flametongue Greatsword").
- **Live-combatant threshold (70)** — used when resolving a target against the combatants already in the fight. This is stricter so that a loosely-typed "goblin" can't accidentally hit a different monster's specific instance, like `kobold_1`, mid-combat.

When nothing clears the relevant threshold, DaD falls back to the LLM's own 5e knowledge rather than failing the action outright.

## Architecture

```
dnd_auto_dmg/
├── pyproject.toml              # package metadata + deps (editable install)
├── requirements.txt
├── data/                       # game data (pydantic-validated at load)
│   ├── characters.json ├── weapons.json ├── spells.json
│   ├── features.json   └── monsters.json
├── src/
│   ├── app.py                  # thin Chainlit entry (chainlit run src/app.py)
│   ├── demo.py                 # thin CLI demo (python src/demo.py)
│   └── dnd_auto_dmg/           # the package
│       ├── config.py           # AppConfig (env: DND_MODEL, DND_DATA_DIR)
│       ├── llm.py               # get_llm/get_llm_with_tools (init_chat_model)
│       ├── schemas.py           # all pydantic models
│       ├── state.py             # CombatState + reducers + UI props contract
│       ├── registry.py          # DataRegistry (loads/validates data/)
│       ├── tools.py             # roll_dice + math tools, extract_json, fuzzy_match
│       ├── nodes/                # begin_turn, relevance, parse, resolve, damage, apply, narrate
│       └── graph.py              # build_graph()
├── public/elements/LanggraphStateDisplay.jsx   # combat roster UI
└── tests/                       # pytest suite (fake LLM, no API key needed)
```

Each turn flows through the graph as: `begin_turn -> check_relevance -> parse_actions -> resolve_combatants -> calculate_damage` (looping through a `roll_dice` tool node as needed) `-> apply_damage` (looping back to `calculate_damage` via a per-action cursor until every resolved action has been applied) `-> narrate`.

The graph's nodes are dependency-injected factories (`make_begin_turn()`, `make_calculate_damage(llm)`, etc.) rather than free functions, so `build_graph()` can wire in either a real provider-backed LLM or a scripted fake. State is a `TypedDict` (`CombatState`) whose channels can still hold pydantic objects — including `Combatant` and its nested `StatusEffect` values — and this round-trips transparently through LangGraph's `SqliteSaver` checkpointer, which is how combat state persists between turns/sessions. The test suite never makes real network calls: it injects a scripted fake chat model that returns pre-programmed structured outputs, so `pytest` runs fast and needs no API key.
