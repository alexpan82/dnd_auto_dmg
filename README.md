# Making DnD combat easy with an Auto-Damage Calculator

Managing damage calculations and combat logistics in DnD is infamously complex and slows down the game. Using popular 3rd DnD Beyond / Roll20 still burdens the player with manual damage roll clicks and requires players to have perfect knowledge of character stats / feats. Additionally, these systems are not flexible to homebrew mechanics / items. They also take away from the player experience by unintentionally promoting online distractions which makes it difficult for players to be "present" during the session. 

We need an automated system that seamlessly integrates into the role-playing experience, does not allow for other distractions, and takes administrative burden away from DMs. Additionally, the system must be flexible to an ever-evolving campaign and generalizable to any (homebrew) campaign.

We present DnD-Auto-Damage (DaD), an LLM-driven agentic workflow that addresses these limitations in managing combat logistics. Notable DaD features include
- User can add any additional attributes / actions / items / spells to json files that allow for combat flexibility and specificity
- Stateful matching of user prompts and user-added json entries
- State persistence so that combat can pick-up where it left off
- Tool usage so that players can feel assured that a real RNG is rolling hit die

![Agent Graph](public/graph.png "Agent Graph")

## Set up

### Create venv and install dependencies using uv
```sh
# First install python 3.10
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Set up API Keys
DaD uses ChatGPT-4o as the default model with support coming to different models in the future. API calls to ChatGPT generally require payment on-file and can be configured on OpenAI's API [website](https://openai.com/api/).

We also provide a chat-interface using Chainlit, which requires a local key generation
```sh
export OPENAI_API_KEY="ENTER_YOUR_KEY"
chainlit create-secret
# Copy and paste key
export CHAINLIT_AUTH_SECRET="ENTER_YOUR_KEY"
```

## Usage
### Front-end UI
Chatbot can be accessed via any web-browser at `http://localhost:8000` after entering the following command
```sh
chainlit run src/app.py
```

### Customization
DaD has robust 5e knowledge including the stats of character races/classes and common weapons, spells, and items. If you have homebrewed features, spells, and weapons you would like to include, or if DaD is incorrectly remembering common weapon/spell/item/feature stats, the system uses JSON files that can be easily modified.

#### JSON Configuration Files
DaD uses two primary JSON files for customization, located in the docs/ directory:

`docs/character.json`

Contains character-specific attributes including stats, proficiencies, class features, and racial traits. Each character entry should follow this structure:
```json
{
  "Character Name": {
    "stats": {
      "strength": 16,
      "dexterity": 14,
      "constitution": 15,
      "intelligence": 10,
      "wisdom": 12,
      "charisma": 8
    },
    "proficiency_bonus": 3,
    "class": ["Fighter"],
    "level": 5,
    "features": [...],
    "additional_attributes": {...}
  }
}
```
`Note`: JSON key names don't need to be exact — the fuzzy matching system will find close matches automatically. However, more specific names improve matching accuracy.

`docs/weapons.json`

Contains weapon, spell, and item definitions with their associated damage dice, damage types, and special properties.

`Note`: JSON key-value names don't need to be exact as long as there is enough context for the DM to understand how to make rolls. Damage rolls and custom descriptions should be as detailed as possible.

```json
{
  "Weapon/Spell Name": {
        "damage": {"slashing": "2d6", "fire": "2d6"},
        "magic_bonus": 1,
        "default_level": 3,
        "scaling_per_higher_level": {"fire": "1d6"},
        "attack_save": "DEX",
        "properties": ["versatile", "finesse"],
        "special_effects": "...",
            }
}
```
#### Fuzzy Matching System
DaD uses intelligent fuzzy matching to handle natural language variations in player inputs. The system:
- Matches partial phrases (e.g., "long sword" matches "Longsword")
- Handles token variations (e.g., "sword of fire" matches "Flaming Longsword")
- Falls back to LLM knowledge when no confident match is found

This means players don't need to remember exact JSON key names during gameplay—natural descriptions work seamlessly.

### Adding Custom Content
To add homebrew content or override existing 5e stats:

1. Add a new character: Open docs/character.json and add a new entry with your character's name as the key
2. Add custom weapons/spells: Open docs/weapons.json and define new entries with unique identifiers
3. Restart the application: Changes take effect on the next app launch