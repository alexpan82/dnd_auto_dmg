# DnD Auto-Damage — Multi-Combatant Refactor Specification

> **Audience:** the project-manager orchestrator agent (Opus 4.8) and its implementation
> subagents (Sonnet 5). This document is the single source of truth for the refactor.
> Progress/checkpoint state lives in `REFACTOR_PROGRESS.md` — read both before doing anything.

## 1. Goal

Refactor the single-file LangGraph agent (`src/agent.py`) into a modular package with a
**full multi-combatant combat tracker**:

1. Track PCs **and** NPCs/monsters as combatants keyed by ID: HP pools, damage automatically
   applied to targets, active features/statuses with round durations, ad-hoc enemies
   ("3 kobolds") created on the fly.
2. Modular package layout (state / nodes / graph / data registry / LLM config) so the system
   is easy to modify and extend.
3. Provider-agnostic LLM layer via `init_chat_model`; `gpt-4o` stays the default.
4. Redesigned, pydantic-validated JSON data schemas; populate the currently-empty
   `features.json` / `spells.json`; add `monsters.json`.
5. A pytest suite (fake-LLM based, no API key needed) plus an end-to-end integration test.

## 2. Known bugs in the current code (all must be fixed)

1. `calculate_damage` system prompt is a plain string containing literal `{tools}` and
   `{roll_dice}` placeholders that are never formatted (`src/agent.py:175`).
2. Nodes return `"messages": state["messages"] + [response]` — with the `add_messages`
   reducer this re-appends fresh-ID system prompts into persisted history every turn.
3. Duplicated condition `state['metadata'] is None or state['metadata'] is None`
   (`src/agent.py:163`).
4. `load_attributes` returns `character` as a string despite the `Dict` type hint; stale
   "implement fuzzy matching" TODO (fuzzy matching already exists in `tools.py`).
5. TypedDict fields with `= None` defaults are meaningless in Python.
6. `narrator_output` is broken (empty string concatenated into a message list) and commented
   out of the graph.
7. Statuses are hardcoded to `None` in damage calc; parsed `active_statuses` / `using_feat`
   are dropped on the floor.
8. Multi-action / multi-target prompts unsupported ("fireball at a group of 3 kobolds").
9. No HP tracking anywhere; damage is calculated but never applied.
10. Hygiene: `getpass` + LLM instantiation at module import time; JSON paths relative to cwd;
    `from tools import ...` relies on cwd; `requirements.txt` bloated (torch, transformers,
    accelerate, tavily-python, wikipedia, trustcall, notebook are unused).

## 3. Target module layout

```
dnd_auto_dmg/
├── pyproject.toml                        # NEW: package metadata + deps (setuptools, editable install)
├── requirements.txt                      # trimmed: langgraph, langgraph-checkpoint-sqlite, langchain,
│                                         #   langchain-openai, chainlit, rapidfuzz, numpy, pyaudio,
│                                         #   pydantic; dev extra: pytest
├── README.md                             # rewritten last (WP7)
├── data/                                 # RENAMED from docs/
│   ├── characters.json                   # migrated + max_hp/ac/full attributes
│   ├── weapons.json                      # migrated (Fireball moves OUT to spells.json)
│   ├── spells.json                       # populated: fireball, tensers_transformation
│   ├── features.json                     # populated: savage_attacks, great_weapon_master
│   └── monsters.json                     # NEW: kobold, goblin, skeleton, zombie, bandit, wolf
├── src/
│   ├── app.py                            # thin Chainlit entry (keeps `chainlit run src/app.py`)
│   ├── demo.py                           # thin CLI entry
│   └── dnd_auto_dmg/                     # the actual package
│       ├── __init__.py
│       ├── config.py                     # AppConfig
│       ├── llm.py                        # get_llm()/get_llm_with_tools() — lazy, no import-time getpass
│       ├── schemas.py                    # ALL pydantic models (runtime + data-file)
│       ├── state.py                      # CombatState TypedDict + reducers + UI props contract
│       ├── registry.py                   # DataRegistry
│       ├── tools.py                      # roll_dice/add/subtract/multiply/divide, extract_json, fuzzy_match
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── begin_turn.py
│       │   ├── relevance.py
│       │   ├── parse.py
│       │   ├── resolve.py
│       │   ├── damage.py
│       │   ├── apply.py
│       │   └── narrate.py
│       └── graph.py                      # build_graph(llm=None, registry=None, config=None)
├── public/elements/LanggraphStateDisplay.jsx   # rewritten combatant roster
└── tests/
    ├── conftest.py                       # ScriptedChatModel fake, registry fixture, tmp data dir
    ├── test_tools.py
    ├── test_schemas.py
    ├── test_registry.py
    ├── test_state.py                     # reducer + SqliteSaver round-trip
    ├── test_nodes_deterministic.py
    ├── test_nodes_llm.py
    └── test_graph_integration.py
```

`pyproject.toml` + `uv pip install -e .` makes `dnd_auto_dmg` importable from any cwd
(fixes bug 10). `src/app.py` / `src/demo.py` stay thin so README commands barely change.

## 4. Runtime pydantic models (`schemas.py`)

```python
class StatusEffect(BaseModel):
    name: str                                  # "tensers_transformation", "unconscious"
    source: Optional[str] = None               # spell/feature id or actor id
    duration_rounds: Optional[int] = None      # None = until removed
    applied_round: Optional[int] = None
    notes: Optional[str] = None

class Combatant(BaseModel):
    id: str                                    # slug: "avantor", "kobold_1"
    name: str
    kind: Literal["pc", "npc", "monster"]
    max_hp: int
    hp: int
    ac: Optional[int] = None
    attributes: Dict[str, int] = Field(default_factory=dict)
    features: List[str] = Field(default_factory=list)      # ids into features.json
    statuses: List[StatusEffect] = Field(default_factory=list)
    inventory: List[str] = Field(default_factory=list)     # ids into weapons.json
    origin: Literal["roster", "adhoc"] = "roster"
    is_alive: bool = True                      # set False when hp reaches 0

class TargetRef(BaseModel):
    name: str                                  # "kobold", "goblin", "self"
    count: int = 1                             # "3 kobolds" -> count=3

class ParsedAction(BaseModel):
    actor: str
    action_type: Literal["attack", "heal", "cast", "apply_status",
                         "remove_status", "advance_round", "other"]
    weapon_or_spell: Optional[str] = None
    targets: List[TargetRef] = Field(default_factory=list)
    spell_level: Optional[int] = None          # "5th level fireball"
    is_critical_hit: bool = False
    statuses_applied: List[str] = Field(default_factory=list)
    statuses_removed: List[str] = Field(default_factory=list)
    features_used: List[str] = Field(default_factory=list)

class ParsedTurn(BaseModel):                   # structured-output wrapper
    actions: List[ParsedAction] = Field(default_factory=list)

class ResolvedAction(BaseModel):
    action: ParsedAction
    actor_id: str
    target_ids: List[str] = Field(default_factory=list)
    item: Optional[dict] = None                # matched weapon/spell def (model_dump) or None
    item_name_raw: Optional[str] = None        # unmatched name -> LLM 5e-knowledge fallback
    feature_texts: Dict[str, str] = Field(default_factory=dict)  # id -> effect text

class PerTargetDamage(BaseModel):
    target_id: str
    damage: int = 0
    healing: int = 0

class DamageReport(BaseModel):
    action_index: int
    total_damage: int = 0
    total_healing: int = 0
    per_target: List[PerTargetDamage] = Field(default_factory=list)
    breakdown: Dict[str, Any] = Field(default_factory=dict)   # damage-type -> rolled amounts
    description: str = ""
```

## 5. Graph state (`state.py`)

```python
def merge_combatants(existing, update):
    # Per-key semantics: value None -> delete key; value Combatant -> insert/replace whole entry.
    # Nodes always return FULL Combatant objects for keys they touch — NO deep field merge.

class CombatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    combatants: Annotated[Dict[str, Combatant], merge_combatants]
    round_number: int                          # last-write-wins
    event_log: Annotated[list[str], operator.add]
    # per-turn scratch — reset by begin_turn on EVERY invocation:
    relevant_query: bool
    parsed_actions: list[ParsedAction]
    resolved_actions: list[ResolvedAction]
    current_action_index: int
    damage_reports: list[DamageReport]
```

Design decisions:
- **Per-key replace, not deep merge.** The graph is linear; nodes read the current combatant,
  `model_copy(update={...})`, and return `{"combatants": {cid: new_combatant}}`.
- **TypedDict state, pydantic values.** SqliteSaver's `JsonPlusSerializer` must round-trip
  pydantic v2 models inside channels — WP3 verifies this FIRST (see §12 risk 1). Fallback if
  it fails: store plain dicts, validate to `Combatant` at node boundaries.
- **Per-turn fields persist across invokes under a checkpointer**, hence `begin_turn` resets
  them every run. Drop all dead fields from the old state (`parsed_action`, `character`,
  `metadata`, `character_id`, `hp`, `status`, `target`, `damage_report`, `log`).
- `state.py`'s docstring carries the **frozen UI props contract** (§9) so UI work can
  proceed in parallel.

### Message hygiene (fixes bug 2) — rule for every node

System prompts are **ephemeral**: build them at invoke time as
`llm.invoke([SystemMessage(prompt)] + state["messages"])` and **never** return them in the
`messages` update. Specifically:
- `check_relevance`, `parse_actions`, `resolve_combatants`: return **no** messages.
- `calculate_damage`: returns only `[response]` (AIMessage, possibly with tool_calls).
- `apply_damage`: emits `RemoveMessage`s deleting the current action's tool-loop scaffolding
  (intermediate AIMessages with tool_calls + ToolMessages) once the report is extracted.
- `narrate`: returns only `[response]`.

## 6. Node contracts and graph wiring (`nodes/`, `graph.py`)

```
START → begin_turn → check_relevance
check_relevance --(not relevant)--> END
check_relevance --(relevant)-----> parse_actions → resolve_combatants → route_actions
route_actions --(no valid actions)--> END   (with an event_log entry)
route_actions --(actions)----------> calculate_damage
calculate_damage --(tools_condition: tool_calls)--> roll_dice(ToolNode) → calculate_damage
calculate_damage --(no tool_calls)--> apply_damage
apply_damage --(current_action_index < len(resolved_actions))--> calculate_damage
apply_damage --(done, narration enabled)--> narrate → END
apply_damage --(done, narration disabled)--> END
```

All nodes are **factories taking dependencies explicitly** —
`make_parse_actions(llm)`, `make_resolve_combatants(registry, config)`, etc.
`build_graph(llm=None, registry=None, config=None)` wires real defaults; tests inject fakes.

- **begin_turn** (deterministic): resets per-turn fields; initializes `combatants={}`,
  `round_number=1` on first turn.
- **check_relevance** (LLM): yes/no gate; parse `"yes"` case-insensitively into
  `relevant_query: bool`; uses the **plain** llm (current code binds tools here for no
  reason); returns no messages.
- **parse_actions** (LLM): `llm.with_structured_output(ParsedTurn)` with `extract_json`
  fallback, over recent history. Prompt must cover: multiple actions per message, target
  lists with counts ("a group of 3 kobolds" → `TargetRef(name="kobold", count=3)`), pronoun
  resolution from history ("They do it again"), spell upcast level, statuses applied/removed,
  and `advance_round` intent ("next round"). Empty `actions` = parse failure.
- **resolve_combatants** (deterministic): for each `ParsedAction`:
  1. *Actor*: fuzzy-match against live `combatants` names, then registry characters; roster
     hit not yet in combat → instantiate `Combatant` from the character sheet. No match →
     drop the action with an event_log entry.
  2. *Targets*: `"self"` → actor. Fuzzy-match against live combatants at
     `config.fuzzy_threshold_combatant` (70 — the old 40 lets "goblin" hit "kobold_1");
     miss → `monsters.json` stat block; miss again → ad-hoc `Combatant` with
     `config.default_adhoc_hp` and `origin="adhoc"`. `count=3` → `kobold_1..kobold_3`
     (reuse *living* instances first, create the remainder).
  3. *Item*: fuzzy-match merged weapons+spells namespace at `config.fuzzy_threshold_lookup`
     (40); miss → keep the raw name in `item_name_raw` (preserves LLM-knowledge fallback).
  4. *Statuses*: apply `statuses_applied`/`statuses_removed` to the actor **here, before
     damage**, so a buff cast this turn boosts this turn's rolls. Look up feature/status
     effect text into `feature_texts`. `advance_round`: increment `round_number`, decrement
     all `duration_rounds`, expire statuses at 0.
- **calculate_damage** (LLM + tool loop): operates on
  `resolved_actions[current_action_index]`. System prompt is a real template (fixes bug 1):
  tool names/descriptions rendered from the bound tool objects, JSON braces escaped,
  instructing output of exactly the `DamageReport` JSON shape (with `per_target` using the
  actual target ids) in a fenced code block, followed by a prose description. Context block:
  actor sheet + current statuses + feature effect texts + item def (or raw name) + is_crit +
  spell level + target ids/ACs. Tool loop wiring unchanged (`tools_condition` → ToolNode →
  back).
- **apply_damage** (deterministic, fixes bug 9): `extract_json` on the final AIMessage →
  validate as `DamageReport`. Tolerant fallback: if `per_target` missing/invalid, split
  `total_damage` evenly across `target_ids`. Per target: `hp = max(0, hp - damage)`; at 0 →
  append `unconscious` (pc) / `dead` (monster/npc) status and `is_alive=False`. Healing:
  `hp = min(max_hp, hp + healing)`. Append event_log lines
  ("Avantor deals 24 fire damage to kobold_2 (0/5 HP, dead)"). Increment
  `current_action_index`. Emit `RemoveMessage`s for this action's scaffolding.
- **narrate** (LLM, fixes bug 6): "summarize this combat turn in 2–3 vivid sentences using
  the event log"; returns `[response]`; gated by `config.enable_narration` (default True).
  This is the primary user-facing stream.

## 7. Config and LLM abstraction (`config.py`, `llm.py`)

- `AppConfig` (pydantic): `model: str = "openai:gpt-4o"` (env `DND_MODEL`),
  `data_dir: Path` (env `DND_DATA_DIR`, default = repo `data/` resolved relative to the
  package, NOT cwd), `fuzzy_threshold_lookup: int = 40`,
  `fuzzy_threshold_combatant: int = 70`, `default_adhoc_hp: int = 10`,
  `enable_narration: bool = True`, `stream_nodes: set[str] = {"calculate_damage", "narrate"}`.
- `get_llm(config)` via `langchain.chat_models.init_chat_model(config.model)` —
  provider-agnostic. **No module-import-time instantiation, no getpass.** Missing API key →
  clear runtime error. Interactive key prompting lives only in the entry points.
- `get_llm_with_tools(config)` binds the dice/math tools with `parallel_tool_calls=False`.

## 8. JSON data schemas (`data/`, data-file models in `schemas.py`)

All files: dict keyed by **slug id**; every entry has a human-readable `name`; the registry
fuzzy-matches against names AND ids.

- **characters.json** (`CharacterSheet`): `name, kind("pc"), class_: [str], subclass, race,
  level: int, max_hp: int, ac: int, attributes: {all six abilities}, proficiency_bonus: int,
  features: [ids], inventory: [ids], spells: [ids]`. Migrate Avantor (invent sensible
  level-10 bladesinger values for max_hp/ac/attributes) and Generic.
- **weapons.json** (`WeaponDef`): `name, damage: {dmg_type: dice_str}, magic_bonus: int = 0,
  ability, properties: [str], special_effects: str`. Dice strings validated with
  `^\d+d\d+([+-]\d+)?$`. Flametongue Greatsword migrates; Fireball moves out.
- **spells.json** (`SpellDef`): `name, level, damage | null, healing: dice_str | null,
  save: {ability, half_on_success} | null, attack_roll: bool,
  scaling_per_higher_level | null, targeting: "single"|"area"|"self",
  applies_status: {name, duration_rounds, effect} | null, description`.
  Populate `fireball` and `tensers_transformation` (status: extra 2d12 force on weapon
  attacks, duration 10 rounds).
- **features.json** (`FeatureDef`): `name, description, effect` (mechanical text consumed by
  the damage LLM). Populate `savage_attacks`, `great_weapon_master`.
- **monsters.json** (`MonsterDef`): `name, kind("monster"), max_hp, ac, attributes`.
  Populate kobold (5 HP, AC 12), goblin (7 HP, AC 15), skeleton, zombie, bandit, wolf.

**registry.py** — `DataRegistry(config)`: loads + validates all five files at construction
(clear per-file/per-entry errors); exposes `find_character(name)`, `find_item(name)` (merged
weapons+spells with a discriminator), `find_monster(name)`, `find_feature(name)`,
`combatant_from_character(id)`, `combatant_from_monster(id, suffix)`,
`adhoc_combatant(name, hp)`. Fuzzy matching delegates to `tools.fuzzy_match` with
configurable threshold.

## 9. Chainlit app + UI (frozen props contract)

- `src/app.py`: imports from `dnd_auto_dmg`; streaming filter changes from hardcoded
  `== "calculate_damage"` to `metadata["langgraph_node"] in config.stream_nodes`. Audio /
  whisper path untouched.
- **Props contract** (frozen in `state.py` docstring at WP3; JSX and app.py both conform):

```json
{"langgraphState": {
    "combatants": {"<id>": {"...Combatant.model_dump()..."}},
    "round": 1,
    "log": ["last", "10", "event_log", "lines"]
}}
```

- `LanggraphStateDisplay.jsx`: rewritten as a roster — one card per combatant: name + kind
  badge, HP bar (hp/max_hp width; green/yellow/red thresholds; grey/skull when dead), status
  badges with remaining duration, round number header, collapsible recent event log.
- `demo.py`: same scripted conversation, drop the `"log": []` key, print roster HP after
  each turn.

## 10. Testing strategy

- **Fake LLM boundary**: nodes receive `llm` by injection. `tests/conftest.py` provides a
  `ScriptedChatModel` returning a queued sequence of `AIMessage`s (including tool_calls) —
  this makes the calculate_damage tool loop testable. For `with_structured_output`, the fake
  returns the pydantic object directly.
- **Determinism**: monkeypatch/seed `random.randint` for roll_dice tests.
- **Unit**: tools (existing behavior locked in), schema validation (dice-regex rejects),
  registry on the REAL data files (doubles as data validation), `merge_combatants`
  (insert/replace/delete/None-on-missing), **SqliteSaver round-trip with pydantic
  Combatants**, resolve behaviors ("3 kobolds" → 3 instances; second volley reuses living
  ones; unknown monster → adhoc; status application; advance_round expiry), apply behaviors
  (HP clamping, death status, heal clamp, malformed-report fallback, cursor increment,
  RemoveMessage emission).
- **Integration**: `build_graph(llm=scripted_fake, registry=real)` runs the demo
  conversation on one thread and asserts: Tenser's status lands on Avantor; goblin HP drops;
  "They do it again" reuses actor/weapon; 5th-level fireball damages all 3 kobolds (dead at
  5 HP); "Literal nonsense" → END with no state change. Explicit termination assertion:
  `current_action_index == len(resolved_actions)`.
- No automated Chainlit tests — manual smoke checklist in WP6.
- Run tests with the repo venv: `.venv/bin/python -m pytest`.

## 11. Work packages

Each WP has **exclusive file ownership** (no two in-flight WPs touch the same file).
Acceptance = listed tests pass + stated smoke checks. Environment: existing `.venv`
(Python 3.10.19); install with `uv pip install -e ".[dev]" --python .venv/bin/python`.

| WP | Files owned | Acceptance criteria | Depends |
|----|-------------|---------------------|---------|
| **1** Scaffolding, config, llm, tools | `pyproject.toml`, `requirements.txt`, `src/dnd_auto_dmg/{__init__,config,llm,tools}.py`, `tests/test_tools.py` | editable install works; `python -c "from dnd_auto_dmg.llm import get_llm"` runs with **no key prompt / no network**; tools tests pass; unused deps removed | — |
| **2** Schemas, data, registry | `src/dnd_auto_dmg/{schemas,registry}.py`, `data/*.json` (5 files), `tests/{test_schemas,test_registry}.py` | registry loads/validates real data; Avantor has max_hp/ac; `find_item("fireball")` hits spells.json; monster lookup + README fuzzy cases pass | 1 |
| **3** State + de-risk + conftest | `src/dnd_auto_dmg/state.py`, `tests/{test_state,conftest}.py` | reducer tests pass; **SqliteSaver round-trip with pydantic Combatants passes** (else: record dict-fallback decision in ledger + implement it); props contract frozen in state.py docstring | 2 |
| **4a** Deterministic nodes | `src/dnd_auto_dmg/nodes/{__init__,begin_turn,resolve,apply}.py`, `tests/test_nodes_deterministic.py` | all resolve/apply/reset behaviors of §10 pass | 3 (∥4b) |
| **4b** LLM nodes | `src/dnd_auto_dmg/nodes/{relevance,parse,damage,narrate}.py`, `tests/test_nodes_llm.py` | with scripted fakes: relevance → bool, no messages appended; parse handles single+multi action; damage prompt contains real tool descriptions and **no literal `{tools}`/`{roll_dice}`**; damage returns only `[response]` | 3 (∥4a) |
| **5** Graph, demo, integration | `src/dnd_auto_dmg/graph.py`, `src/demo.py`, `tests/test_graph_integration.py` | §10 integration scenario passes incl. cursor-termination test; `demo.py` runs against real LLM as manual smoke (kobolds at 0 HP) | 4a+4b |
| **6** Chainlit + roster UI | `src/app.py`, `public/elements/LanggraphStateDisplay.jsx` | app boots via `chainlit run src/app.py`; roster renders HP bars/statuses after a turn; streaming from `config.stream_nodes`; audio path still imports | 3 (JSX), 5 (app) |
| **7** README + cleanup | `README.md`, `chainlit.md`, delete `src/agent.py` + old `src/tools.py`, remove `docs/` | README commands work verbatim from fresh clone; `grep -r "docs/character" src/` empty | all |
| **8** Stretch (optional) | initiative/turn-order (`turn_order` field + "roll initiative" action), LLM-estimated HP for unknown monsters, death saves, `@pytest.mark.live` suite | each behind its own tests | 5 |

Parallelism: 1 → 2 → 3 → {4a ∥ 4b ∥ 6-JSX} → 5 → {6-app ∥ 7} → 8.

## 12. Risks / ordering constraints

1. **Pydantic-in-state serialization** is the load-bearing assumption — WP3 tests it before
   any node code; dict-fallback decision recorded in the ledger.
2. **Per-turn state persists under the checkpointer** — `begin_turn` reset is structural;
   test it.
3. **Multi-action cursor loop**: infinite-loop risk if the index isn't incremented —
   explicit termination test in WP5; set a `recursion_limit` guard when compiling.
4. **RemoveMessage pruning** must happen only in `apply_damage` (never mid-tool-loop) and
   only for the current action's scaffolding. If flaky on the installed langgraph, degrade
   to keeping tool messages (correctness unaffected).
5. **Fuzzy threshold 40 too permissive for targets** — dual thresholds (40 lookup / 70
   combatant); tune against the demo script in WP5.
6. **Node-name coupling**: `app.py` streams by node name — `config.stream_nodes` is the
   single source of truth; WP5/WP6 must agree.
7. **`with_structured_output` provider variance** — keep `extract_json` fallback path.
8. Ordering: state (3) before nodes (4x) before graph (5); graph wiring last among core WPs.

## 12a. WP9 — Ollama as default model (added mid-project, 2026-07-20)

User decision: make **`ollama:minimax-m3:cloud`** the default chat model (replacing
`openai:gpt-4o`), with gpt-4o still reachable via `DND_MODEL=openai:gpt-4o`. The LLM layer
is already provider-agnostic (`init_chat_model`), so this is a config + dependency +
entry-point change, not an architecture change. Files owned by WP9:
`src/dnd_auto_dmg/config.py`, `src/dnd_auto_dmg/llm.py`, `src/app.py`, `src/demo.py`,
`requirements.txt`, `pyproject.toml`, `README.md`, and a small test.

Required changes and the gotchas each addresses:
1. **`config.py`**: `_model_default()` returns `os.environ.get("DND_MODEL",
   "ollama:minimax-m3:cloud")`. (`init_chat_model` splits provider from model on the FIRST
   colon, so `"ollama:minimax-m3:cloud"` → provider `ollama`, model `minimax-m3:cloud` —
   correct.)
2. **`llm.py`**: `bind_tools(tools, parallel_tool_calls=False)` uses an OpenAI-only kwarg
   that `ChatOllama.bind_tools` rejects. Make binding provider-robust: try with
   `parallel_tool_calls=False`, and on `TypeError` fall back to `bind_tools(tools)`. Keep it
   lazy (no construction-time network).
3. **`app.py` + `demo.py`**: both currently prompt for `OPENAI_API_KEY` unconditionally.
   Gate that prompt on the configured provider — only prompt when
   `AppConfig().model.split(":", 1)[0] == "openai"`. Ollama cloud auth is handled by the
   `ollama` daemon/CLI (`ollama signin`), NOT an env var this app manages, so no key prompt
   for the ollama default. Always keep the `CHAINLIT_AUTH_SECRET` prompt in `app.py`.
4. **Whisper caveat**: `app.py`'s speech-to-text still calls OpenAI Whisper
   (`AsyncOpenAI().audio.transcriptions`). That is independent of the chat model. Document
   that the OPTIONAL audio-input path needs `OPENAI_API_KEY` even when the chat model is
   Ollama; the text path does not. Do NOT rip out the audio path.
5. **`requirements.txt` + `pyproject.toml`**: add `langchain-ollama` so
   `init_chat_model("ollama:...")` can construct. (`ChatOllama` construction is lazy — no
   server ping until invoke — so `get_llm()` on the ollama default no longer raises at
   construction the way the openai default did; that's fine and expected.)
6. **`README.md`**: document Ollama as the default (pull/`ollama signin` for cloud models),
   switching back via `DND_MODEL=openai:gpt-4o`, and the audio/Whisper OpenAI caveat.
7. **Test**: assert `AppConfig().model == "ollama:minimax-m3:cloud"` (and that `DND_MODEL`
   overrides it); optionally assert the provider parse. Do NOT add a test that requires a
   running Ollama server. The existing 110 tests inject fakes and must stay green.

Acceptance: full suite green; `env -u OPENAI_API_KEY .venv/bin/python -c "from
dnd_auto_dmg.llm import get_llm; get_llm()"` constructs the ollama model without prompting or
network; `DND_MODEL=openai:gpt-4o` still selects OpenAI; README/entry-point key-prompting
matches provider. Live Ollama smoke (`python src/demo.py` with the daemon running / signed
in) is deferred to the user.

## 13. Orchestration protocol (PM agent duties)

- **Phases and pause gates** (one phase per run, then STOP and report — never continue into
  the next phase in the same run; the user reviews diffs / submits PRs / waits out usage
  limits between phases):
  - Phase A: WP1 → pause
  - Phase B: WP2 → pause
  - Phase C: WP3 → pause (includes serialization de-risk decision)
  - Phase D: WP4a ∥ WP4b ∥ WP6-JSX (parallel subagents) → pause
  - Phase E: WP5 → pause
  - Phase F: WP6-app + WP7 → pause
  - Phase G1: WP9 (Ollama default, §12a) → pause
  - Phase G2 (optional): WP8 stretch → done
- **Ledger discipline**: update `REFACTOR_PROGRESS.md` after EVERY subagent completes and
  before every pause — status, files touched, test command + result, decisions, resume
  notes. The ledger + this spec must be sufficient for a cold session to resume.
- **Delegation**: spawn Sonnet 5 subagents per WP with the relevant spec sections pasted
  into their prompts (subagents must not need to re-derive design decisions). Verify each
  WP's acceptance criteria yourself (run the tests) before marking `done`.
- **Resume protocol**: on "continue", re-read both docs, re-run the last `done` phase's
  test suite to confirm the base is green, then execute the first non-`done` phase.
- **Do not commit or push** — the user handles git at the pause gates.
- **Escalate, don't improvise**: if a design decision in this spec proves unworkable,
  record the problem + recommendation in the ledger under `blocked` and stop the phase.

---

# PART 2 — Combat Overhaul (WP10–WP16, Phases H1–H5; added 2026-07-26)

> Orchestrated by Opus 5 with Sonnet 5 subagents. Same protocol as Part 1 (§13): one phase
> per run, verify acceptance yourself, update `REFACTOR_PROGRESS.md` after every subagent
> and before every pause, never commit, escalate blockers to the ledger.

## §14. Why (live-run failures, all root-caused with file:line evidence)

First real Chainlit run on `ollama:minimax-m3:cloud` exposed:

1. **Statuses never land** — `resolve.py:204-217` applies only statuses the parse LLM lists
   in `statuses_applied`; real LLMs list none for a named spell, and the matched
   `SpellDef.applies_status` is never auto-applied (tests masked this by scripting the
   field). Status application also writes NO event_log line even when it works, and
   statuses always land on the ACTOR regardless of `SpellDef.targeting`.
2. **Silent turns** — `relevance.py:39` `startswith("yes")` breaks on `<think>` blocks
   (ChatOllama `reasoning` never set → think text lands in content), markdown (`**Yes**`),
   or prose imitation once AI narration accumulates; turn routes to END; `app.py` sends an
   EMPTY bubble (no fallback text path exists).
3. **Duplicated output** — both `calculate_damage` (raw ```json + prose + tool-call
   scratch) and `narrate` stream into ONE Chainlit message; narrate paraphrases the report
   AIMessage kept directly above it in history.
4. **Slow turns** — 5–7+ LLM calls/turn (relevance + parse + k×damage-tool-loop + narrate);
   every call carries FULL history that grows with kept report JSON; `llm.py:23` passes
   ZERO kwargs → server-default num_ctx (2048: silent truncation by ~turn 3),
   temperature 0.8, thinking enabled.
5. **B-list** (30 verified bugs, see ledger discussion). Fixed in this part: B1 (module-level
   `AsyncOpenAI()` blocks keyless ollama boot), B2 (dead combatants valid targets), B3
   (healing never revives), B4 (`extract_json` gives up on first brace candidate), B5
   (narrate reads stale cross-turn event lines), B6 (AoE divided across targets instead of
   full to each), B7 (`advance_round` needs an actor → rounds/durations never tick), B8
   (empty bubble), B9 (`divide()` returns 1 for ≤0), B10 (`subtract` docstring says
   "Adds"), B11 (repeat adhoc multi-target attacks reset HP), B12 (status ignores
   targeting), B13 (graph.py dead try/except bind), B14 (prior action's report in next
   prompt), B15/B16 (tool-scaffolding message pollution/orphans), B17 (case-sensitive
   fuzzy), B18 (threshold 40 matches "longsword"→flametongue, "bob"→kobold), B20 (registry
   drops proficiency/level), B21 (all features presented as in-play), B22/B23 (app.py
   shadowing / audio NameError), B27 (demo/app divergence), B29 (flametongue double-counts
   fire; weapons.json too sparse). Deferred: B19 (real persistence — README wording fixed
   only), B24 (auth), B26 (minor config knobs), B28 cosmetics, B30.

User-confirmed scope: player-turn bugs only (NO monster-AI turns); approved speed levers =
merge relevance into parse, deterministic Python dice for matched items, single combined
output call; add temp_hp.

## §15. Target graph topology

```
START → begin_turn → parse_turn
parse_turn  --route_parse-->  resolve_combatants   if turn_status == "combat"
                              respond              otherwise ("irrelevant" | "parse_failed")
resolve_combatants --route_action--> damage_det | calculate_damage | respond
damage_det → apply_damage
calculate_damage --tools_condition(damage_messages)--> roll_dice → calculate_damage
calculate_damage --(no tool calls)--> apply_damage
apply_damage --route_action--> damage_det | calculate_damage | respond
respond → END
```

- `route_action(state)` (ONE shared function in graph.py): `idx = current_action_index`;
  `idx >= len(resolved_actions)` → `"respond"`; elif `resolved_actions[idx].item is not
  None` → `"damage_det"`; else → `"calculate_damage"` (item_name_raw / homebrew only).
- `respond` is UNCONDITIONAL — every turn ends with exactly one streamed AIMessage.
- DELETED: `nodes/relevance.py`, `nodes/narrate.py`, inline `no_actions` node,
  `state.relevant_query`, `config.enable_narration` (WP15 holds sole deletion rights;
  earlier WPs keep them so the suite stays green at every gate).
- `roll_dice` node = `ToolNode(_TOOLS, messages_key="damage_messages")`; routing via
  `lambda s: tools_condition(s, messages_key="damage_messages")` (langgraph 1.0.1 supports
  both — verified).
- Common matched-item turn = exactly 2 LLM calls (parse_turn + respond), zero tool loops.

## §16. Schema / state / config changes (ALL additive until WP15)

Runtime models (`schemas.py`):
- `Combatant`: `temp_hp: int = 0`, `level: Optional[int] = None`,
  `proficiency_bonus: Optional[int] = None`.
- `StatusEffect`: `damage_rider: Optional[Dict[str, str]] = None` (dmg_type → dice; copied
  from AppliedStatus at cast time so the engine never needs a registry lookup).
- `ParsedAction.actor: str = ""` (actorless advance_round).
- `ParsedTurn`: `relevant: bool = True`.
- `ResolvedAction`: `item_id: Optional[str] = None`,
  `item_kind: Optional[Literal["weapon","spell"]] = None`,
  `feature_defs: Dict[str, dict] = {}` (id → FeatureDef.model_dump()),
  `features_invoked: List[str] = []` (canonical ids resolve matched from
  action.features_used). `feature_texts` kept for the LLM fallback prompt.

Data-file models (`schemas.py`):
- `AppliedStatus`: `damage_rider: Optional[Dict[str, str]] = None` (dice-string validated),
  `grants_temp_hp: Optional[int] = None`.
- `FeatureDef`: `crit_extra_die: bool = False`, `flat_damage_bonus: int = 0`,
  `opt_in: bool = False`.

Data edits (`data/`):
- `spells.json` tensers_transformation.applies_status gains
  `"damage_rider": {"force": "2d12"}, "grants_temp_hp": 50` (keep free-text `effect`).
- `features.json`: savage_attacks + `"crit_extra_die": true`; great_weapon_master +
  `"flat_damage_bonus": 10, "opt_in": true`.
- `weapons.json`: flametongue `special_effects` rewritten so it no longer restates the 2d6
  fire already in `damage` (e.g. "The blade is wreathed in flame."). ADD generic weapons:
  longsword 1d8 slashing (versatile), greatsword 2d6 slashing, dagger 1d4 piercing
  (finesse), shortbow 1d6 piercing (dexterity), mace 1d6 bludgeoning, handaxe 1d6 slashing.
- `registry.combatant_from_character` must populate `level` and `proficiency_bonus`
  (currently dropped, registry.py:173-184).

`CombatState` (`state.py`) — new scratch (reset by begin_turn each invoke):
- `turn_status: Literal["combat","irrelevant","parse_failed"]` (begin_turn default:
  `"parse_failed"` as safe value).
- `turn_event_start: int` — begin_turn sets `len(state.get("event_log") or [])`; respond
  uses `event_log[turn_event_start:]`.
- `pending_report: Optional[DamageReport]` — written by damage_det, consumed+cleared by
  apply_damage.
- `damage_messages: Annotated[list[BaseMessage], add_messages]` — isolated tool-loop
  channel; cleared with `RemoveMessage(id=REMOVE_ALL_MESSAGES)` by apply_damage per action
  and begin_turn per turn.
- `relevant_query` stays until WP15 deletes it.

`AppConfig` (`config.py`) — new fields, env-overridable via default_factory like `model`:
- `temperature: float = 0.0` (`DND_TEMPERATURE`)
- `num_ctx: int = 16384` (`DND_NUM_CTX`, ollama only)
- `keep_alive: str = "10m"` (`DND_KEEP_ALIVE`, ollama only)
- `model_kwargs: dict = {}` (`DND_MODEL_KWARGS` as JSON; merged LAST, wins)
- `fuzzy_threshold_lookup: 40 → 60`; `fuzzy_threshold_combatant: 70 → 80`
- `stream_nodes` default → `{"respond"}` (WP15 flips it; WP10 keeps current default so the
  old graph still streams until then — WP10 adds ONLY the new fields).

## §17. Deterministic damage engine (`engine.py`, new; pure, no LangGraph imports)

```python
def parse_dice(expr) -> (num, sides, modifier)          # "2d6+1" -> (2,6,1); ValueError on bad
def roll_dice_expr(expr, rng, *, double_dice=False, extra_dice=0) -> (total, [rolls])
def ability_mod(score) -> int                            # (score - 10) // 2
def is_deterministic(resolved) -> bool                   # resolved.item is not None
def compute_damage(resolved, combatants, action_index, rng=None) -> DamageReport
```

Rules:
1. Weapon: roll each type in `WeaponDef.damage`; add `ability_mod(actor.attributes[weapon
   .ability])` + `magic_bonus` ONCE to the primary (first) type. Independent roll per target.
2. Crit (`action.is_critical_hit`): every damage die rolled twice (weapon, rider, and
   attack-roll spell dice). Modifiers/flat bonuses NOT doubled. Save spells never crit.
3. Riders: each actor StatusEffect with `damage_rider` adds those dice (weapon attacks
   only); rider dice double on crit.
4. Feature hooks from `resolved.feature_defs`: `crit_extra_die` → +1 primary weapon die on
   crit; `flat_damage_bonus` always if `opt_in=False`, else only when id ∈
   `resolved.features_invoked`.
5. Spells: `SpellDef.damage` dice; upcast adds `scaling_per_higher_level` ×
   `max(0, action.spell_level - spell.level)`. No ability mod on spell damage. Saves
   assumed failed / attacks assumed to hit (documented simplification, matches current
   behavior).
6. AoE (`targeting == "area"`): ONE roll, FULL total to EACH target. `"single"` with
   multiple targets: independent roll per target.
7. Healing: `SpellDef.healing` dice per target.
8. Zero-damage casts (pure buffs): 0-total DamageReport with description
   "X casts Y." — flows through apply_damage harmlessly.
9. Per-type totals clamp ≥ 0. `breakdown` human-readable
   (`{"slashing": "2d6+5 = 12 [4,3]+5"}`); `description` deterministic one-liner.

`nodes/damage_det.py`: thin node factory — `{"pending_report": engine.compute_damage(...)}`.

## §18. Node contracts

**begin_turn** (extended): resets scratch incl. new fields (`turn_status="parse_failed"`,
`pending_report=None`, `turn_event_start=len(event_log)`), clears `damage_messages`
(REMOVE_ALL_MESSAGES), and ORPHAN-SWEEPS main `messages`: emit RemoveMessage for EVERY
ToolMessage and every AIMessage with tool_calls anywhere in history (heals threads poisoned
by aborted turns / pre-migration checkpoints — under the new policy none may legally exist).

**parse_turn** (`nodes/parse.py`, merged relevance+parse): one structured-output call
(`ParsedTurn` now has `relevant`), extract_json fallback. Status mapping:
- both paths fail/raise → `turn_status="parse_failed"`
- `relevant is False` OR `actions == []` → `"irrelevant"`
- else → `"combat"` (advance_round-only and status-only turns ARE combat).
Prompt additions: `relevant` semantics; self-targeting sentinel
(`targets: [{"name": "self"}]`); spell name always in `weapon_or_spell` for casts; do NOT
list statuses_applied for named spells (system auto-applies; statuses_applied is only for
ad-hoc conditions); advance_round needs no actor. Input = SystemMessage + last 10 persisted
messages (clean under the new policy). Apply `strip_think` before extract_json.

**resolve_combatants** (overhaul):
- AUTO-APPLY `SpellDef.applies_status` on cast: status lands per `SpellDef.targeting`
  (`self` → actor; else → resolved target_ids — fixes Hold-Person-on-caster), with
  `damage_rider` copied onto the StatusEffect and `grants_temp_hp` applied to the status
  target as `temp_hp = max(existing, grant)` (5e no-stack). Event_log line for every status
  application/removal and temp-HP grant. `statuses_applied` still honored for ad-hoc
  conditions (dedup by canonical name).
- Dead combatants filtered from fuzzy target candidates EXCEPT heal actions.
- Adhoc multi-target reuse: match existing living combatants by slug before creating
  (no full-HP resurrection of `cultist_1..3` on the second volley).
- `advance_round` is actorless: process even when actor is ""/unresolvable; produces no
  ResolvedAction (unchanged).
- Populate `item_id`/`item_kind`/`feature_defs`/`features_invoked` on ResolvedAction.

**calculate_damage** (LLM fallback only): invokes on
`[SystemMessage(action prompt)] + state["damage_messages"]` — NEVER main history (kills
cross-action pollution). Returns `{"damage_messages": [response]}`. Prompt trimmed: drop
duplicate status dump; only `features_invoked`+relevant riders as "in play"; keep
`is_crit`/`spell_level`/target ACs.

**apply_damage**: report source = `pending_report` (use+clear) else extract from
`damage_messages` final AIMessage (strip_think first), then clear channel. Temp HP:
`absorbed = min(temp_hp, damage)`; `temp_hp -= absorbed`; `hp -= damage - absorbed`; event
line mentions absorption. Healing revives: `hp > 0` → strip dead/unconscious statuses,
`is_alive=True`; healing never restores temp_hp. AoE full-damage handled upstream by
engine; the even-split fallback stays ONLY for degraded LLM reports and must distribute
`total_healing` too. Never touches main `messages`.

**respond** (`nodes/respond.py`, new; replaces narrate + no_actions): the ONE user-facing
LLM call, streamed. Inputs: turn_status, `event_log[turn_event_start:]`, damage_reports,
round_number, HP snapshot lines (`name: hp/max_hp (+N temp)`) for combatants touched this
turn. Four modes (computed in Python, one template):
- combat with material → 2–3 vivid sentences that MUST include the numbers (damage per
  target, remaining HP) sourced ONLY from the slice/reports;
- combat with empty slice → explain what couldn't be matched;
- irrelevant → one short in-character DM nudge;
- parse_failed → brief apology + one-line example action.
Deterministic fallback INSIDE the node: LLM raises or returns empty (after strip_think) →
synthesize AIMessage from the event slice / canned mode line. Returns
`{"messages": [AIMessage]}` — the only AIMessage persisted per turn.

**Message-persistence policy**: main history grows by exactly [HumanMessage, respond
AIMessage] per turn. Reports live in `damage_reports` + `event_log` only. System prompts
ephemeral everywhere (unchanged convention).

## §19. tools.py / llm.py

- `strip_think(text) -> str`: removes `<think>...</think>` blocks + leading unclosed
  `<think>` prefix. Applied to EVERY raw LLM text read (parse fallback, damage final
  message, respond guard).
- `extract_json` rewrite: pre-strip think blocks + markdown fences; scan ALL `{`/`[`
  candidates in order; attempt each balanced span; return first that json.loads (current
  impl returns None on first JSONDecodeError — tools.py:90-93).
- `fuzzy_match`: `processor=rapidfuzz.utils.default_process` (case/punct insensitive);
  score = `max(token_set_ratio, 0.9 * partial_ratio)`; default threshold param 60.
- `divide`: `max(0, math.floor(a / b))` — halving-save semantics; half of 1 is 0; document.
- `subtract` docstring: "Subtracts b from a."
- `llm.get_llm`: `kwargs = {"temperature": config.temperature}`; if provider == "ollama":
  `|= {"num_ctx": config.num_ctx, "keep_alive": config.keep_alive, "reasoning": False}`;
  `|= config.model_kwargs` (wins); pass to init_chat_model. (langchain-ollama 0.3.10
  supports `reasoning=False` — verified. strip_think stays as the second layer since cloud
  models may ignore it.) Keep the provider-gated `parallel_tool_calls` from the WP9 fix;
  WP15 deletes graph.py's dead `_bind_tools` try/except (B13) by reusing llm.py's gate.

## §20. app.py / demo.py / JSX (WP16)

- `async for chunk in app.astream(...)` (sync `app.stream` blocks the event loop).
- `openai_client = AsyncOpenAI()` moves lazily INSIDE `speech_to_text` → keyless ollama
  boot (B1). `CHAINLIT_AUTH_SECRET` prompt stays.
- Stream filter: `config.stream_nodes` (now `{"respond"}`); rename the shadowed `msg` loop
  variable (B22); guard: if nothing streamed, send last AIMessage content or canned line.
- Audio: fix `wav_file` NameError on empty buffer (B23); send the transcription message.
- JSX: render `temp_hp` (e.g. `62/62 +50` and/or a shield chip next to the HP bar).
  Existing status badges already work once statuses actually land.
- demo.py: print temp_hp in roster; note 2-LLM-call expectation in docstring.
- README: Part-2 features note; fix persistence overclaim (in-memory checkpointer, resume
  = new combat) (B19 wording only).

## §21. Work packages / phases (pause gate after EVERY phase)

| Phase | WP | Owns (exclusive) | Depends |
|---|---|---|---|
| H1 | WP10 contracts & data | schemas.py, state.py, config.py, registry.py, data/{spells,features,weapons}.json, tests/{test_schemas,test_state,test_registry}.py | — |
| H2 | WP11 tools+llm | tools.py, llm.py, tests/{test_tools,test_llm_config}.py | H1 |
| H2 | WP12 engine | engine.py (new), tests/test_engine.py (new) | H1 |
| H3 | WP13 deterministic nodes | nodes/{begin_turn,resolve,apply,damage_det}.py, tests/test_nodes_deterministic.py | H2 |
| H3 | WP14 LLM nodes | nodes/{parse,respond,damage}.py, tests/test_nodes_llm.py (must NOT delete relevance.py/narrate.py or touch nodes/__init__.py) | H2 |
| H4 | WP15 graph+integration | graph.py, nodes/__init__.py, tests/test_graph_integration.py; SOLE deletion rights: nodes/relevance.py, nodes/narrate.py, state.relevant_query, config.enable_narration, stream_nodes default flip | H3 |
| H5 | WP16 app+UI+docs | src/app.py, src/demo.py, public/elements/LanggraphStateDisplay.jsx, README.md, chainlit.md | H4 |

Acceptance per WP (orchestrator verifies itself; suite green at EVERY gate):
- WP10: full suite green (additive fields default cleanly); new tests: temp_hp default,
  damage_rider dice validation, tensers/savage/GWM JSON round-trip, registry level/PB
  passthrough. stream_nodes default UNCHANGED this phase.
- WP11: strip_think cases; extract_json recovers JSON after broken first candidate and
  inside think-polluted text; fuzzy: "longsword" ≠ flametongue at 60, "bob" ≠ kobold at 80,
  "AVANTOR" matches avantor; divide(0,2)==0, divide(1,2)==0, divide(-4,2)==0; monkeypatched
  init_chat_model capture asserts ollama kwargs + temperature.
- WP12: seeded-RNG tests for every §17 rule (crit doubles dice not modifiers; L5 fireball
  = 10d6; tensers rider 2d12→4d12 on crit; savage +1 die crit-only; GWM +10 only when
  invoked; AoE full per target; heal; clamp).
- WP13: regression test per bug: auto-apply status incl. targeting + event lines + temp-HP
  grant no-stack; dead-target filter (heals exempt); adhoc reuse; actorless advance_round
  ticks durations; orphan sweep removes mid-history scaffolding; apply temp-HP absorption +
  revive; even-split fallback distributes healing.
- WP14: parse status mapping (relevant-false → irrelevant; garbage → parse_failed;
  think-wrapped fallback JSON parses); respond 4 modes + deterministic fallback + exactly
  one AIMessage; damage uses damage_messages only (consciously deletes ~6 relevance/narrate
  tests, replaces with these).
- WP15: integration — matched-weapon turn consumes EXACTLY 2 scripted LLM responses;
  homebrew turn exercises tool loop, main messages NEVER holds a ToolMessage; post-turn
  messages == [Human, AI]; irrelevant turn yields an AIMessage; pre-poisoned checkpoint
  healed next turn; mixed det+LLM 2-action turn has no cross-action bleed (risk 1).
- WP16: py_compile; `env -u OPENAI_API_KEY` boot-path import works on ollama default;
  stream filter matches respond; JSX shows temp_hp; README claims match code. Manual
  Chainlit/demo smokes deferred to user.

## §22. Risks
1. damage_messages channel under SqliteSaver (REMOVE_ALL + re-add per action) → WP15 mixed
   det+LLM integration test is mandatory.
2. Small-model `relevant` flag unreliability → actions==[] mapping is the net; worst case a
   polite nudge.
3. `reasoning=False` ignored by some cloud models → strip_think mandatory everywhere.
4. Old checkpoints: additive schema hydrates via pydantic defaults; begin_turn sweep heals
   poisoned histories; enable_narration removal affects code only.
5. WP13/WP14 parallel coupling → orchestrator pastes §16 contracts verbatim into both
   prompts; NEITHER may edit schemas.py.
