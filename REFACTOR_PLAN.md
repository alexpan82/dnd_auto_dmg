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
  - Phase G (optional): WP8 → done
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
