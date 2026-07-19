# Refactor Progress Ledger

> Checkpoint state for the multi-combatant refactor. Spec: `REFACTOR_PLAN.md` (§13 defines
> how this file is maintained). Updated by the orchestrator after every subagent completes
> and before every pause. A cold session resumes by reading the spec + this file only.

**Current phase:** D (WP4a ∥ WP4b ∥ WP6-JSX) — awaiting user go-ahead
**Base health:** WP1–WP3 done and green (69/69 tests); WP1 merged as PR #7, WP2 merged as
  PR #8 (`bf12c3a`); old code intact; `docs/` untouched (removed in WP7)
**Last updated:** 2026-07-19

## Phase / WP status

| Phase | WP | Status | Notes |
|-------|----|--------|-------|
| A | WP1 scaffolding/config/llm/tools | `done` | 23/23 tool tests pass; deps trimmed; lazy llm/config verified |
| B | WP2 schemas/data/registry | `done` | 50/50 tests pass; registry validates real data; fireball resolves as spell |
| C | WP3 state + serialization de-risk + conftest | `done` | 69/69 tests; serialization de-risk PASSED — pydantic kept in channels |
| D | WP4a deterministic nodes | `pending` | — |
| D | WP4b LLM nodes | `pending` | — |
| D | WP6-JSX roster element | `pending` | — |
| E | WP5 graph/demo/integration | `pending` | — |
| F | WP6-app Chainlit wiring | `pending` | — |
| F | WP7 README + cleanup | `pending` | — |
| G | WP8 stretch (optional) | `pending` | — |

## Decisions log

*(record irreversible/architectural decisions here — e.g. WP3 serialization outcome)*

- **WP1 / config env handling:** `pydantic-settings` is NOT installed and was NOT added.
  `AppConfig` is a plain pydantic v2 `BaseModel` reading `DND_MODEL` / `DND_DATA_DIR` from
  `os.environ` via `Field(default_factory=...)` callables (lazy, at construction). Keeps the
  dependency footprint minimal.
- **WP1 / data_dir resolution:** default `data_dir = Path(__file__).resolve().parents[2] /
  "data"` (package-relative, not cwd), per §7. `data/` does not exist yet (WP2 creates it);
  `AppConfig()` does not require it to exist.
- **WP1 / llm error timing (observed):** langchain-openai's `ChatOpenAI` validates the API
  key at *construction*, so a missing `OPENAI_API_KEY` raises a clear `openai.OpenAIError`
  when `get_llm()` is called (not at first `.invoke()`). This is spec-compliant (§7 "missing
  key → clear runtime error"): no import-time instantiation, no getpass, no network (error is
  raised before any HTTP request). Importing `get_llm` and constructing `AppConfig()` remain
  side-effect-free.
- **WP1 / requirements+deps:** kept `langsmith` and `langgraph-cli[inmem]` (part of the
  langgraph toolchain; §11 permitted this at discretion). Removed torch, transformers,
  accelerate, tavily-python, wikipedia, trustcall, notebook. Added pydantic. pyproject
  `dependencies` mirror the trimmed `requirements.txt`; `dev` extra = pytest.
- **WP2 / schemas layout:** runtime models (§4) AND data-file models (§8) both live in
  `schemas.py` (single source of truth). Shared `DICE_STRING_RE = ^\d+d\d+([+-]\d+)?$`
  enforced via `field_validator` on `WeaponDef.damage`, `SpellDef.damage/healing/
  scaling_per_higher_level`. `CharacterSheet.class_` uses `Field(alias="class")` with
  `populate_by_name=True` so JSON keeps the natural `"class"` key.
- **WP2 / registry API shapes:** `find_character/find_monster/find_feature` return
  `Optional[Tuple[id, model]]`; `find_item` returns `Optional[Tuple[id,
  "weapon"|"spell", model]]` (discriminator per §8). Load/validation failures raise
  `DataRegistryError(ValueError)` with `"<file>.json entry '<id>': ..."` context. Fuzzy
  matching delegates to `tools.fuzzy_match` at `config.fuzzy_threshold_lookup`, over a
  candidate list of BOTH slug ids and display names mapped back to ids.
- **WP2 / combatant builders:** builders deep-copy mutable fields (attributes/features/
  inventory) so instances never share state; `combatant_from_monster(id, n)` yields id
  `"{id}_{n}"` / name `"Kobold 2"`; `adhoc_combatant` slugifies the name (fallback
  `"combatant"`), `kind="monster"`, `origin="adhoc"`; unknown ids raise `KeyError`.
- **WP3 / SERIALIZATION DE-RISK OUTCOME (§12 risk 1): PASS — pydantic models in state
  channels are KEPT.** Orchestrator ran a standalone experiment (2026-07-19) before any
  state.py code was written: a minimal StateGraph with `combatants: Annotated[Dict[str,
  Combatant], merge_combatants]` compiled with `SqliteSaver(sqlite3.connect(file))`,
  invoked on thread t1; then a FRESH connection + fresh SqliteSaver + fresh graph over the
  same DB file. Result: `get_state().values["combatants"]["avantor"]` re-hydrated as a
  `Combatant` INSTANCE (not dict), nested `statuses[0]` as `StatusEffect`, and a node
  running on the resumed thread received a real `Combatant` and successfully
  `.model_copy(update=...)`-ed it (hp 62→52). JsonPlusSerializer handles pydantic v2 fine.
  **No dict fallback needed** — §5's "TypedDict state, pydantic values" design stands.
- **WP2 / data content:** Avantor migrated with invented level-10 Bladesinger stats
  (max_hp 62, ac 15, all six abilities keeping source strength 18, proficiency_bonus 4);
  "Blade-signing" typo fixed to "Bladesinging"; Fireball moved weapons→spells;
  tensers_transformation applies status (2d12 force rider, 10 rounds); six monsters
  (kobold 5/12, goblin 7/15, skeleton 13/13, zombie 22/8, bandit 11/12, wolf 11/13).

## Per-WP details

### WP1 — Scaffolding, config, llm, tools
- Status: `done`
- Files touched (all NEW except requirements.txt which was trimmed):
  - `pyproject.toml` (setuptools; package under `src/dnd_auto_dmg`; `[project.optional-dependencies].dev = [pytest]`; `[tool.setuptools.packages.find] where=["src"] include=["dnd_auto_dmg*"]`)
  - `requirements.txt` (trimmed — see decisions log)
  - `src/dnd_auto_dmg/__init__.py`
  - `src/dnd_auto_dmg/config.py` (`AppConfig` per §7)
  - `src/dnd_auto_dmg/llm.py` (`get_llm` / `get_llm_with_tools` via `init_chat_model`; lazy, no import-time getpass/instantiation; `bind_tools(..., parallel_tool_calls=False)`)
  - `src/dnd_auto_dmg/tools.py` (ported from old `src/tools.py`; `fuzzy_match(query, matches, threshold=40)` — threshold now a parameter, default preserves old behavior; otherwise byte-for-byte behavior)
  - `tests/test_tools.py` (23 tests; `random.randint` monkeypatched for roll_dice determinism)
- NOT touched (old code kept working): `src/agent.py`, `src/app.py`, `src/demo.py`, old
  `src/tools.py`, `docs/`, `public/`, `README.md`.
- Verification (orchestrator ran these directly, 2026-07-19):
  - `uv pip install -e ".[dev]" --python .venv/bin/python` → built + installed `dnd-auto-dmg==0.1.0`, pulled in pytest 9.1.1.
  - `.venv/bin/python -c "from dnd_auto_dmg.llm import get_llm; from dnd_auto_dmg.config import AppConfig; print(AppConfig())"` (stdin=/dev/null) → EXIT 0, printed full AppConfig repr, **no key prompt, no network**.
  - `.venv/bin/python -m pytest tests/test_tools.py -v` → **23 passed in 0.14s**.
  - Removed-deps grep on requirements.txt → torch/transformers/accelerate/tavily-python/wikipedia/trustcall/notebook all confirmed **removed**.
  - `.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import tools"` → old flat module still imports (EXIT 0), confirming old code untouched.
- Resume notes: WP1 was merged by the user as PR #7 (`e5c126a`) before Phase B started.

### WP2 — Schemas, data, registry
- Status: `done`
- Files touched (all NEW; no existing file modified):
  - `src/dnd_auto_dmg/schemas.py` — §4 runtime models (StatusEffect, Combatant, TargetRef,
    ParsedAction, ParsedTurn, ResolvedAction, PerTargetDamage, DamageReport) + §8 data-file
    models (CharacterSheet, WeaponDef, SpellSave, AppliedStatus, SpellDef, FeatureDef,
    MonsterDef) + shared dice-string regex validation
  - `src/dnd_auto_dmg/registry.py` — `DataRegistry(config)` + `DataRegistryError`; loads/
    validates all five data files at construction; find_character/find_item/find_monster/
    find_feature + combatant_from_character/combatant_from_monster/adhoc_combatant
  - `data/characters.json` (avantor, generic), `data/weapons.json` (flametongue_greatsword
    only), `data/spells.json` (fireball, tensers_transformation), `data/features.json`
    (savage_attacks, great_weapon_master), `data/monsters.json` (kobold, goblin, skeleton,
    zombie, bandit, wolf)
  - `tests/test_schemas.py` (15 tests), `tests/test_registry.py` (12 tests)
- NOT touched: all WP1 files, `src/agent.py`, `src/app.py`, `src/demo.py`, old
  `src/tools.py`, `docs/` (source data read-only; removed in WP7), README.
- Verification (orchestrator ran these directly, 2026-07-19):
  - `.venv/bin/python -m pytest tests/ -v` → **50 passed in 0.17s** (23 WP1 + 27 WP2).
  - Direct registry checks against real `data/`: registry loads (2 chars / 1 weapon /
    2 spells / 2 features / 6 monsters); `find_character("Avantor")` → max_hp=62, ac=15,
    6 attributes, pb=4; `find_item("fireball")` → `("fireball", "spell", SpellDef)`;
    `find_item("flametongue")` → weapon; README-style fuzzy hits: "flame sword",
    "greatsword" → flametongue_greatsword; "fire ball" → fireball; "tensers" →
    tensers_transformation; `find_monster("kobold")` → 5 HP / AC 12;
    `find_monster("xyzzy")` → None; builders: `combatant_from_character("avantor")`
    hp==max_hp==62/pc, `combatant_from_monster("kobold", 2)` → id kobold_2 hp 5,
    `adhoc_combatant("Weird Beast", 10)` → id weird_beast, origin adhoc.
  - Schema-level dice rejection: `"2d6+"`, `"d6"`, `"abc"`, `"2x6"` all raise
    `ValidationError`; corrupted tmp-dir weapons.json raises `DataRegistryError`
    naming `weapons.json` and entry `flametongue_greatsword`.
  - Legacy intact: `git diff --stat` vs HEAD empty (no tracked file modified);
    old `src/tools.py` still imports.
- Resume notes: base is green. Next phase = **C (WP3)**: state + serialization de-risk +
  conftest (see §5, §10, §11 WP3 row, §12 risk 1). WP3 owns `src/dnd_auto_dmg/state.py`,
  `tests/test_state.py`, `tests/conftest.py`. It must FIRST verify SqliteSaver round-trips
  pydantic `Combatant`s inside state channels (risk 1); if that fails, record the
  dict-fallback decision in this ledger and implement it. `state.py` docstring must freeze
  the §9 UI props contract. `conftest.py` provides the `ScriptedChatModel` fake, registry
  fixture, tmp data dir. Await user go-ahead before starting.

### WP3 — State, serialization de-risk, conftest
- Status: `done` (implementation + verification by orchestrator; ledger completion by main
  session after the orchestrator hit an API connection error post-verification)
- Files touched (all NEW; no existing file modified):
  - `src/dnd_auto_dmg/state.py` — `CombatState` TypedDict per §5 (`messages` add_messages,
    `combatants` merge_combatants, `round_number` last-write-wins, `event_log` operator.add,
    five per-turn scratch fields documented as begin_turn-reset); `merge_combatants` reducer
    (per-key REPLACE, `None` deletes, delete-of-missing is silent no-op, never mutates
    existing); module docstring freezes the §9 UI props contract verbatim.
  - `tests/test_state.py` — 9 reducer unit tests; 2 SqliteSaver round-trip lock-in tests
    (fresh connection/saver/graph over the same DB file re-hydrates `Combatant` +
    nested `StatusEffect` as real instances; resumed-thread node receives a live model and
    `.model_copy()`s it); 1 test demonstrating scratch fields persist across invokes
    without an explicit reset (motivates begin_turn); 7 ScriptedChatModel behavior tests.
  - `tests/conftest.py` — `ScriptedChatModel` (queued AIMessages incl. tool_calls;
    `with_structured_output` returns pydantic objects directly; `bind_tools` shares the
    queue; tracks calls), `make_scripted_llm` factory fixture, `app_config` fixture,
    `registry` fixture over real `data/`, `tmp_data_dir` fixture.
- Verification: orchestrator ran the de-risk experiment standalone (PASS — see Decisions
  log) and reported 69/69 green + conftest importing without an API key before its
  connection dropped; main session independently re-ran `.venv/bin/python -m pytest
  tests/ -q` → **69 passed** and reviewed all three files against §5/§9/§10.
- Resume notes: base is green. Next phase = **D**, three parallel Sonnet subagents:
  **WP4a** deterministic nodes (`nodes/{__init__,begin_turn,resolve,apply}.py` +
  `tests/test_nodes_deterministic.py`), **WP4b** LLM nodes
  (`nodes/{relevance,parse,damage,narrate}.py` + `tests/test_nodes_llm.py`), **WP6-JSX**
  roster element (`public/elements/LanggraphStateDisplay.jsx` only, built against the
  frozen props contract in `state.py`). File ownership is disjoint — safe to parallelize.
  Await user go-ahead before starting.

*(add a section per WP as work begins)*
