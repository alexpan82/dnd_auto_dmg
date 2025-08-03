import os, getpass
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage
from typing import TypedDict, Optional, List, Dict, Any, Literal
from langgraph.checkpoint.memory import MemorySaver
from tools import roll_dice, extract_json
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# -------- Set up OpenAI Key and Model -------- #
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

tools = [roll_dice]
llm = ChatOpenAI(model="gpt-4o")
llm = llm.bind_tools(tools, parallel_tool_calls=False)


# -------- STATE DEFINITION -------- #
class CombatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    parsed_action: Optional[Dict]
    character: Optional[Dict]
    character_id: str
    hp: int
    status: Optional[Dict]
    target: Optional[Dict]
    damage_report: Optional[Dict]
    log: List[str]


# --------- ASSISTANT NODE --------- #
# Base agent system calls
sys_msg = '''You are a knowledgeable DnD DM tasked with calculating damage rolls. 
The player will provide a piece of action dialogue containing who is attacking and with what.
You are to calculate the total damage of the action and provide the damage-type breakdown based on provided json-formatted parameters (features, weapon, stats, etc) that determine the damage.'''
def assistant(state: CombatState):
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

# A router that determines the relevancy of the user input
# and decides whether to calculate dmg or not
def decide_relevance(state: CombatState) -> Literal["parse_action", "end"]:
    prompt = f"""Is the following content a DnD-related combat or action dialogue? Simply answer with only "Yes" or "No"
    Content: 
    {state['user_input']}"""

    response = llm.invoke([SystemMessage(content=prompt)])
    if "yes" in response.content.lower():
        return "parse_action"
    else:
        return "end"


# --------- PARSER NODE --------- #
# TODO: We should load the character and target prior to this node
# So it's obvious to the LLM how to create the action json
# Consider giving the LLM the choice to choose who are the 
# target(s) given the keys of a json file
# TODO: Consider how to heal / get temp hp
def parse_action(state: CombatState) -> CombatState:
    # Ask the LLM to extract structured action info
    prompt = f"""You are a D&D action interpreter. Parse the user natural language input into JSON:
    User input: {state['user_input']}
    Output format:
    {{"character": "...", "action_type": "...", "weapon_id": "...", "target_id": "...", "is_critical_hit": true/false, "using_feat": [...]}}
    If no match, return null."""

    response = llm.invoke([SystemMessage(content=prompt)])
    cleaned_response = extract_json(response.content) if "{" in response.content else None

    state["parsed_action"] = cleaned_response
    return state


# --------- CHARACTER LOADER NODE --------- #
def load_character(state: CombatState) -> CombatState:
    # Mock character data
    char_db = {
        "Avantor": {
            "attributes": {"strength": 18},
            "features": ["Savage Attacks", "Great Weapon Master"],
            "inventory": {
                "Flametongue Greatsword": {
                    "damage": ["2d6", "2d6"],
                    "type": "slashing",
                    "magic_bonus": 1
                }
            }
        }
    }
    char_id = state["parsed_action"]["character"]
    state["character"] = char_db.get(char_id)
    state["character_id"] = char_id
    return state


# --------- STATUS EFFECT LOADER NODE --------- #
def load_status(state: CombatState) -> CombatState:
    # Mock status context (e.g., target has fire resistance)
    state["status"] = {
        "target_id": state["parsed_action"]["target_id"],
        "resistances": ["fire"],
        "buffs": [],
        "homebrew_modifiers": []
    }
    return state


# --------- DAMAGE CALCULATOR NODE --------- #
# TODO: Make this a react agent call rather than rely on a for-loop
# to roll for damage for each dmg type
def calculate_damage(state: CombatState) -> CombatState:
    char = state["character"]
    char_id = state["character_id"]
    action = state["parsed_action"]
    status = state["status"]
    is_crit = action['is_critical_hit']

    # TODO: Write a fuzzy match helper for this
    # Example: A user might say "Attack with my sword"
    # But the char["inventory"] json has a name attr w/ Flametongue Greatsword
    # Should return a json
    # weapon = char["inventory"][action["weapon_id"]]
    weapon = {"damage": ["2d6", "2d6"], "type": "slashing", "magic_bonus": 1}
    
    prompt = f"""Roll resultant die for {char_id} given the following information:
    {char_id} action: {action}
    weapon/spell json: {weapon}
    {char_id} attributes: {char}
    {char_id} status: {status}
    is_crit: {is_crit}

    Output in json format:
    {{"total_damage": int, 
     "breakdown": json,
     "notes": str}}
     """

    response = llm.invoke([SystemMessage(content=prompt)])
    # TODO: Get AI message from this tool call. response content is empty
    print("calculate_damage", response)

    cleaned_response = extract_json(response.content) if "{" in response.content else None

    state["damage_report"] = cleaned_response
    return state


# --------- OUTPUT NODE --------- #
def narrator_output(state: CombatState) -> CombatState:
    char_name = state["character"]
    dmg = state["damage_report"]
    # desc = f"{char_name} hits for {dmg['total_damage']} damage! ({', '.join(f'{k}: {v}' for k,v in dmg['breakdown'].items())})"
    # state["log"].append(desc)
    # print("🧙 " + desc)
    print("🧙 " + dmg)
    return state


# --------- BUILD LANGGRAPH --------- #
graph = StateGraph(CombatState)
graph.add_node("assistant", assistant)
graph.add_node("roll_dice", ToolNode(tools))
graph.add_node("parse_action", parse_action)
graph.add_node("load_character", load_character)
graph.add_node("load_status", load_status)
graph.add_node("calculate_damage", calculate_damage)
graph.add_node("narrate", narrator_output)

graph.set_entry_point("assistant")
graph.add_edge("assistant", "parse_action")
# graph.add_edge("assistant", END)
graph.add_conditional_edges(
    "assistant",
    decide_relevance,
    {
        "parse_action": "parse_action",
        "end": END
    }
)
graph.add_edge("parse_action", "load_character")
graph.add_edge("load_character", "load_status")
graph.add_edge("load_status", "calculate_damage")
graph.add_edge("roll_dice", "calculate_damage")
graph.add_conditional_edges(
    "calculate_damage",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
    {
        'tools': 'roll_dice',
        END: 'narrate'
    }
)
graph.add_edge("narrate", END)


# --------- COMPILE AND RUN --------- #
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


if __name__ == "__main__":
    # Specify a thread
    config = {"configurable": {"thread_id": "1"}}
    
    app.get_graph().draw_mermaid_png(output_file_path='docs/graph.png')

    # user_input = input("🎲 Describe your attack: ")

    user_input = 'Avantor attacks with his greatsword'
    result = app.invoke({
        "user_input": user_input,
        "log": []
        },
        config)
    