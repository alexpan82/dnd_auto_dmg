import os, getpass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage
from typing import TypedDict, Optional, List, Dict, Any, Literal
from tools import roll_dice, extract_json, add, subtract, multiply, divide, fuzzy_match
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages
import json


# -------- Set up OpenAI Key and Model -------- #
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o")
tools = [roll_dice, add, subtract, multiply, divide]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


# -------- Import relevant files -------- #
with open('docs/character.json', 'r') as f:
        character_json = json.load(f)
with open('docs/weapons.json', 'r') as f:
        weapons_json = json.load(f)


# -------- STATE DEFINITION -------- #
# TODO: Allow for multiple actions in the same prompt
class CombatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    parsed_action: Optional[Dict]
    character: Optional[Dict]
    metadata: Optional[Dict]
    character_id: str
    hp: int
    status: Optional[Dict]
    target: Optional[Dict]
    damage_report: Optional[Dict]
    log: List[str]
    relevant_query: str


# --------- RELEVANCE ROUTER --------- #
# A router that determines the relevancy of the user input
# and decides whether to calculate dmg or not
def is_relevant_query(state: CombatState) -> CombatState:
    print('Deciding relevance...')

    prompt = f"""Is the previous user query a DnD-related combat or action dialogue given the query history context? 
    Simply answer with only "Yes" or "No".
    Be permissive since DnD language has wide variance, but answer "No" to clearly irrelevant queries.
    """

    message = state["messages"] + [HumanMessage(content=state['user_input'])] + [SystemMessage(content=prompt)]
    response = llm_with_tools.invoke(message)

    return {
        "messages": message + [response],
        "relevant_query": response.content.lower(),
    }


def decide_relevance(state: CombatState) -> Literal["parse_action", "__end__"]:
    if state['relevant_query'] != 'yes':
        return "__end__"
    else:
        return "parse_action"


# --------- PARSER NODE --------- #
def parse_action(state: CombatState) -> CombatState:
    # Ask the LLM to extract structured action info
    print('Parsing action...')
    '''chat_history = [msg.content for msg in state["messages"] if msg.type == 'human']
    chat_history = chat_history[:-1] # Get all previous queries
    
    # Only look back 2 messages
    if len(chat_history) >= 2:
        chat_history = chat_history[-2:]'''

    prompt = f"""You are a knowledgeable D&D dungeon master assistant. 
    Parse the most recent user DnD combat / action query into JSON.
    Take note of the current character performing the action, the type of action, the spell or weapon they are using (weapon_id), if the prompt contains a "crit" (critical hit) reference, and if they activate a feature or gain/lose a status effect.
    Look at the previous query history to determine whether the character has an active feature or status effect.
    Output format:
    {{  "character": "...", 
        "action_type": "attack" or "heal" or "other", 
        "weapon_id": "...", 
        "target_id": "..." or "self" or "" if none, 
        "is_critical_hit": true or false,
        "active_statuses": [...]
        "using_feat": [...]
        }}

    If no match, return null.
    """

    message = state["messages"] + [SystemMessage(content=prompt)]
    response = llm.invoke(message)
    cleaned_response = extract_json(response.content) if "{" in response.content else None
    
    return {
        "messages": message + [response],
        "parsed_action": cleaned_response
        }


# --------- CHARACTER LOADER NODE --------- #
# TODO: Implement fuzzy matching here
def load_attributes(state: CombatState) -> CombatState:
    """
    Tries to find the corresponding characters / spells / actions / weapons
    referenced in the user query from imported json files.

    If there is a confident match, then pull that json entry into
    calculate_damage()
    """
    print('Matching and loading relevant JSON attributes...')
    metadata= {}

    if state['parsed_action'] is None:
        return {
        "metadata": None
        }

    parsed_query_json = state['parsed_action']
    
    # Find best character match
    # End the interaction if character is not found  
    best_match, score = fuzzy_match(parsed_query_json["character"], 
                             character_json.keys())
    if best_match is None:
        return {
        "metadata": None
        }
    metadata['character_attributes'] = {best_match: character_json[best_match]}

    # Find best weapon / spell / item match
    # If no good match, leverage the in-built knowledge of the LLM
    best_match, score = fuzzy_match(parsed_query_json["weapon_id"], 
                             weapons_json.keys())
    
    matched_weapon = parsed_query_json["weapon_id"] if best_match is None else weapons_json[best_match]
    metadata['weapon_spell_attributes'] = {best_match: matched_weapon}

    return {
        "messages": [SystemMessage(content=f"Retrieved data from JSONs:\n{metadata}")],
        "metadata": metadata
        }

# --------- DAMAGE CALCULATOR NODE --------- #
def calculate_damage(state: CombatState) -> CombatState:
    print('Rolling damage 🎲 ...')

    if state['metadata'] is None:
        return {
        "messages": [SystemMessage(content=f"Error in parsing user query / retrieving key data from JSONs")]
        }

    char = state['metadata']['character_attributes']
    action = state["parsed_action"]
    # TODO: Change state to keep track of active character statuses
    status = None
    is_crit = action['is_critical_hit']
    weapon = state['metadata']['weapon_spell_attributes']

    sys_prompt = '''
    You are a knowledgeable DnD (version 5e) DM tasked with accurately calculating damage / healing rolls. You have access to the following tools:
    {tools}

    You will be provided json-formatted parameters determining who is attacking / healing and with what along with relevant features, stats, etc.
    
    Action: Roll the resultant die using [{roll_dice}] to calculate the total damage / healing of the action given the parameters.
    
    Then return the damage-type breakdown in json format:
    {"total_damage": int, 
     "total_heal": int, 
     "breakdown": json,
     "notes": str}
    '''

    user_prompt = f"""DnD action context:
    Action: {action}
    weapon/spell json: {weapon}
    Character attributes: {char}
    Character status: {status}
    is_crit: {is_crit}"""

    return {"messages": [llm_with_tools.invoke([SystemMessage(content=sys_prompt)] + [SystemMessage(user_prompt)] + state["messages"])]}


# --------- OUTPUT NODE --------- #
def narrator_output(state: CombatState) -> CombatState:
    # Base agent system calls
    sys_msg = ''''''
    
    return {"messages": [SystemMessage(content=sys_msg)] + 
            [llm_with_tools.invoke(state["messages"] + [sys_msg] + state["messages"])]}



# --------- BUILD LANGGRAPH --------- #
def build_graph():
    graph = StateGraph(CombatState)
    graph.add_node("is_relevant_query", is_relevant_query)
    graph.add_node("roll_dice", ToolNode(tools))
    graph.add_node("parse_action", parse_action)
    graph.add_node("load_attributes", load_attributes)
    graph.add_node("calculate_damage", calculate_damage)
    # graph.add_node("narrate", narrator_output)

    graph.set_entry_point("is_relevant_query")
    graph.add_conditional_edges(
        "is_relevant_query",
        decide_relevance
        )

    graph.add_edge("parse_action", "load_attributes")
    graph.add_edge("load_attributes", "calculate_damage")
    # graph.add_edge("calculate_damage", "assistant")
    graph.add_conditional_edges(
        "calculate_damage",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
        {
            'tools': 'roll_dice',
            END: END
        }
    )
    graph.add_edge("roll_dice", "calculate_damage")

    return(graph)

