import os, getpass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage
from typing import TypedDict, Optional, List, Dict, Any, Literal
from tools import roll_dice, extract_json, add, subtract, multiply, divide, fuzzy_match
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
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
    last_user_prompt = state['messages'][-1].content
    
    chat_history = [msg.content for msg in state["messages"] if msg.type == 'human']
    chat_history = chat_history[:-1] # Get all previous queries
    
    # Only look back 2 messages
    if len(chat_history) >= 2:
        chat_history = chat_history[-2:]

    prompt = f"""Is the following content a DnD-related combat or action dialogue given the query history context? Simply answer with only "Yes" or "No"
    Query history:
    {chat_history}
    
    Current query: 
    {last_user_prompt}
    """

    message = [SystemMessage(content=prompt)]
    response = llm.invoke(message)

    return {
        "messages": message + [response],
        "relevant_query": response.content.lower(),
        "user_input": last_user_prompt
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
    chat_history = [msg.content for msg in state["messages"] if msg.type == 'human']
    chat_history = chat_history[:-1] # Get all previous queries
    
    # Only look back 2 messages
    if len(chat_history) >= 2:
        chat_history = chat_history[-2:]

    prompt = f"""You are a D&D action interpreter. Parse the user natural language input into JSON.
    Take note of the current character performing the action, the type of action, the spell or weapon they are using (weapon_id), if the prompt contains a "crit" (critical hit) reference, and if they activate / are using a feature.
    Output format:
    {{"character": "...", "action_type": "attack" or "heal" or "other", "weapon_id": "...", "target_id": "..." or "self" or "" if none, "is_critical_hit": true or false, "using_feat": [...]}}
    If no match, return null.
    Query history:
    {chat_history}

    Current query:
    {state['user_input']}
    """

    message = [SystemMessage(content=prompt)]
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

    parsed_query_json = state['parsed_action']
    
    # TODO: Make a router to end the interaction if character is not found  
    best_match, score = fuzzy_match(parsed_query_json["character"], 
                             character_json.keys())
    metadata['character_attributes'] = {best_match: character_json[best_match]}

    best_match, score = fuzzy_match(parsed_query_json["weapon_id"], 
                             weapons_json.keys())
    metadata['weapon_spell_attributes'] = {best_match: weapons_json[best_match]}

    return {
        "messages": [SystemMessage(content=f"Retrieved data from JSONs:\n{metadata}")],
        "metadata": metadata
        }

# --------- DAMAGE CALCULATOR NODE --------- #
def calculate_damage(state: CombatState) -> CombatState:
    print('Rolling damage 🎲 ...')

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
            [llm.invoke([sys_msg] + state["messages"])]}



# --------- BUILD LANGGRAPH --------- #
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


# --------- COMPILE AND RUN --------- #
conn = sqlite3.connect("test_checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)


if __name__ == "__main__":
    # Specify a thread
    config = {"configurable": {"thread_id": "1"}}
    
    app.get_graph().draw_mermaid_png(output_file_path='docs/graph.png')

    user_input = [HumanMessage(content="Avantor attacks the goblin with his greatsword")]
    result = app.invoke({
        "messages": user_input,
        "log": []
        },
        config)
    
    result = app.invoke({"messages": [HumanMessage(content="They do it again")]},
                        config)
    
    for m in result['messages']:
        m.pretty_print()



