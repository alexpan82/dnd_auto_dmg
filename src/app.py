from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
import chainlit as cl
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from agent import build_graph, _set_env
from typing import Dict, Any
from time import sleep

_set_env("CHAINLIT_AUTH_SECRET")

# Compile agent graph
graph = build_graph()
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)


async def update_state(state):
    return {"langgraphState":{"character": state["character"],
                 "metadata": state["metadata"]}
                 }


'''
@cl.on_chat_start
async def start():
    ...
'''

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass


@cl.on_message
async def on_message(msg: cl.Message):
    # Loading dot is visible after the first streaming token is added into the message.
    await msg.stream_token(" ")

    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    current_state = None

    for chunk in app.stream({"messages": [HumanMessage(content=msg.content)]}, 
                                    stream_mode=["messages", "values"], 
                                    config=RunnableConfig(callbacks=[cb], **config)):
        mode, data = chunk

        # Only print AI messages from the "calculate_damage" node
        if mode == 'messages':
            msg, metadata = data
            if (
                msg.content
                and not isinstance(msg, HumanMessage)
                and not isinstance(msg, SystemMessage)
                and metadata["langgraph_node"] == "calculate_damage"
            ):
                
                await final_answer.stream_token(msg.content)

        # Get most recent state
        elif mode == 'values':
            current_state = data

    # Stream AI tokens
    await final_answer.send()

    # Update custom UI element to show state info
    props = await update_state(current_state)
    element = cl.CustomElement(
        name="LanggraphStateDisplay",
        props = props)
    
    await cl.Message(
        content="Updated state:",
        elements=[element]
        ).send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Roll for initiative!",
            message="Generic swings an axe at a kobold",
            # icon="/public/write.svg",
        )
    ]