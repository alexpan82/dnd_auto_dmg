from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
import chainlit as cl
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from agent import build_graph, _set_env

load_dotenv()

graph = build_graph()
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)

_set_env("CHAINLIT_AUTH_SECRET")

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
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    for msg, metadata in app.stream({"messages": [HumanMessage(content=msg.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and not isinstance(msg, SystemMessage)
            and metadata["langgraph_node"] == "calculate_damage"
        ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Roll for initiative!",
            message="Generic swings an axe at a kobold",
            # icon="/public/write.svg",
        )
    ]