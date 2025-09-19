import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from agent import build_graph
from langchain_core.messages import HumanMessage


if __name__ == "__main__":
    # --------- COMPILE AND RUN --------- #
    graph = build_graph()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    app = graph.compile(checkpointer=memory)
    app.get_graph().draw_mermaid_png(output_file_path='docs/graph.png')

    # Specify a thread
    config = {"configurable": {"thread_id": "1"}}

    # Example of user query chain
    result = app.invoke({
        "messages": [HumanMessage("Avantor casts Tenser's Transformation on themself")], 
        "log": []}, config)
    result = app.invoke({"messages": [HumanMessage("Avantor attacks the goblin with his greatsword")]}, config)
    result = app.invoke({"messages": [HumanMessage("They do it again")]}, config)
    result = app.invoke({"messages": [HumanMessage("He then casts 5th level fireball at a group of 3 kobolds")]}, config)
    result = app.invoke({"messages": [HumanMessage("Literal nonsense")]}, config)
    
    for m in result['messages']:
        m.pretty_print()



