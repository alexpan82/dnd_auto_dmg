import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from agent import build_graph


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
        "user_input": "Avantor casts Tenser's Transformation on themself", 
        "log": []}, config)
    result = app.invoke({"user_input": "Avantor attacks the goblin with his greatsword"}, config)
    result = app.invoke({"user_input": "They do it again"}, config)
    result = app.invoke({"user_input": "He then casts 5th level fireball at a group of 3 kobolds"}, config)
    result = app.invoke({"user_input": "Literal nonsense"}, config)
    
    for m in result['messages']:
        m.pretty_print()



