from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
import chainlit as cl
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dnd_auto_dmg.graph import build_graph, RECURSION_LIMIT
from dnd_auto_dmg.config import AppConfig
import io
from openai import AsyncOpenAI
import wave
import numpy as np
import audioop
import getpass
import os


def _set_env(var: str) -> None:
    """Prompt for an env var if unset. Entry-point-level interactive
    prompting is allowed here (Spec §7) - and ONLY here and demo.py."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("CHAINLIT_AUTH_SECRET")
_set_env("OPENAI_API_KEY")

# Compile agent graph
config = AppConfig()
graph = build_graph(config=config)
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)
app = graph.compile(checkpointer=memory)


async def update_state(state):
    state = state or {}
    combatants = state.get("combatants") or {}
    return {
        "langgraphState": {
            "combatants": {
                cid: (c.model_dump() if hasattr(c, "model_dump") else c)
                for cid, c in combatants.items()
            },
            "round": state.get("round_number", 1),
            "log": (state.get("event_log") or [])[-10:],
        }
    }


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

    config_dict = {
        "configurable": {"thread_id": cl.context.session.id},
        "recursion_limit": RECURSION_LIMIT,
    }
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    current_state = None

    for chunk in app.stream({"messages": [HumanMessage(content=msg.content)]},
                                    stream_mode=["messages", "values"],
                                    config=RunnableConfig(callbacks=[cb], **config_dict)):
        mode, data = chunk

        # Only print AI messages from the streamed nodes (config.stream_nodes)
        if mode == 'messages':
            msg, metadata = data
            if (
                msg.content
                and not isinstance(msg, HumanMessage)
                and not isinstance(msg, SystemMessage)
                and metadata.get("langgraph_node") in config.stream_nodes
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

openai_client = AsyncOpenAI()

@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )

    return response.text


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])
    return True


# Define a threshold for detecting silence and a timeout for ending a turn
SILENCE_THRESHOLD = (
    3500  # Adjust based on your audio level (e.g., lower for quieter audio)
)
SILENCE_TIMEOUT = 1300.0  # Seconds of silence to consider the turn finished
@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    audio_chunks = cl.user_session.get("audio_chunks")
    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(
        chunk.data, 2
    )  # Assumes 16-bit audio (2 bytes per sample)

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)


async def process_audio():
    # Get the audio buffer from the session
    if audio_chunks := cl.user_session.get("audio_chunks"):
        # Concatenate all chunks
        concatenated = np.concatenate(list(audio_chunks))

        # Create an in-memory binary stream
        wav_buffer = io.BytesIO()

        # Create WAV file with proper parameters
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # sample rate (24kHz PCM)
            wav_file.writeframes(concatenated.tobytes())

        # Reset buffer position
        wav_buffer.seek(0)

        cl.user_session.set("audio_chunks", [])

    frames = wav_file.getnframes()
    rate = wav_file.getframerate()

    duration = frames / float(rate)
    if duration <= 1.71:
        print("The audio is too short, please try again.")
        return

    audio_buffer = wav_buffer.getvalue()

    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")

    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    transcription = await speech_to_text(whisper_input)

    # await cl.Message(content=HumanMessage(transcription)).send()
    # msg = cl.Message(content=HumanMessage(transcription))
    # await msg.send()
    # on_message(msg)
    '''await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()'''

    await on_message(cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ))
    

@cl.on_audio_end
async def on_audio_end():
    print()