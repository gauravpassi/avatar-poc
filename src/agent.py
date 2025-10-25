import os
import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent, AgentSession, JobContext, JobProcess,
    MetricsCollectedEvent, RoomInputOptions, RoomOutputOptions,
    WorkerOptions, cli, metrics,
)
from livekit.plugins import noise_cancellation, silero, simli, google
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful AI assistant. Always speak and respond in English only. "
                "Keep responses concise, natural, and friendly."
            )
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Detect console mode (mock room) vs real room
    room_name = getattr(ctx.room, "name", "")
    is_console = room_name == "mock_room" or os.getenv("LK_AGENT_MODE") == "console"

    if room_name:
        ctx.log_context_fields = {"room": room_name}

    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-live-2.5-flash-preview",
            voice="Aoede",
            temperature=0.6,
            instructions=(
                "You are a helpful, friendly AI assistant. "
                "Always speak, listen, and respond only in English. "
                "Do not switch to any other language, even if the user speaks another one."
            ),
        ),
    )

    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    if is_console:
        # Console mode: NO avatar, NO RoomInput/OutputOptions
        logger.info("Running in CONSOLE mode: disabling avatar & room I/O")
        await session.start(agent=Assistant())
    else:
        # Real room mode: avatar + room I/O are allowed
        logger.info("Running in ROOM mode: enabling avatar & room I/O")
        # Start Simli avatar (requires simli_API_KEY and simli_AVATAR_ID)
        avatar = simli.AvatarSession(
            simli_config=simli.SimliConfig(
                api_key=os.getenv("simli_API_KEY"),
                face_id=os.getenv("simli_AVATAR_ID"),
            ),
        )
        await avatar.start(session, room=ctx.room)

        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
                # turn_detector=MultilingualModel(language="en"),  # optional: force English speech detection
            ),
            # room_output_options=RoomOutputOptions(audio_enabled=True),  # enable if you want audible replies in room
        )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
