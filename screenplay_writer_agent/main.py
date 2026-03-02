# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""screenplay-writer-agent - A Bindu Agent for Screenplay Writing."""

import argparse
import asyncio
import json
import os
import re
import sys
import traceback
from logging import getLogger
from pathlib import Path
from textwrap import dedent
from typing import Any, cast

from bindu.penguin.bindufy import bindufy
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Error constants
ERROR_NO_API_KEY = "No API key available"
ERROR_CREW_NOT_INITIALIZED = "Crew not initialized"
ERROR_API_CONFIG = "API key configuration error"

# Global variables
crew: Crew | None = None
_initialized = False
_init_lock = asyncio.Lock()
_logger = getLogger(__name__)


def load_config() -> dict:
    """Load agent configuration from project root."""
    config_path = Path(__file__).parent / "agent_config.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                return cast(dict[str, Any], json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("Failed to load config from %s", config_path, exc_info=exc)

    print("⚠️  No agent_config.json found, using default configuration")
    return {
        "name": "screenplay-writer",
        "description": "AI screenplay writing agent for professional script development",
        "version": "1.0.0",
        "deployment": {
            "url": "http://127.0.0.1:3773",
            "expose": True,
            "protocol_version": "1.0.0",
            "proxy_urls": ["127.0.0.1"],
            "cors_origins": ["*"],
        },
        "environment_variables": [
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key for LLM calls", "required": True},
            {"key": "MEM0_API_KEY", "description": "Mem0 API key for memory operations", "required": False},
        ],
    }


def _handle_character_dialogue(
    line: str,
    in_dialogue: bool,
    current_character: str,
    result_lines: list[str],
) -> tuple[bool, str]:
    """Handle character dialogue and parenthetical formatting."""
    # Check for character name (all caps, short)
    if (
        line.isupper()
        and len(line) < 30
        and "(" not in line
        and ")" not in line
        and not line.startswith(("INT.", "EXT.", "FADE"))
    ):
        # Character name - center it
        centered = " " * 20 + line
        result_lines.append(centered)
        current_character = line
        in_dialogue = True
        return in_dialogue, current_character

    # Check for parenthetical
    if line.startswith("(") and line.endswith(")"):
        parenthetical = " " * 15 + line
        result_lines.append(parenthetical)
        return in_dialogue, current_character

    # If we're in dialogue mode and have a character
    if in_dialogue and current_character:
        # This is dialogue
        dialogue = " " * 10 + line
        result_lines.append(dialogue)
        result_lines.append("")
        in_dialogue = False
        current_character = ""

    return in_dialogue, current_character


def _handle_action_description(line: str, result_lines: list[str]) -> None:
    """Handle action description lines with wrapping."""
    # Split long action lines into shorter ones
    if len(line) > 60:
        words = line.split()
        current_line = []
        current_len = 0

        for word in words:
            if current_len + len(word) + 1 > 60:
                result_lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
            else:
                current_line.append(word)
                current_len += len(word) + 1

        if current_line:
            result_lines.append(" ".join(current_line))
    else:
        result_lines.append(line)

    result_lines.append("")


async def initialize_crew() -> None:
    """Initialize the screenplay writing crew with proper model and agents."""
    global crew

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-4o")

    from crewai import LLM

    try:
        if openrouter_api_key:
            llm = LLM(
                model=model_name,
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.7,
            )
            print(f"✅ Using OpenRouter via CrewAI LLM: {model_name}")

        else:
            error_msg = (
                "No API key provided. Set OPENROUTER_API_KEY environment variable.\n"
                "For OpenRouter: https://openrouter.ai/keys\n"
            )
            raise ValueError(error_msg)  # noqa: TRY301

    except Exception as e:
        print(f"❌ LLM initialization error: {e}")
        print("🔄 Trying alternative configuration...")

        try:
            # SIMPLIFIED: Just use CrewAI LLM directly
            if openrouter_api_key:
                from crewai import LLM as CrewAI_LLM

                llm = CrewAI_LLM(
                    model="gpt-4o",
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.7,
                )
                print("✅ Using OpenRouter via CrewAI LLM (fallback)")
            else:
                raise ValueError(ERROR_NO_API_KEY)  # noqa: TRY301

        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")

            class MockLLM:
                def __call__(self, *args, **kwargs):
                    return "Mock response for testing"

            llm = MockLLM()
            print("⚠️ Using mock LLM for testing only")

    # Define Agent - STRICT FORMATTER
    screenwriter = Agent(
        role="Strict Screenplay Formatter",
        goal="Output ONLY perfectly formatted screenplay text following EXACT industry standards",
        backstory=dedent("""
            You are a robotic screenplay formatter. You follow formatting rules with 100% precision.
            You never write prose or paragraphs. You only write in proper screenplay format.
            Every line follows exact screenplay conventions.
        """),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # Define Task - ULTRA-STRICT FORMATTING
    writing_task = Task(
        description=dedent("""
            Create a screenplay based on: {input}

            FORMATTING RULES - MUST FOLLOW 100%:

            1. ALWAYS start with: FADE IN:
            2. Scene headers: "INT. LOCATION - TIME" or "EXT. LOCATION - TIME" (ALL CAPS)
            3. Action descriptions: Write what we SEE/HEAR, present tense, short lines
            4. Character names: CENTERED, ALL CAPS, on own line
            5. Dialogue: Under character names, indented

            EXAMPLE OF CORRECT OUTPUT:
            FADE IN:

            EXT. CITY STREET - NIGHT

            Rain pours down heavily. Headlights cut through darkness.

                         JAMES
                We can't stop now. They're right behind us.

            James sprints down the alley.

            EXT. DARK ALLEY - NIGHT

            James ducks into a narrow passage.

                         JAMES
                This way!

            Sarah motions to a fire escape.

            FADE OUT.

            IMPORTANT:
            - NO paragraphs or prose
            - NO run-on sentences in action
            - NO dialogue mixed with action
            - NO "We see" or "We hear"
            - Each element on its own line
            - Character names ALWAYS centered
            - Action lines ALWAYS short and visual

            Write ONLY the screenplay in this exact format.
            Return NOTHING else.
        """),
        expected_output="Perfectly formatted screenplay text only.",
        agent=screenwriter,
    )

    # Create crew
    crew = Crew(
        agents=[screenwriter],
        tasks=[writing_task],
        verbose=True,
        process=Process.sequential,
        memory=False,
    )

    print("✅ Screenplay Writing Crew initialized")


def enforce_screenplay_format(text: str) -> str:  # noqa: C901
    """Enforce proper screenplay formatting with zero tolerance for errors."""
    if not text:
        return "FADE IN:\n\nEXT. LOCATION - NIGHT\n\nNo content.\n\nFADE OUT."

    # Start with FADE IN
    result_lines = ["FADE IN:", ""]

    # Remove any markdown
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Split into sentences/paragraphs
    content = text.strip()

    # Extract FADE IN if present
    if "FADE IN" in content.upper():
        content = re.sub(r"FADE IN.*?\n", "", content, flags=re.IGNORECASE)

    # Process each line
    lines = content.split("\n")
    current_scene = None
    in_dialogue = False
    current_character = ""
    last_line_was_dialogue = False

    for line in lines:
        line = line.strip()
        if not line:
            if result_lines[-1] != "":
                result_lines.append("")
            last_line_was_dialogue = False
            continue

        # Check for scene header
        scene_match = re.match(
            r"^(INT\.|EXT\.|INT/EXT\.)\s+(.+?)\s*-\s*(DAY|NIGHT|CONTINUOUS|LATER)",
            line.upper(),
        )
        if scene_match:
            if current_scene:
                result_lines.append("")
            current_scene = line.upper()
            result_lines.append(current_scene)
            result_lines.append("")
            in_dialogue = False
            current_character = ""
            last_line_was_dialogue = False
            continue

        # Check for character name (all caps, short)
        if (
            line.isupper()
            and len(line) < 30
            and "(" not in line
            and ")" not in line
            and not line.startswith(("INT.", "EXT.", "FADE"))
        ):
            # Character name - center it
            centered = " " * 20 + line
            result_lines.append(centered)
            current_character = line
            in_dialogue = True
            last_line_was_dialogue = False
            continue

        # Check for parenthetical
        if line.startswith("(") and line.endswith(")"):
            parenthetical = " " * 15 + line
            result_lines.append(parenthetical)
            last_line_was_dialogue = False
            continue

        # If we're in dialogue mode and have a character
        if in_dialogue and current_character and not last_line_was_dialogue:
            # This is dialogue - remove duplicates
            dialogue = line.strip()
            # Remove duplicate lines (check if same as previous line)
            if result_lines and dialogue in result_lines[-1]:
                continue  # Skip duplicate

            dialogue_formatted = " " * 10 + dialogue
            result_lines.append(dialogue_formatted)
            result_lines.append("")
            in_dialogue = False
            current_character = ""
            last_line_was_dialogue = True
            continue

        # Otherwise it's action description
        # Remove duplicate action lines
        if result_lines and line in result_lines[-1]:
            continue  # Skip duplicate

        # Handle action description
        _handle_action_description(line, result_lines)
        last_line_was_dialogue = False

    # Ensure we have at least one scene
    if not any("INT." in line or "EXT." in line for line in result_lines) and len(result_lines) > 2:
        result_lines.insert(2, "EXT. LOCATION - NIGHT")
        result_lines.insert(3, "")

    # Add FADE OUT (only once)
    if result_lines[-1] != "":
        result_lines.append("")

    # Check if FADE OUT already exists
    if not any("FADE OUT" in line.upper() for line in result_lines):
        result_lines.append("FADE OUT.")

    # Join and clean
    result = "\n".join(result_lines)

    # Remove excessive blank lines
    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)

    return result


async def run_crew(input_text: str) -> str:
    """Run the crew and get the screenplay."""
    global crew

    if not crew:
        raise RuntimeError(ERROR_CREW_NOT_INITIALIZED)

    try:
        print(f"🎬 Running crew with input: {input_text}")

        # Run the crew
        result = await crew.kickoff_async(inputs={"input": input_text})

        # Get the text - CrewAI returns the result directly
        screenplay = str(result)

        print(f"📊 Raw output: {len(screenplay)} chars")

        # Apply STRICT formatting enforcement
        screenplay = enforce_screenplay_format(screenplay)

        print(f"📊 Formatted: {len(screenplay)} chars")

    except Exception as e:
        error_msg = f"Crew execution failed: {e!s}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return "FADE IN:\n\nEXT. ERROR - NIGHT\n\nAn error occurred.\n\nFADE OUT."
    else:
        return screenplay


async def handler(messages: list[dict[str, str]]) -> str:
    """Handle incoming agent messages."""
    global _initialized

    # Type checking for messages
    if not isinstance(messages, list):
        return "FADE IN:\n\nEXT. ERROR - DAY\n\nInvalid input: messages must be a list.\n\nFADE OUT."

    # Lazy initialization
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Screenplay Writing Crew...")
            await initialize_crew()
            _initialized = True

    # Extract user input
    user_input = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_input = msg.get("content", "").strip()
            break

    if not user_input:
        return "FADE IN:\n\nEXT. OFFICE - DAY\n\nPlease provide a story idea.\n\nFADE OUT."

    print(f"✅ Processing: {user_input}")

    try:
        screenplay = await run_crew(user_input)

        if screenplay:
            print("✅ Success! Generated screenplay")
            return screenplay
        else:
            return "FADE IN:\n\nEXT. OFFICE - DAY\n\nNo screenplay generated.\n\nFADE OUT."

    except Exception as e:
        error_msg = f"Handler error: {e!s}"
        print(f"❌ {error_msg}")
        return f"FADE IN:\n\nEXT. ERROR - NIGHT\n\n{error_msg}\n\nFADE OUT."


async def cleanup() -> None:
    """Clean up resources."""
    global crew
    print("🧹 Cleaning up...")
    crew = None
    print("✅ Cleanup complete")


def main() -> None:
    """Run the main entry point for the Screenplay Writing Agent."""
    parser = argparse.ArgumentParser(description="Bindu Screenplay Writing Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "openai/gpt-4o"),
        help="Model ID",
    )
    args = parser.parse_args()

    # Set environment variables
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Screenplay Writer Agent")
    print("📝 Generates perfectly formatted screenplays")

    config = load_config()

    try:
        print("🚀 Starting server...")
        os.environ["CREWAI_TRACING_ENABLED"] = "false"
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        asyncio.run(cleanup())


if __name__ == "__main__":
    main()
