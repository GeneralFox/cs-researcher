"""cs-researcher: multi-agent cybersecurity research on local Ollama models.

A coordinator model delegates each question to a team of analysts, each
running a different model on trusted sources only, then compares their
reports and writes a weighted synthesis.
"""

import os
import warnings
from importlib.metadata import version

# CrewAI sends anonymous usage telemetry by default: turned off here, before
# any module of the package imports it, so that nothing leaves the machine
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

__version__ = version("cs-researcher")
