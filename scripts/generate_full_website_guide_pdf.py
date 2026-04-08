# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Website_User_Guide.pdf"


def page_number(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def p(text: str, style: Any) -> Paragraph:
    return Paragraph(text, style)


def code_block(text: str, style: Any) -> Preformatted:
    return Preformatted(text, style)


def bullets(items: list[str], style: Any) -> ListFlowable:
    flow_items = [ListItem(Paragraph(item, style), leftIndent=10) for item in items]
    return ListFlowable(flow_items, bulletType="bullet", leftIndent=16)


def build_pdf():
    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#12263A"),
        spaceAfter=10,
    )

    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#12355B"),
        spaceBefore=10,
        spaceAfter=6,
    )

    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1D5C63"),
        spaceBefore=8,
        spaceAfter=4,
    )

    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#1B1B1B"),
        spaceAfter=6,
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body,
        fontName="Helvetica-Bold",
    )

    code = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=9,
        leading=12,
        backColor=colors.HexColor("#F4F6F8"),
        borderWidth=0.5,
        borderColor=colors.HexColor("#D0D7DE"),
        borderPadding=6,
    )

    story = []

    story.append(p("Linux SRE Website - Complete User Guide", title))
    story.append(p("How to use the full website, including LLM and RL modes", body_bold))
    story.append(p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
    story.append(Spacer(1, 8))

    story.append(p("1) What This Website Does", h1))
    story.append(p(
        "This website is a Linux SRE simulation platform where users solve incident scenarios manually, "
        "with RL agents, or with an LLM assistant. It includes four major experiences:",
        body,
    ))
    story.append(bullets([
        "ChaosHub home page to browse scenarios and launch sessions.",
        "Playground to execute terminal commands step-by-step in a live sandbox.",
        "Arena to compare two agents (script, RL, or LLM) on the same scenario.",
        "Builder to inspect scenario faults, cascades, and objectives visually."
    ], body))

    story.append(p("2) Prerequisites", h1))
    story.append(p("You need backend + frontend running, and environment variables for LLM mode.", body))

    story.append(p("Required LLM environment variables", h2))
    story.append(code_block(
        "$env:API_BASE_URL = \"https://api.openai.com/v1\"\n"
        "$env:MODEL_NAME = \"gpt-4.1-mini\"\n"
        "$env:HF_TOKEN = \"<your-api-key>\"",
        code,
    ))

    story.append(p("Install dependencies", h2))
    story.append(code_block(
        "# Backend\n"
        "cd C:\\Users\\KIIT0001\\Downloads\\meta mod\\meta-harryson\n"
        "c:/Users/KIIT0001/Downloads/meta mod/.venv/Scripts/python.exe -m pip install -r requirements.txt\n\n"
        "# Frontend\n"
        "cd frontend\n"
        "npm install",
        code,
    ))

    story.append(p("Run backend and frontend", h2))
    story.append(code_block(
        "# Terminal 1: Backend (FastAPI)\n"
        "cd C:\\Users\\KIIT0001\\Downloads\\meta mod\\meta-harryson\n"
        "c:/Users/KIIT0001/Downloads/meta mod/.venv/Scripts/python.exe -m uvicorn src.server:app --host 127.0.0.1 --port 8000\n\n"
        "# Terminal 2: Frontend (Next.js)\n"
        "cd C:\\Users\\KIIT0001\\Downloads\\meta mod\\meta-harryson\\frontend\n"
        "npm run dev",
        code,
    ))

    story.append(p("Health checks", h2))
    story.append(code_block(
        "Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing\n"
        "Invoke-WebRequest http://localhost:3000 -UseBasicParsing",
        code,
    ))

    story.append(PageBreak())

    story.append(p("3) Website Navigation", h1))
    story.append(p("Main routes:", body))
    story.append(bullets([
        "Home: http://localhost:3000/",
        "Playground: http://localhost:3000/playground?scenario=log_analysis",
        "Arena: http://localhost:3000/arena",
        "Builder: http://localhost:3000/builder"
    ], body))

    story.append(p("4) Home (ChaosHub) - Start Here", h1))
    story.append(bullets([
        "Browse all scenarios as cards with difficulty and max steps.",
        "Use search and complexity filter to quickly find incidents.",
        "Click a scenario card to open Playground for that specific scenario.",
        "Use the featured scenario for high-difficulty full incident response."
    ], body))

    story.append(p("Recommended demo action", h2))
    story.append(p("From home, open log_analysis first so viewers see a quick success path.", body))

    story.append(p("5) Playground - Manual, RL, and LLM", h1))
    story.append(p("Playground is the core terminal interface. It has three ways to operate:", body))
    story.append(bullets([
        "Manual mode: type shell commands in terminal input and press Enter.",
        "RL Auto-Solve mode: select an RL model and let it execute commands automatically.",
        "LLM Auto-Solve mode: click LLM to let the OpenAI-backed agent solve the scenario."
    ], body))

    story.append(p("Manual mode workflow", h2))
    story.append(bullets([
        "Open playground with scenario query parameter.",
        "Read task description in left panel.",
        "Enter command at prompt and press Enter.",
        "Watch score and step counter update in real time.",
        "Repeat until score reaches completion."
    ], body))

    story.append(p("RL mode workflow", h2))
    story.append(bullets([
        "Click RL in the Auto-Solve section.",
        "Choose model (for example PPO, Q-Learning, Heuristic).",
        "System posts to /api/v1/env/{env_id}/agent_run.",
        "Commands stream live into terminal feed.",
        "Use this to show autonomous policy behavior."
    ], body))

    story.append(p("LLM mode workflow", h2))
    story.append(bullets([
        "Ensure API_BASE_URL, MODEL_NAME, HF_TOKEN are set.",
        "If LLM is configured, LLM button becomes enabled.",
        "Click LLM in Auto-Solve.",
        "Backend boots Autonomous LLM via OpenAI client and streams actions.",
        "You can compare output quality against RL mode."
    ], body))

    story.append(p("AI Assistant chat panel", h2))
    story.append(bullets([
        "Use chat input on right panel to ask: what command should I run next?",
        "Backend endpoint /api/v1/chat/{env_id} returns explanation plus suggested_command.",
        "Click suggested command to paste it into terminal input.",
        "Press Enter to execute instantly.",
        "This is the easiest way to demonstrate interactive LLM assistance."
    ], body))

    story.append(PageBreak())

    story.append(p("6) Arena - Agent vs Agent", h1))
    story.append(p("Arena lets you compare two agents side-by-side on the same scenario.", body))
    story.append(bullets([
        "Choose scenario at top.",
        "For Agent Alpha and Beta choose behavior type: script, LLM, or RL.",
        "For script mode provide one command per line.",
        "For RL mode select model for each side.",
        "Click RUN RACE and observe winner, score rings, and timelines."
    ], body))

    story.append(p("Best demo idea", h2))
    story.append(p("Run RL vs LLM on the same scenario and show final winner and command history differences.", body))

    story.append(p("7) Builder - Scenario Intelligence View", h1))
    story.append(bullets([
        "Select any scenario from dropdown.",
        "Inspect injected faults, cascade rules, and objectives.",
        "Use visual graph to explain why incidents cascade.",
        "Export scenario YAML for documentation/reproducibility.",
        "Launch selected scenario directly into Playground."
    ], body))

    story.append(p("8) Inference Script (Hackathon Baseline)", h1))
    story.append(p("Your root-level inference script is custom and compliant.", body))
    story.append(bullets([
        "File location: inference.py at repository root.",
        "Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN.",
        "Emits [START], [STEP], [END] lines with required fields.",
        "Ensures score is within [0, 1].",
        "Can run single task or all tasks."
    ], body))

    story.append(code_block(
        "# Single task\n"
        "c:/Users/KIIT0001/Downloads/meta mod/.venv/Scripts/python.exe inference.py --task log_analysis\n\n"
        "# All tasks\n"
        "c:/Users/KIIT0001/Downloads/meta mod/.venv/Scripts/python.exe inference.py --all",
        code,
    ))

    story.append(p("9) Pre-Submission Validation", h1))
    story.append(p("Use your custom validator script before submission:", body))
    story.append(code_block(
        "# Run custom validator\n"
        "bash scripts/validate-submission.sh https://your-space.hf.space .",
        code,
    ))

    story.append(bullets([
        "Checks space reset endpoint response.",
        "Checks Docker image build.",
        "Runs openenv validate.",
        "Runs inference and validates output contract format.",
        "Confirms score range and line format assumptions."
    ], body))

    story.append(PageBreak())

    story.append(p("10) Troubleshooting", h1))
    story.append(p("Backend fails with code 1 (most common: port already in use)", h2))
    story.append(code_block(
        "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |\n"
        "  Select-Object LocalAddress,LocalPort,State,OwningProcess\n\n"
        "Stop-Process -Id <PID> -Force\n\n"
        "c:/Users/KIIT0001/Downloads/meta mod/.venv/Scripts/python.exe -m uvicorn src.server:app --host 127.0.0.1 --port 8000",
        code,
    ))

    story.append(p("LLM button disabled", h2))
    story.append(bullets([
        "Set all three vars: API_BASE_URL, MODEL_NAME, HF_TOKEN.",
        "Restart backend after setting variables.",
        "Call /api/v1/models and verify llm.available is true."
    ], body))

    story.append(p("Chat returns errors", h2))
    story.append(bullets([
        "Verify API key and model permissions.",
        "Test /health and /api/v1/models first.",
        "Inspect backend terminal logs for 4xx/5xx from model endpoint."
    ], body))

    story.append(p("11) Quick Demo Script for Presentation", h1))
    story.append(code_block(
        "1. Open home page and choose log_analysis.\n"
        "2. In Playground run one manual command.\n"
        "3. Open AI Assistant, ask for next command, paste suggestion, execute.\n"
        "4. Click LLM auto-solve and show autonomous run.\n"
        "5. Switch to Arena, run RL vs LLM and show winner.\n"
        "6. Open Builder, show faults/cascades/objectives and export YAML.\n"
        "7. End with inference.py and validator commands as proof of submission readiness.",
        code,
    ))

    story.append(p("End of guide", body_bold))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.8 * cm,
        title="Linux SRE Website User Guide",
        author="ChaosLab Team",
    )

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(f"PDF generated: {OUT}")


if __name__ == "__main__":
    build_pdf()
