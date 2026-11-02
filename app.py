import os
import time

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

# The two fixed models
GOV_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPP_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

MAX_ROUNDS = 10  # hard limit required by the assignment
DELAY_SECONDS = 3  # avoid free-tier rate limits
MAX_TOKENS = 2500  # reasoning models spend tokens on thinking before answering

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY, timeout=120, max_retries=0)

GOV_SYSTEM = """You are a senior spokesperson for the GOVERNMENT of {country}, in a live political debate.
Defend the government's policies and record.
Rules:
- Support every argument with valid, verifiable data and statistics (GDP growth, inflation, unemployment, budgets, indices, etc.), naming the source and year (World Bank, IMF, UN, national statistics...).
- Never invent numbers. If unsure of an exact figure, say it is approximate.
- Directly rebut the opposition's last points.
- Stay respectful and professional.
- Keep each reply under 150 words."""

OPP_SYSTEM = """You are a senior spokesperson for the OPPOSITION in {country}, in a live political debate.
Criticize the government's policies and propose better alternatives.
Rules:
- Support every argument with valid, verifiable data and statistics (GDP growth, inflation, unemployment, budgets, indices, etc.), naming the source and year (World Bank, IMF, UN, national statistics...).
- Never invent numbers. If unsure of an exact figure, say it is approximate.
- Directly rebut the government's last points.
- Stay respectful and professional.
- Keep each reply under 150 words."""


def build_messages(system_prompt, transcript, speaker, merge_system):
    messages = []
    for who, text in transcript:
        messages.append({"role": "assistant" if who == speaker else "user", "content": text})
    if merge_system:  # some models reject the system role
        messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def call_model(model, messages):
    """One API call. On 429 (free models are often busy) wait and retry the same model."""
    waits = (0, 10, 20, 30)
    for i, wait in enumerate(waits):
        if wait:
            time.sleep(wait)
        try:
            return client.chat.completions.create(
                model=model, messages=messages, max_tokens=MAX_TOKENS, temperature=0.7
            )
        except Exception as e:
            if "429" in str(e) and i < len(waits) - 1:
                print(f"429 from {model}, retrying in {waits[i + 1]}s ...", flush=True)
                continue
            raise


def ask_llm(model, system_prompt, transcript, speaker):
    """Returns (text, None) on success or (None, error_message) on failure."""
    last_error = "unknown error"
    print(f"[{speaker}] asking {model} ...", flush=True)
    for merge in (False, True):  # second try merges the system prompt into the user message
        try:
            resp = call_model(model, build_messages(system_prompt, transcript, speaker, merge))
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text, None
            last_error = f"{model}: empty response"
        except Exception as e:
            last_error = f"{model}: {e}"
        print(f"[{speaker}] failed: {last_error}", flush=True)
    return None, last_error


def run_debate(country, rounds):
    country = (country or "").strip()
    if not API_KEY:
        raise gr.Error("OPENROUTER_API_KEY is missing. Add it to your .env file.")
    if not country:
        raise gr.Error("Please enter a country.")

    rounds = min(int(rounds), MAX_ROUNDS)

    # The country is specified in the user prompt
    user_prompt = (
        f"The debate topic is the current political and economic situation in {country}. "
        f"Give your opening statement."
    )
    transcript = [("user", user_prompt)]
    chat = []
    yield [{"role": "assistant", "content": f"⏳ Starting a new debate about **{country}** ..."}]

    sides = (
        ("gov", GOV_MODEL, GOV_SYSTEM.format(country=country), "🏛️", "Government"),
        ("opp", OPP_MODEL, OPP_SYSTEM.format(country=country), "⚔️", "Opposition"),
    )

    for r in range(1, rounds + 1):
        for side, model, sys_prompt, icon, title in sides:
            yield chat + [{"role": "assistant", "content": f"⏳ {title} is thinking (round {r}) ..."}]
            text, error = ask_llm(model, sys_prompt, transcript, side)
            if text is None:
                chat.append({"role": "assistant", "content": f"⚠️ Debate stopped. Error: {error}"})
                yield chat
                return
            transcript.append((side, text))
            chat.append({
                "role": "assistant",
                "content": f"{icon} **{title} (Round {r})** · _{model}_\n\n{text}",
            })
            yield chat
            time.sleep(DELAY_SECONDS)


with gr.Blocks(title="Government vs Opposition Debate") as demo:
    gr.Markdown("# 🗣️ Government vs Opposition\nTwo free LLMs debate politics using data and statistics.")
    with gr.Row():
        country_in = gr.Textbox(label="Country", placeholder="e.g. Egypt", scale=3)
        rounds_in = gr.Slider(1, MAX_ROUNDS, value=3, step=1, label="Rounds (max 10)", scale=2)
    start_btn = gr.Button("Start Debate", variant="primary")
    chatbot = gr.Chatbot(label="Debate", height=550, type="messages")
    start_btn.click(lambda: gr.update(interactive=False), None, start_btn, queue=False).then(
        run_debate, inputs=[country_in, rounds_in], outputs=chatbot
    ).then(lambda: gr.update(interactive=True), None, start_btn, queue=False)

if __name__ == "__main__":
    demo.launch()