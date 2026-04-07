import os
import requests
from openai import OpenAI

# ---------------- CONFIG ----------------
BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize client only if key exists
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------------- PROMPTS ----------------
SYSTEM_PROMPT = """You are an e-commerce customer support decision agent.

Choose exactly ONE action from:
- approve_refund
- approve_replacement
- reject
- request_info

Output ONLY the action string.
No explanation.
"""

VALID_ACTIONS = {
    "approve_refund",
    "approve_replacement",
    "reject",
    "request_info"
}


def build_prompt(observation: str, stage: str) -> str:
    if stage == "followup":
        extra = "This is additional information after requesting more details. Make the FINAL decision."
    else:
        extra = "This is the initial customer ticket."

    return f"""Customer support ticket:

{observation}

{extra}

Think carefully about fraud signals, missing evidence, and return history.

Respond with ONLY one of:
approve_refund / approve_replacement / reject / request_info
"""


# ---------------- FALLBACK AGENT ----------------
def fallback_action(observation: str) -> str:
    obs = observation.lower()

    # Extract signals
    has_damage = "damaged" in obs or "broken" in obs
    has_images = "images provided: true" in obs
    no_images = "images provided: false" in obs

    # Try extracting return count safely
    return_count = 0
    if "past returns:" in obs:
        try:
            return_count = int(obs.split("past returns:")[1].strip().split()[0])
        except:
            pass

    # -------- Decision logic --------

    # Fraud risk
    if return_count > 4:
        return "reject"

    # Damaged but no proof → ask info
    if has_damage and no_images:
        return "request_info"

    # Damaged with proof → refund
    if has_damage and has_images:
        return "approve_refund"

    # Medium risk cases
    if return_count > 2:
        return "request_info"

    # Default safe action
    return "approve_refund"


# ---------------- LLM / DECISION ----------------
def get_action(observation: str, stage: str = "initial") -> str:

    # 🔥 No API key → fallback
    if not OPENAI_API_KEY:
        return fallback_action(observation)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(observation, stage)}
            ]
        )

        action = response.choices[0].message.content.strip().lower()

        if action not in VALID_ACTIONS:
            return "request_info"

        return action

    except Exception as e:
        print(f"LLM error: {e}")
        return "request_info"


# ---------------- ENV CALLS ----------------
def reset():
    return requests.get(f"{BASE_URL}/reset").json()


def step(action: str):
    return requests.post(
        f"{BASE_URL}/step",
        json={"action_type": action}
    ).json()


# ---------------- EPISODE LOOP ----------------
def run_episode(debug=False):
    state = reset()

    total_reward = 0.0
    done = False
    step_count = 0

    while not done and step_count < 3:  # safety limit
        obs_obj = state.get("observation")

        if isinstance(obs_obj, dict):
            observation = obs_obj.get("ticket", "")
        else:
            observation = ""

        if not observation:
            raise ValueError("No valid observation returned from environment")

        # 🔥 detect stage
        info = state.get("info", {})
        stage = info.get("stage", "initial")

        action = get_action(observation, stage)

        if debug:
            print("\n--- STEP ---")
            print("STAGE:", stage)
            print("OBS:", observation[:200])
            print("ACT:", action)

        state = step(action)

        reward = state.get("reward", 0.0)
        done = state.get("done", False)

        total_reward += reward
        step_count += 1

    return total_reward


# ---------------- EVALUATION ----------------
def evaluate(num_episodes=30, debug=False):
    scores = []

    for i in range(num_episodes):
        score = run_episode(debug=debug)
        scores.append(score)
        print(f"Episode {i+1}: {score:.2f}")

    avg_score = sum(scores) / len(scores)

    print("\n==============================")
    print(f"🔥 Baseline Score: {avg_score:.3f}")
    print("==============================\n")

    return avg_score


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("⚠️ Running in fallback mode (no OpenAI key)")

    evaluate(num_episodes=30, debug=False)