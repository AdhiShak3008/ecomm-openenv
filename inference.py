import os
import requests
from openai import OpenAI

API_BASE = os.getenv("API_BASE_URL")
MODEL = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE,
    api_key=HF_TOKEN
)

ACTIONS = [
    "approve_refund",
    "approve_replacement",
    "reject",
    "request_info"
]

TASK_NAME = "ecommerce-support"
ENV_NAME = "openenv"


def choose_action(observation):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a customer support agent."},
                {"role": "user", "content": observation}
            ],
        )

        text = (response.choices[0].message.content or "").lower()

        for action in ACTIONS:
            if action in text:
                return action

    except Exception:
        pass

    return "request_info"


def run_episode():
    rewards = []
    steps = 0
    success = False

    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL}", flush=True)

    try:
        res = requests.get(f"{API_BASE}/reset").json()
        obs = res["observation"]["ticket"]
        done = False

        while not done and steps < 10:
            steps += 1

            action = choose_action(obs)

            try:
                res = requests.post(
                    f"{API_BASE}/step",
                    json={"action_type": action}
                ).json()

                reward = float(res.get("reward", 0.0))
                obs = res.get("observation", {}).get("ticket", "")
                done = bool(res.get("done", True))
                error = None

            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)

            print(
                f"[STEP] step={steps} action={action} reward={reward:.2f} "
                f"done={str(done).lower()} error={error if error else 'null'}",
                flush=True
            )

        score = requests.get(f"{API_BASE}/grader").json().get("score", 0.0)
        score = max(0.0, min(1.0, float(score)))

        success = score > 0.0

    except Exception as e:
        score = 0.0
        success = False
        print(f"[STEP] step=0 action=null reward=0.00 done=true error={str(e)}", flush=True)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True
    )


if __name__ == "__main__":
    run_episode()
