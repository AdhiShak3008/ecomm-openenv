from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import random
import json

app = FastAPI()


# -------- Load Dataset --------
def load_cases():
    all_cases = []

    for level in ["easy", "medium", "hard"]:
        with open(f"data/{level}.json") as f:
            cases = json.load(f)

            for c in cases:
                c["difficulty"] = level

            all_cases.extend(cases)

    return all_cases


CASES = load_cases()


# -------- Global State --------
current_case = None
current_task = None
last_reward = None
last_action = None
awaiting_followup = False


# -------- Models --------
class Action(BaseModel):
    action_type: str


class Observation(BaseModel):
    ticket: str


class State(BaseModel):
    case_id: str
    difficulty: str
    observation: Observation


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


# -------- Helpers --------
def build_observation(case):
    return Observation(
        ticket=f"""
Customer: {case['customer_message']}
Product: {case['product']}
Days since delivery: {case['days_since_delivery']}
Images provided: {case['has_images']}
Past returns: {case['return_history']}
        """
    )


def build_followup_observation(case):
    if case["issue_type"] == "damaged":
        return Observation(
            ticket="""
Customer follow-up: Uploaded images showing visible damage to the product.
            """
        )

    return Observation(
        ticket="""
Customer follow-up: Provided additional clarification about the issue.
        """
    )


def compute_reward(action, case):
    expected = case["expected_action"]

    if case["return_history"] > 4:
        expected = "reject"

    if case["issue_type"] == "damaged" and not case["has_images"]:
        expected = "request_info"

    if action == expected:
        return 0.9

    if expected == "approve_refund" and action == "approve_replacement":
        return 0.6

    if expected == "request_info" and action == "reject":
        return 0.2

    if case["return_history"] > 3 and action == "approve_refund":
        return 0.1

    return 0.2


# -------- Routes --------

@app.get("/")
def root():
    return {"message": "Ecomm Env Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def get_tasks():
    return [
        {
            "id": "refund_task",
            "difficulty": "easy",
            "grader": "/grader/refund_task"
        },
        {
            "id": "replacement_task",
            "difficulty": "medium",
            "grader": "/grader/replacement_task"
        },
        {
            "id": "fraud_task",
            "difficulty": "hard",
            "grader": "/grader/fraud_task"
        },
    ]


@app.api_route("/reset", methods=["GET", "POST"], response_model=State)
def reset(task_id: Optional[str] = None):
    global current_case, current_task, last_reward, last_action, awaiting_followup

    # Use validator-provided task_id if available
    if task_id in ["refund_task", "replacement_task", "fraud_task"]:
        current_task = task_id
    else:
        current_task = random.choice(["refund_task", "replacement_task", "fraud_task"])

    # Select case pool based on task
    if current_task == "refund_task":
        pool = [c for c in CASES if c["expected_action"] == "approve_refund"]
    elif current_task == "replacement_task":
        pool = [c for c in CASES if c["expected_action"] == "approve_replacement"]
    else:
        pool = [c for c in CASES if c["expected_action"] == "reject"]

    current_case = random.choice(pool) if pool else random.choice(CASES)

    last_reward = None
    last_action = None
    awaiting_followup = False

    return {
        "case_id": current_case["id"],
        "difficulty": current_case["difficulty"],
        "observation": build_observation(current_case)
    }


@app.get("/state", response_model=Optional[State])
def get_state():
    if not current_case:
        return None

    return {
        "case_id": current_case["id"],
        "difficulty": current_case["difficulty"],
        "observation": build_observation(current_case)
    }


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    global current_case, last_reward, last_action, awaiting_followup

    if not current_case:
        return {
            "observation": Observation(ticket=""),
            "reward": 0.1,
            "done": True,
            "info": {}
        }

    if action.action_type == "request_info" and not awaiting_followup:
        awaiting_followup = True

        return {
            "observation": build_followup_observation(current_case),
            "reward": 0.1,
            "done": False,
            "info": {"stage": "followup"}
        }

    reward = compute_reward(action.action_type, current_case)

    last_reward = reward
    last_action = action.action_type

    return {
        "observation": Observation(ticket="Episode finished"),
        "reward": reward,
        "done": True,
        "info": {"stage": "final"}
    }


@app.get("/grader")
def grader():
    if current_task == "refund_task":
        return grader_refund()
    elif current_task == "replacement_task":
        return grader_replacement()
    elif current_task == "fraud_task":
        return grader_fraud()

    return {"score": 0.1}


@app.get("/grader/refund_task")
def grader_refund():
    if last_reward is None:
        return {"score": 0.1}

    score = float(last_reward) * 0.95
    return {"score": max(0.1, min(0.9, score))}


@app.get("/grader/replacement_task")
def grader_replacement():
    if last_reward is None:
        return {"score": 0.1}

    score = float(last_reward) * 0.9
    return {"score": max(0.1, min(0.9, score))}


@app.get("/grader/fraud_task")
def grader_fraud():
    if last_reward is None:
        return {"score": 0.1}

    score = float(last_reward) * 0.85
    return {"score": max(0.1, min(0.9, score))}


@app.get("/baseline")
def baseline():
    total_score = 0
    n = 10

    for _ in range(n):
        case = random.choice(CASES)

        if case["issue_type"] == "damaged" and not case["has_images"]:
            action = "request_info"
        elif case["return_history"] > 4:
            action = "reject"
        else:
            action = case["expected_action"]

        score = compute_reward(action, case)
        total_score += score

    return {"baseline_score": total_score / n}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
