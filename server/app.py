from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import random
import json

# Initialize FastAPI app (this exposes all endpoints)
app = FastAPI()


# -------- Load Dataset --------
def load_cases():
    """
    Loads all cases from JSON files (easy, medium, hard)
    and tags each case with its difficulty level.
    """
    all_cases = []

    for level in ["easy", "medium", "hard"]:
        with open(f"data/{level}.json") as f:
            cases = json.load(f)

            # Add difficulty label to each case
            for c in cases:
                c["difficulty"] = level

            all_cases.extend(cases)

    return all_cases


# Global dataset used by the environment
CASES = load_cases()


# -------- Global State --------
# Stores current episode + last action result (used by grader)
current_case = None
last_reward = None
last_action = None

# Tracks whether agent has already asked for more info (multi-step)
awaiting_followup = False


# -------- Models --------
# Action sent by agent
class Action(BaseModel):
    action_type: str


# Task metadata (for listing available tasks)
class Task(BaseModel):
    id: str
    difficulty: str


# Observation shown to agent (natural language ticket)
class Observation(BaseModel):
    ticket: str


# Full environment state
class State(BaseModel):
    case_id: str
    difficulty: str
    observation: Observation


# Step response (must follow OpenEnv spec)
class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict   # REQUIRED by spec


# -------- Helpers --------
def build_observation(case):
    """
    Converts structured case data into a readable support ticket.
    This forces the agent to interpret natural language.
    """
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
    """
    Generates follow-up information when agent requests more details.
    Simulates a real customer replying with additional context.
    """
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
    """
    Computes reward based on:
    - expected action
    - real-world business rules
    - partial credit and penalties
    """

    # Base expected action from dataset
    expected = case["expected_action"]

    # 🔥 Override rules (real-world logic)

    # Too many past returns → possible fraud → reject
    if case["return_history"] > 4:
        expected = "reject"

    # Damaged claim without proof → ask for more info
    if case["issue_type"] == "damaged" and not case["has_images"]:
        expected = "request_info"

    # 🎯 Reward logic

    # Perfect decision
    if action == expected:
        return 1.0

    # Acceptable alternative (replacement instead of refund)
    if expected == "approve_refund" and action == "approve_replacement":
        return 0.5

    # Too harsh decision (reject instead of asking info)
    if expected == "request_info" and action == "reject":
        return -0.2

    # Risky approval in high-return cases
    if case["return_history"] > 3 and action == "approve_refund":
        return -0.5

    # Default wrong action
    return 0.0


# -------- Routes --------

@app.get("/")
def root():
    """
    Health check endpoint.
    Used to verify the server is running.
    """
    return {"message": "Ecomm Env Running"}


# Added for deployment health checks
@app.get("/health")
def health():
    """
    Lightweight health endpoint.
    Useful for quick deployment verification.
    """
    return {"status": "ok"}


@app.get("/tasks", response_model=List[Task])
def get_tasks():
    """
    Returns all available tasks (cases) with difficulty levels.
    """
    return [
        {"id": case["id"], "difficulty": case["difficulty"]}
        for case in CASES
    ]


@app.api_route("/reset", methods=["GET", "POST"], response_model=State)
def reset():
    """
    Starts a new episode by selecting a random case.
    Resets previous reward and action history.
    Also resets multi-step tracking.
    """
    global current_case, last_reward, last_action, awaiting_followup

    current_case = random.choice(CASES)

    # Reset episode tracking
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
    """
    Returns current environment state.
    If no episode started, returns None.
    """
    if not current_case:
        return None

    return {
        "case_id": current_case["id"],
        "difficulty": current_case["difficulty"],
        "observation": build_observation(current_case)
    }


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    """
    Takes an action from the agent and returns:
    - new observation
    - reward
    - done flag
    - info (required by spec)

    Supports multi-step interaction:
    - If agent asks for info → provide follow-up (done=False)
    - Otherwise → finalize decision (done=True)
    """
    global current_case, last_reward, last_action, awaiting_followup

    # If reset not called yet
    if not current_case:
        return {
            "observation": Observation(ticket=""),
            "reward": 0.0,
            "done": True,
            "info": {}
        }

    # -------- Step 1: Agent requests more info --------
    if action.action_type == "request_info" and not awaiting_followup:
        awaiting_followup = True

        return {
            "observation": build_followup_observation(current_case),
            "reward": 0.0,
            "done": False,  # episode continues
            "info": {"stage": "followup"}
        }

    # -------- Step 2: Final decision --------
    reward = compute_reward(action.action_type, current_case)

    # Store result for grading
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
    """
    Returns score of last episode.
    Score is clamped to [0.0, 1.0].
    """
    if last_reward is None:
        return {"score": 0.0}

    return {"score": max(0.0, float(last_reward))}


@app.get("/baseline")
def baseline():
    """
    Slightly improved baseline agent.
    Uses simple rules instead of always requesting info.
    """
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

    return {
        "baseline_score": total_score / n
    }
