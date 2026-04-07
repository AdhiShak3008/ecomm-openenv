---
title: Ecomm OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---


E-Commerce Customer Support Decision Environment (OpenEnv)

Overview

This project is an OpenEnv environment for testing how an agent handles customer support decisions in an e-commerce setting.

The idea is to simulate cases where the decision isn’t obvious — things like missing proof, suspicious return patterns, or unclear customer messages. Instead of just picking the “correct” label, the agent has to make a reasonable call based on the situation.

---

Problem

Most customer support decisions involve trade-offs rather than clear rules.

Customer support isn’t clean or structured in real life. Some examples:

- product reported as damaged but no images  
- customer has a lot of past returns (possible abuse)  
- request is technically valid but context is messy  
- message itself is vague  

This environment tries to reflect that kind of ambiguity and see how an agent deals with it.

---

Actions

The agent selects one of the following actions:

approve_refund  
approve_replacement  
reject  
request_info  

---

Observations

Each case is turned into a support ticket (plain text). It includes:

- customer message  
- product  
- days since delivery  
- whether images were provided  
- return history  

So the agent has to read and interpret instead of just using structured fields directly.

---

Reward logic

The reward isn’t just strict right/wrong.

- correct decision → 1.0  
- reasonable alternative → 0.5  
- wrong → 0.0  
- bad decisions → negative penalties  

There are also some overrides:

- high return history → lean towards reject  
- damaged claim without images → ask for more info  

Final score from /grader is always between 0 and 1.

---

Tasks

Dataset is split into:

easy → obvious cases  
medium → needs some reasoning  
hard → ambiguous, possible fraud, missing info  

---

API

GET /reset (also supports POST)  
POST /step  
GET /state  
GET /tasks  
GET /grader  
GET /baseline  
GET /health  

---

Running locally (Docker)

docker build -t openenv .  
docker run -p 7860:7860 openenv  

API will be at:

http://localhost:7860

---

Inference

inference.py is the required evaluation script.

It:
- uses OpenAI client via API_BASE_URL, MODEL_NAME, HF_TOKEN  
- interacts with /reset and /step  
- outputs logs in [START], [STEP], [END] format  

---

Baseline

baseline.py uses OpenAI to choose actions from the ticket text.

Run it like this:

export OPENAI_API_KEY=your_key  
export ENV_URL=http://localhost:7860  

python baseline.py  

Expected score is roughly:

Average score: 0.45 (20 episodes, fallback agent)  
Note: OpenAI-based baseline expected to perform higher  

---

Structure

data/  
  easy.json  
  medium.json  
  hard.json  

env/  
  app.py  

baseline.py  
inference.py  
openenv.yaml  
Dockerfile  
README.md  

---

Possible extensions

- multi-step conversations  
- stricter policy constraints  
- larger dataset  

---

Notes

This environment is designed to stay simple while still capturing realistic decision-making scenarios. The goal is not perfect accuracy, but reasonable behavior under uncertainty.
