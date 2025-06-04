# GP-Parser
A parser for both grammar AND cohesive calculus.

### Prerequisites

Requires Docker. Trust me, it's just easier this way.

Start the container in the root directory of this repo via
```bash
docker compose up -d
```

Run the python files in the container via
```bash
docker exec -it depccg_parser python3 /workspace/src/depccg_example.py
```