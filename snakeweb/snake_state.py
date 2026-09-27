"""Prompt builder for the board-reading Laya snake models (snake-trap). Stdlib only.

Builds the exact `state` text and `questions` the model was trained on (gen_snake_dataset.py
--state relative-room --traps), from a snake body (head first) and a food cell, and picks the
move from the answers. Drop it next to snakeweb/game.py; it needs nothing else.

    from snake_state import build_request, pick_move
    state, questions = build_request(list(game.body), game.food, shield=False)
    answers = client.predict(state, questions)["answers"]
    move = pick_move(answers, shield=False)

Coordinates: (x, y), x = column from the left, y = row from the top, both 0-based. Board 24 x 16.
"""

from collections import deque

WIDTH, HEIGHT = 24, 16
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
VECTORS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
HEADINGS = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}

MOVE_QUESTION = {"move": {
    "type": "choice",
    "instructions": "Read the board and choose the snake's next move.",
    "criteria": {
        "UP": "Move the head one cell up.",
        "DOWN": "Move the head one cell down.",
        "LEFT": "Move the head one cell left.",
        "RIGHT": "Move the head one cell right.",
    },
}}
TRAP_QUESTIONS = {f"trap_{d.lower()}": {
    "type": "noul", "criteria": {},
    "instructions": f"If the head moves one cell {d.lower()}, is the snake trapped: dead, or cut off from its own tail?",
} for d in DIRECTIONS}


def _inside(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def advance(body, direction, food, width=WIDTH, height=HEIGHT):
    """Body after one move, or None if the move is fatal (wall, own body; the tail moves away unless eating)."""
    dx, dy = VECTORS[direction]
    target = (body[0][0] + dx, body[0][1] + dy)
    grows = target == food
    occupied = set(body) if grows else set(list(body)[:-1])
    if not _inside(target, width, height) or target in occupied:
        return None
    new = [target] + list(body)
    return new if grows else new[:-1]


def reachable_area(body, width=WIDTH, height=HEIGHT):
    """Cells the head can reach (itself included), treating the tail cell as free."""
    blocked = set(list(body)[:-1])
    head = body[0]
    seen, queue = {head}, deque([head])
    while queue:
        x, y = queue.popleft()
        for dx, dy in VECTORS.values():
            nxt = (x + dx, y + dy)
            if _inside(nxt, width, height) and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen)


def _offset(src, dst):
    (sx, sy), (dx, dy) = src, dst
    v = f"{abs(dy - sy)} rows {'down' if dy > sy else 'up'}" if dy != sy else "same row"
    h = f"{abs(dx - sx)} columns {'right' if dx > sx else 'left'}" if dx != sx else "same column"
    return f"{v}, {h}"


def board_state(body, food, width=WIDTH, height=HEIGHT):
    """The `state` string. Facts relative to the head only, no verdicts. body: head first, len >= 2."""
    body = [tuple(c) for c in body]
    food = tuple(food) if food else None
    (hx, hy), (nx, ny) = body[0], body[1]
    cells = {c: i for i, c in enumerate(body)}
    lines = [f"Head at column {hx + 1}, row {hy + 1} (row 1 is the top). "
             f"Moving {HEADINGS[(hx - nx, hy - ny)]}."]
    rays = []
    for d in DIRECTIONS:
        dx, dy = VECTORS[d]
        x, y, n = hx + dx, hy + dy, 0
        while _inside((x, y), width, height) and (x, y) not in cells:
            x, y, n = x + dx, y + dy, n + 1
        if not _inside((x, y), width, height):
            what = "wall"
        else:
            what = "tail" if cells[(x, y)] == len(body) - 1 else "body"
        ray = f"{d.capitalize()}: {what} 1 away" if n == 0 else f"{d.capitalize()}: open for {n}, then {what}"
        after = advance(body, d, food, width, height)
        if after is not None:
            ray += f"; room {reachable_area(after, width, height)}"
        rays.append(ray + ".")
    lines.append(" ".join(rays))
    lines.append(f"Tail: {_offset(body[0], body[-1])}.")
    lines.append(f"Food: {_offset(body[0], food)}. Length {len(body)}." if food else f"No food. Length {len(body)}.")
    return "\n".join(lines)


def build_request(body, food, shield=False):
    """(state, questions) for POST /v1/systemone. shield=True adds the 4 trap questions (5x the work)."""
    questions = {**MOVE_QUESTION, **TRAP_QUESTIONS} if shield else dict(MOVE_QUESTION)
    return board_state(body, food), questions


def pick_move(answers, shield=False):
    """Most likely move; with shield, among the moves the model does not rate as traps (p < 0.5)."""
    probs = answers["move"]["probabilities"]
    pool = list(DIRECTIONS)
    if shield:
        safe = [d for d in DIRECTIONS if answers[f"trap_{d.lower()}"]["noul"] < 0.5]
        pool = safe or pool
    return max(pool, key=lambda d: probs.get(d, 0.0))
