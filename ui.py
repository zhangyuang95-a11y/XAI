"""
ui.py -- Pac-Man themed Tkinter GUI for the XAI demo.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext

from agent import HeuristicAgent
from environment import GameState, MazeEnvironment, WALL, manhattan_distance
from evidence_recorder import EvidenceRecorder
from explanation_engine import ExplanationEngine
from question_parser import QuestionParser


APP_BG = "#07111d"
PANEL_BG = "#0f1b2d"
CARD_BG = "#14243c"
CARD_ACCENT = "#20395f"
BOARD_BG = "#02060d"
CORRIDOR = "#06111f"
WALL_FILL = "#1d5bff"
WALL_EDGE = "#77a6ff"
DANGER_FILL = "#32101a"
DOT_FILL = "#ffd166"
PLAYER_FILL = "#ffe45e"
EXIT_LOCKED = "#64748b"
EXIT_OPEN = "#34d399"
TEXT_MAIN = "#ecf2ff"
TEXT_MUTED = "#8ea3c0"
TEXT_ACCENT = "#8bc6ff"
STATUS_COLORS = {
    GameState.READY: "#94a3b8",
    GameState.RUNNING: "#60a5fa",
    GameState.PAUSED: "#fbbf24",
    GameState.WON: "#4ade80",
    GameState.LOST: "#fb7185",
}
MONSTER_COLORS = ["#ff5c8a", "#4fd1c5", "#a78bfa", "#f97316", "#38bdf8", "#f43f5e"]
STEP_DELAY_MS = 280


class MazeGameUI:
    def __init__(
        self,
        root: tk.Tk,
        env: MazeEnvironment,
        agent: HeuristicAgent,
        recorder: EvidenceRecorder,
        parser: QuestionParser,
        engine: ExplanationEngine,
    ):
        self.root = root
        self.env = env
        self.agent = agent
        self.recorder = recorder
        self.parser = parser
        self.engine = engine

        self.auto_running = False
        self.after_id: str | None = None
        self.last_action = "RIGHT"
        self.cell_size = max(20, min(34, 760 // self.env.grid_size))

        self.root.title("Pac-Man XAI Demo")
        self.root.configure(bg=APP_BG)
        self.root.geometry("1460x940")
        self.root.minsize(1180, 780)

        self._build_layout()
        self._render()
        self._update_status()
        self._sync_controls()

    def _build_layout(self) -> None:
        self.main = tk.Frame(self.root, bg=APP_BG)
        self.main.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        self.left_panel = tk.Frame(self.main, bg=APP_BG)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 14))

        self.right_panel = tk.Frame(self.main, bg=APP_BG)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title = tk.Label(
            self.left_panel,
            text="Pac-Man Arena",
            bg=APP_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 20),
            anchor="w",
        )
        title.pack(fill=tk.X, pady=(0, 6))

        subtitle = tk.Label(
            self.left_panel,
            text="Collect every dot, open the gate, and ask why the agent moved the way it did.",
            bg=APP_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        )
        subtitle.pack(fill=tk.X, pady=(0, 10))

        board_shell = tk.Frame(self.left_panel, bg=CARD_ACCENT, bd=0, highlightthickness=0)
        board_shell.pack(fill=tk.BOTH, expand=True)

        board_size = self.cell_size * self.env.grid_size
        self.canvas = tk.Canvas(
            board_shell,
            width=board_size,
            height=board_size,
            bg=BOARD_BG,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(padx=2, pady=2)

        self.legend_label = tk.Label(
            self.left_panel,
            text="Yellow = Pac-Man, gold = dots, green = open exit, gray = locked exit, red haze = danger zone.",
            bg=APP_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.legend_label.pack(fill=tk.X, pady=(10, 0))

        self._build_right_panel()

    def _build_right_panel(self) -> None:
        hero = tk.Frame(self.right_panel, bg=PANEL_BG, padx=14, pady=14)
        hero.pack(fill=tk.X, pady=(0, 10))

        hero_title = tk.Label(
            hero,
            text="Decision HUD",
            bg=PANEL_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 18),
            anchor="w",
        )
        hero_title.pack(fill=tk.X)

        hero_subtitle = tk.Label(
            hero,
            text=f"NLP backend: {self.parser.backend}",
            bg=PANEL_BG,
            fg=TEXT_ACCENT,
            font=("Consolas", 10),
            anchor="w",
        )
        hero_subtitle.pack(fill=tk.X, pady=(4, 0))

        controls_outer = self._make_card("Controls")
        controls_outer.pack(fill=tk.X, pady=(0, 10))
        self.controls_card = controls_outer.content
        self._build_controls(self.controls_card)

        status_outer = self._make_card("Live Status")
        status_outer.pack(fill=tk.X, pady=(0, 10))
        self.status_card = status_outer.content
        self._build_status(self.status_card)

        ask_outer = self._make_card("Ask Why")
        ask_outer.pack(fill=tk.X, pady=(0, 10))
        self.ask_card = ask_outer.content
        self._build_question_box(self.ask_card)

        explanation_outer = self._make_card("Explanation Output")
        explanation_outer.pack(fill=tk.BOTH, expand=True)
        self.explanation_card = explanation_outer.content
        self._build_explanation_box(self.explanation_card)

    def _make_card(self, title: str) -> tk.Frame:
        outer = tk.Frame(self.right_panel, bg=CARD_ACCENT, padx=1, pady=1)
        inner = tk.Frame(outer, bg=CARD_BG, padx=12, pady=12)
        inner.pack(fill=tk.BOTH, expand=True)

        label = tk.Label(
            inner,
            text=title,
            bg=CARD_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 13),
            anchor="w",
        )
        label.pack(fill=tk.X, pady=(0, 8))
        outer.content = inner
        return outer

    def _build_controls(self, parent: tk.Frame) -> None:
        row1 = tk.Frame(parent, bg=CARD_BG)
        row1.pack(fill=tk.X, pady=(0, 6))
        row2 = tk.Frame(parent, bg=CARD_BG)
        row2.pack(fill=tk.X)

        self.btn_start = self._make_button(row1, "Start", self._on_start, "#2563eb")
        self.btn_start.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_pause = self._make_button(row1, "Pause", self._on_pause, "#f59e0b")
        self.btn_pause.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_resume = self._make_button(row1, "Resume", self._on_resume, "#22c55e")
        self.btn_resume.pack(side=tk.LEFT)

        self.btn_step = self._make_button(row2, "Step", self._on_step, "#38bdf8")
        self.btn_step.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_reset = self._make_button(row2, "Reset", self._on_reset, "#ef4444")
        self.btn_reset.pack(side=tk.LEFT)

    def _make_button(self, parent: tk.Frame, text: str, command, fill: str) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=fill,
            fg="white",
            activebackground=fill,
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=8,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        )

    def _build_status(self, parent: tk.Frame) -> None:
        self.status_values: dict[str, tk.Label] = {}
        for key, title in [
            ("step", "Step"),
            ("state", "State"),
            ("phase", "Phase"),
            ("dots", "Dots"),
            ("exit", "Exit"),
            ("threat", "Threat"),
            ("action", "Action"),
        ]:
            row = tk.Frame(parent, bg=CARD_BG)
            row.pack(fill=tk.X, pady=2)
            label = tk.Label(
                row,
                text=title,
                width=9,
                anchor="w",
                bg=CARD_BG,
                fg=TEXT_MUTED,
                font=("Segoe UI", 10),
            )
            label.pack(side=tk.LEFT)
            value = tk.Label(
                row,
                text="--",
                anchor="w",
                bg=CARD_BG,
                fg=TEXT_MAIN,
                font=("Consolas", 10),
            )
            value.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.status_values[key] = value

        self.reason_value = tk.Label(
            parent,
            text="Planner trace will appear here after the first move.",
            justify=tk.LEFT,
            anchor="w",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            wraplength=440,
            font=("Segoe UI", 10),
        )
        self.reason_value.pack(fill=tk.X, pady=(8, 0))

    def _build_question_box(self, parent: tk.Frame) -> None:
        self.q_entry = tk.Entry(
            parent,
            bg="#0a1424",
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 11),
        )
        self.q_entry.pack(fill=tk.X, pady=(0, 8), ipady=8)
        self.q_entry.bind("<Return>", lambda _event: self._on_ask())

        self.btn_ask = self._make_button(parent, "Ask", self._on_ask, "#8b5cf6")
        self.btn_ask.pack(anchor="w")

        hint = tk.Label(
            parent,
            text='Try: "Why not go right?", "为什么去吃那个豆子？", "Is it safe here?"',
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify=tk.LEFT,
            anchor="w",
            wraplength=440,
            font=("Segoe UI", 9),
        )
        hint.pack(fill=tk.X, pady=(8, 0))

    def _build_explanation_box(self, parent: tk.Frame) -> None:
        self.exp_text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#08111f",
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            relief=tk.FLAT,
            bd=0,
            height=20,
        )
        self.exp_text.pack(fill=tk.BOTH, expand=True)
        self.exp_text.config(state=tk.DISABLED)

    def _sync_controls(self) -> None:
        finished = self.env.game_state in (GameState.WON, GameState.LOST)
        has_evidence = self.recorder.get_latest() is not None
        ready = self.env.game_state == GameState.READY
        paused = self.env.game_state == GameState.PAUSED

        if self.auto_running:
            self.btn_start.config(state=tk.DISABLED)
            self.btn_pause.config(state=tk.NORMAL)
            self.btn_resume.config(state=tk.DISABLED)
            self.btn_step.config(state=tk.DISABLED)
            self.q_entry.config(state=tk.DISABLED)
            self.btn_ask.config(state=tk.DISABLED)
            return

        self.btn_pause.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.NORMAL)

        if finished:
            self.btn_start.config(state=tk.DISABLED)
            self.btn_resume.config(state=tk.DISABLED)
            self.btn_step.config(state=tk.DISABLED)
            self.q_entry.config(state=tk.NORMAL)
            self.btn_ask.config(state=tk.NORMAL if has_evidence else tk.DISABLED)
            return

        self.btn_step.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL if ready else tk.DISABLED)
        self.btn_resume.config(state=tk.NORMAL if paused and has_evidence else tk.DISABLED)
        self.q_entry.config(state=tk.NORMAL if has_evidence else tk.DISABLED)
        self.btn_ask.config(state=tk.NORMAL if has_evidence else tk.DISABLED)

    def _refresh_canvas_size(self) -> None:
        self.cell_size = max(20, min(34, 760 // self.env.grid_size))
        board_size = self.cell_size * self.env.grid_size
        self.canvas.config(width=board_size, height=board_size)

    def _render(self) -> None:
        self._refresh_canvas_size()
        self.canvas.delete("all")

        grid = self.env.grid
        gs = self.env.grid_size
        dots = self.env.dots
        monsters = self.env.monsters
        danger = self._build_danger_zone()

        self.canvas.create_rectangle(
            0,
            0,
            self.cell_size * gs,
            self.cell_size * gs,
            fill=BOARD_BG,
            outline="",
        )

        for row in range(gs):
            for col in range(gs):
                x0, y0 = col * self.cell_size, row * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size

                if grid[row][col] == WALL:
                    self.canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        y1,
                        fill=WALL_FILL,
                        outline=WALL_EDGE,
                        width=1,
                    )
                else:
                    self.canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        y1,
                        fill=CORRIDOR,
                        outline="#091525",
                        width=1,
                    )
                    if (row, col) in danger:
                        self.canvas.create_rectangle(
                            x0 + 1,
                            y0 + 1,
                            x1 - 1,
                            y1 - 1,
                            fill=DANGER_FILL,
                            outline="",
                            stipple="gray50",
                        )

        self._draw_start_tile()
        self._draw_dots(dots)
        self._draw_exit()
        self._draw_monsters(monsters)
        self._draw_player()
        self._draw_focus_hint()

    def _build_danger_zone(self) -> set[tuple[int, int]]:
        danger: set[tuple[int, int]] = set()
        radius = self.agent.danger_radius
        for monster in self.env.monsters:
            for row in range(max(0, monster.row - radius), min(self.env.grid_size, monster.row + radius + 1)):
                for col in range(max(0, monster.col - radius), min(self.env.grid_size, monster.col + radius + 1)):
                    if manhattan_distance((row, col), (monster.row, monster.col)) <= radius:
                        danger.add((row, col))
        return danger

    def _draw_start_tile(self) -> None:
        row, col = self.env.start
        x0, y0 = col * self.cell_size, row * self.cell_size
        x1, y1 = x0 + self.cell_size, y0 + self.cell_size
        self.canvas.create_rectangle(
            x0 + 4,
            y0 + 4,
            x1 - 4,
            y1 - 4,
            fill="",
            outline="#38bdf8",
            width=2,
        )

    def _draw_dots(self, dots: set[tuple[int, int]]) -> None:
        radius = max(2, self.cell_size // 9)
        for row, col in dots:
            cx = col * self.cell_size + self.cell_size / 2
            cy = row * self.cell_size + self.cell_size / 2
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=DOT_FILL,
                outline="",
            )

    def _draw_exit(self) -> None:
        row, col = self.env.exit
        x0, y0 = col * self.cell_size, row * self.cell_size
        x1, y1 = x0 + self.cell_size, y0 + self.cell_size

        fill = EXIT_OPEN if self.env.exit_open else EXIT_LOCKED
        self.canvas.create_oval(
            x0 + 3,
            y0 + 3,
            x1 - 3,
            y1 - 3,
            outline=fill,
            width=3,
        )
        self.canvas.create_oval(
            x0 + 8,
            y0 + 8,
            x1 - 8,
            y1 - 8,
            outline=fill,
            width=2,
        )
        self.canvas.create_text(
            (x0 + x1) / 2,
            y1 - 6,
            text="OPEN" if self.env.exit_open else "LOCK",
            fill=fill,
            font=("Consolas", max(7, self.cell_size // 4), "bold"),
        )

    def _draw_monsters(self, monsters) -> None:
        radius = self.cell_size * 0.38
        for monster in monsters:
            color = MONSTER_COLORS[monster.id % len(MONSTER_COLORS)]
            cx = monster.col * self.cell_size + self.cell_size / 2
            cy = monster.row * self.cell_size + self.cell_size / 2
            self._draw_ghost(cx, cy, radius, color, monster.id)

    def _draw_ghost(self, cx: float, cy: float, radius: float, color: str, monster_id: int) -> None:
        self.canvas.create_arc(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            start=0,
            extent=180,
            style=tk.CHORD,
            fill=color,
            outline="",
        )
        self.canvas.create_rectangle(
            cx - radius,
            cy,
            cx + radius,
            cy + radius,
            fill=color,
            outline="",
        )
        for offset in (-0.68, -0.22, 0.22, 0.68):
            self.canvas.create_oval(
                cx + radius * (offset - 0.22),
                cy + radius * 0.56,
                cx + radius * (offset + 0.22),
                cy + radius * 1.0,
                fill=BOARD_BG,
                outline="",
            )

        eye_radius = radius * 0.2
        pupil_radius = eye_radius * 0.45
        for dx in (-radius * 0.32, radius * 0.12):
            self.canvas.create_oval(
                cx + dx - eye_radius,
                cy - radius * 0.25 - eye_radius,
                cx + dx + eye_radius,
                cy - radius * 0.25 + eye_radius,
                fill="white",
                outline="",
            )
            self.canvas.create_oval(
                cx + dx - pupil_radius / 2,
                cy - radius * 0.22 - pupil_radius / 2,
                cx + dx + pupil_radius * 1.5,
                cy - radius * 0.22 + pupil_radius * 1.5,
                fill="#111827",
                outline="",
            )

        self.canvas.create_text(
            cx,
            cy + radius * 0.08,
            text=str(monster_id),
            fill="#111827",
            font=("Consolas", max(8, int(radius * 0.6)), "bold"),
        )

    def _draw_player(self) -> None:
        row, col = self.env.player_pos
        cx = col * self.cell_size + self.cell_size / 2
        cy = row * self.cell_size + self.cell_size / 2
        radius = self.cell_size * 0.38

        open_angle = 32 if self.env.step_count % 2 == 0 else 12
        facing = {
            "RIGHT": 0,
            "DOWN": 270,
            "LEFT": 180,
            "UP": 90,
            "STAY": 0,
        }.get(self.last_action, 0)

        self.canvas.create_arc(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            start=facing + open_angle,
            extent=360 - open_angle * 2,
            style=tk.PIESLICE,
            fill=PLAYER_FILL,
            outline="",
        )

        eye_x = cx + (radius * 0.05 if self.last_action in ("RIGHT", "STAY") else -radius * 0.05)
        eye_y = cy - radius * 0.4
        self.canvas.create_oval(
            eye_x - 2,
            eye_y - 2,
            eye_x + 2,
            eye_y + 2,
            fill="#111827",
            outline="",
        )

    def _draw_focus_hint(self) -> None:
        latest = self.recorder.get_latest()
        if latest is None:
            return

        if latest.dots_remaining > 0 and latest.nearest_dot_distance >= 0:
            target_direction = latest.nearest_dot_direction
            target_text = "DOT"
            color = DOT_FILL
        else:
            target_direction = latest.exit_direction
            target_text = "EXIT"
            color = EXIT_OPEN if self.env.exit_open else EXIT_LOCKED

        row, col = self.env.player_pos
        cx = col * self.cell_size + self.cell_size / 2
        cy = row * self.cell_size + self.cell_size / 2
        self.canvas.create_text(
            cx,
            cy - self.cell_size * 0.62,
            text=f"{target_text} {target_direction}",
            fill=color,
            font=("Consolas", max(8, self.cell_size // 4), "bold"),
        )

    def _update_status(self) -> None:
        state = self.env.get_state()
        latest = self.recorder.get_latest()

        phase = "Collect dots" if state["dots"] else "Exit sprint"
        exit_state = "OPEN" if state["exit_open"] else "LOCKED"
        status_text = {
            GameState.READY: "READY",
            GameState.RUNNING: "RUNNING",
            GameState.PAUSED: "PAUSED",
            GameState.WON: "WON",
            GameState.LOST: "LOST",
        }[state["game_state"]]

        self.status_values["step"].config(text=str(state["step_count"]))
        self.status_values["state"].config(
            text=status_text,
            fg=STATUS_COLORS[state["game_state"]],
        )
        self.status_values["phase"].config(text=phase)
        self.status_values["dots"].config(
            text=f"{state['collected_dots']}/{state['total_dots']} cleared | {len(state['dots'])} left"
        )
        self.status_values["exit"].config(
            text=f"{exit_state} | {state['exit_pos']} | dist {manhattan_distance(state['player_pos'], state['exit_pos'])}"
        )

        if latest is None:
            threat_text = "--"
            action_text = "--"
            reasoning = "The planner trace will populate after the first step."
        else:
            chosen_risk = dict(latest.collision_risks).get(latest.chosen_action, 0.0)
            threat_text = (
                f"#{latest.nearest_monster_id} | {latest.nearest_monster_direction} | "
                f"dist {latest.nearest_monster_distance}"
            )
            action_text = f"{latest.chosen_action} | risk {chosen_risk:.0%}"
            reasoning = latest.reasoning or "No extra planner trace available."

        self.status_values["threat"].config(text=threat_text)
        self.status_values["action"].config(text=action_text)
        self.reason_value.config(text=reasoning)

    def _cancel_timer(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _game_step(self) -> None:
        if not self.auto_running:
            return
        if self.env.game_state in (GameState.WON, GameState.LOST):
            self._on_game_over()
            return

        self._execute_one_step()

        if self.env.game_state in (GameState.WON, GameState.LOST):
            self._on_game_over()
            return

        self.after_id = self.root.after(STEP_DELAY_MS, self._game_step)

    def _execute_one_step(self) -> None:
        state = self.env.get_state()
        action = self.agent.choose_action(state)
        self.last_action = action
        next_state = self.env.step(action)
        self.recorder.record(next_state, self.agent, action)
        self._render()
        self._update_status()

    def _on_start(self) -> None:
        self.auto_running = True
        self._sync_controls()
        self._game_step()

    def _on_pause(self) -> None:
        self.auto_running = False
        self._cancel_timer()
        if self.env.game_state not in (GameState.WON, GameState.LOST):
            self.env.game_state = GameState.PAUSED
        self._update_status()
        self._sync_controls()
        if self.recorder.get_latest() is not None:
            self.q_entry.focus_set()

    def _on_resume(self) -> None:
        if self.env.game_state in (GameState.WON, GameState.LOST):
            return
        self.auto_running = True
        self._sync_controls()
        self._game_step()

    def _on_step(self) -> None:
        if self.env.game_state in (GameState.WON, GameState.LOST):
            return
        self._execute_one_step()
        if self.env.game_state in (GameState.WON, GameState.LOST):
            self._on_game_over()
            return
        self.env.game_state = GameState.PAUSED
        self._update_status()
        self._sync_controls()

    def _on_reset(self) -> None:
        self.auto_running = False
        self._cancel_timer()
        self.env.reset()
        self.last_action = "RIGHT"
        self.recorder._history.clear()
        self._clear_explanation()
        self._render()
        self._update_status()
        self._sync_controls()

    def _on_game_over(self) -> None:
        self.auto_running = False
        self._cancel_timer()
        self._update_status()
        self._sync_controls()

        message = (
            f"Pac-Man escaped in {self.env.step_count} steps."
            if self.env.game_state == GameState.WON
            else f"Pac-Man was caught at step {self.env.step_count}."
        )
        self._append_explanation(f"\n{'=' * 56}\n{message}\n{'=' * 56}\n")

    def _on_ask(self) -> None:
        question_text = self.q_entry.get().strip()
        if not question_text:
            return

        latest = self.recorder.get_latest()
        if latest is None:
            self._append_explanation(
                "No evidence yet. Take at least one step before asking a question.\n"
            )
            return

        parsed = self.parser.parse(question_text)
        result = self.engine.generate_explanation(latest, parsed)

        self._clear_explanation()
        self._append_explanation(f"Q: {question_text}\n")
        self._append_explanation(
            f"Intent: {parsed.intent.value} | confidence={parsed.confidence:.3f} | backend={self.parser.backend}\n\n"
        )

        layer1 = result["all_evidence"]
        self._append_explanation(f"{'=' * 56}\n{layer1['label']}\n{'=' * 56}\n")
        for factor in layer1["factors"]:
            marks = "".join(
                [
                    "T" if factor["is_true"] else "-",
                    "F" if factor["is_faithful"] else "-",
                    "C" if factor["is_contrastive"] else "-",
                ]
            )
            self._append_explanation(f"[{marks}] {factor['name']}\n  {factor['description']}\n")

        layer2 = result["evidence_used"]
        self._append_explanation(f"\n{'=' * 56}\n{layer2['label']}\n{'=' * 56}\n")
        for factor in layer2["factors"]:
            self._append_explanation(f"* {factor['name']}\n  {factor['description']}\n")

        layer3 = result["explanation_text"]
        self._append_explanation(f"\n{'=' * 56}\n{layer3['label']}\n{'=' * 56}\n")
        self._append_explanation(f"{layer3['text']}\n")

        self._append_explanation(f"\n{'=' * 56}\nValidation\n{'=' * 56}\n")
        for key, value in result["validation"].items():
            marker = "OK" if value else "FAIL"
            self._append_explanation(f"[{marker:4s}] {key}\n")

        self.q_entry.delete(0, tk.END)

    def _append_explanation(self, text: str) -> None:
        self.exp_text.config(state=tk.NORMAL)
        self.exp_text.insert(tk.END, text)
        self.exp_text.see(tk.END)
        self.exp_text.config(state=tk.DISABLED)

    def _clear_explanation(self) -> None:
        self.exp_text.config(state=tk.NORMAL)
        self.exp_text.delete("1.0", tk.END)
        self.exp_text.config(state=tk.DISABLED)
