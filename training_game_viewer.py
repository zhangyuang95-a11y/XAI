"""
training_game_viewer.py -- Live Tkinter game window for RL training.
"""

from __future__ import annotations

import tkinter as tk

from environment import WALL, manhattan_distance

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
MONSTER_COLORS = ["#ff5c8a", "#4fd1c5", "#a78bfa", "#f97316", "#38bdf8", "#f43f5e"]


class TrainingGameViewer:
    def __init__(self, grid_size: int, danger_radius: int = 3, enabled: bool = True):
        self.enabled = enabled
        self.danger_radius = danger_radius
        self.closed = False
        self.last_action = "RIGHT"
        self.grid_size = grid_size
        self.cell_size = max(18, min(28, 620 // grid_size))

        self.root: tk.Tk | None = None
        self.canvas: tk.Canvas | None = None
        self.status_labels: dict[str, tk.Label] = {}
        self.q_label: tk.Label | None = None

        if not enabled:
            return

        try:
            self.root = tk.Tk()
            self.root.title("RL Training Arena")
            self.root.configure(bg=APP_BG)
            self.root.geometry("1100x820")
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self._build_layout()
        except Exception as exc:
            print(f"[game ] viewer disabled: {exc}")
            self.enabled = False
            self.root = None

    def _build_layout(self) -> None:
        assert self.root is not None
        main = tk.Frame(self.root, bg=APP_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left = tk.Frame(main, bg=APP_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 12))

        right = tk.Frame(main, bg=APP_BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        title = tk.Label(
            left,
            text="RL Training Arena",
            bg=APP_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 19),
            anchor="w",
        )
        title.pack(fill=tk.X, pady=(0, 6))

        subtitle = tk.Label(
            left,
            text="Live game playback during DQN training.",
            bg=APP_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        )
        subtitle.pack(fill=tk.X, pady=(0, 10))

        board_shell = tk.Frame(left, bg=CARD_ACCENT, padx=1, pady=1)
        board_shell.pack(fill=tk.BOTH, expand=True)
        board_size = self.cell_size * self.grid_size
        self.canvas = tk.Canvas(
            board_shell,
            width=board_size,
            height=board_size,
            bg=BOARD_BG,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()

        legend = tk.Label(
            left,
            text="Yellow = Pac-Man, gold = dots, green = open exit, gray = locked exit, red haze = danger zone.",
            bg=APP_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
        )
        legend.pack(fill=tk.X, pady=(10, 0))

        self._build_status_card(right)

    def _build_status_card(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg=CARD_ACCENT, padx=1, pady=1)
        frame.pack(fill=tk.BOTH, expand=True)
        inner = tk.Frame(frame, bg=CARD_BG, padx=12, pady=12)
        inner.pack(fill=tk.BOTH, expand=True)

        header = tk.Label(
            inner,
            text="Training Monitor",
            bg=CARD_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 16),
            anchor="w",
        )
        header.pack(fill=tk.X, pady=(0, 8))

        for key, title in [
            ("mode", "Mode"),
            ("episode", "Episode"),
            ("step", "Step"),
            ("global_step", "Global Step"),
            ("epsilon", "Epsilon"),
            ("episode_reward", "Episode Reward"),
            ("last_reward", "Last Reward"),
            ("phase", "Phase"),
            ("dots", "Dots"),
            ("exit", "Exit"),
            ("action", "Action"),
            ("status", "Status"),
        ]:
            row = tk.Frame(inner, bg=CARD_BG)
            row.pack(fill=tk.X, pady=2)
            label = tk.Label(
                row,
                text=title,
                width=12,
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
                justify=tk.LEFT,
                bg=CARD_BG,
                fg=TEXT_MAIN,
                font=("Consolas", 10),
                wraplength=340,
            )
            value.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.status_labels[key] = value

        self.q_label = tk.Label(
            inner,
            text="Q-values will appear here after the first rendered step.",
            bg=CARD_BG,
            fg=TEXT_ACCENT,
            justify=tk.LEFT,
            anchor="w",
            wraplength=360,
            font=("Consolas", 10),
        )
        self.q_label.pack(fill=tk.X, pady=(12, 0))

    def update(
        self,
        state: dict,
        *,
        mode: str,
        episode: int,
        total_episodes: int,
        episode_step: int,
        global_step: int,
        epsilon: float,
        episode_reward: float,
        last_reward: float,
        action: str,
        q_values: list[tuple[str, float]] | None = None,
    ) -> None:
        if not self.enabled or self.closed or self.root is None or self.canvas is None:
            return

        self.last_action = action
        self._render_board(state)
        self._update_status(
            state,
            mode=mode,
            episode=episode,
            total_episodes=total_episodes,
            episode_step=episode_step,
            global_step=global_step,
            epsilon=epsilon,
            episode_reward=episode_reward,
            last_reward=last_reward,
            action=action,
            q_values=q_values or [],
        )
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
            self.enabled = False

    def close(self) -> None:
        if self.root is None or self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.destroy()
        except tk.TclError:
            pass
        self.closed = True

    def _on_close(self) -> None:
        self.close()

    def _render_board(self, state: dict) -> None:
        assert self.canvas is not None
        self.canvas.delete("all")
        grid = state["grid"]
        size = state["grid_size"]
        dots = state.get("dots", frozenset())
        monsters = state["monsters"]
        danger = self._build_danger_zone(monsters, size)

        self.canvas.create_rectangle(
            0,
            0,
            self.cell_size * size,
            self.cell_size * size,
            fill=BOARD_BG,
            outline="",
        )

        for row in range(size):
            for col in range(size):
                x0, y0 = col * self.cell_size, row * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size

                if grid[row][col] == WALL:
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1,
                        fill=WALL_FILL,
                        outline=WALL_EDGE,
                        width=1,
                    )
                else:
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1,
                        fill=CORRIDOR,
                        outline="#091525",
                        width=1,
                    )
                    if (row, col) in danger:
                        self.canvas.create_rectangle(
                            x0 + 1, y0 + 1, x1 - 1, y1 - 1,
                            fill=DANGER_FILL,
                            outline="",
                            stipple="gray50",
                        )

        self._draw_start_tile(state["start_pos"])
        self._draw_dots(dots)
        self._draw_exit(state["exit_pos"], state["exit_open"])
        self._draw_monsters(monsters)
        self._draw_player(state["player_pos"])

    def _build_danger_zone(self, monsters: list[tuple[int, int, int]], size: int) -> set[tuple[int, int]]:
        danger: set[tuple[int, int]] = set()
        for _, mr, mc in monsters:
            for row in range(max(0, mr - self.danger_radius), min(size, mr + self.danger_radius + 1)):
                for col in range(max(0, mc - self.danger_radius), min(size, mc + self.danger_radius + 1)):
                    if manhattan_distance((row, col), (mr, mc)) <= self.danger_radius:
                        danger.add((row, col))
        return danger

    def _draw_start_tile(self, start_pos: tuple[int, int]) -> None:
        assert self.canvas is not None
        row, col = start_pos
        x0, y0 = col * self.cell_size, row * self.cell_size
        x1, y1 = x0 + self.cell_size, y0 + self.cell_size
        self.canvas.create_rectangle(
            x0 + 4, y0 + 4, x1 - 4, y1 - 4,
            fill="",
            outline="#38bdf8",
            width=2,
        )

    def _draw_dots(self, dots) -> None:
        assert self.canvas is not None
        radius = max(2, self.cell_size // 9)
        for row, col in dots:
            cx = col * self.cell_size + self.cell_size / 2
            cy = row * self.cell_size + self.cell_size / 2
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill=DOT_FILL,
                outline="",
            )

    def _draw_exit(self, exit_pos: tuple[int, int], exit_open: bool) -> None:
        assert self.canvas is not None
        row, col = exit_pos
        x0, y0 = col * self.cell_size, row * self.cell_size
        x1, y1 = x0 + self.cell_size, y0 + self.cell_size
        fill = EXIT_OPEN if exit_open else EXIT_LOCKED
        self.canvas.create_oval(x0 + 3, y0 + 3, x1 - 3, y1 - 3, outline=fill, width=3)
        self.canvas.create_oval(x0 + 8, y0 + 8, x1 - 8, y1 - 8, outline=fill, width=2)
        self.canvas.create_text(
            (x0 + x1) / 2,
            y1 - 6,
            text="OPEN" if exit_open else "LOCK",
            fill=fill,
            font=("Consolas", max(7, self.cell_size // 4), "bold"),
        )

    def _draw_monsters(self, monsters: list[tuple[int, int, int]]) -> None:
        radius = self.cell_size * 0.38
        for monster_id, row, col in monsters:
            color = MONSTER_COLORS[monster_id % len(MONSTER_COLORS)]
            cx = col * self.cell_size + self.cell_size / 2
            cy = row * self.cell_size + self.cell_size / 2
            self._draw_ghost(cx, cy, radius, color, monster_id)

    def _draw_ghost(self, cx: float, cy: float, radius: float, color: str, monster_id: int) -> None:
        assert self.canvas is not None
        self.canvas.create_arc(
            cx - radius, cy - radius, cx + radius, cy + radius,
            start=0, extent=180, style=tk.CHORD,
            fill=color, outline="",
        )
        self.canvas.create_rectangle(cx - radius, cy, cx + radius, cy + radius, fill=color, outline="")
        for offset in (-0.68, -0.22, 0.22, 0.68):
            self.canvas.create_oval(
                cx + radius * (offset - 0.22),
                cy + radius * 0.56,
                cx + radius * (offset + 0.22),
                cy + radius,
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

    def _draw_player(self, player_pos: tuple[int, int]) -> None:
        assert self.canvas is not None
        row, col = player_pos
        cx = col * self.cell_size + self.cell_size / 2
        cy = row * self.cell_size + self.cell_size / 2
        radius = self.cell_size * 0.38

        open_angle = 32
        facing = {
            "RIGHT": 0,
            "DOWN": 270,
            "LEFT": 180,
            "UP": 90,
            "STAY": 0,
        }.get(self.last_action, 0)

        self.canvas.create_arc(
            cx - radius, cy - radius, cx + radius, cy + radius,
            start=facing + open_angle,
            extent=360 - open_angle * 2,
            style=tk.PIESLICE,
            fill=PLAYER_FILL,
            outline="",
        )
        eye_x = cx + (radius * 0.05 if self.last_action in ("RIGHT", "STAY") else -radius * 0.05)
        eye_y = cy - radius * 0.4
        self.canvas.create_oval(eye_x - 2, eye_y - 2, eye_x + 2, eye_y + 2, fill="#111827", outline="")

    def _update_status(
        self,
        state: dict,
        *,
        mode: str,
        episode: int,
        total_episodes: int,
        episode_step: int,
        global_step: int,
        epsilon: float,
        episode_reward: float,
        last_reward: float,
        action: str,
        q_values: list[tuple[str, float]],
    ) -> None:
        self.status_labels["mode"].config(text=mode.upper())
        self.status_labels["episode"].config(text=f"{episode}/{total_episodes}")
        self.status_labels["step"].config(text=str(episode_step))
        self.status_labels["global_step"].config(text=str(global_step))
        self.status_labels["epsilon"].config(text=f"{epsilon:.3f}")
        self.status_labels["episode_reward"].config(text=f"{episode_reward:.2f}")
        self.status_labels["last_reward"].config(text=f"{last_reward:.2f}")
        self.status_labels["phase"].config(text="Collect dots" if state["dots"] else "Exit sprint")
        self.status_labels["dots"].config(
            text=f"{state['collected_dots']}/{state['total_dots']} cleared | {len(state['dots'])} left"
        )
        exit_state = "OPEN" if state["exit_open"] else "LOCKED"
        self.status_labels["exit"].config(
            text=f"{exit_state} | dist {manhattan_distance(state['player_pos'], state['exit_pos'])}"
        )
        self.status_labels["action"].config(text=action)
        self.status_labels["status"].config(text=state["game_state"].value.upper())

        if self.q_label is not None:
            if q_values:
                top_text = ", ".join(f"{name}={value:.2f}" for name, value in q_values[:4])
                self.q_label.config(text=f"Top Q-values: {top_text}")
            else:
                self.q_label.config(text="Q-values unavailable for this frame.")
