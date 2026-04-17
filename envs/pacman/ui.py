"""
ui.py -- Pac-Man themed Tkinter GUI for the XAI demo.
"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext

from .agent import RLAgent
from .environment import GameState, MazeEnvironment, WALL, manhattan_distance
from .evidence_recorder import EvidenceRecorder
from .explanation_engine import (
    ExplanationEngine,
    SYMBOLIC_MATCH_VALIDATION_KEY,
    SYMBOLIC_SUPPORT_VALIDATION_KEY,
)
from .question_parser import ParsedQuestion, QuestionIntent, QuestionParser


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

ANSWER_LANGUAGE_OPTIONS = {
    "Auto / 自动": "auto",
    "中文": "zh",
    "English": "en",
    "中英双语": "both",
}

INTENT_LABELS = {
    QuestionIntent.WHY_THIS_ACTION: "Why this action / 为什么这样走",
    QuestionIntent.WHY_NOT_OTHER: "Why not another action / 为什么不选别的动作",
    QuestionIntent.MONSTER_INFLUENCE: "Monster influence / 怪物影响",
    QuestionIntent.PATH_REASON: "Path reason / 路径原因",
    QuestionIntent.SAFETY_REASON: "Safety / 安全性",
    QuestionIntent.GOAL_REASON: "Goal progress / 目标进度",
    QuestionIntent.DOT_COLLECTION: "Dot collection / 吃豆策略",
    QuestionIntent.GENERAL: "General / 总结",
    QuestionIntent.IRRELEVANT: "Irrelevant / 无关问题",
}

VALIDATION_HELP = {
    "E ⊆ S_t": "Used evidence comes from the full evidence set. / 实际使用证据来自当前总证据集合。",
    "True_t(E)": "The selected evidence is true at this step. / 选中的证据在当前时刻为真。",
    "Faithful_π(E, a_t)": "The evidence faithfully supports the chosen action. / 这些证据确实支持当前动作。",
    "Contrastive_π(E, a_t, Δ_t)": "The evidence distinguishes this action from alternatives. / 这些证据能区分当前动作和替代动作。",
    "Basis_{u,t}(E, Q)": "The evidence set satisfies the basis conditions. / 证据集合满足形式化基础条件。",
    "Minimal(E)": "No smaller subset still satisfies the basis. / 再删掉证据就不满足定义。",
    "x = R_u(E, Q)": "The answer matches the rendering function. / 最终回答等于渲染函数输出。",
    "Readable_u(x)": "The answer is readable for the user. / 回答对用户是可读的。",
    "Explain_u(Q, t, x)": "The final explanation definition holds. / 最终 explanation 定义成立。",
    SYMBOLIC_SUPPORT_VALIDATION_KEY: "The symbolic surrogate supports the chosen action. / 符号代理支持当前动作。",
    SYMBOLIC_MATCH_VALIDATION_KEY: "The symbolic surrogate matches the neural policy at this step. / 符号代理在这一帧与神经策略一致。",
}

GUIDE_TEXT = (
    "Workflow / 使用流程: train the RL model first, then run auto mode and ask questions during the live game.\n"
    "Ask / 提问: type your own Chinese or English question. Asking will pause the game automatically.\n"
    "Answer Language / 回答语言: choose 自动、中文、English or 中英双语.\n"
    "Step / 步数: current decision step.\n"
    "Phase / 阶段: collect dots first, then rush the exit.\n"
    "Threat / 威胁: nearest monster and its distance.\n"
    "Action / 动作: chosen move and immediate collision risk.\n"
    "Confidence / 置信度: parser confidence from 0 to 1.\n"
    "Validation / 验证: checks whether the answer really comes from current evidence and satisfies the formal explanation definition.\n"
    "Question Log / 提问日志: every real user question is saved with its answer and validation result.\n"
    "S_t: all evidence at time t.  E: evidence actually used.  x: final natural-language answer.\n"
    "T / F / C: True, Faithful, Contrastive. / 真、忠实、可对比。"
)


INTENT_LABELS[QuestionIntent.POLICY_SUMMARY] = "Policy summary / 整体策略"


class MazeGameUI:
    def __init__(
        self,
        root: tk.Tk,
        env: MazeEnvironment,
        agent: RLAgent,
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
        self.answer_language_label_var = tk.StringVar(value="Auto / 自动")
        self.cell_size = max(20, min(34, 760 // self.env.grid_size))
        self.question_log_path = Path("artifacts/user_question_log.jsonl")
        self.question_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.show_technical_details = False

        self.root.title("Pac-Man XAI Demo")
        self.root.configure(bg=APP_BG)
        self.root.geometry("1880x940")
        self.root.minsize(1450, 780)

        self._build_layout()
        self._render()
        self._update_status()
        self._sync_controls()

    def _build_layout(self) -> None:
        self.main = tk.Frame(self.root, bg=APP_BG)
        self.main.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        self.left_panel = tk.Frame(self.main, bg=APP_BG)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 14))

        self.diagnostics_shell = tk.Frame(self.main, bg=APP_BG, width=430)
        self.diagnostics_shell.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        self.diagnostics_shell.pack_propagate(False)

        self.right_shell = tk.Frame(self.main, bg=APP_BG)
        self.right_shell.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 14))

        self.right_canvas = tk.Canvas(
            self.right_shell,
            bg=APP_BG,
            highlightthickness=0,
            bd=0,
        )
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_scrollbar = tk.Scrollbar(
            self.right_shell,
            orient=tk.VERTICAL,
            command=self.right_canvas.yview,
        )
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        self.right_panel = tk.Frame(self.right_canvas, bg=APP_BG)
        self._right_window_id = self.right_canvas.create_window(
            (0, 0),
            window=self.right_panel,
            anchor="nw",
        )
        self.right_panel.bind("<Configure>", self._on_right_panel_configure)
        self.right_canvas.bind("<Configure>", self._on_right_canvas_configure)
        self.right_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        title = tk.Label(
            self.left_panel,
            text="Pac-Man Arena / 吃豆人解释界面",
            bg=APP_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 20),
            anchor="w",
        )
        title.pack(fill=tk.X, pady=(0, 6))

        subtitle = tk.Label(
            self.left_panel,
            text="Run the fixed RL demo, then type your own question whenever you want an explanation.",
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
            text="Yellow=Pac-Man | Gold=dots | Green=open exit | Gray=locked exit | Red haze=danger zone",
            bg=APP_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.legend_label.pack(fill=tk.X, pady=(10, 0))

        self._build_right_panel()
        self._build_diagnostics_panel()

    def _build_right_panel(self) -> None:
        hero = tk.Frame(self.right_panel, bg=PANEL_BG, padx=14, pady=14)
        hero.pack(fill=tk.X, pady=(0, 10))

        hero_title = tk.Label(
            hero,
            text="Ask, Answer, Validate / 提问、回答、验证",
            bg=PANEL_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 18),
            anchor="w",
        )
        hero_title.pack(fill=tk.X)

        hero_subtitle = tk.Label(
            hero,
            text=f"NLP backend: {self.parser.backend} | This demo always uses the fixed trained RL model.",
            bg=PANEL_BG,
            fg=TEXT_ACCENT,
            font=("Consolas", 10),
            anchor="w",
        )
        hero_subtitle.pack(fill=tk.X, pady=(4, 0))

        controls_outer = self._make_card("Controls / 控制")
        controls_outer.pack(fill=tk.X, pady=(0, 10))
        self.controls_card = controls_outer.content
        self._build_controls(self.controls_card)

        status_outer = self._make_card("Live Status / 当前状态")
        status_outer.pack(fill=tk.X, pady=(0, 10))
        self.status_card = status_outer.content
        self._build_status(self.status_card)

        ask_outer = self._make_card("Ask A Question / 自由提问")
        ask_outer.pack(fill=tk.X, pady=(0, 10))
        self.ask_card = ask_outer.content
        self._build_question_box(self.ask_card)

        guide_outer = self._make_card("Guide / 参数说明")
        guide_outer.pack(fill=tk.X, pady=(0, 10))
        self.guide_card = guide_outer.content
        self._build_guide(self.guide_card)

        explanation_outer = self._make_card("Answer / 回答")
        explanation_outer.pack(fill=tk.BOTH, expand=True)
        self.explanation_card = explanation_outer.content
        self._build_explanation_box(self.explanation_card)

    def _build_diagnostics_panel(self) -> None:
        outer = tk.Frame(self.diagnostics_shell, bg=CARD_ACCENT, padx=1, pady=1)
        outer.pack(fill=tk.BOTH, expand=True)

        inner = tk.Frame(outer, bg=CARD_BG, padx=12, pady=12)
        inner.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            inner,
            text="Evidence & Metrics / Evidence and Metrics",
            bg=CARD_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 13),
            anchor="w",
        )
        title.pack(fill=tk.X, pady=(0, 4))

        subtitle = tk.Label(
            inner,
            text="All evidence, selected evidence, parser grounding, validation, and symbolic diagnostics.",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            anchor="w",
            wraplength=390,
        )
        subtitle.pack(fill=tk.X, pady=(0, 8))

        self.metrics_text = scrolledtext.ScrolledText(
            inner,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#06101d",
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            relief=tk.FLAT,
            bd=0,
            height=20,
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.insert(
            tk.END,
            "Ask a question after at least one game step to inspect evidence and metrics here.\n",
        )
        self.metrics_text.config(state=tk.DISABLED)

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

        self.btn_start = self._make_button(row1, "Start / 开始", self._on_start, "#2563eb")
        self.btn_start.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_pause = self._make_button(row1, "Pause / 暂停", self._on_pause, "#f59e0b")
        self.btn_pause.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_resume = self._make_button(row1, "Resume / 继续", self._on_resume, "#22c55e")
        self.btn_resume.pack(side=tk.LEFT)

        self.btn_step = self._make_button(row2, "Step / 单步", self._on_step, "#38bdf8")
        self.btn_step.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_reset = self._make_button(row2, "Reset / 重开", self._on_reset, "#ef4444")
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

    def _on_right_panel_configure(self, _event=None) -> None:
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def _on_right_canvas_configure(self, event) -> None:
        self.right_canvas.itemconfigure(self._right_window_id, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if not self.right_canvas.winfo_exists():
            return
        target = self.root.winfo_containing(event.x_root, event.y_root)
        if target is None:
            return
        current = target
        inside_right_panel = False
        while current is not None:
            if current == self.right_shell:
                inside_right_panel = True
                break
            current = current.master
        if inside_right_panel:
            self.right_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_guide(self, parent: tk.Frame) -> None:
        guide = tk.Label(
            parent,
            text=GUIDE_TEXT,
            justify=tk.LEFT,
            anchor="w",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            wraplength=460,
            font=("Segoe UI", 9),
        )
        guide.pack(fill=tk.X)

    def _build_status(self, parent: tk.Frame) -> None:
        self.status_values: dict[str, tk.Label] = {}
        for key, title in [
            ("step", "Step / 步数"),
            ("state", "State / 状态"),
            ("phase", "Phase / 阶段"),
            ("dots", "Dots / 豆子"),
            ("exit", "Exit / 出口"),
            ("threat", "Threat / 威胁"),
            ("action", "Action / 动作"),
        ]:
            row = tk.Frame(parent, bg=CARD_BG)
            row.pack(fill=tk.X, pady=2)
            label = tk.Label(
                row,
                text=title,
                width=14,
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
            text="Decision note / 决策说明 will appear here after the first move.",
            justify=tk.LEFT,
            anchor="w",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            wraplength=440,
            font=("Segoe UI", 10),
        )
        self.reason_value.pack(fill=tk.X, pady=(8, 0))

    def _build_question_box(self, parent: tk.Frame) -> None:
        tip = tk.Label(
            parent,
            text="Type your own question below. If the game is running, asking will pause it automatically.\n直接输入你的问题即可；如果游戏正在运行，提问时会自动暂停。",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            justify=tk.LEFT,
            anchor="w",
            wraplength=440,
            font=("Segoe UI", 9),
        )
        tip.pack(fill=tk.X, pady=(0, 8))

        input_label = tk.Label(
            parent,
            text="Question Input / 提问输入",
            bg=CARD_BG,
            fg=TEXT_MAIN,
            font=("Segoe UI Semibold", 10),
            anchor="w",
        )
        input_label.pack(fill=tk.X, pady=(2, 6))

        lang_row = tk.Frame(parent, bg=CARD_BG)
        lang_row.pack(fill=tk.X, pady=(0, 8))

        lang_label = tk.Label(
            lang_row,
            text="Answer Language / 回答语言",
            bg=CARD_BG,
            fg=TEXT_MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        )
        lang_label.pack(side=tk.LEFT)

        self.answer_language_menu = tk.OptionMenu(
            lang_row,
            self.answer_language_label_var,
            *ANSWER_LANGUAGE_OPTIONS.keys(),
        )
        self.answer_language_menu.config(
            bg="#0a1424",
            fg=TEXT_MAIN,
            activebackground="#162235",
            activeforeground=TEXT_MAIN,
            relief=tk.FLAT,
            highlightthickness=0,
            bd=0,
            font=("Segoe UI", 10),
        )
        self.answer_language_menu["menu"].config(
            bg="#0a1424",
            fg=TEXT_MAIN,
            activebackground="#162235",
            activeforeground=TEXT_MAIN,
            font=("Segoe UI", 10),
        )
        self.answer_language_menu.pack(side=tk.RIGHT)
        self.answer_language_menu.pack_forget()

        input_shell = tk.Frame(parent, bg=TEXT_ACCENT, padx=1, pady=1)
        input_shell.pack(fill=tk.X, pady=(0, 8))

        self.q_entry = tk.Text(
            input_shell,
            bg="#0a1424",
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 11),
            height=3,
            wrap=tk.WORD,
        )
        self.q_entry.pack(fill=tk.X, ipady=4)
        self.q_entry.bind("<Control-Return>", lambda _event: self._on_ask())

        self.btn_ask = self._make_button(parent, "Ask / 提问", self._on_ask, "#8b5cf6")
        self.btn_ask.pack(anchor="w")

        for row_items in [
            ["Why not go right?", "为什么去吃那个豆子？"],
            ["Is it safe here?", "怪物#2影响了这次决策吗？"],
        ]:
            examples_row = tk.Frame(parent, bg=CARD_BG)
            examples_row.pack(fill=tk.X, pady=(8, 0))
            examples_row.pack_forget()
            for text in row_items:
                button = tk.Button(
                    examples_row,
                    text=text,
                    command=lambda value=text: self._use_example_question(value),
                    bg="#0a1424",
                    fg=TEXT_ACCENT,
                    activebackground="#162235",
                    activeforeground=TEXT_MAIN,
                    relief=tk.FLAT,
                    bd=0,
                    padx=8,
                    pady=6,
                    font=("Segoe UI", 9),
                    cursor="hand2",
                )
                button.pack(side=tk.LEFT, padx=(0, 6))

        hint = tk.Label(
            parent,
            text="Examples above fill the input box. Press Ctrl+Enter or click Ask. / 点击示例会填入输入框；按 Ctrl+Enter 或点击 Ask 提问。",
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
            self.q_entry.config(state=tk.NORMAL)
            self.btn_ask.config(state=tk.NORMAL)
            return

        self.btn_pause.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.NORMAL)

        if finished:
            self.btn_start.config(state=tk.DISABLED)
            self.btn_resume.config(state=tk.DISABLED)
            self.btn_step.config(state=tk.DISABLED)
            self.q_entry.config(state=tk.NORMAL)
            self.btn_ask.config(state=tk.NORMAL)
            return

        self.btn_step.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL if ready else tk.DISABLED)
        self.btn_resume.config(state=tk.NORMAL if paused and has_evidence else tk.DISABLED)
        self.q_entry.config(state=tk.NORMAL)
        self.btn_ask.config(state=tk.NORMAL)

    def _use_example_question(self, text: str) -> None:
        self.q_entry.config(state=tk.NORMAL)
        self.q_entry.delete("1.0", tk.END)
        self.q_entry.insert("1.0", text)
        self.q_entry.focus_set()

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

        phase = "Collect dots / 吃豆阶段" if state["dots"] else "Exit sprint / 冲向出口"
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
            reasoning = "Decision note / 决策说明 will appear after the first step."
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
        self.recorder.record(state, self.agent, action)
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
        self.recorder.clear()
        self._clear_explanation()
        self._clear_diagnostics()
        self._render()
        self._update_status()
        self._sync_controls()

    def _on_game_over(self) -> None:
        self.auto_running = False
        self._cancel_timer()
        self._update_status()
        self._sync_controls()

        message = (
            f"Pac-Man escaped in {self.env.step_count} steps. / Pac-Man 在 {self.env.step_count} 步内逃脱。"
            if self.env.game_state == GameState.WON
            else f"Pac-Man was caught at step {self.env.step_count}. / Pac-Man 在第 {self.env.step_count} 步被抓住。"
        )
        self._append_explanation(f"\n{'=' * 56}\n{message}\n{'=' * 56}\n")

    def _on_ask(self) -> None:
        if self.auto_running:
            self._on_pause()

        question_text = self.q_entry.get("1.0", tk.END).strip()
        if not question_text:
            return

        latest = self.recorder.get_latest()
        if latest is None:
            self._clear_diagnostics()
            self._append_diagnostics("No evidence yet. Take at least one step before asking a question.\n")
            self._append_explanation(
                "No evidence yet. Take at least one step before asking a question.\n还没有可用证据，请先至少走一步再提问。\n"
            )
            return

        parsed = self.parser.parse(question_text)
        mode = "manual_input"
        primary_result, secondary_result = self._generate_answer_results(latest, parsed, mode)
        self._render_answer_output(question_text, parsed, primary_result, secondary_result)
        self._log_question_event(latest, question_text, parsed, mode, primary_result, secondary_result)

        self.q_entry.delete("1.0", tk.END)

    def _generate_answer_results(
        self,
        evidence,
        parsed: ParsedQuestion,
        mode: str,
    ) -> tuple[dict, dict | None]:
        if mode in {"auto", "manual_input"}:
            return self.engine.generate_explanation(evidence, parsed), None
        if mode == "zh":
            return self.engine.generate_explanation(evidence, replace(parsed, language="zh")), None
        if mode == "en":
            return self.engine.generate_explanation(evidence, replace(parsed, language="en")), None

        zh_result = self.engine.generate_explanation(evidence, replace(parsed, language="zh"))
        en_result = self.engine.generate_explanation(evidence, replace(parsed, language="en"))
        return zh_result, en_result

    def _render_answer_output(
        self,
        question_text: str,
        parsed: ParsedQuestion,
        primary_result: dict,
        secondary_result: dict | None,
    ) -> None:
        self._clear_explanation()
        self._clear_diagnostics()
        self._append_explanation("Question / 提问\n")
        self._append_explanation(f"{question_text}\n\n")
        self._append_explanation(
            f"Intent / 类型: {INTENT_LABELS.get(parsed.intent, parsed.intent.value)}\n"
            f"Confidence / 置信度: {parsed.confidence:.3f}\n"
            f"Backend / 解析后端: {self.parser.backend}\n"
            f"Question Log / 提问日志: {self.question_log_path}\n"
        )

        self._append_explanation(f"\nShort Answer / 简明回答\n{'-' * 56}\n")
        self._append_explanation(
            f"{self._answer_language_title(primary_result['language'])}\n{primary_result['explanation_text']['text']}\n"
        )
        if secondary_result is not None:
            self._append_explanation(
                f"\n{self._answer_language_title(secondary_result['language'])}\n"
                f"{secondary_result['explanation_text']['text']}\n"
            )

        self._render_diagnostics_output(question_text, parsed, primary_result, secondary_result)

        if not self.show_technical_details:
            return

        symbolic_rule = primary_result.get("symbolic_rule", {})
        symbolic_trace = primary_result.get("symbolic_trace", {})
        policy_summary = primary_result.get("policy_summary", {})

        self._append_explanation(f"\nSymbolic Rule / 符号规则\n{'-' * 56}\n")
        self._append_explanation(f"symbolic_match = {primary_result.get('symbolic_match')}\n")
        if symbolic_rule.get("fallback_used"):
            self._append_explanation("fallback_used = True\n")
        if symbolic_rule.get("text"):
            self._append_explanation(f"{symbolic_rule['text']}\n")
        if symbolic_rule.get("python"):
            self._append_explanation("\n```python\n")
            self._append_explanation(f"{symbolic_rule['python']}\n")
            self._append_explanation("```\n")

        self._append_explanation(f"\nDecision Trace / 决策轨迹\n{'-' * 56}\n")
        for item in symbolic_trace.get("trace", []):
            self._append_explanation(
                f"- {item['condition']} | {item.get('description', '')} | sources={', '.join(item.get('source', []))}\n"
            )
        if symbolic_trace.get("approximate_trace"):
            self._append_explanation("\nApproximate Symbolic Trace / 近似符号轨迹\n")
            for item in symbolic_trace.get("approximate_trace", []):
                self._append_explanation(
                    f"- {item['condition']} | {item.get('description', '')} | sources={', '.join(item.get('source', []))}\n"
                )

        self._append_explanation(f"\nHuman-Readable Explanation / 人类可读解释\n{'-' * 56}\n")
        self._append_explanation(
            f"{self._answer_language_title(primary_result['language'])}\n{primary_result['explanation_text']['text']}\n"
        )
        if secondary_result is not None:
            self._append_explanation(
                f"\n{self._answer_language_title(secondary_result['language'])}\n"
                f"{secondary_result['explanation_text']['text']}\n"
            )

        self._append_explanation(f"\nGlobal Policy Summary / 全局策略摘要\n{'-' * 56}\n")
        for bullet in policy_summary.get("bullets", []):
            self._append_explanation(f"- {bullet}\n")
        if policy_summary.get("python_snippet"):
            self._append_explanation("\n```python\n")
            self._append_explanation(f"{policy_summary['python_snippet']}\n")
            self._append_explanation("```\n")

        layer2 = primary_result["evidence_used"]
        self._append_explanation(f"\nEvidence Used / 实际使用证据\n{'-' * 56}\n")
        for factor in layer2["factors"]:
            self._append_explanation(f"- {factor['name']}: {factor['description']}")
            if factor.get("sources"):
                self._append_explanation(f" [sources: {', '.join(factor['sources'])}]")
            self._append_explanation("\n")

        self._append_explanation(f"\nValidation / 形式化验证\n{'-' * 56}\n")
        for key, value in primary_result["validation"].items():
            marker = "OK" if value else "FAIL"
            help_text = VALIDATION_HELP.get(key, "")
            self._append_explanation(f"[{marker:4s}] {key}\n")
            if help_text:
                self._append_explanation(f"      {help_text}\n")

        layer1 = primary_result["all_evidence"]
        self._append_explanation(f"\nAll Evidence / 全部证据 (T=True, F=Faithful, C=Contrastive)\n{'-' * 56}\n")
        for factor in layer1["factors"]:
            marks = "".join(
                [
                    "T" if factor["is_true"] else "-",
                    "F" if factor["is_faithful"] else "-",
                    "C" if factor["is_contrastive"] else "-",
                ]
            )
            self._append_explanation(f"[{marks}] {factor['name']}: {factor['description']}")
            if factor.get("sources"):
                self._append_explanation(f" [sources: {', '.join(factor['sources'])}]")
            self._append_explanation("\n")

    def _render_diagnostics_output(
        self,
        question_text: str,
        parsed: ParsedQuestion,
        primary_result: dict,
        secondary_result: dict | None,
    ) -> None:
        latest = self.recorder.get_latest()
        validation = primary_result.get("validation", {})
        symbolic_rule = primary_result.get("symbolic_rule", {})
        symbolic_trace = primary_result.get("symbolic_trace", {})
        policy_summary = primary_result.get("policy_summary", {})
        metrics = primary_result.get("distillation_metrics", {}) or {}
        all_factors = primary_result.get("all_evidence", {}).get("factors", [])
        used_factors = primary_result.get("evidence_used", {}).get("factors", [])
        semantic_frame = getattr(parsed, "semantic_frame", {}) or {}

        self._append_diagnostics("QUESTION PARSE\n")
        self._append_diagnostics("-" * 54 + "\n")
        self._append_diagnostics(f"question: {question_text}\n")
        self._append_diagnostics(f"intent: {parsed.intent.value}\n")
        self._append_diagnostics(f"mentioned_action: {parsed.mentioned_action or 'none'}\n")
        self._append_diagnostics(f"language: {parsed.language}\n")
        self._append_diagnostics(f"confidence: {parsed.confidence:.3f}\n")
        self._append_diagnostics(f"backend: {self.parser.backend}\n")
        self._append_diagnostics(f"grounded: {getattr(parsed, 'grounded', True)}\n")
        reason = getattr(parsed, "relevance_reason", "")
        if reason:
            self._append_diagnostics(f"relevance_reason: {reason}\n")
        if semantic_frame:
            self._append_diagnostics("semantic_frame:\n")
            self._append_diagnostics(json.dumps(semantic_frame, ensure_ascii=False, indent=2, default=str) + "\n")

        self._append_diagnostics("\nCURRENT STATE METRICS\n")
        self._append_diagnostics("-" * 54 + "\n")
        if latest is None:
            self._append_diagnostics("No latest evidence record is available.\n")
        else:
            risks = dict(latest.collision_risks)
            self._append_diagnostics(f"step: {latest.step}\n")
            self._append_diagnostics(f"player_pos: {latest.player_pos}\n")
            self._append_diagnostics(f"chosen_action: {latest.chosen_action}\n")
            self._append_diagnostics(f"available_actions: {', '.join(latest.available_actions)}\n")
            self._append_diagnostics(f"collision_risks: {self._format_metrics_dict(risks, percent=True)}\n")
            self._append_diagnostics(f"has_safer_alternative: {latest.has_safer_alternative}\n")
            self._append_diagnostics(
                "nearest_monster: "
                f"id={latest.nearest_monster_id}, "
                f"direction={latest.nearest_monster_direction}, "
                f"distance={latest.nearest_monster_distance}\n"
            )
            self._append_diagnostics(
                f"dots: remaining={latest.dots_remaining}, "
                f"collected={latest.dots_collected}, "
                f"total={latest.total_dots}, "
                f"nearest_direction={latest.nearest_dot_direction}, "
                f"nearest_distance={latest.nearest_dot_distance}\n"
            )
            self._append_diagnostics(
                f"exit: open={latest.exit_open}, "
                f"direction={latest.exit_direction}, "
                f"distance={latest.exit_distance}, "
                f"pos={latest.exit_pos}\n"
            )
            if latest.reasoning:
                self._append_diagnostics(f"agent_reasoning: {latest.reasoning}\n")

        self._append_diagnostics("\nEXPLANATION STATUS\n")
        self._append_diagnostics("-" * 54 + "\n")
        self._append_diagnostics(f"answer_language: {primary_result.get('language')}\n")
        self._append_diagnostics(f"symbolic_match: {primary_result.get('symbolic_match')}\n")
        self._append_diagnostics(f"symbolic_support: {validation.get(SYMBOLIC_SUPPORT_VALIDATION_KEY)}\n")
        self._append_diagnostics(f"fallback_used: {symbolic_rule.get('fallback_used', False)}\n")
        if secondary_result is not None:
            self._append_diagnostics(f"secondary_language: {secondary_result.get('language')}\n")
        if metrics:
            self._append_diagnostics("distillation_metrics:\n")
            for key, value in sorted(metrics.items()):
                self._append_diagnostics(f"  {key}: {value}\n")
        else:
            self._append_diagnostics("distillation_metrics: none loaded\n")

        self._append_diagnostics("\nEVIDENCE USED (E)\n")
        self._append_diagnostics("-" * 54 + "\n")
        if not used_factors:
            self._append_diagnostics("No selected evidence factors.\n")
        for factor in used_factors:
            self._append_factor_line(factor)

        self._append_diagnostics("\nALL EVIDENCE (S_t)\n")
        self._append_diagnostics("-" * 54 + "\n")
        if not all_factors:
            self._append_diagnostics("No evidence factors were returned.\n")
        for factor in all_factors:
            marks = "".join(
                [
                    "T" if factor.get("is_true") else "-",
                    "F" if factor.get("is_faithful") else "-",
                    "C" if factor.get("is_contrastive") else "-",
                ]
            )
            self._append_factor_line(factor, prefix=f"[{marks}] ")

        self._append_diagnostics("\nVALIDATION\n")
        self._append_diagnostics("-" * 54 + "\n")
        for key, value in validation.items():
            marker = "OK" if value else "FAIL"
            self._append_diagnostics(f"[{marker}] {key}\n")

        self._append_diagnostics("\nSYMBOLIC RULE AND TRACE\n")
        self._append_diagnostics("-" * 54 + "\n")
        self._append_diagnostics(f"chosen_action: {symbolic_trace.get('chosen_action')}\n")
        self._append_diagnostics(f"predicted_action: {symbolic_trace.get('predicted_action')}\n")
        self._append_diagnostics(f"alternative_action: {symbolic_trace.get('alternative_action')}\n")
        if symbolic_rule.get("text"):
            self._append_diagnostics(f"rule_text: {symbolic_rule['text']}\n")
        if symbolic_rule.get("python"):
            self._append_diagnostics("rule_python:\n")
            self._append_diagnostics(symbolic_rule["python"] + "\n")
        self._append_trace_items("chosen_trace", symbolic_trace.get("trace", []))
        self._append_trace_items("approximate_trace", symbolic_trace.get("approximate_trace", []))

        bullets = policy_summary.get("bullets", [])
        if bullets:
            self._append_diagnostics("\nPOLICY SUMMARY\n")
            self._append_diagnostics("-" * 54 + "\n")
            for bullet in bullets:
                self._append_diagnostics(f"- {bullet}\n")
        if policy_summary.get("python_snippet"):
            self._append_diagnostics("policy_python_snippet:\n")
            self._append_diagnostics(policy_summary["python_snippet"] + "\n")
        if hasattr(self, "metrics_text"):
            self.metrics_text.yview_moveto(0)

    def _append_factor_line(self, factor: dict, prefix: str = "") -> None:
        name = factor.get("name", "unknown")
        description = factor.get("description", "")
        sources = factor.get("sources") or []
        suffix = f" | sources={', '.join(sources)}" if sources else ""
        self._append_diagnostics(f"{prefix}{name}: {description}{suffix}\n")

    def _append_trace_items(self, title: str, items: list[dict]) -> None:
        if not items:
            self._append_diagnostics(f"{title}: none\n")
            return
        self._append_diagnostics(f"{title}:\n")
        for item in items:
            condition = item.get("condition", "")
            description = item.get("description", "")
            sources = item.get("source") or []
            suffix = f" | sources={', '.join(sources)}" if sources else ""
            self._append_diagnostics(f"  - {condition} | {description}{suffix}\n")

    @staticmethod
    def _format_metrics_dict(values: dict, percent: bool = False) -> str:
        if not values:
            return "{}"
        parts = []
        for key, value in sorted(values.items()):
            if percent and isinstance(value, (int, float)):
                parts.append(f"{key}={value:.0%}")
            else:
                parts.append(f"{key}={value}")
        return "{ " + ", ".join(parts) + " }"

    @staticmethod
    def _answer_language_title(language: str) -> str:
        if language == "zh":
            return "Chinese Answer / 中文回答"
        if language == "en":
            return "English Answer / 英文回答"
        return "Answer / 回答"
        if language == "zh":
            return "Chinese Answer / 中文回答"
        if language == "en":
            return "English Answer / 英文回答"
        return "Answer / 回答"

    def _selected_answer_mode(self) -> str:
        label = self.answer_language_label_var.get()
        if label in ANSWER_LANGUAGE_OPTIONS:
            return ANSWER_LANGUAGE_OPTIONS[label]

        lowered = label.lower()
        if "both" in lowered or "双语" in label:
            return "both"
        if "english" in lowered or label == "en":
            return "en"
        if "中文" in label or "chinese" in lowered or label == "zh":
            return "zh"
        return "auto"

    def _log_question_event(
        self,
        evidence,
        question_text: str,
        parsed: ParsedQuestion,
        mode: str,
        primary_result: dict,
        secondary_result: dict | None,
    ) -> None:
        payload = {
            "step": evidence.step,
            "player_pos": list(evidence.player_pos),
            "chosen_action": evidence.chosen_action,
            "question": question_text,
            "intent": parsed.intent.value,
            "confidence": parsed.confidence,
            "parser_backend": self.parser.backend,
            "answer_mode": mode,
            "primary_language": primary_result.get("language"),
            "primary_answer": primary_result["explanation_text"]["text"],
            "primary_validation": primary_result["validation"],
            "evidence_used": primary_result["evidence_used"]["factors"],
            "symbolic_match": primary_result.get("symbolic_match"),
            "symbolic_rule": primary_result.get("symbolic_rule", {}),
            "symbolic_trace": primary_result.get("symbolic_trace", {}),
            "policy_summary": primary_result.get("policy_summary", {}),
        }
        if secondary_result is not None:
            payload["secondary_language"] = secondary_result.get("language")
            payload["secondary_answer"] = secondary_result["explanation_text"]["text"]
            payload["secondary_validation"] = secondary_result["validation"]

        with self.question_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _append_explanation(self, text: str) -> None:
        self.exp_text.config(state=tk.NORMAL)
        self.exp_text.insert(tk.END, text)
        self.exp_text.see(tk.END)
        self.exp_text.config(state=tk.DISABLED)

    def _clear_explanation(self) -> None:
        self.exp_text.config(state=tk.NORMAL)
        self.exp_text.delete("1.0", tk.END)
        self.exp_text.config(state=tk.DISABLED)

    def _append_diagnostics(self, text: str) -> None:
        if not hasattr(self, "metrics_text"):
            return
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.see(tk.END)
        self.metrics_text.config(state=tk.DISABLED)

    def _clear_diagnostics(self) -> None:
        if not hasattr(self, "metrics_text"):
            return
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.config(state=tk.DISABLED)
