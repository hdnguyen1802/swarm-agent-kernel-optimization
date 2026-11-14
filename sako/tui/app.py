"""Set up a harness run and inspect its saved measurements in the terminal."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import sqlite3
import subprocess

from rich import box
from rich.table import Table
from rich.console import Group
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import Screen
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option
from filelock import FileLock, Timeout

from .. import __version__
from ..core import load_config, read_json
from .colab import ColabSession, GPUS, cli_command
from .runner import MODES, controller_busy, launch, probe_gpu
from .runs import (
    latest_attempt, list_runs, read_if_present, run_path, run_snapshot, saved_config, text_preview,
)
from .timeline import AMBER, LIME, MUTED, Timeline, brief

STEPS = ("Input", "GPU", "Mode", "Output", "Review")


def ribbon(labels, current):
    text = Text()
    for index, label in enumerate(labels):
        if index:
            text.append("  ")
        if index == current:
            text.append(" " + label + " ", style=f"bold #10130e on {LIME}")
        else:
            text.append(label, style=MUTED)
    return text


def choice(key, title, description):
    line = Text(title, style="bold")
    line.append("\n" + description, style=MUTED)
    return Option(line, id=key)


def worker_label(settings):
    if settings.get("transport") == "local":
        return f"Local worker  ·  GPU {settings.get('gpu', 0)}"
    return f"{settings.get('host', 'SSH worker')}  ·  GPU {settings.get('gpu', 0)}"


def workload_label(config):
    return ", ".join((f"batch {w['batch_size']}, generate {w['new_tokens']} tokens with KV cache"
                      if w.get("mode") == "cached_generation" else f"batch {w['batch_size']}, length {w['sequence_length']}")
                     for w in config.get("workloads", [])) or "Unknown"


def fields(rows):
    table = Table.grid(padding=(0, 2), expand=False)
    table.add_column(style=MUTED, no_wrap=True)
    table.add_column()
    for label, value in rows:
        table.add_row(Text(label), Text(str(value)))
    return table


class Masthead(Horizontal):
    def compose(self) -> ComposeResult:
        # Draw the mark with terminal cells so it works in a plain terminal.
        mark = Text()
        for row in ("   ▗██████▖ ", " ▗██▘    ▝██", " ██  ▪  ▪ ██", " ▝██▖ ▀▀ ▗██", "   ▜██  ██▛ "):
            for cell in row:
                mark.append(cell, style=LIME if cell in "▪▀" else "#dfe7da")
            mark.append("\n")
        yield Static(mark, id="mark")
        brand = Text("SAKO", style=f"bold {LIME}")
        brand.append(f"  v{__version__}\n", style=MUTED)
        brand.append("Swarm Agent Kernel Optimization\n", style="bold #e6ebe2")
        brand.append("Triton kernels  /  GPU evaluation  /  Model checks", style=MUTED)
        yield Static(brand, id="brand")


class TerminalScreen(Screen):
    def on_resize(self, event: Resize):
        self.set_class(event.size.height < 27, "short")


class Wizard(TerminalScreen):
    BINDINGS = [("left", "back", "Back"), ("right", "next", "Next"),
                ("escape", "back", "Back"), ("p", "probe", "Check GPU")]

    def __init__(self, *, existing=None, suggested=None, colab=None, source_config=None):
        super().__init__()
        self.existing = existing
        self.run = existing or suggested
        self.step = 2 if existing else 1 if colab else 0
        self.colab = colab
        self.source_config = source_config
        self.gpu_menu_id = "managed"
        self.subpage = ""
        self.mode = "resume" if existing else "search"
        self.config = {}
        self.config_error = ""
        self.connections = []
        self.worker_path = None
        self.folder = str(suggested) if suggested else "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.note = ""
        self.saved = []
        self.probes = {}
        self.launching = False

    def compose(self) -> ComposeResult:
        yield Masthead()
        yield Static(id="steps")
        with VerticalScroll(id="form"):
            yield Static(id="question")
            yield OptionList(id="choices")
            with Vertical(id="editor"):
                yield Label("", id="field-label")
                yield Input(id="field")
                yield Label("Run note (optional)", id="note-label")
                yield Input(id="note", placeholder="What are you testing?")
            yield Static(id="review")
            yield Static(id="feedback", markup=False)
            yield Button("Continue", id="confirm")
        yield Static(id="keys")

    def on_mount(self):
        self.base_path = self.source_config or self.app.config_path
        path = saved_config(self.existing) if self.existing else self.base_path
        try:
            if path is None:
                raise ValueError("This run has no saved config.")
            self.config = load_config(path)
            self.worker_path = path
            self.worker_settings = self.config["worker"]
        except (OSError, ValueError, KeyError) as error:
            self.config_error = str(error)
        self.connections = sorted((self.app.root / ".local/colab-connections").glob("*/config.json"))
        self.show_step()

    def options(self, question, items, selected=None):
        self.query_one("#question", Static).update(question)
        menu = self.query_one("#choices", OptionList)
        menu.display = True
        menu.set_options(items)
        if items:
            menu.highlighted = next((i for i, item in enumerate(items) if item.id == selected), 0)
        menu.focus()

    def show_step(self):
        self.query_one("#form", VerticalScroll).scroll_home(animate=False)
        self.query_one("#steps", Static).update(ribbon(STEPS, self.step))
        for selector in ("#choices", "#editor", "#review", "#confirm"):
            self.query_one(selector).display = False
        self.feedback("")
        keys = "↑ ↓  Select     Enter  Confirm     ← →  Steps     Esc  Back"
        if self.subpage == "saved":
            self.saved = list_runs(self.app.root)
            options = [choice(f"run-{i}", path.name, "Open saved measurements")
                       for i, path in enumerate(self.saved)]
            if not options:
                options = [choice("back", "Back to input", "No saved runs in runs/")]
            self.options("Which run do you want to open?", options)
        elif self.subpage == "accelerator":
            options = [Option(Text(gpu[0]), id=key) for key, gpu in GPUS.items()]
            self.options("Which Colab GPU do you want to request?", options, self.colab or "a100-40")
            self.feedback("Colab checks availability when you start the run.")
            keys = "↑ ↓  Select GPU     Enter  Use GPU     Esc  Back"
        elif self.subpage == "connection":
            self.editor("Connect a Colab worker", "Worker config path", "", note=False)
            self.feedback("Use the config from scripts.configure_colab. See docs/colab.md.")
            keys = "Enter  Use connection     Esc  Back"
        elif self.step == 0:
            model = self.config.get("model_id", "Model config unavailable")
            items = [choice("new", "Optimize a model", model),
                     choice("saved", "Open a saved run", "View progress, candidates, logs, and measurements")]
            items[0].disabled = bool(self.config_error)
            self.options("What do you want to work on?", items, "saved" if self.config_error else "new")
            self.feedback(self.config_error)
            keys = "↑ ↓  Select     Enter  Confirm     Esc  Exit"
        elif self.step == 1:
            items = [choice("managed", "Start a Colab session", "Request a GPU and release it after the run"),
                     choice("default", "Configured worker", worker_label(self.config["worker"]))]
            items += [choice(f"connection-{i}", "Colab  ·  " + p.parent.name, "Use this prepared SSH connection")
                      for i, p in enumerate(self.connections)]
            items.append(choice("other", "Open a Colab connection file", "Select a prepared worker by its config path"))
            active = (f"connection-{self.connections.index(self.worker_path)}"
                      if self.gpu_menu_id == "other" and self.worker_path in self.connections else self.gpu_menu_id)
            self.options("Which worker should run the model?", items, active)
            self.feedback(self.probes.get(self.worker_path, ""))
            keys = "↑ ↓  Select     P  Check GPU     Enter  Use worker     Esc  Back"
        elif self.step == 2:
            if self.config_error:
                self.options("This run supports viewing only.", [choice("back", "Open the run", self.config_error)])
            else:
                items = ([choice("resume", "Resume search", "Continue with the saved settings and token budget"),
                          choice("integrate", "Test kept kernels", "Measure the complete model against torch.compile")]
                         if self.existing else
                         [choice("search", "Search and test the model", "Generate kernels and measure accepted changes"),
                          choice("baseline", "Profile and measure the baseline", "Measure the model without agent requests"),
                          choice("doctor", "Verify the GPU worker", "Install and check the worker on a new Colab session"
                                 if self.colab else "Check the worker and its installed software")])
                self.options("What should the harness run?", items, self.mode)
        elif self.step == 3:
            self.editor("Where should the harness save this run?", "New folder under runs/", self.folder, note=True)
            keys = "Tab  Move focus     Enter  Review     Esc  Back"
        elif self.step == 4:
            self.query_one("#question", Static).update("Review the run")
            rows = [("Action", MODES[self.mode]), ("Model", self.config["model_id"]),
                    ("Revision", self.config.get("revision", "main")),
                    ("Worker", "New Colab " + GPUS[self.colab][0] if self.colab else worker_label(self.worker_settings)),
                    ("Workload", workload_label(self.config)), ("Precision", self.config["dtype"]),
                    ("Kernels", self.config.get("kernel_backend", "triton")),
                    (("Token limit", f"{self.config['agent']['max_total_tokens']:,}") if self.mode in {"search", "resume"}
                     else ("Agent calls", "None")),
                    ("Speed reference", "torch.compile"), ("Run folder", self.run.relative_to(self.app.root)),
                    ("Profile", self.profile_path().relative_to(self.app.root))]
            if self.note:
                rows.append(("Note", self.note))
            if self.mode in {"search", "resume"}:
                rows.extend((("Agent provider", self.config["agent"].get("provider", "vertex")),
                             ("Agent model", self.config["agent"]["model"])))
            review = self.query_one("#review", Static)
            review.display = True
            review.update(fields(rows))
            if self.colab:
                self.feedback("Colab uses compute units during setup and the job. SAKO saves the evidence and releases the GPU.")
            self.button("Start " + ("search" if self.mode in {"search", "resume"} else "worker check"
                                    if self.mode == "doctor" else "measurement")).focus()
            keys = "Enter  Start     ← / Esc  Back"
        self.query_one("#keys", Static).update(Text(keys, style=MUTED))

    def profile_path(self):
        if self.app.profile is not None:
            return self.app.profile
        if self.existing:
            attempt = latest_attempt(self.existing)
            plan = read_if_present(attempt / "plan.json") if attempt else {}
            if plan.get("profile"):
                path = (self.app.root / plan["profile"]).resolve()
                path.relative_to(self.app.root)
                return path
        return self.run / "profile"

    def editor(self, question, label, value, *, note):
        self.query_one("#question", Static).update(question)
        self.query_one("#editor").display = True
        self.query_one("#field-label", Label).update(label)
        field = self.query_one("#field", Input)
        field.placeholder = ".local/colab-connections/session-001/config.json" if not note else "my-run"
        field.value = value
        self.query_one("#note-label").display = note
        self.query_one("#note", Input).display = note
        self.query_one("#note", Input).value = self.note
        self.button("Review" if note else "Use connection")
        field.focus()

    def button(self, label):
        button = self.query_one("#confirm", Button)
        button.label = label
        button.display = True
        return button

    def feedback(self, message):
        widget = self.query_one("#feedback", Static)
        widget.update(message)
        widget.display = bool(message)

    def selected(self):
        option = self.query_one("#choices", OptionList).highlighted_option
        return option.id if option and not option.disabled else None

    def connection(self, key):
        if key == "default":
            return self.base_path
        return self.connections[int(key.split("-")[1])]

    def use_connection(self, path):
        worker = read_json(path)["worker"]
        if worker.get("transport") not in {"local", "ssh"}:
            raise ValueError("Choose a config for a local or SSH worker.")
        self.worker_path, self.worker_settings = path, worker
        self.colab = None
        if path != self.base_path and path not in self.connections:
            self.connections.append(path)
        self.subpage = ""
        self.step = 2

    @on(OptionList.OptionSelected, "#choices")
    def choose(self, event):
        if self.query_one("#choices").display:
            self.advance(event.option.id)

    @on(Input.Submitted)
    @on(Button.Pressed, "#confirm")
    def confirm(self):
        self.advance(self.selected())

    def action_next(self):
        if self.step != 4 and not isinstance(self.focused, Input):
            self.advance(self.selected())

    def advance(self, key):
        if self.launching:
            return
        try:
            if key == "back" and self.query_one("#choices").display:
                self.action_back()
                return
            if self.subpage == "saved":
                if key:
                    self.app.switch_screen(RunView(self.saved[int(key.split("-")[1])]))
                return
            if self.subpage == "connection":
                value = self.query_one("#field", Input).value.strip().strip('"')
                if not value:
                    raise ValueError("Enter the path to the worker config.")
                self.use_connection((self.app.root / Path(value).expanduser()).resolve())
            elif self.subpage == "accelerator":
                if key not in GPUS:
                    return
                cli_command(self.app.root)
                self.colab = key
                self.subpage = ""
                self.step = 2
            elif self.step == 0:
                if key == "saved":
                    self.subpage = "saved"
                elif key == "new" and not self.config_error:
                    self.step = 1
                else:
                    return
            elif self.step == 1:
                self.gpu_menu_id = key
                if key == "managed":
                    self.subpage = "accelerator"
                elif key == "other":
                    self.subpage = "connection"
                else:
                    self.use_connection(self.connection(key))
            elif self.step == 2:
                if key not in MODES:
                    return
                self.mode = key
                self.step = 4 if self.existing else 3
            elif self.step == 3:
                self.folder = self.query_one("#field", Input).value.strip()
                self.note = self.query_one("#note", Input).value.strip()
                self.run = run_path(self.app.root, self.folder)
                if self.run.exists():
                    raise ValueError("This folder already exists. Choose a new name or open the saved run.")
                self.step = 4
            else:
                self.launching = True
                config = {**self.config, "worker": self.worker_settings}
                process, attempt = launch(self.app.root, self.run, self.profile_path(), self.mode,
                                          config, self.note, self.colab)
                self.app.switch_screen(RunView(self.run, process, attempt))
                return
            self.show_step()
        except (OSError, ValueError, KeyError, TypeError, IndexError, sqlite3.Error) as error:
            self.launching = False
            self.feedback(str(error))

    def action_back(self):
        if self.subpage:
            self.subpage = ""
        elif self.existing and self.step == 2:
            self.app.switch_screen(RunView(self.existing))
            return
        elif self.step == 0:
            self.app.exit()
            return
        elif self.existing:
            self.step = 2
        else:
            self.step -= 1
        self.show_step()

    @work(exclusive=True, group="gpu")
    async def action_probe(self):
        if self.step != 1 or self.subpage or self.selected() in {"managed", "other", None}:
            return
        path = self.connection(self.selected())
        self.feedback("Checking GPU connection…")
        try:
            config = {**self.config, "worker": read_json(path)["worker"]}
            result = await asyncio.to_thread(probe_gpu, config, self.app.root)
            message = "Connected  ·  " + result
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            message = f"GPU check failed. {error}"
        self.probes[path] = message
        if self.is_mounted and self.step == 1 and not self.subpage and self.selected() not in {"managed", "other", None}:
            if self.connection(self.selected()) == path:
                self.feedback(message)

    @on(OptionList.OptionHighlighted, "#choices")
    def highlighted(self, event):
        if self.step == 1 and not self.subpage:
            key = event.option.id
            if key in {"other", "managed"}:
                self.feedback("")
            elif key == "default" or key.startswith("connection-"):
                self.feedback(self.probes.get(self.connection(key), ""))


class RunView(TerminalScreen):
    BINDINGS = [("t", "pane(0)", "Timeline"), ("o", "pane(1)", "Overview"), ("c", "pane(2)", "Candidates"),
                ("l", "pane(3)", "Log"), ("r", "pane(4)", "Report"),
                ("left", "previous", "Previous"), ("right", "next", "Next"),
                ("a", "actions", "Run actions"), ("p", "stop_work", "Stop"), ("f", "release", "Retry release"),
                ("escape", "home", "Setup"), ("q", "quit_view", "Exit")]
    PANES = ("Timeline", "Overview", "Candidates", "Log", "Report")

    def __init__(self, run, process=None, attempt=None):
        super().__init__()
        self.run_dir, self.process, self.attempt = run, process, attempt
        self.pane = 0
        self.data = {}
        self.reading = False
        self.config = {}
        self.note = ""

    def compose(self) -> ComposeResult:
        yield Static(id="run-title", markup=False)
        yield Static(id="steps")
        yield Static(id="live-summary", markup=False)
        yield Timeline()
        with VerticalScroll(id="run-body"):
            yield Static(id="content", markup=False)
        yield Static(id="message", markup=False)
        yield Static(id="keys")

    def on_mount(self):
        config_path = saved_config(self.run_dir)
        if config_path:
            try:
                self.config = read_json(config_path)
            except (OSError, ValueError):
                pass
        attempt = self.attempt or latest_attempt(self.run_dir)
        if attempt:
            self.note = read_if_present(attempt / "plan.json").get("note", "")
            if not self.config:
                self.config = read_if_present(attempt / "requested-config.json")
        self.query_one("#run-title", Static).update(Text.assemble(
            ("SAKO", f"bold {LIME}"), ("   /   ", MUTED), self.run_dir.name))
        self.message("")
        self.set_interval(1.5, self.refresh_run)
        self.action_pane(0)

    def message(self, text):
        message = self.query_one("#message", Static)
        message.update(text)
        message.display = bool(text)

    def action_pane(self, pane):
        self.pane = pane
        self.query_one("#steps", Static).update(ribbon(self.PANES, pane))
        keys = "↑ ↓ Scroll   End Follow   " if pane == 0 else ""
        self.query_one("#keys", Static).update(keys + "← → Views   A Actions   P Stop   Esc Setup   Q Exit")
        timeline = self.query_one(Timeline)
        body = self.query_one("#run-body", VerticalScroll)
        timeline.display = self.query_one("#live-summary").display = pane == 0
        body.display = pane != 0
        if pane == 0:
            timeline.focus()
        else:
            body.focus()
            body.scroll_home(animate=False)
        self.refresh_run()

    def action_previous(self):
        self.action_pane((self.pane - 1) % len(self.PANES))

    def action_next(self):
        self.action_pane((self.pane + 1) % len(self.PANES))

    def read_pane(self, pane, event_offset):
        data = run_snapshot(self.run_dir, self.attempt, event_offset=event_offset, compact=True)
        preview = ""
        if pane == 3:
            attempt = self.attempt or latest_attempt(self.run_dir)
            preview = text_preview(attempt / "controller.log" if attempt else None, tail=True)
        elif pane == 4:
            preview = text_preview(self.run_dir / "report.md")
        return data, preview

    @work(group="read")
    async def refresh_run(self):
        if self.reading:
            return
        self.reading = True
        pane = self.pane
        try:
            timeline = self.query_one(Timeline)
            offset = timeline.event_offset if pane == 0 else None
            data, preview = await asyncio.to_thread(self.read_pane, pane, offset)
            if not self.is_mounted or pane != self.pane:
                return
            previous_warning = self.data.get("warning")
            self.data = data
            if self.process is not None and self.process.poll() is not None:
                if self.process.returncode:
                    self.message("The controller stopped. Press L to read the log.")
                else:
                    self.message("")
                self.process = None
            content = self.query_one("#content", Static)
            if pane == 0:
                self.query_one("#live-summary", Static).update(self.live_summary())
                timeline.show_snapshot(data)
            elif pane == 1:
                content.update(self.overview())
            elif pane == 2:
                decode = self.config.get("generation_objective") == "decode"
                table = Table("Model", "Candidate", "Target", "Writer", "GPU", "Test", "Local ×", "Rank",
                    box=None, padding=(0, 1), caption="Local decode vs torch.compile · ranks are per target" if decode
                    else "Local kernel ratio · see the run files for each timing reference")
                for identity, task, writer, outcome, ratio, _reason in data["candidates"]:
                    name = self.target_name(task)
                    rank = data["shortlist"].get(identity)
                    used = identity in data.get("selected_candidates", [])
                    table.add_row(Text("USED" if used else "—", style=LIME if used else MUTED),
                        identity[:9], name, writer or "—", str(data["candidate_gpus"].get(identity, "—")), Text(outcome,
                        style=LIME if outcome == "passed" else MUTED if outcome == "pending" else AMBER),
                        ratio, f"#{rank}" if rank else "—")
                content.update(Group(self.model_summary(), Text(""), table)
                               if data["candidates"] else "No candidates recorded.")
            elif pane == 3:
                content.update(("Last 12 KB of the controller log\n\n" + preview) if preview else
                               "No terminal log for this run.")
            else:
                content.update(("report.md  ·  first 12 KB\n\n" + preview) if preview else
                               "The harness has not saved a report yet.")
            if data["warning"]:
                self.message(data["warning"])
            elif previous_warning and self.query_one("#message", Static).content == previous_warning:
                self.message("")
        except (OSError, ValueError) as error:
            self.message(str(error))
        finally:
            self.reading = False
            if self.is_mounted and pane != self.pane:
                self.refresh_run()

    def target_name(self, task):
        from ..operation_names import operation_name
        return operation_name(task, self.data.get("operation_names", {}))

    def model_summary(self):
        blocks = []
        metrics = self.data.get("decode_metrics", [])
        if metrics:
            for row in metrics:
                workload = row.get("workload", {})
                title = workload_label({"workloads": [workload]}) if len(metrics) > 1 else None
                table = Table("Decode", "PyTorch", "torch.compile", "Our optimization", box=box.SIMPLE_HEAD,
                              padding=(0, 2), title=title, title_justify="left")
                for column in table.columns[1:]:
                    column.justify = "right"
                table.columns[3].style = LIME
                original_ms = row["original_ms_per_token"]
                original_rate = 1000 * workload.get("batch_size", 1) / original_ms
                eager_ms = row.get("pytorch_ms_per_token")
                eager_rate = 1000 * workload.get("batch_size", 1) / eager_ms if eager_ms else None
                table.add_row("Time / token (ms)", f"{eager_ms:.3f}" if eager_ms else "—",
                              f"{original_ms:.3f}", f"{row['ms_per_token']:.3f}")
                table.add_row("Tokens / second", f"{eager_rate:.2f}" if eager_rate else "—",
                              f"{original_rate:.2f}", f"{row['tokens_per_second']:.2f}")
                table.add_row("Speedup vs PyTorch", "1.000×" if eager_ms else "—",
                              f"{eager_ms / original_ms:.3f}×" if eager_ms else "—",
                              f"{eager_ms / row['ms_per_token']:.3f}×" if eager_ms else "—")
                if not eager_ms:
                    table.caption = "PyTorch timing has not been measured for this run."
                blocks.append(table)
        elif self.config.get("generation_objective") == "decode":
            blocks.append(Text("Awaiting first validated model", style=MUTED))
        else:
            blocks.append(Text("Full model vs torch.compile  " + self.data["model"], style="bold"))
        progress = self.data.get("final_progress")
        if progress and progress.get("active"):
            label = "Final combined test" if progress.get("phase") == "combined" else "Model test · " + self.target_name(progress.get("target"))
            blocks.append(Text(label + "  ·  " + "  ·  ".join(f"#{r['rank']} on GPU {r['gpu']}" for r in progress["active"]),
                               style=MUTED))
        return blocks[0] if len(blocks) == 1 else Group(*blocks)

    def live_summary(self):
        data = self.data
        stage = data["stage"]
        status = stage.get("status") or data["status"] or "Saved run"
        color = AMBER if status.lower() in {"failed", "stopped", "request_failure", "infrastructure_failure"} else LIME
        header = Text()
        header.append("●  " + status.replace("_", " ").capitalize(), style=f"bold {color}")
        if stage.get("stage"):
            header.append("  ·  " + stage["stage"])
        setup = [self.config.get("model_id"), data["gpu"], self.config.get("kernel_backend"), self.config.get("dtype")]
        if any(setup):
            header.append("\n" + "  ·  ".join(item for item in setup if item), style=MUTED)
        if data["task"]:
            position = " " + data["task_position"] if data["task_position"] else ""
            label = "Current task" if status.lower() == "running" else "Last task"
            header.append(f"\n{label}{position}  {self.target_name(data['task'])}", style="bold")
        counts = []
        if data["tested"] is not None:
            counts.extend((f"{data['passed']} passed", f"{data['partial']} partial",
                           f"{data['tested'] - data['passed'] - data['partial']} failed",
                           f"{data['pending']} pending"))
        if data["writers"] is not None:
            counts.append(f"Open writer calls  {data['writers']}")
        if counts:
            header.append("\n" + "  ·  ".join(counts), style=MUTED)
        if data["planning"]:
            header.append("\nCoordinator is planning.")
        pending = next((row for row in data["candidates"] if row[3] == "pending"), None)
        if pending:
            header.append(f"\nAwaiting kernel result  ·  {pending[0][:12]}  ·  {self.target_name(pending[1])}")
        if stage.get("detail"):
            header.append("\n" + brief(stage["detail"]), style=AMBER)
        summary = self.model_summary()
        if isinstance(summary, Text):
            header.append("\n").append_text(summary)
            return header
        return Group(header, summary)

    def overview(self):
        data = self.data
        stage = data["stage"]
        gpu = data["gpu"] or "Awaiting GPU facts"
        if data["memory"]:
            gpu += f"  ·  {data['memory'] / 2**30:.1f} GiB"
        agents = self.config.get("agent", {})
        tokens = "—" if data["tokens"] is None else f"{data['tokens']:,}"
        if agents.get("max_total_tokens") is not None:
            tokens += f" / {agents['max_total_tokens']:,}"
        counts = ("Awaiting evaluation" if data["tested"] is None else
                  f"{data['tested']} tested  ·  {data['passed']} passed  ·  {data['partial']} partial  ·  {data['pending']} pending")
        rows = [("Model", self.config.get("model_id", "Unknown")), ("GPU", gpu),
                ("Workload", workload_label(self.config) + "  ·  " + self.config.get("dtype", "Unknown"))]
        if stage:
            rows.append(("Controller", "  ·  ".join(str(stage[k]) for k in ("status", "stage") if stage.get(k))))
        rows += [("Search", data["status"] or "No search result"), ("Active task", self.target_name(data["task"])),
                 ("Pending writers", data["writers"] if data["writers"] is not None else "—"),
                 ("Candidates", counts), ("Accounted tokens", tokens),
                 ("Kernel ratios", "See Candidates for local results.")]
        if agents:
            rows.extend((("Agent provider", agents.get("provider", "vertex")),
                         ("Agent model", agents.get("model", "Unknown"))))
        if stage.get("detail"):
            rows.append(("Details", stage["detail"]))
        colab = data.get("colab", {})
        if colab:
            rows.append(("Colab request", colab["requested"]))
            if colab.get("released"):
                rows.append(("Colab session", "Released" if colab.get("allocated") else "No active allocation"))
            elif colab.get("release_error"):
                rows.append(("Colab session", "Release unconfirmed. Press F to retry."))
            elif colab.get("allocated"):
                rows.append(("Colab session", "Allocated"))
            if colab.get("evidence"):
                rows.append(("GPU evidence", colab["evidence"]))
        if self.note:
            rows.append(("Note", self.note))
        return Group(fields(rows), Text(""), self.model_summary())

    def action_actions(self):
        if (self.process is not None and self.process.poll() is None) or controller_busy(self.run_dir):
            self.message("A controller is working on this run.")
        elif self.data.get("colab", {}).get("release_error"):
            self.message("Press F to retry release of this Colab session.")
        elif self.data.get("colab"):
            attempt = latest_attempt(self.run_dir)
            self.app.switch_screen(Wizard(colab=self.data["colab"]["choice"],
                                          source_config=attempt / "requested-config.json"))
        elif self.data.get("status") == "running" and self.data.get("stage", {}).get("status") != "Running":
            self.message("The saved search is running. Stop it before starting another controller.")
        else:
            self.app.switch_screen(Wizard(existing=self.run_dir))

    def action_stop_work(self):
        if ((self.process is not None and self.process.poll() is None)
                or controller_busy(self.run_dir) or self.data.get("status") == "running"):
            attempt = self.attempt or latest_attempt(self.run_dir)
            if attempt:
                (attempt / "stop-requested").touch()
            (self.run_dir / "STOP").touch(exist_ok=True)
            self.message("Stop requested. The harness will finish active work at a saved boundary.")
        else:
            self.message("There is no active run to stop.")

    @work(exclusive=True, group="release")
    async def action_release(self):
        if controller_busy(self.run_dir):
            self.message("The controller will release its Colab session.")
            return
        attempt = latest_attempt(self.run_dir)
        receipt = read_if_present(attempt / "colab.json") if attempt else {}
        if not receipt.get("allocation_attempted") or receipt.get("released"):
            self.message("There is no Colab session to release.")
            return
        self.message("Checking the Colab session…")
        try:
            owner = ColabSession(self.app.root, attempt, receipt["choice"], resume_cleanup=True)
            def release():
                with FileLock(str(self.run_dir / "ui/controller.lock"), timeout=0):
                    owner.release()
            await asyncio.to_thread(release)
            self.message("Colab confirmed release of this session.")
        except Timeout:
            self.message("The controller is already handling this session.")
        except (OSError, ValueError, RuntimeError) as error:
            self.message(str(error))
        self.refresh_run()

    def action_home(self):
        self.app.switch_screen(Wizard())

    def action_quit_view(self):
        self.app.exit()


class HarnessApp(App):
    TITLE = "SAKO"
    CSS_PATH = "terminal.tcss"
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, root, config_path, run=None, profile=None):
        super().__init__()
        self.root = Path(root).resolve()
        self.config_path = (self.root / config_path).resolve()
        self.initial_run = run_path(self.root, run) if run is not None else None
        self.profile = (self.root / profile).resolve() if profile else None
        if self.profile is not None:
            self.profile.relative_to(self.root)

    def on_mount(self):
        if self.initial_run and self.initial_run.is_dir():
            self.push_screen(RunView(self.initial_run))
        else:
            self.push_screen(Wizard(suggested=self.initial_run))
