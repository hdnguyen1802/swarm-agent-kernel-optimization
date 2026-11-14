"""Check the terminal workflow and the boundary between UI and evaluation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from sako.core import read_json, write_json
from sako.store import Store
from sako.tui.colab import ColabSession, verify_gpu
from sako.tui.runs import run_path, run_snapshot

ROOT = Path(__file__).resolve().parents[1]
HAS_UI = all(importlib.util.find_spec(module) for module in ("textual", "filelock"))


class RunEvidenceTests(unittest.TestCase):
    def test_view_does_not_create_missing_run_or_interpret_kernel_as_model_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            run = run_path(Path(directory), "trial")
            self.assertEqual(run_snapshot(run)["model"], "Not measured")
            self.assertFalse(run.exists())
            store = Store(run / "state.sqlite")
            with store.connect() as db:
                db.execute("INSERT INTO candidates(id,accepted,result) VALUES (?,?,?)",
                           ("kernel", 1, '{"decision":{"accepted":true,"speedup":1.25}}'))
                db.execute("INSERT INTO candidates(id) VALUES ('pending')")
            evidence = run_snapshot(run)
            self.assertEqual((evidence["tested"], evidence["passed"], evidence["pending"]), (1, 1, 1))
            self.assertEqual(evidence["kernel"], "1.250×")
            self.assertEqual(evidence["model"], "Not measured")
            model = {"manifest": {"entries": [{}]}, "result": {"decision": {"accepted": False, "speedup": 1.1}}}
            store.set("last_model_measurement", model)
            self.assertEqual(run_snapshot(run)["model"], "No gain accepted")
            model["result"]["decision"]["accepted"] = True
            store.set("best_model_measurement", model)
            self.assertEqual(run_snapshot(run)["model"], "1.100×  ·  accepted")


    def test_partial_speed_pass_stays_distinct_from_model_acceptance(self):
        from sako.tui.timeline import kernel_result
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "trial"
            store = Store(run / "state.sqlite")
            decision = {"accepted": False, "speedup": 1.071, "reasons": ["Prompt is slower."]}
            result = {"decision": decision, "shape_decisions": {
                "prompt": {"accepted": False, "speedup": 0.797},
                "decode": {"accepted": True, "speedup": 1.242}}}
            with store.connect() as db:
                db.execute("INSERT INTO candidates(id,accepted,result) VALUES (?,?,?)",
                           ("partial", 0, json.dumps(result)))
            store.set("swarm_events", [{"kind": "candidate_evaluated", "candidate_id": "partial",
                                       "decision": decision, "correct": True}])
            evidence = run_snapshot(run, event_offset=0)
            self.assertEqual((evidence["passed"], evidence["partial"], evidence["tested"]), (0, 1, 1))
            self.assertEqual(evidence["candidates"][0][3], "partial 1/2")
            self.assertEqual(evidence["kernel"], "1.242×")
            self.assertEqual(evidence["model"], "Not measured")
            rows = kernel_result(evidence["events"][2][0], "compute")
            self.assertIn("Kernel test partial 1/2", rows[0][1])
            self.assertIn("only passing shapes can be used", rows[1][1])


class ColabOwnershipTests(unittest.TestCase):
    def test_install_uses_the_named_notebook_and_requires_a_successful_exit(self):
        with tempfile.TemporaryDirectory() as temp, patch("sako.tui.colab.cli_command", return_value=["colab"]):
            root = Path(temp)
            session = ColabSession(root, root, "a100-40")
            session.call = Mock(side_effect=['SAKO_INSTALL={"exit_code":0,"log":"installed"}',
                                             'SAKO_INSTALL={"exit_code":3,"log":"setup failed"}', "no result"])
            session.install()
            session.call.assert_called_once_with("exec", "-s", session.name, "-f", str(root / "scripts/colab_install.py"),
                                                 "--timeout", "1800", timeout=1830)
            with self.assertRaisesRegex(RuntimeError, "exit code 3"):
                session.install()
            with self.assertRaisesRegex(RuntimeError, "installer result"):
                session.install()

    def test_requires_requested_hardware(self):
        verify_gpu("a100-40", "NVIDIA A100-SXM4-40GB", 40960)
        verify_gpu("a100-80", "NVIDIA A100-SXM4-80GB", 81920)
        for choice, name, memory in [("a100-40", "NVIDIA A100-SXM4-80GB", 81920),
                                     ("a100-80", "NVIDIA L4", 23034)]:
            with self.assertRaises(ValueError):
                verify_gpu(choice, name, memory)

    def test_release_only_addresses_the_session_this_run_created(self):
        with tempfile.TemporaryDirectory() as temp, patch("sako.tui.colab.cli_command", return_value=["colab"]):
            root = Path(temp)
            session = ColabSession(root, root, "a100-80")
            other = ("another-session", "other-endpoint")
            session.save(allocation_attempted=True, allocated=True, endpoint_hash=session.endpoint_hash("own"))
            session.sessions = Mock(side_effect=[[other, (session.name, "own")], [other]])
            session.call = Mock()
            session.release()
            session.call.assert_called_once_with("stop", "-s", session.name)
            self.assertTrue(read_json(session.receipt)["released"])

    def test_evidence_is_published_only_after_checksum_matches(self):
        with tempfile.TemporaryDirectory() as temp, patch("sako.tui.colab.cli_command", return_value=["colab"]):
            root = Path(temp)
            session = ColabSession(root, root, "l4")
            digest = hashlib.sha256(b"evidence").hexdigest()
            chunks = iter([b"partial", b"evidence"])

            def call(action, *args, **kwargs):
                if action == "exec":
                    return 'SAKO_ARCHIVE={"bytes":8,"sha256":"' + digest + '"}'
                Path(args[-1]).write_bytes(next(chunks))
            session.call = call
            session.collect()
            self.assertEqual((root / "gpu-evidence.tar.gz").read_bytes(), b"evidence")
            self.assertEqual(read_json(session.receipt)["evidence_sha256"], digest)


@unittest.skipUnless(HAS_UI, "Install the ui extra to test the terminal app.")
class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root / "runs" / "trial"
        self.attempt = self.run / "ui" / "attempt-test"
        write_json(self.attempt / "plan.json", {"mode": "search", "colab": "a100-40",
                   "config": "runs/trial/launch-config.json", "profile": "runs/trial/profile"})
        write_json(self.attempt / "requested-config.json", read_json(ROOT / "configs/project.json"))

    def exercise(self, failure=None, stop_after=None):
        from sako.tui.runner import execute
        events = []
        run = self.run
        class Session:
            config = None
            def allocate(self):
                events.append("allocate")
                if failure == "allocate":
                    raise RuntimeError("Service Unavailable")
            def connect(self, config):
                events.append("verify")
                self.config = config
                return config
            def install(self):
                events.append("install")
                if failure == "install":
                    raise RuntimeError("Install failed.")
            def collect(self):
                events.append("collect")
                if failure == "collect":
                    raise RuntimeError("Download failed.")
            def release(self):
                events.append("release")

        def command(args, cwd):
            stage = args[4]
            events.append(stage)
            if stage == "search":
                write_json(run / "result.json", {"status": "completed_kernel_search",
                           "integration_status": "pending", "kernel_manifest": {"entries": [{}]}})
            if stop_after == stage:
                (run / "STOP").touch()
            return subprocess.CompletedProcess(args, 0)
        status = execute(self.attempt, self.root, command, lambda *_: Session())
        return status, events

    def test_allocates_installs_runs_collects_then_releases(self):
        result, steps = self.exercise()
        self.assertEqual(result, 0)
        self.assertEqual(steps, ["allocate", "verify", "sync", "install", "prepare", "baseline",
                                 "search", "integrate", "report", "collect", "release"])

    def test_keep_session_preserves_the_runtime_after_a_full_run(self):
        path = self.attempt / "plan.json"
        write_json(path, {**read_json(path), "keep_session": True})
        status, steps = self.exercise()
        self.assertEqual(status, 0)
        self.assertIn("search", steps)
        self.assertIn("integrate", steps)
        self.assertIn("collect", steps)
        self.assertNotIn("release", steps)
        self.assertEqual(read_json(self.attempt / "progress.json")["status"], "Finished")

    def test_errors_and_stop_release_the_gpu_and_keep_the_reason(self):
        for failure in ["allocate", "install", "collect"]:
            with self.subTest(failure=failure):
                status, steps = self.exercise(failure)
                self.assertEqual(status, 1)
                self.assertEqual(steps[-1], "release")
                progress = read_json(self.attempt / "progress.json")
                self.assertEqual(progress["status"], "Failed")
                self.assertTrue(progress["detail"])
        status, steps = self.exercise(stop_after="prepare")
        self.assertEqual(status, 0)
        self.assertEqual(steps[-2:], ["collect", "release"])
        self.assertNotIn("search", steps)
        self.assertEqual(read_json(self.attempt / "progress.json")["status"], "Stopped")


@unittest.skipUnless(HAS_UI, "Install the ui extra to test the terminal app.")
class WizardTests(unittest.IsolatedAsyncioTestCase):
    async def test_keyboard_colab_review_precedes_launch_and_preserves_contract(self):
        from sako.tui.app import HarnessApp, RunView
        from textual.widgets import Input

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = read_json(ROOT / "configs/project.json")
            write_json(root / "configs/project.json", config)
            app = HarnessApp(root, "configs/project.json")
            with patch("sako.tui.app.cli_command", return_value=["colab"]), patch("sako.tui.app.launch") as start:
                async with app.run_test(size=(92, 33)) as pilot:
                    await pilot.press("enter", "enter", "down", "enter", "enter")
                    self.assertEqual(app.screen.step, 3)
                    app.screen.query_one("#field", Input).value = "../elsewhere"
                    await pilot.press("enter")
                    self.assertEqual(app.screen.step, 3)
                    app.screen.query_one("#field", Input).value = "trial"
                    await pilot.press("enter")
                    self.assertEqual(app.screen.step, 4)
                    self.assertFalse((root / "runs").exists())
                    start.assert_not_called()
                    run = root / "runs/trial"
                    attempt = run / "ui/attempt-test"
                    start.return_value = (Mock(poll=lambda: None), attempt)
                    await pilot.press("enter")
                    self.assertIsInstance(app.screen, RunView)
                    start.assert_called_once_with(root, run, run / "profile", "search", config, "", "a100-80")

    async def test_saved_run_opens_without_a_default_config_or_a_worker(self):
        from sako.tui.app import HarnessApp, RunView
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs/archive"
            write_json(run / "hardware.json", {"facts": {"name": "Recorded GPU", "total_memory_bytes": 2**30}})
            (run / "report.md").write_text("Saved report", encoding="utf-8")
            app = HarnessApp(root, "missing.json", run)
            with patch("sako.tui.app.launch") as start, patch("sako.tui.app.probe_gpu") as probe:
                async with app.run_test(size=(60, 22)) as pilot:
                    await pilot.pause()
                    self.assertIsInstance(app.screen, RunView)
                    self.assertEqual(app.screen.data["gpu"], "Recorded GPU")
                    await pilot.press("r")
                    await app.workers.wait_for_complete()
                    self.assertIn("Saved report", str(app.screen.query_one("#content").content))
                    start.assert_not_called()
                    probe.assert_not_called()
                    self.assertFalse((run / "state.sqlite").exists())


@unittest.skipUnless(HAS_UI, "Install the ui extra to test the terminal app.")
class TimelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_events_follow_the_reader_and_keep_kernel_and_model_results_separate(self):
        from sako.tui.app import HarnessApp
        from sako.tui.timeline import Timeline

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs/trial"
            store = Store(run / "state.sqlite")
            plan = {"id": 0, "hypotheses": [{"id": "h", "algorithm": "tile [M]", "specialty": "memory",
                                             "expected_operation_speedup": 1.5}]}
            events = [{"kind": "task_started", "task_id": "linear", "region_id": "mlp"},
                      {"kind": "coordinator_decision", "plan_id": 0, "hypothesis_count": 1}]
            events.extend({"kind": "writer_started", "assignment_id": f"a{slot}", "slot": slot,
                           "hypothesis_id": "h"} for slot in range(5))
            events.extend({"kind": "writer_finished", "assignment_id": f"a{slot}", "slot": slot,
                           "status": "generated"} for slot in range(5))
            store.set_many({"status": "running", "active_task": "linear", "task_progress": {"queue": ["linear"]},
                            "swarm_plans": [plan], "swarm_events": events})
            app = HarnessApp(root, "missing.json", run)
            async with app.run_test(size=(64, 21)) as pilot:
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                log = app.screen.query_one(Timeline)
                self.assertEqual(app.screen.PANES[app.screen.pane], "Timeline")
                self.assertEqual(log.event_offset, len(events))
                self.assertGreater(log.max_scroll_y, 0)
                self.assertTrue(log.is_vertical_scroll_end)
                self.assertIn("estimate 1.500×", " ".join(row[2] for row in log.history))
                await pilot.press("home")
                await pilot.pause(0.3)
                decision = {"accepted": False, "speedup": 0.8, "reasons": ["Below the speed limit."]}
                result = {"decision": decision, "timing_baseline": {"kind": "kept_kernel"}}
                with store.connect() as db:
                    db.execute("INSERT INTO candidates(id,task,island,accepted,result) VALUES (?,?,?,?,?)",
                               ("slow", "linear", "memory", 0, json.dumps(result)))
                events.append({"kind": "candidate_evaluated", "assignment_id": "a0", "candidate_id": "slow",
                               "decision": decision, "correct": True})
                store.set("swarm_events", events)
                # Let the screen's timer discover the new trial while the reader looks back.
                await pilot.pause(1.7)
                await app.workers.wait_for_complete()
                self.assertEqual(log.scroll_y, 0)
                saved = "\n".join(row[2] for row in log.history)
                self.assertIn("Kernel test failed  ·  slow  ·  0.800× vs kept kernel", saved)
                self.assertIn("Writer 0  ·  memory  ·  tile [M]", saved)
                self.assertIn("Below the speed limit.", saved)
                self.assertIn("1 failed", str(app.screen.query_one("#live-summary").content))
                await pilot.press("end", "o")
                events.extend([
                    {"kind": "candidate_evaluated", "candidate_id": "fast", "decision": {"accepted": True, "speedup": 1.3}},
                    {"kind": "kernel_kept", "candidate_id": "fast", "acceptance_scope": "measured_shapes_only"},
                    {"kind": "model_checked", "decision": {"accepted": True, "speedup": 9}, "manifest": {"entries": []}},
                ])
                store.set("swarm_events", events)
                await pilot.press("t")
                await app.workers.wait_for_complete()
                await pilot.pause()
                saved = "\n".join(row[2] for row in log.history)
                self.assertEqual(saved.count("Kernel test failed"), 1)
                self.assertIn("1.300× vs trial baseline", saved)
                self.assertIn("Kept kernel  fast  ·  measured shapes", saved)
                self.assertIn("Full model vs torch.compile  ·  Not measured", saved)
                self.assertNotIn("9.000×", saved)
                self.assertTrue(log.is_vertical_scroll_end)
                events.append({"kind": "model_checked", "decision": {"accepted": True, "speedup": 1.08},
                               "manifest": {"entries": [{"candidate_id": "fast"}]}})
                # Finish any periodic refresh before requesting a snapshot of this write.
                await app.workers.wait_for_complete()
                store.set("swarm_events", events)
                await app.screen.refresh_run().wait()
                self.assertIn("1.080×  ·  accepted", log.history[-1][2])

    async def test_old_run_shows_saved_candidates_without_search_events(self):
        from sako.tui.app import HarnessApp
        from sako.tui.timeline import Timeline

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs/archive"
            store = Store(run / "state.sqlite")
            result = {"decision": {"accepted": False, "speedup": None}, "error": "Compiler failed."}
            with store.connect() as db:
                db.execute("INSERT INTO candidates(id,task,island,accepted,result) VALUES (?,?,?,?,?)",
                           ("old", "linear", "memory", 0, json.dumps(result)))
            app = HarnessApp(root, "missing.json", run)
            async with app.run_test(size=(60, 22)) as pilot:
                await pilot.pause()
                preview = "\n".join(line.text for line in app.screen.query_one(Timeline).lines)
                self.assertIn("old", preview)
                self.assertIn("failed", preview)
                self.assertIn("Compiler failed.", preview)
