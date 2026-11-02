"""Exercise the actual dashboard save functions without a browser or network.

Run: python -B -m unittest discover -s tests -p test_dashboard_ui.py -v
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest


NODE = shutil.which("node")
TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "index.html"
SARVAM_KEY = "test-only-sarvam-secret"
NVIDIA_KEY = "test-only-nvidia-secret"

NODE_HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const logs = [];
const timers = [];
const requests = [];
let refreshes = 0;
let prevented = false;
let htmlWrites = 0;
let pending;
function node(value = '', classes = []) {
    const tokens = new Set(classes);
    return {
        value, textContent: '', disabled: false,
        classList: {
            add: name => tokens.add(name),
            remove: name => tokens.delete(name),
            contains: name => tokens.has(name),
        },
        set innerHTML(value) {
            htmlWrites++;
            throw new Error('Use textContent, not innerHTML');
        },
    };
}
const nodes = {
    configSaveStatus: node(),
    configSaveToast: node(), // A stale success toast must be hidden on failure.
    btnSaveConfig: node(),
    cfgMode: node('all'),
    cfgSttProvider: node('sarvam'),
    cfgSarvamApiKey: node(input.sarvamKey),
    cfgNvApiKey: node(input.nvidiaKey),
};
const context = vm.createContext({
    document: { getElementById: id => nodes[id] || null },
    console: Object.fromEntries(['log', 'error', 'warn', 'info', 'debug'].map(
        method => [method, (...args) => logs.push([method, ...args])]
    )),
    alert: message => logs.push(['alert', message]),
    logSimulatorMessage: message => logs.push(['simulator', message]),
    fetchStatus: async () => { refreshes++; },
    setTimeout: (callback, delay) => { timers.push(delay); return timers.length; },
    fetch: async (url, options) => {
        requests.push({url, method: options.method, body: JSON.parse(options.body)});
        pending = {
            disabled: nodes.btnSaveConfig.disabled,
            dimmed: nodes.btnSaveConfig.classList.contains('opacity-50'),
            status: nodes.configSaveStatus.textContent,
            keysRetained: nodes.cfgSarvamApiKey.value === input.sarvamKey &&
                nodes.cfgNvApiKey.value === input.nvidiaKey,
        };
        if (input.reject) throw new Error('Transport failed: ' + input.sarvamKey);
        return {
            ok: input.status >= 200 && input.status < 300,
            status: input.status,
            json: async () => input.data,
        };
    },
});
vm.runInContext(input.source, context, {timeout: 1000});
(async () => {
    if (input.quick) {
        await context.quickUpdateConfig({
            stt_provider: 'sarvam', sarvam_api_key: input.sarvamKey,
            nvidia_api_key: input.nvidiaKey,
        });
    } else {
        await context.saveConfig({preventDefault() { prevented = true; }});
    }
    const body = requests[0].body;
    process.stdout.write(JSON.stringify({
        status: nodes.configSaveStatus.textContent,
        toastHidden: nodes.configSaveToast.classList.contains('hidden'),
        disabled: nodes.btnSaveConfig.disabled,
        dimmed: nodes.btnSaveConfig.classList.contains('opacity-50'),
        keysCleared: nodes.cfgSarvamApiKey.value === '' && nodes.cfgNvApiKey.value === '',
        keysRetained: nodes.cfgSarvamApiKey.value === input.sarvamKey &&
            nodes.cfgNvApiKey.value === input.nvidiaKey,
        secretsSubmitted: body.sarvam_api_key === input.sarvamKey &&
            body.nvidia_api_key === input.nvidiaKey,
        requestCount: requests.length,
        url: requests[0].url,
        method: requests[0].method,
        pending, prevented, htmlWrites, refreshes, logs, timers,
    }));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""


@unittest.skipUnless(NODE, "Node.js is not available on PATH")
class DashboardUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        html = TEMPLATE.read_text(encoding="utf-8")
        match = re.search(
            r"async\s+function\s+quickUpdateConfig\s*\(.*?(?=//\s*Utility:\s*XSS)",
            html,
            re.DOTALL,
        )
        if match is None:
            raise AssertionError("Dashboard config function range not found")
        cls.source = match.group(0)
        if not re.search(r"async\s+function\s+saveConfig\s*\(", cls.source):
            raise AssertionError("saveConfig missing from extracted function range")

    def run_case(self, *, quick=False, status=200, data=None, reject=False):
        result = subprocess.run(
            [NODE, "-e", NODE_HARNESS],
            input=json.dumps({
                "source": self.source, "quick": quick, "status": status,
                "data": data, "reject": reject,
                "sarvamKey": SARVAM_KEY, "nvidiaKey": NVIDIA_KEY,
            }),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        state = json.loads(result.stdout)
        self.assertEqual(state["requestCount"], 1)
        self.assertEqual(state["url"], "/api/config")
        self.assertEqual(state["method"], "POST")
        self.assertTrue(state["secretsSubmitted"])
        self.assertFalse(state["disabled"])
        self.assertFalse(state["dimmed"])
        self.assertEqual(state["htmlWrites"], 0)
        self.assertTrue(state["pending"]["keysRetained"])
        self.assertIn("Saving", state["pending"]["status"])
        self.assertEqual(state["prevented"], not quick)
        if not quick:
            self.assertTrue(state["pending"]["disabled"])
            self.assertTrue(state["pending"]["dimmed"])
        rendered = json.dumps(state["logs"]) + state["status"] + result.stderr
        for secret in (SARVAM_KEY, NVIDIA_KEY):
            self.assertNotIn(secret, rendered)
        return state

    def assert_failed(self, state):
        self.assertTrue(state["status"])
        self.assertNotIn("Saved", state["status"])
        self.assertTrue(state["toastHidden"])
        self.assertTrue(state["keysRetained"])
        self.assertFalse(state["keysCleared"])
        self.assertEqual(state["refreshes"], 0)
        self.assertEqual(state["timers"], [])
        self.assertFalse(any(entry[0] == "simulator" for entry in state["logs"]))

    def test_confirmed_durable_success_and_key_clearing(self):
        for quick in (False, True):
            with self.subTest(quick=quick):
                state = self.run_case(quick=quick, data={"success": True, "persisted": True})
                self.assertIn("Saved", state["status"])
                self.assertIn("survives server restarts", state["status"])
                self.assertEqual(state["refreshes"], 1)
                self.assertEqual(state["keysCleared"], not quick)
                self.assertEqual(state["keysRetained"], quick)
                self.assertEqual(state["toastHidden"], quick)
                self.assertEqual(state["timers"], [] if quick else [4000])

    def test_http_503_detail_is_plain_text_not_success(self):
        detail = '<img src=x onerror="alert(1)"> Cannot save configuration.'
        for quick in (False, True):
            # Even affirmative body flags must not override a failed HTTP status.
            for flags in ({}, {"success": True, "persisted": True}):
                with self.subTest(quick=quick, flags=flags):
                    state = self.run_case(
                        quick=quick, status=503, data={"detail": detail, **flags},
                    )
                    self.assert_failed(state)
                    self.assertEqual(state["status"], detail)

    def test_http_200_requires_success_and_strict_persisted_true(self):
        for quick in (False, True):
            for data in (
                {"success": True},
                {"success": True, "persisted": False},
                {"success": True, "persisted": "true"},
                {"success": False, "persisted": True},
            ):
                with self.subTest(quick=quick, data=data):
                    state = self.run_case(quick=quick, data=data)
                    self.assert_failed(state)
                    self.assertIn("Failed to save configuration", state["status"])

    def test_thrown_fetch_shows_error_preserves_keys_and_resets_button(self):
        for quick in (False, True):
            with self.subTest(quick=quick):
                state = self.run_case(quick=quick, reject=True)
                self.assert_failed(state)
                self.assertIn("Failed to save configuration", state["status"])


if __name__ == "__main__":
    unittest.main()
