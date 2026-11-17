"""Running nai in-process against a fake NovelAI client, shared by the tests."""

import io
import json
from pathlib import Path

from PIL import Image
from samples import png_bytes

from nai_cli import api, cli, config, ledger


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = json.loads(capsys.readouterr().out)
    return code, out


def ledger_entry(file):
    return ledger.by_file().get(Path(file).resolve())


class FakeClient:
    def __init__(self, token, endpoint=api.OFFICIAL, before_send=None):
        self.endpoint = endpoint
        self.retries = []
        self.anlas = 1000
        self.calls = []
        self._before_send = before_send or (lambda: None)

    def _sent(self, call):
        self._before_send()
        self.calls.append(call)

    def account(self):
        self._sent("account")
        return api.Account(3, True, self.anlas, 0, 50.0, False, None)

    def generate(self, body):
        self._sent("generate")
        self.anlas -= 30
        return [b"\x89PNG\r\n\x1a\nimg"]

    def upscale(self, data):
        self._before_send()
        self.anlas -= 2
        with Image.open(io.BytesIO(data)) as img:
            return [png_bytes(img.width * 2, img.height * 2)]

    def augment(self, data, req_type, w, h, prompt, defry, expected=1):
        self._sent((req_type, w, h, prompt, defry))
        return [png_bytes(w, h)] * expected

    def encode_vibe(self, image, model, information_extracted=1.0, mask=None):
        self._sent("encode_vibe")
        self.anlas -= 2
        return "ENCODED"

    def suggest_tags(self, prompt, model, lexicon, lang):
        return [{"tag": f"{prompt} {model} {lexicon}", "count": 1}]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_defaults(text: str) -> None:
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8") + "[defaults]\n" + text, encoding="utf-8")


def set_cap(anlas: int) -> None:
    path = config.default_path()
    path.write_text(path.read_text(encoding="utf-8") + f"[cost]\nmax_per_request = {anlas}\n", encoding="utf-8")


def fake_clients(monkeypatch) -> list[FakeClient]:
    made: list[FakeClient] = []

    def client(token, endpoint=api.OFFICIAL, before_send=None):
        made.append(FakeClient(token, endpoint, before_send))
        return made[-1]

    monkeypatch.setattr(api, "Client", client)
    return made


def set_filename(template: str) -> None:
    path = config.default_path()
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("[output]\n", f"[output]\nfilename = {json.dumps(template)}\n", 1), encoding="utf-8")


def run_batch(capsys, *argv):
    code = cli.main(["batch", *argv])
    return code, [json.loads(line) for line in capsys.readouterr().out.splitlines()]
