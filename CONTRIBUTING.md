# Contributing

Bug reports, fixes and features are welcome. nai-cli aims to send the same requests as the official NovelAI web client, so a change to how requests are built, priced or read back needs the official client's code as its source, not images made with other clients.

## Reporting a bug

Open an issue with the bug template. Include `nai --version`, your operating system, the command and its JSON output. The output never holds your token, but remove it if you paste a config file. Security problems go to a private report instead; see [SECURITY.md](SECURITY.md).

## Changing the code

```sh
git clone https://github.com/obzawa/nai-cli.git
cd nai-cli
pip install -e ".[dev]"
pytest
ruff check
ruff format
```

Read [docs/architecture.md](docs/architecture.md) before changing the structure; it sets the layers, the owner of each rule and the checklist a change is reviewed against. [AGENTS.md](AGENTS.md) lists the project rules, for people and coding agents alike. Tests never contact NovelAI; use `--dry-run`, or the `with_token` fixture and the fake client in `tests/helpers.py`.

`tests/private/` may hold generation settings taken from real images, used to check that requests rebuilt from an image match its record. It is not under version control, and the tests that need it are skipped when it is absent.

Add user-visible changes under `## Unreleased` at the top of [CHANGELOG.md](CHANGELOG.md).
