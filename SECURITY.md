# Security

nai-cli runs on your computer with your permissions and talks to NovelAI, or to a relay you configure, with your token. Anyone who can change your files, your environment or the config file can already use that token, so reports that depend on such access are not vulnerabilities in nai-cli.

What nai-cli protects:

- Tokens and access keys are sent only to the account's own `api_url` and `image_url`. They never appear in the output, in error messages or in the request body shown by `--show-body`.
- The config file and `accounts.json` are created readable only by you on macOS and Linux. Email logins keep the derived access key and the session token, never the password.
- Images and Vibe files from other people are read as data: the settings in them are imported only when you ask, priced before anything is sent, and never make nai-cli write outside the folders you chose or replace an existing file. A batch file is a list of commands, so run only those you have read, as with a shell script.

## Reporting a vulnerability

Report privately through [GitHub Security Advisories](https://github.com/obzawa/nai-cli/security/advisories/new) rather than in a public issue. Include the nai-cli version (`nai --version`), your operating system, the steps that reproduce the problem and what it lets someone do. Leave your token out of the report.
