# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project follows [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-09-19

### Added

- `--jobs N` keeps N LLM calls in flight. For Ollama, start the server with
  `OLLAMA_NUM_PARALLEL` set to match, or the extra calls only queue.
- The report is rewritten every 30 seconds during a scan and marked as
  partial until the scan completes.

### Changed

- Ollama models run with thinking turned off. On the bench this made
  qwen3:32b about 3x faster with better recall.
- The docs and example configs use the `ollama_chat/` model prefix, which
  sends the scanner instructions as a system message. `ollama/` still works.

### Fixed

- Ollama scans silently truncated long conversations: Ollama's default
  context window is as small as 4096 tokens. looselips now requests 16k.
  **Rerun Ollama scans made with earlier versions.**
- Single messages longer than a chunk, such as pasted documents, were sent
  whole and truncated. They are now split.
- Text of files attached to Claude messages was never scanned.
- Claude exports downloaded as separate zips (`conversations-000.zip`) were
  misread as ChatGPT exports.
- Newer thinking models (qwen3.5 and later) with the `ollama/` prefix
  returned empty output, so every conversation was recorded as an error.
- A scan that failed at the end, for example on a missing output directory,
  lost all its results.
- The report left out conversations whose LLM calls failed and counted them
  as clean. They are now listed, with an Errors count.

Cached `looselips-bench` results from earlier versions were produced with
thinking on and without the context fix. Use a fresh `--db` or `--force`.
