# PaperMCP

MCP server with three tools for research papers from arXiv and Semantic Scholar. SQLite caches search results and extracted papers; persistent Chroma retrieval returns excerpts with physical PDF page citations.

| Tool | Purpose |
| --- | --- |
| `search_papers(query, source="arxiv", limit=5)` | Search `arxiv` or `semantic_scholar`; returns paper IDs and metadata. |
| `fetch_paper(paper_id)` | Fetch metadata and extract text page by page from an arXiv PDF. |
| `ask_paper(paper_id, question, top_k=5)` | Retrieve relevant excerpts with page links; optionally synthesize an answer using Gemini. |

## Run

Requires Python 3.11 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
papermcp
```

The server uses stdio: configure your MCP client to launch the absolute path to `.venv/bin/papermcp`. For example, clients supporting `mcpServers` configuration can use:

```json
{"mcpServers":{"papers":{"command":"/absolute/path/to/papermcp/.venv/bin/papermcp"}}}
```

Search first, then pass a returned ID to the other tools. Accepted IDs include `arxiv:2301.12345`, bare arXiv IDs, arXiv abstract/PDF URLs, and `s2:<40-character-paper-ID>`.

No API key is required for excerpt answers. Chroma's default local embedding model is downloaded on first use, requiring network access and disk space. Search and PDF fetching also require network access. Semantic Scholar may rate-limit unauthenticated requests. Optionally export `SEMANTIC_SCHOLAR_API_KEY`.

For Gemini synthesis, install `pip install -e '.[gemini]'`, then export `GEMINI_API_KEY`. `PAPERMCP_GEMINI_MODEL` selects the model (default `gemini-2.5-flash`). Questions and retrieved excerpts are sent to Google when enabled. Otherwise `ask_paper` returns cited excerpts, not a synthesized answer. `.env` files are not loaded automatically.

`PAPERMCP_CACHE_DIR` controls storage (default `.paper-cache` relative to the working directory). Search entries expire after one day; papers after 30 days. Delete the cache directory while the server is stopped to reset it.

Only arXiv-hosted PDFs are downloaded. Semantic Scholar results without an arXiv ID return metadata only. Scanned PDFs need external OCR. Download limits are 30 MiB and 500 pages. Citations refer to physical PDF pages, which can differ from printed page numbers. Gemini answers are generated text; the returned excerpts let callers verify the citations.

## Development

```sh
pip install -e '.[dev]'
pytest -q
```

Tests use mocked HTTP and retrieval, exercise MCP tool discovery, and run in GitHub Actions on Python 3.11–3.13. They do not require API keys or a model download.

Keep keys in environment variables, never in source. `.env` and caches are ignored. Before publishing, scan the actual Git history as well as the working tree; this source scaffold alone cannot certify a remote repository's history.
