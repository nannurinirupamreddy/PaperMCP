"""Exactly three tools, served over stdio."""
from functools import lru_cache
from fastmcp import FastMCP
from papermcp.core import Papers

mcp = FastMCP("PaperMCP")

@lru_cache(maxsize=1)
def service():
    return Papers()

@mcp.tool()
def search_papers(query: str, source: str = "arxiv", limit: int = 5) -> list[dict]:
    """Search arxiv or semantic_scholar; returns IDs for fetch_paper and ask_paper."""
    return service().search(query, source, limit)

@mcp.tool()
def fetch_paper(paper_id: str) -> dict:
    """Fetch metadata and text split by physical PDF page. Accept arXiv IDs/URLs or s2: IDs."""
    return service().fetch(paper_id)

@mcp.tool()
def ask_paper(paper_id: str, question: str, top_k: int = 5) -> dict:
    """Retrieve relevant paper excerpts with page citations; optionally synthesize with Gemini."""
    return service().ask(paper_id, question, top_k)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
