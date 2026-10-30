"""Shared tools for the cybersecurity deep search agents.

Contains the trusted site list, the search tool (DuckDuckGo), the page
reader tool (HTML and PDF) with a per-run cache, and the system prompt.
Copied from github.com/GeneralFox/ai-experiment, where the same module
serves the deepagents, LangGraph, ADK and AG2 versions: keep the two copies
identical, so that the comparison uses the same tools and prompts.
"""

import io
import re

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pypdf import PdfReader

SITES = [
    "nist.gov",
    "cisa.gov",
    "attack.mitre.org",
    "isc.sans.edu",
    "bleepingcomputer.com",
    "thehackernews.com",
]

MAX_PAGE_CHARS = 8000
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) deep-search-experiment"


def search_sites(query: str, site: str = "") -> str:
    """Search the trusted cybersecurity sites for a query.

    Args:
        query: What to search for.
        site: Optional single domain (e.g. "cisa.gov") to restrict the search
            to one site. Leave empty to search across all trusted sites.

    Returns:
        A list of results with title, URL and snippet.
    """
    if site:
        q = f"{query} site:{site}"
    else:
        site_filter = " OR ".join(f"site:{s}" for s in SITES)
        q = f"{query} ({site_filter})"

    try:
        results = DDGS().text(q, max_results=10)
    except DDGSException:
        # ddgs raises "No results found." instead of returning an empty list
        results = []
    if not results:
        return "No results found. Try different keywords or a specific site."

    lines = []
    for r in results:
        lines.append(f"- {r['title']}\n  URL: {r['href']}\n  {r['body']}")
    return "\n".join(lines)


# download cache shared by all agents: a page is downloaded only once per
# process, even when several analysts read it
_download_cache: dict[str, str] = {}

ALREADY_READ = (
    "[NOTE: you already read this page earlier in this conversation and its "
    "text is above. Do not fetch it again: use what you have, fetch a "
    "different page, or write your answer.]"
)


def _html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _pdf_text(data: bytes) -> str:
    """Text of a PDF, page by page until MAX_PAGE_CHARS (advisories are often PDFs)."""
    parts, size = [], 0
    for page in PdfReader(io.BytesIO(data)).pages:
        text = page.extract_text() or ""
        parts.append(text)
        size += len(text)
        if size > MAX_PAGE_CHARS:
            break
    return "\n".join(parts)


def _download(url: str) -> str:
    """Download a trusted page or PDF and return its readable text (cached)."""
    if url in _download_cache:
        return _download_cache[url]

    host = re.sub(r"^https?://", "", url).split("/")[0].lower()
    if not any(host == s or host.endswith("." + s) for s in SITES):
        return f"Refused: {host} is not in the trusted site list {SITES}."

    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=20,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    try:
        # the content, not the URL, decides: a .pdf link can be an HTML page
        if content_type == "application/pdf" or resp.content.startswith(b"%PDF"):
            text = _pdf_text(resp.content)
        elif content_type in ("", "text/html", "application/xhtml+xml", "text/plain"):
            text = _html_text(resp.text)
        else:
            # a binary file would reach the model as garbage bytes
            return f"Refused: {url} is {content_type}, not an HTML page or a PDF."
    except Exception as e:
        return f"Error fetching {url}: could not extract the text ({type(e).__name__}: {e})"
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n[...truncated...]"
    text = text or "The page appears to be empty."
    _download_cache[url] = text
    return text


def make_fetch_page():
    """Return a fetch_page tool with its own memory of the pages already read.

    Give each agent its own instance: "already read" must mean read by *this*
    agent (parallel analysts would otherwise be told they saw pages they
    never received). A repeated fetch returns only a short note, not the page
    again: the text is already in the agent's context, and re-sending it made
    a looping model's context grow by a whole page per call.
    """
    read: set[str] = set()

    def fetch_page(url: str) -> str:
        """Download a web page and return its readable text content.

        Only URLs belonging to the trusted cybersecurity sites are allowed.

        Args:
            url: Full URL of the page to read.

        Returns:
            The main text of the page (truncated if very long).
        """
        if url in read:
            return ALREADY_READ
        text = _download(url)
        if not text.startswith(("Refused:", "Error fetching")):
            read.add(url)
        return text

    return fetch_page


# default instance for the single-agent examples
fetch_page = make_fetch_page()


SYSTEM_PROMPT = f"""You are a cybersecurity research analyst. You answer questions
by researching ONLY these trusted sources: {", ".join(SITES)}.

Workflow:
1. Plan your research with the todo list.
2. Use `search_sites` to find relevant pages (you can narrow to a single site).
3. Use `fetch_page` to read the 2-4 most promising results in depth.
4. If the first search is not enough, refine the query and search again.

Final answer format (in the same language as the user's question):
- A concise summary (a few paragraphs) synthesizing what you found.
- A "References" section listing every URL you actually used, one per line.

Never invent URLs or facts: only cite pages you fetched or search results you saw.
"""

# for the analysts of the multi-agent examples: the coordinator has already
# planned the work, the analyst only researches its brief (no own plan)
ANALYST_PROMPT = f"""You are a cybersecurity research analyst in a team. A coordinator
gives you a research brief: research it directly, the coordinator has already
planned the work. Use ONLY these trusted sources: {", ".join(SITES)}.

How to work:
- Use `search_sites` to find relevant pages (you can narrow to a single site).
- Use `fetch_page` to read the most promising results. You decide how many
  trusted pages to search and read (typically 2-4): stop as soon as you can
  answer the brief.
- If the first search is not enough, refine the query and search again.

Report format (in the same language as the brief):
- A concise summary (a few paragraphs) synthesizing what you found.
- A "References" section listing every URL you actually used, one per line.

Never invent URLs or facts: only cite pages you fetched or search results you saw.
"""
