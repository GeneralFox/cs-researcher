"""Research tools: web search and page reading, restricted to trusted sites."""

import io
import re
from collections.abc import Iterable

import httpx
from bs4 import BeautifulSoup
from crewai.tools import BaseTool, tool
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pypdf import PdfReader

from . import __version__

USER_AGENT = f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) cs-researcher/{__version__}"


class TrustedSites:
    """The domains the agents may search and read, subdomains included."""

    def __init__(self, domains: Iterable[str]):
        self.domains = tuple(d.lower() for d in domains)

    def allows(self, url: str) -> bool:
        host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0].lower()
        return any(host == d or host.endswith("." + d) for d in self.domains)

    def search_filter(self) -> str:
        return " OR ".join(f"site:{d}" for d in self.domains)

    def __str__(self) -> str:
        return ", ".join(self.domains)


class SiteSearch:
    """DuckDuckGo search on the trusted sites."""

    def __init__(self, sites: TrustedSites, max_results: int):
        self.sites = sites
        self.max_results = max_results

    def search_sites(self, query: str, site: str = "") -> str:
        """Search the trusted cybersecurity sites for a query.

        Args:
            query: What to search for.
            site: Optional single domain (e.g. "cisa.gov") to restrict the search
                to one site. Leave empty to search across all trusted sites.

        Returns:
            A list of results with title, URL and snippet.
        """
        q = f"{query} site:{site}" if site else f"{query} ({self.sites.search_filter()})"
        try:
            results = DDGS().text(q, max_results=self.max_results)
        except DDGSException:
            # ddgs raises "No results found." instead of returning an empty list
            results = []
        if not results:
            return "No results found. Try different keywords or a specific site."
        return "\n".join(f"- {r['title']}\n  URL: {r['href']}\n  {r['body']}" for r in results)

    def as_tool(self) -> BaseTool:
        return tool("search_sites")(self.search_sites)


class PageUnavailable(Exception):
    """A page that was refused or could not be read: the message goes to the agent."""


class PageReader:
    """Downloads trusted pages and PDFs as plain text.

    One instance serves all the agents: its cache downloads a page only once,
    even when several analysts read it.
    """

    def __init__(self, sites: TrustedSites, max_chars: int, timeout: float):
        self.sites = sites
        self.max_chars = max_chars
        self.timeout = timeout
        self._cache: dict[str, str] = {}

    def read(self, url: str) -> str:
        """The readable text of a trusted page; raises PageUnavailable."""
        if url not in self._cache:
            self._cache[url] = self._download(url)
        return self._cache[url]

    def _download(self, url: str) -> str:
        if not self.sites.allows(url):
            raise PageUnavailable(f"Refused: {url} is not on the trusted sites ({self.sites}).")
        try:
            resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=self.timeout, follow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            raise PageUnavailable(f"Error fetching {url}: {e}") from e

        content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        try:
            # the content, not the URL, decides: a .pdf link can be an HTML page
            if content_type == "application/pdf" or resp.content.startswith(b"%PDF"):
                text = self._pdf_text(resp.content)
            elif content_type in ("", "text/html", "application/xhtml+xml", "text/plain"):
                text = self._html_text(resp.text)
            else:
                # a binary file would reach the model as garbage bytes
                raise PageUnavailable(f"Refused: {url} is {content_type}, not an HTML page or a PDF.")
        except PageUnavailable:
            raise
        except Exception as e:
            raise PageUnavailable(f"Error fetching {url}: could not extract the text ({type(e).__name__}: {e})") from e

        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n[...truncated...]"
        return text or "The page appears to be empty."

    @staticmethod
    def _html_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    def _pdf_text(self, data: bytes) -> str:
        """Text of a PDF, page by page until max_chars (advisories are often PDFs)."""
        parts, size = [], 0
        for page in PdfReader(io.BytesIO(data)).pages:
            text = page.extract_text() or ""
            parts.append(text)
            size += len(text)
            if size > self.max_chars:
                break
        return "\n".join(parts)


class PageFetcher:
    """The fetch_page tool of one agent, with its memory of the pages read.

    Give each agent its own instance: "already read" must mean read by *this*
    agent (parallel analysts would otherwise be told they saw pages they
    never received). A repeated fetch returns only a short note, not the page
    again: the text is already in the agent's context, and re-sending it made
    a looping model's context grow by a whole page per call.
    """

    ALREADY_READ = (
        "[NOTE: you already read this page earlier in this conversation and its "
        "text is above. Do not fetch it again: use what you have, fetch a "
        "different page, or write your answer.]"
    )

    def __init__(self, reader: PageReader):
        self.reader = reader
        self._read: set[str] = set()

    def fetch_page(self, url: str) -> str:
        """Download a web page and return its readable text content.

        Only URLs belonging to the trusted cybersecurity sites are allowed.

        Args:
            url: Full URL of the page to read.

        Returns:
            The main text of the page (truncated if very long).
        """
        if url in self._read:
            return self.ALREADY_READ
        try:
            text = self.reader.read(url)
        except PageUnavailable as e:
            return str(e)
        self._read.add(url)
        return text

    def as_tool(self) -> BaseTool:
        return tool("fetch_page")(self.fetch_page)
