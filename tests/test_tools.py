import pytest

from cs_researcher.tools import PageFetcher, PageReader, PageUnavailable, TrustedSites

SITES = TrustedSites(["cisa.gov", "attack.mitre.org"])


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://www.cisa.gov/news", True),
        ("https://cisa.gov", True),
        ("https://attack.mitre.org:443/techniques/T1190/", True),
        ("https://mitre.org/", False),
        ("https://evilcisa.gov/", False),
        ("https://cisa.gov.evil.com/", False),
    ],
)
def test_trusted_sites(url, allowed):
    assert SITES.allows(url) is allowed


class FakeReader(PageReader):
    def __init__(self):
        super().__init__(SITES, max_chars=100, timeout=1)
        self.downloads = 0

    def _download(self, url):
        if not self.sites.allows(url):
            raise PageUnavailable(f"Refused: {url}")
        self.downloads += 1
        return "page text"


def test_reader_downloads_each_page_once():
    reader = FakeReader()
    assert reader.read("https://cisa.gov/a") == reader.read("https://cisa.gov/a") == "page text"
    assert reader.downloads == 1


def test_untrusted_page_is_refused_without_network():
    reader = PageReader(SITES, max_chars=100, timeout=1)
    with pytest.raises(PageUnavailable, match="Refused"):
        reader.read("https://example.com/")


def test_fetcher_remembers_pages_per_agent():
    reader = FakeReader()
    first, second = PageFetcher(reader), PageFetcher(reader)
    assert first.fetch_page("https://cisa.gov/a") == "page text"
    assert first.fetch_page("https://cisa.gov/a") == PageFetcher.ALREADY_READ
    assert second.fetch_page("https://cisa.gov/a") == "page text"  # another agent: not read yet


def test_refused_page_is_not_remembered():
    fetcher = PageFetcher(FakeReader())
    assert fetcher.fetch_page("https://example.com/").startswith("Refused")
    assert fetcher.fetch_page("https://example.com/").startswith("Refused")


def test_tools_keep_their_names_and_arguments():
    from cs_researcher.tools import SiteSearch

    search = SiteSearch(SITES, max_results=5).as_tool()
    fetch = PageFetcher(FakeReader()).as_tool()
    assert (search.name, fetch.name) == ("search_sites", "fetch_page")
    assert set(search.args_schema.model_json_schema()["properties"]) == {"query", "site"}
