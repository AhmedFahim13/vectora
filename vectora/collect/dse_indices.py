"""DSE homepage index snapshot parser.

The homepage (verified 2026-07-12) shows current index values in
div.midrow blocks: m_col-1 = name (split by <font> tags), m_col-2 = value,
m_col-3 = signed change. Market totals follow in m_col-wid* divs.
Names normalize to bare index codes: "DSEX Index" -> "DSEX".
"""
import re

from bs4 import BeautifulSoup

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE


def fetch_homepage(session: PoliteSession) -> str:
    return session.get(f"{DSE_BASE}/")


def _clean_name(text: str) -> str:
    return re.sub(r"\s*Index\s*$", "", " ".join(text.split())).strip()


def _float(text: str) -> float | None:
    t = text.strip().replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def parse_homepage(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    indices = []
    totals = None
    for row in soup.find_all("div", class_="midrow"):
        name_div = row.find("div", class_="m_col-1")
        val_div = row.find("div", class_="m_col-2")
        chg_div = row.find("div", class_="m_col-3")
        if name_div and val_div:
            value = _float(val_div.get_text())
            change = _float(chg_div.get_text()) if chg_div else None
            name = _clean_name(name_div.get_text())
            if name and value is not None:
                indices.append({"index_name": name, "value": value, "change": change})
            continue
        # totals values row: three m_col-wid* divs holding plain numbers
        wid_divs = row.find_all("div", class_=re.compile(r"^m_col-wid"))
        if len(wid_divs) == 3 and totals is None:
            nums = [_float(d.get_text()) for d in wid_divs]
            if all(n is not None for n in nums):
                totals = {
                    "total_trades": int(nums[0]),
                    "total_volume": int(nums[1]),
                    "total_value_mn": nums[2],
                }
    return {"indices": indices, "totals": totals}
