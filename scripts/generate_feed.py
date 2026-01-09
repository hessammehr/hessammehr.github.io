#!/usr/bin/env python3

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, register_namespace

from bs4 import BeautifulSoup

POSTS_DIR = Path(".build/blog/posts")
SITE_URL = "https://hessammehr.github.io"
POSTS_URL = f"{SITE_URL}/blog/posts"
NS = "http://www.w3.org/2005/Atom"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def atom_date(value):
    return value.isoformat(timespec="seconds")


def title_for(md_path):
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            candidate = line[2:].strip()
            if candidate:
                return candidate
            break
    return md_path.stem


def safe_html(text):
    soup = BeautifulSoup(text, "html.parser")
    fragment = soup.select_one("div.markdown-body") or soup.body or soup
    fragment_html = fragment.decode_contents() if hasattr(fragment, "decode_contents") else str(fragment)
    clean = BeautifulSoup(fragment_html, "html.parser")
    for tag in clean.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in {"href", "src", "alt", "title"}}
    fragment = ANSI_RE.sub("", str(clean))
    return "".join(
        ch
        for ch in fragment
        if ch in "\t\n\r" or 0x20 <= ord(ch) <= 0xD7FF or 0xE000 <= ord(ch) <= 0x10FFFF
    )


def posts_from(paths):
    posts = []
    for path in paths:
        html = Path(path)
        posts.append(
            {
                "date": datetime.fromisoformat(html.name[:10]).replace(
                    tzinfo=timezone.utc
                ),
                "name": html.name,
                "title": title_for(POSTS_DIR / html.with_suffix(".md").name),
                "summary": safe_html(html.read_text(encoding="utf-8")),
            }
        )
    return sorted(posts, key=lambda post: post["date"], reverse=True)


def build_feed(posts):
    register_namespace("", NS)
    ns = f"{{{NS}}}"
    feed = Element(f"{ns}feed", {"xml:lang": "en-us"})
    SubElement(feed, f"{ns}title").text = "Hessam's blog"
    SubElement(feed, f"{ns}link", {"href": f"{SITE_URL}/blog/", "rel": "alternate"})
    SubElement(feed, f"{ns}link", {"href": f"{SITE_URL}/feed.xml", "rel": "self"})
    SubElement(feed, f"{ns}id").text = f"{SITE_URL}/"
    SubElement(SubElement(feed, f"{ns}author"), f"{ns}name").text = "Hessam Mehr"
    SubElement(feed, f"{ns}updated").text = atom_date(
        posts[0]["date"] if posts else datetime.now(timezone.utc)
    )
    for post in posts:
        link = f"{POSTS_URL}/{post['name']}"
        entry = SubElement(feed, f"{ns}entry")
        SubElement(entry, f"{ns}title").text = post["title"]
        SubElement(entry, f"{ns}link", {"href": link, "rel": "alternate"})
        SubElement(entry, f"{ns}id").text = link
        SubElement(entry, f"{ns}published").text = atom_date(post["date"])
        SubElement(entry, f"{ns}updated").text = atom_date(post["date"])
        SubElement(entry, f"{ns}summary", {"type": "html"}).text = post["summary"]
    return feed


def main():
    posts = posts_from(sys.argv[1:])
    ElementTree(build_feed(posts)).write(
        sys.stdout.buffer, encoding="utf-8", xml_declaration=True
    )


if __name__ == "__main__":
    main()
