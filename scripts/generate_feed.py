#!/usr/bin/env python3
import re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, register_namespace

SITE = "https://hessammehr.github.io"
POSTS = f"{SITE}/blog/posts"
NS = "http://www.w3.org/2005/Atom"
ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RELATIVE_URL = re.compile(r'((?:src|href)=")(?!https?://|/)')


def md_to_html(text):
    text = ANSI.sub("", text)
    html = subprocess.run(
        ["pandoc", "--to=html", "--no-highlight"],
        input=text, capture_output=True, text=True, check=True
    ).stdout
    return RELATIVE_URL.sub(rf"\1{POSTS}/", html).replace("/posts/../", "/")


def parse_post(path):
    text = path.read_text()
    title = m.group(1) if (m := re.search(r"^# (.+)", text, re.MULTILINE)) else path.stem
    return {
        "title": title,
        "date": datetime.fromisoformat(path.name[:10]).replace(tzinfo=timezone.utc),
        "link": f"{POSTS}/{path.stem}.html",
        "content": md_to_html(text),
    }


def build_feed(posts):
    register_namespace("", NS)
    n = f"{{{NS}}}"
    feed = Element(f"{n}feed", {"xml:lang": "en-us"})
    SubElement(feed, f"{n}title").text = "Hessam's blog"
    SubElement(feed, f"{n}link", {"href": f"{SITE}/blog/", "rel": "alternate"})
    SubElement(feed, f"{n}link", {"href": f"{SITE}/feed.xml", "rel": "self"})
    SubElement(feed, f"{n}id").text = f"{SITE}/"
    SubElement(SubElement(feed, f"{n}author"), f"{n}name").text = "Hessam Mehr"
    SubElement(feed, f"{n}updated").text = posts[0]["date"].isoformat(timespec="seconds")
    for p in posts:
        e = SubElement(feed, f"{n}entry")
        SubElement(e, f"{n}title").text = p["title"]
        SubElement(e, f"{n}link", {"href": p["link"], "rel": "alternate"})
        SubElement(e, f"{n}id").text = p["link"]
        SubElement(e, f"{n}published").text = p["date"].isoformat(timespec="seconds")
        SubElement(e, f"{n}updated").text = p["date"].isoformat(timespec="seconds")
        SubElement(e, f"{n}content", {"type": "html"}).text = p["content"]
    return feed


posts = sorted([parse_post(Path(p)) for p in sys.argv[1:]], key=lambda p: p["date"], reverse=True)
ElementTree(build_feed(posts)).write(sys.stdout.buffer, encoding="utf-8", xml_declaration=True)
