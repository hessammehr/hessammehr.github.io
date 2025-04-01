#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "scholarly",
#     "requests",
# ]
# ///

import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import sys

ORCID_ID = "0000-0001-7710-3102"
GOOGLE_SCHOLAR_URL = "https://scholar.google.com/citations?hl=en&user=HeyhCHEAAAAJ"
ORCID_API_URL = f"https://pub.orcid.org/v3.0/{ORCID_ID}/record"

HEADERS_ORCID = {
    'Accept': 'application/vnd.orcid+xml'
}
HEADERS_SCHOLAR = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_orcid_works_count_api(orcid_id):
    url = f"https://pub.orcid.org/v3.0/{orcid_id}/record"
    response = requests.get(url, headers=HEADERS_ORCID, timeout=15)
    response.raise_for_status()
    namespaces = {
        'activities': 'http://www.orcid.org/ns/activities',
        'record': 'http://www.orcid.org/ns/record'
    }
    root = ET.fromstring(response.text)
    works_groups = root.findall('./activities:activities-summary/activities:works/activities:group', namespaces)
    if works_groups is not None:
        return len(works_groups)
    return None

def get_scholar_metrics_scrape(url):
    response = requests.get(url, headers=HEADERS_SCHOLAR, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    stats_table = soup.find('table', {'id': 'gsc_rsb_st'})
    if not stats_table:
        return None, None
    
    all_value_cells = stats_table.find_all('td', class_='gsc_rsb_std')
    if len(all_value_cells) < 3:
        return None, None

    try:
        citations = int(all_value_cells[0].text)
        h_index = int(all_value_cells[2].text)
        return citations, h_index
    except (ValueError, IndexError):
        return None, None


works_count = get_orcid_works_count_api(ORCID_ID)
citations, h_index = get_scholar_metrics_scrape(GOOGLE_SCHOLAR_URL)

if works_count is None or citations is None or h_index is None:
    sys.exit(1)

works_str = str(works_count)
h_index_str = str(h_index)
citations_str = str(citations)

content = sys.stdin.read()

content = content.replace("{{WORKS_COUNT}}", works_str)
content = content.replace("{{H_INDEX}}", h_index_str)
content = content.replace("{{CITATIONS}}", citations_str)

print(content)