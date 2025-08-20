import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

visited = set()

def crawl(url, base_url):
    if url in visited:
        return []
    try:
        r = requests.get(url, verify=False)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to access {url}:{e}")
        return []
    
    soup = BeautifulSoup(r.text, 'html.parser')
    text = soup.get_text(separator="\n")

    print(f"\nVisiting: {url}")
    print(f"Content snippet: {text[:200].replace(chr(10), ' ')}...")

    links = [urljoin(base_url, a.get('href')) for a in soup.find_all('a', href=True)]
    links = [l for l in links if l.startswith(base_url)]

    results = [(url, text)]
    for link in links:
        results.extend(crawl(link, base_url))

    return results

base_url = "https://refbejuso.tlex.ch/app/de/systematic/texts_of_law"
pages = crawl(base_url, base_url)