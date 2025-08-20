from playwright.sync_api import sync_playwright
import time
timer = time.time()
def extract_links(url):
    with sync_playwright() as s:
        browser = s.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")

        links = page.eval_on_selector_all(
            "a[href]", "elements => elements.map(e => e.href)")
        browser.close()

        exclude = ["/fr", "/search", "/overview", "sitrox", "chronology"]
        links = [link for link in links if not any(sub in link for sub in exclude)]

        return links
    
def extract_content(links):
    """Visiting each link and extracting content"""
    results = {}
    with sync_playwright() as s:
        browser = s.chromium.launch(headless=False)
        page = browser.new_page()
        for link in links:
            print(f"Visiting {link}")
            try:
                page.goto(link, timeout=60000)
                page.wait_for_load_state("networkidle")

                text = page.inner_text("main")
                results[link] = text[:1500]
            except Exception as e:
                print(f"Failed to scrape {link}: {e}")
        browser.close()
    return results

if __name__ == "__main__":
    baseurl = "https://refbejuso.tlex.ch/app/de/systematic/texts_of_law"
    all_links = extract_links(baseurl)
    # for link in all_links:
    #     print(link)

    all_content = extract_content(all_links)
    print("Content preview: ")
    for url, text in list(all_content.items())[5:10]:
        print(f"\nURL: {url}\n{text}...\n")

    end_timer = time.time()
    runtime = end_timer - timer
    print(f"Request processed in {runtime:.2f} seconds")