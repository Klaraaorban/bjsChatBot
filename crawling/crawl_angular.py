from playwright.sync_api import sync_playwright

def extract_links(url):
    with sync_playwright() as s:
        browser = s.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")

        links = page.eval_on_selector_all(
            "a[href]", "elements => elements.map(e => e.href)")
        browser.close()
        return links
    
if __name__ == "__main__":
    baseurl = "https://refbejuso.tlex.ch/app/de/systematic/texts_of_law"
    all_links = extract_links(baseurl)
    for link in all_links:
        print(link)