from playwright.sync_api import sync_playwright

def extract_elements(url):
    try:
        with sync_playwright() as p:

            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            page.goto(url)

            # Wait for the page to load network requests fully
            page.wait_for_load_state("networkidle")

            title_elem = page.query_selector('h2[data-testid="simpler-jobTitle"]')
            title = title_elem.inner_text()
            description = (page.locator("#jobDescriptionText")).inner_text()

            print("\n")
            print(title)
            print("-----------------------------------------------")
            print(description)
            # ---------------------
            context.close()
            browser.close()
    except:
        print("Page error, please try another page")
        

if __name__ == "__main__":
    while(True):
        url = input("Enter job URL: ")
        extract_elements(url)
        #test with: https://www.indeed.com/viewjob?jk=b9a21b341a84ad53&tk=1ijtuerlnl53487a&from=serp&vjs=3