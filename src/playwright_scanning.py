from playwright.sync_api import sync_playwright

def scrape_indeed_jobs():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set headless=False for debugging
        page = browser.new_page()
        
        url = "https://www.indeed.com/jobs?q=software+engineer&l="
        page.goto(url)
        
        # Wait for the page to load network requests fully
        page.wait_for_load_state("networkidle")
        
        # Optional: take a screenshot to inspect the page state
        page.screenshot(path="indeed_page.png")
        
        # Increase the timeout to 60 seconds for debugging purposes
        try:
            page.wait_for_selector("div.job_seen_beacon", timeout=60000)
        except Exception as e:
            print("Selector not found. Please check the page structure and update your selector.")
            browser.close()
            return
        
        # Once confirmed, query for job cards
        job_cards = page.query_selector_all("div.job_seen_beacon")
        
        for job_card in job_cards:
            # Extract the job title
            job_title_elem = job_card.query_selector("h2.jobTitle")
            job_title = job_title_elem.inner_text() if job_title_elem else "N/A"
            
            
            # Extract location
            location_elem = job_card.query_selector("div.company_location")
            location = location_elem.inner_text().strip() if location_elem else "N/A"
            
            print("Job Title:\n", job_title)
            print("Company & Location:\n", location)
            print("-" * 80)
        
        browser.close()

if __name__ == '__main__':
    scrape_indeed_jobs()
