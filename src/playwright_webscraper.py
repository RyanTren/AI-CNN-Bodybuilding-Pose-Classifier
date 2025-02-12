from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables at the start
load_dotenv()

class JobScraper:
    def __init__(self):
        self.jobs = []
        
    def login_to_linkedin(self, page, email, password):
        try:
            # Go to LinkedIn login page
            page.goto("https://www.linkedin.com/login")
            
            # Fill in credentials
            page.fill("#username", email)
            page.fill("#password", password)
            
            # Click sign in button
            page.click(".btn__primary--large")
            
            # Wait for navigation
            page.wait_for_load_state("networkidle")
            print("Successfully logged in")
            
        except Exception as e:
            print(f"Login failed: {e}")

    def scrape_linkedin(self, search_term="software engineer", location="United States", email=None, password=None):
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=False)
                context = browser.new_context(
                    viewport={'width': 1080, 'height': 720},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                
                page.set_default_timeout(60000)
                page.set_default_navigation_timeout(60000)
                
                if email and password:
                    self.login_to_linkedin(page, email, password)
                    # Add delay after login
                    time.sleep(5)
                
                url = f"https://www.linkedin.com/jobs/search?keywords={search_term}&location={location}"
                print(f"Navigating to: {url}")
                page.goto(url)
                
                print("Waiting for results to load...")
                # Updated selectors based on current LinkedIn structure
                selectors = [
                    ".jobs-search__results-list",
                    ".scaffold-layout__list-container",
                    ".jobs-search-results-list",
                    ".jobs-search-results__list",
                    ".jobs-search-two-pane__results",
                    "ul.jobs-search__results-list",
                    "[data-test-id='job-card']",
                    ".job-search-card"
                ]
                
                selector_found = False
                for selector in selectors:
                    try:
                        print(f"Trying selector: {selector}")
                        page.wait_for_selector(selector, timeout=10000)
                        print(f"Found selector: {selector}")
                        selector_found = True
                        break
                    except Exception as e:
                        print(f"Selector {selector} not found: {str(e)}")
                        continue
                
                if not selector_found:
                    print("Could not find job listings. Taking screenshot for debugging...")
                    page.screenshot(path="debug_screenshot.png")
                    # Print page content for debugging
                    print("\nPage HTML:")
                    print(page.content())
                    return
                
                # Scroll multiple times with longer pauses
                for i in range(5):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(3)
                    print(f"Scroll {i+1}/5 complete")
                
                # Try different job card selectors
                job_cards = page.query_selector_all(".job-card-container, .jobs-search-results__list-item")
                print(f"Found {len(job_cards)} job cards")
                
                for card in job_cards:
                    try:
                        # Updated selectors with multiple fallbacks
                        title = card.query_selector('.job-card-list__title, .job-card-container__link')
                        company = card.query_selector('.job-card-container__company-name, .job-card-container__primary-description')
                        location = card.query_selector('.job-card-container__metadata-item, .job-card-container__secondary-description')
                        link = card.query_selector('a')
                        
                        if all([title, company, location, link]):
                            job = {
                                'title': title.inner_text().strip(),
                                'company': company.inner_text().strip(),
                                'location': location.inner_text().strip(),
                                'link': link.get_attribute('href')
                            }
                            self.jobs.append(job)
                            print(f"Scraped job: {job['title']}")
                    except Exception as e:
                        print(f"Error scraping job card: {e}")
            
            except Exception as e:
                print(f"Error during scraping: {e}")
                # Take screenshot on error
                try:
                    page.screenshot(path="error_screenshot.png")
                    print("Error screenshot saved as error_screenshot.png")
                except:
                    pass
            finally:
                browser.close()
            
    def save_jobs(self, filename="src/data/jobs.json"):
        # Create the data directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.jobs, f, indent=2)

if __name__ == "__main__":
    EMAIL = os.getenv("LINKEDIN_EMAIL")
    PASSWORD = os.getenv("LINKEDIN_PASSWORD")
    
    if not EMAIL or not PASSWORD:
        print("Please set LINKEDIN_EMAIL and LINKEDIN_PASSWORD in .env file")
        exit(1)
    
    scraper = JobScraper()
    print("Starting LinkedIn scraper...")
    scraper.scrape_linkedin(email=EMAIL, password=PASSWORD)
    print(f"Scraped {len(scraper.jobs)} jobs")
    scraper.save_jobs()
    print("Jobs saved to src/data/jobs.json")
