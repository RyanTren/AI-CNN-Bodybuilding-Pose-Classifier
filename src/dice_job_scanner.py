# https://www.dice.com/job-detail/50e92f08-bebb-4641-a726-1521a41eb8ba?searchlink=search%2F%3Fq%3Dsoftware%26radius%3D30%26radiusUnit%3Dmi%26page%3D1%26pageSize%3D20%26language%3Den%26eid%3D2649&searchId=63f96e51-17df-45dd-97ac-ad340f4098d1

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

            # Wait for the job description element to load
            page.wait_for_selector('div[data-testid="jobDescriptionHtml"]')
            
            # Extract the text from the element
            try: 
                job_description = page.query_selector('div[data-testid="jobDescriptionHtml"]').inner_text()
            except:
                job_description = "UNAVAILABLE_DESCRIPTION"
            #Get company
            try: 
                job_company = page.query_selector('a[data-cy="companyNameLink"]').inner_text()
            except:
                job_company = "UNAVAILABLE_COMPANY"
            # Get position title
            try: 
                job_title = page.query_selector('h1[data-cy="jobTitle"]').inner_text()
            except:
                job_title = "UNAVAILABLE_TITLE"
            


            print("\n")
            print(job_title)
            print("company: "+ str(job_company))
            print("-----------------------------------------------")
            print(job_description)
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