# AI-Based-Fake-Job-Posting-Detector

this is a rough draft...

### Problem Definition
* Many job seekers fall for fraudulent job listings, leading to wasted time, financial loss, and identity theft.
* Fake job postings use deceptive language, request upfront payments, or mimic legitimate companies.
* Current methods rely on manual reporting, which is slow and inefficient.

### Background & Motivation
* Job seekers (especially new graduates) struggle to identify fake listings.
* Employers face reputation damage from fraudulent postings under their name.
* AI-based classification could provide real-time fraud detection.

### Literature Review
* Previous research on spam detection (email phishing, social media scams).
* Studies on adversarial attacks in NLP to bypass AI-based security systems.
* Common linguistic markers of fraud (e.g., excessive urgency, vague job descriptions).

### Proposed Approach
* Data Collection: Scrape job postings from LinkedIn, Indeed, and Glassdoor, labeling them as real or fake (manually or via crowdsourcing).

Playwright/BeautifulSoup for Webscraping - Colin 

* Feature Extraction: Identify key factors:
* Text-based: Keywords like "quick money," "no experience needed," "startup fee."
* Structural: Unusual formatting, missing company details.
* Behavioral: Employer response times, email domains.
* AI Model: Train a supervised learning classifier (Random Forest, BERT, or LSTM) to predict fraud likelihood.
* Adversarial ML: Simulate how fraudsters might alter postings to bypass detection.
* Evaluation Metrics: Accuracy, precision-recall, F1-score.

### Implementation
* Tech Stack: Python, Playwright, TensorFlow/PyTorch, BeautifulSoup/Scrapy for web scraping.
* Dataset: Kaggle’s fake job posting dataset + newly collected listings.
* Training & Testing: Split into 80/20 for model validation.

### Expected Outcomes
* A web-based API or browser extension that flags suspicious job postings.
* Dashboard for recruiters to verify job authenticity.
* Research contribution on adversarially resilient fraud detection
