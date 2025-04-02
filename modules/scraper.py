from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

def extract_text_from_website(url):
    """Scrapes all visible text from a given website."""
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    service = Service()  
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        time.sleep(3)  
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        page_text = driver.find_element(By.TAG_NAME, "body").text

    except Exception as e:
        page_text = f"Error fetching website data: {e}"

    finally:
        driver.quit()

    return page_text
