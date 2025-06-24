# =============================================================
# Python libraries
# =============================================================
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import undetected_chromedriver as uc

# USE CASE: Open an event the moment it appears on BookMyShow

driver = uc.Chrome()
url = "https://in.bookmyshow.com/events/coldplay-music-of-the-spheres-world-tour/ET00419733"
driver.get(url)

try:
    while True:  # Keep trying until the "Book" button appears
        try:
            # Set a short timeout to check for the "Book" button
            WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Book')]"))
            ).click()
            print("Clicked on 'Book' button")
            break  # Exit the loop once the button is clicked

        except Exception:
            print("Book button not found. Refreshing the page...")
            driver.refresh()  # Refresh the page

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    time.sleep(5)  # no quit

