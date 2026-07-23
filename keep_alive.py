"""
Keeps a Streamlit Community Cloud app awake by loading it in a real
headless browser session and clicking the "wake up" button if asleep.
"""

import os
import time
import sys

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

STREAMLIT_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://diabetes-risk-screener-isp9hnylrpdggpwdedxqfu.streamlit.app/",
)


def main():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        print(f"Opening {STREAMLIT_URL}")
        driver.get(STREAMLIT_URL)
        time.sleep(5)

        try:
            wake_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(., 'get this app back up')]")
                )
            )
            print("App is asleep. Clicking wake-up button...")
            wake_button.click()
            time.sleep(20)
            print("App should be waking up now.")
        except TimeoutException:
            print("No wake-up button found — app is already awake.")

        time.sleep(10)
        print("Done. Page title:", driver.title)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
