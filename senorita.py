import webbrowser

def open_google():
    url = "https://www.facebook.com"
    # Specify the path to Chrome executable if it's not in the default location
    chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
    webbrowser.get(chrome_path).open(url)

if __name__ == "__main__":
    open_google()
