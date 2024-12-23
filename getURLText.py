import requests
from bs4 import BeautifulSoup #import to better read html sites
import pandas as pd

csv_path = 'your_csv_path'
reports = pd.read_csv(csv_path)

def get_full_text_from_url(url, ticker, date):
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
       
        full_text = soup.get_text(separator="\n", strip=True)
        
        #Observed the first 370 lines are always meaningless
        lines = full_text.split("\n")[370:]
    
        cleaner_text = "\n".join(lines)

        cleaned_text = cleaner_text.split("Stocks Mentioned")[0]

        file_path = f"file_path{ticker}_{date}.txt"

   
        with open(file_path, 'w') as file:
            file.write(cleaned_text)
        print("*")
        return f"output_{ticker}_{date}.txt"
    else:
        print(f"Failed to retrieve content: {response.status_code}")
        return None

reports["Report_Text"] = reports.apply(
    lambda row: get_full_text_from_url(row["URL"], row["Ticker"], row["Date"]), axis=1
)
path = 'your_path'
reports.to_csv(path, index = False)
print(reports)