import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

base_url = 'https://worldathletics.org/records/all-time-toplists/sprints/100-metres/all/men/senior'

request_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

all_data = []
max_pages = 500  

for page in range(1, max_pages + 1):
    print(f'Fetching page {page}')

    params = {
        'regionType': 'world',
        'timing': 'electronic',
        'windReading': 'regular',
        'page': page,  
        'bestResultsOnly': 'false',
        'firstDay': '1977-01-01',
        'lastDay': '2024-07-10',
        'maxResultsByCountry': 'all',
        'eventId': '10229630',
        'ageCategory': 'senior'
    }

    try:
        response = requests.get(base_url, headers=request_headers, params=params, timeout=10)

        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', class_='records-table')

        if not table:
            print(f'Could not find the results table on page {page}')
            continue

        if page == 1:
            table_headers = [header.text.strip() for header in table.find_all('th')]

        rows = table.find_all('tr')[1:] 
        for row in rows:
            cols = [col.text.strip() for col in row.find_all('td')]
            all_data.append(cols)

        time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f'Failed to fetch page {page}. Error: {e}')
        continue
    except Exception as ex:
        print(f'An error occurred while processing page {page}. Error: {ex}')
        continue

df = pd.DataFrame(all_data, columns=table_headers)

print(df.head())

df.to_csv('1977_2024.csv', index=False)


