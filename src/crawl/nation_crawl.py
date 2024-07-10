import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# 基础 URL
base_url = 'https://worldathletics.org/records/all-time-toplists/sprints/100-metres/all/men/senior'

# 请求头
request_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                  ' Chrome/91.0.4472.124 Safari/537.36'
}

all_data = []
max_pages = 10  # 设置为需要爬取的最大页数

for page in range(1, max_pages + 1):
    print(f'Fetching page {page}')

    params = {
        'regionType': 'world',
        'timing': 'electronic',
        'windReading': 'regular',
        'page': page,  # 直接传递整数页码
        'bestResultsOnly': 'true',
        'firstDay': '1900-01-01',
        'lastDay': '2024-07-10',
        'maxResultsByCountry': '5',
        'eventId': '10229630',
        'ageCategory': 'senior'
    }

    try:
        # 发送请求
        response = requests.get(base_url, headers=request_headers, params=params, timeout=10)

        # 检查响应状态
        response.raise_for_status()

        # 解析页面内容
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', class_='records-table')

        # 检查是否找到表格
        if not table:
            print(f'Could not find the results table on page {page}')
            continue

        # 提取表头
        if page == 1:
            table_headers = [header.text.strip() for header in table.find_all('th')]

        # 提取数据行
        rows = table.find_all('tr')[1:]  # 跳过表头行
        for row in rows:
            cols = [col.text.strip() for col in row.find_all('td')]
            all_data.append(cols)

        # 延时
        time.sleep(2)

    except requests.exceptions.RequestException as e:
        print(f'Failed to fetch page {page}. Error: {e}')
        continue
    except Exception as ex:
        print(f'An error occurred while processing page {page}. Error: {ex}')
        continue

# 创建 DataFrame
df = pd.DataFrame(all_data, columns=table_headers)

# 打印前几行数据
print(df)

# 保存到 CSV 文件
df.to_csv('../../database/crawl_data/nation5_data.csv', index=False)