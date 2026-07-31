from src.features.openaq_client import OpenAQClient
from src.features.aqi_calculator import add_aqi_from_pm25

c = OpenAQClient()
cities = ['Lahore', 'Karachi', 'Islamabad', 'Sukkur']

for city in cities:
    df = c.fetch_city_historical(city, '2026-07-01', '2026-07-07')
    if df.empty:
        print(f'{city}: NO DATA')
        continue
    df = add_aqi_from_pm25(df)
    aqi_min = df['aqi'].min()
    aqi_max = df['aqi'].max()
    print(f'{city}: {len(df)} rows, AQI range {aqi_min:.0f}-{aqi_max:.0f}')