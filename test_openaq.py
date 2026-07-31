from src.features.openaq_client import OpenAQClient


client = OpenAQClient()

print("=" * 70)
print("OPENAQ PAKISTAN LOCATIONS")
print("=" * 70)

df = client.search_locations(
    country="PK",
    limit=100,
)

print()
print(df.to_string(index=False))
print()
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())