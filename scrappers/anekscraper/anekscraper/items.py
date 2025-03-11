# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
from scrapy.item import Item, Field

# def serialize_anek(value):
#     value = w3lib.html.remove_tags(value.get()).replace('\n', '')
#     return re.sub(r'\s+', ' ', value).strip()

class AnekItem(Item):
    date = Field()
    anek = Field()
