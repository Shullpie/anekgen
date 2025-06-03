# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import re
import w3lib.html
from itemadapter import ItemAdapter


class VseshutochkiPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        for field_name in adapter.field_names():
            if field_name == 'anek':
                value = adapter.get(field_name)
                value = w3lib.html.remove_tags(value).replace('\n', '')
                adapter[field_name] = re.sub(r'\s+', ' ', value).strip().lower()
        
        
        return item
