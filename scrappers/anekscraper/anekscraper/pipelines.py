# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import re
import w3lib.html
from itemadapter import ItemAdapter

class AnekscraperPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        for field_name in adapter.field_names():

            if field_name == 'date':
                value = adapter.get(field_name)
                adapter[field_name] = value.rsplit('/', maxsplit=1)[1].split('.')[0]
                
            # remove HTML tags, replace multiple spaces and switch to lowercase
            elif field_name == 'anek':
                value = adapter.get(field_name)
                # value = w3lib.html.remove_tags(value).replace('\n', '')
                adapter[field_name] = re.sub(r'\s+', ' ', value).strip()
        
        
        return item
