import scrapy
from ..items import VseshutochkiItem


class ShutockiSpider(scrapy.Spider):
    name = "shutocki"
    allowed_domains = ["vse-shutochki.ru"]
    start_urls = ["https://vse-shutochki.ru/anekdoty"]

    def parse(self, response):
        aneks = response.xpath('//div[@class="post noSidePadding"]/*/p')
        next_page = response.xpath('//a[@class="btn btn-warning btn-large"]/attribute::href').get()
        for anek in aneks:
            anek_item = VseshutochkiItem()
            anek_item['anek'] = anek.get()
            yield anek_item
        
        if next_page is not None: 
        #     # next_page = "https://www.anekdot.ru" + next_page
        #     next_page = "https://v1.anekdot.ru/" + next_page
            yield response.follow(next_page, callback=self.parse)
