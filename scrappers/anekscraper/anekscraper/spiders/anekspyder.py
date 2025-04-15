import scrapy
from ..items import AnekItem


class AnekspyderSpider(scrapy.Spider):
    name = "anekspyder"
    allowed_domains = ["anekdotov.net"]
    # start_urls = ["https://anekdotov.net/anekdot/arc/180704.html"]
    start_urls = ["https://anekdotov.net/aforizm/today.html"]

    def parse(self, response):
        aneks = response.xpath('//div[@class="anekdot"]')
        next_page = response.xpath('//table[@class="pagenavibig"]//td[3]/a/attribute::href').get()
        for anek in aneks:
            anek_item = AnekItem()
            anek_item['date'] = next_page
            anek_item['anek'] = anek.get()
            yield anek_item
        
        if next_page is not None and next_page not in ('/aforizm/today.html'): 
            next_page = "https://anekdotov.net" + next_page
            yield response.follow(next_page, callback=self.parse)
