import scrapy
from ..items import AnekdotruItem

class AnekdotspiderSpider(scrapy.Spider):
    name = "anekdotspider"
    allowed_domains = ["anekdot.ru", "v1.anekdot.ru"]
    start_urls = ['https://v1.anekdot.ru/an/an0212/021230.html']

    def parse(self, response):
        aneks = response.xpath('//pre')
        # next_page = response.xpath('//div[@class="voteresult"]/a/attribute::href').get()
        next_page = response.xpath('//tr/*/*/a/attribute::href').get()
        for anek in aneks:
            anek_item = AnekdotruItem()
            anek_item['anek'] = anek.get()
            yield anek_item
        
        if next_page is not None: 
            # next_page = "https://www.anekdot.ru" + next_page
            next_page = "https://v1.anekdot.ru/" + next_page
            yield response.follow(next_page, callback=self.parse)

