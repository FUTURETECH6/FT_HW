import scrapy
import re
from scrapy import Selector

class wdzjSpider(scrapy.Spider):
    name = "wdzj"
    allowed_domains = ["wdzj.com"]
    start_urls = [
        "https://www.wdzj.com/bank/",
    ]

    def parse(self, response):
        # filename = response.url.split("/")[-2] + '.html'
        selector = Selector(text=response.body)

        productName = selector.xpath('//div[@class="bank-col bank-title"]/text()').getall()
        raw_interestRate = selector.xpath('//div[@class="bank-col w100"]//div[@class="bank-value color-red"]/text()').getall()
        term = selector.xpath('//div[@class="tip-box"]//span/text()').getall()

        print(type(raw_interestRate[0]))
        for i in range(len(raw_interestRate)):
            raw_interestRate[i] = re.sub("\n\s+", "", raw_interestRate[i])
        interestRate = []
        for i in raw_interestRate:
            if i != '':
                interestRate.append(i)

        print(productName)
        print(interestRate)
        print(term)

        with open('wdzj_bank.html', 'wb') as f:
            f.write(response.body)
        # for quote in response.css('div.quote'):
        for i in range(len(term)):
            yield {
                'productName': productName[i],
                'interestRate': interestRate[i],
                'term': term[i],
            }