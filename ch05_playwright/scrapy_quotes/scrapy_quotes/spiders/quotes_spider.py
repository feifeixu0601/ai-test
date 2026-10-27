"""
名言爬虫 Spider
爬取 quotes.toscrape.com 的名人名言数据
"""

import scrapy
from scrapy_quotes.items import QuoteItem


class QuotesSpider(scrapy.Spider):
    # 爬虫名称（运行时用来指定跑哪个爬虫）
    name = "quotes"

    # 允许爬取的域名（防止爬到别的网站去）
    allowed_domains = ["quotes.toscrape.com"]

    # 起始 URL（从这里开始爬）
    start_urls = ["https://quotes.toscrape.com/"]

    def parse(self, response):
        """
        解析页面，提取数据和下一页链接
        Scrapy 会自动调用这个方法来处理每个响应
        """
        # 提取当前页所有名言
        quote_divs = response.css("div.quote")

        for div in quote_divs:
            item = QuoteItem()
            item["text"] = div.css("span.text::text").get()
            item["author"] = div.css("small.author::text").get()
            item["tags"] = div.css("a.tag::text").getall()
            yield item

        # 查找下一页并继续爬取
        next_page = response.css("li.next a::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)