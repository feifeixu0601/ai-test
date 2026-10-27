"""
数据模型定义
定义爬取的名言数据包含哪些字段
"""

import scrapy


class QuoteItem(scrapy.Item):
    """名言数据模型"""
    text = scrapy.Field()      # 名言内容
    author = scrapy.Field()    # 作者
    tags = scrapy.Field()      # 标签列表