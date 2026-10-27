"""
数据处理管道
接收 Spider 提取的数据，负责清洗和存储
"""

import json


class JsonExportPipeline:
    """将数据导出为 JSON 文件的管道"""

    def open_spider(self, spider):
        """爬虫启动时调用，打开文件"""
        self.file = open("quotes_scrapy.json", "w", encoding="utf-8")
        self.data = []

    def close_spider(self, spider):
        """爬虫结束时调用，保存并关闭文件"""
        json.dump(self.data, self.file, ensure_ascii=False, indent=2)
        self.file.close()
        spider.logger.info(f"数据已保存，共 {len(self.data)} 条")

    def process_item(self, item, spider):
        """
        每提取到一条数据就会调用一次
        可以在这里做数据清洗、验证、存储
        """
        # 简单的数据清洗
        cleaned_item = {
            "text": item.get("text", "").strip(),
            "author": item.get("author", "未知").strip(),
            "tags": item.get("tags", [])
        }

        # 如果数据不合法就丢弃
        if not cleaned_item["text"]:
            spider.logger.warning(f"丢弃无效数据：{cleaned_item}")
            return None    # 返回 None 表示丢弃这条数据

        self.data.append(cleaned_item)
        return cleaned_item