# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

from twisted.enterprise import adbapi
import MySQLdb
import MySQLdb.cursors
import codecs
import json
from logging import log

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

class WdzjPipeline(object):

    def __init__(self):
        self.dbpool = adbapi.ConnectionPool('MySQLdb',
                                            host='127.0.0.1',
                                            db='WDZJ_DB',
                                            user='root',
                                            passwd='',
                                            cursorclass=MySQLdb.cursors.DictCursor,
                                            charset='utf8',
                                            use_unicode=False)

    def process_item(self, item, spider):
        query = self.dbpool.runInteraction(self._conditional_insert, item)
        query.addErrback(self._handle_error, item, spider)
        return item

    def _conditional_insert(self, tx, item):
        sql = "insert into product values(%s, %s, %s)"
        params = (item["productName"], item["interestRate"], item["term"])
        print(item["productName"], item["interestRate"], item["term"])
        tx.execute(sql, params)

    def _handle_error(self, failue, item, spider):
        print("ERROR")