# -*- coding:utf-8 -*-

"""
@author: hanjunming
@time: 2018/11/30 11:36
"""
import configparser
import os
from configparser import ConfigParser

import pymysql

from DBUtils.PooledDB import PooledDB


class Mysql:
    """
    MYSQL 数据库对象，负责产生连接，此类中的连接采用连接池实现
    获取连接对象: conn = Mysql.getconn()
    释放连接对象: conn.close() 或 del conn
    """

    def __init__(self, config_file):
        """
        数据库构造函数，从连接池中去除连接，并生成操作游标
        """
        # 连接池对象
        self.__pool = None
        self._conn = self._getConn(self._get_config(config_file))
        self._cursor = self._conn.cursor()

    def _getConn(self, config):
        """
        @summary: 静态方法，从连接池中取出连接
        :return: Mysql.connetion
        """
        if self.__pool is None:
            self.__pool = PooledDB(
                creator=pymysql,
                mincached=config['db_min_cached'],
                maxcached=config['db_max_cached'],
                maxshared=config['db_max_shared'],
                maxconnections=config['db_max_connecyions'],
                blocking=config['db_blocking'],
                maxusage=config['db_max_usage'],
                host=config['db_host'],
                port=config['db_port'],
                user=config['db_user'],
                passwd=config['db_pwd'],
                db=config['db_name'],
                use_unicode=False,
                charset=config['db_encoding'],
                cursorclass=pymysql.cursors.DictCursor
            )

        return self.__pool.connection()

        # 读取配置文件
    def _get_config(self, config_file, encoding='utf-8'):
        parser = ConfigParser()
        parser.read(config_file, encoding=encoding)

        sections = parser.sections()

        results = []

        for section in sections:
            if section == 'strings':
                _conf = [(key, str(value)) for key, value in parser.items(section)]
            elif section == 'ints':
                _conf = [(key, int(value)) for key, value in parser.items(section)]
            elif section == 'floats':
                _conf = [(key, float(value)) for key, value in parser.items(section)]
            elif section == 'bools':
                _conf = [(key, bool(value)) for key, value in parser.items(section)]

            results.extend(_conf)

        return dict(results)

    def getAll(self, sql, param=None):
        """
        @summary: 执行查询， 并取出所有的结果集
        :param sql: 查询sql，如果有查询条件，请只指定条件列表，并将条件使用参数[param]传递进来
        :param param: 可选参数，条件列表值(元组/列表)
        :return: list/bool 查询到的结果集
        """
        count = self._cursor.execute(sql) if param is None else self._cursor.execute(sql, param)

        result = self._cursor.fetchall() if count > 0 else False

        return result

    def getOne(self, sql, param=None):
        """
        @summary: 执行查询，并取出第一条
        :param sql: 查询sql，如果有查询条件，请将条件值使用参数[param]传递进来
        :param param: 可选参数，条件列表值(元组/列表)
        :return: list/bool 查询到的结果集
        """
        count = self._cursor.execute(sql) if param is None else self._cursor.execute(sql, param)

        result = self._cursor.fetchone() if count and count > 0 else False

        return result

    def getMany(self, sql, num, param=None):
        """
        @summary: 执行查询，并取出num条结果
        :param sql: 查询sql，如果有查询条件，请只指定条件列表，并将条件值使用参数[param]传递进来
        :param num: 取得的结果条数
        :param param: 可选参数，条件列表值(元组/列表)
        :return: list/bool 查询到的结果集
        """
        count = self._cursor.execute(sql) if param is None else self._cursor.execute(sql, param)

        result = self._cursor.fetchmany(num) if count > 0 else False

        return result

    def insertOne(self, sql, value=None):
        """
        @summary: 向数据表中插入一条记录
        :param sql: 要插入的sql
        :param value: 要插入的记录数据tuple/list
        :return: insertId 受影响的行数
        """
        self._cursor.execute(sql) if value is None else self._cursor.execute(sql, value)

        self.end()

        return self.__getInsertId()

    def insertMany(self, sql, values):
        """
        @summary: 向数据表中插入多条记录
        :param sql: 要插入的sql
        :param values: 要插入的记录数据tuple/list
        :return: count 受影响的行数
        """
        count = self._cursor.executemany(sql, values)

        self.end(option='commit')

        return count

    def __getInsertId(self):
        """

        :return: 当前连接最后一次插入操作生成的id，如果没有则为0
        """
        self._cursor.execute(
            """
            SELECT @@IDENTITY AS id
            """
        )

        result = self._cursor.fetchall()

        return result[0]['id']

    def __query(self, sql, param=None):
        count = self._cursor.execute(sql) if param is None else self._cursor.execute(sql, param)

        return count

    def update(self, sql, param=None):
        """
        @summary: 删除数据表记录
        :param sql: sql格式及条件，使用(%s, %s)
        :param param: 要更新的值tuple/list
        :return: count 受影响的行数
        """
        return self.__query(sql, param)

    def delete(self, sql, param=None):
        """
        @summary: 删除数据表记录
        :param sql: sql格式及条件，使用(%s, %s)
        :param param: 要删除的条件值tuple/list
        :return: count 受影响的行数
        """
        return self.__query(sql, param)

    def end(self, option='commit'):
        self._conn.commit() if option == 'commit' else self._conn.rollback()

    def dispose(self, isEnd=1):
        """
        @summary: 释放连接池资源
        :param isEnd:
        :return:
        """
        self.end('commit') if isEnd >= 1 else self.end('rollback')

        self._cursor.close()
        self._conn.close()
