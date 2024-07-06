import configparser

import pymysql
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np
import datetime


dbconfig = {}

def get_tag_text_with_spaces(tag):
    text_parts = []
    for child in tag:
        if isinstance(child, str):
            text_parts.append(child.strip())
        elif child.name == 'br':
            text_parts.append(' ')
        else:
            text_parts.append(get_tag_text_with_spaces(child))
    return ' '.join(text_parts)


def getNum(str_array):
    int_array = [int(num) for num in str_array]
    return int_array


def get_lotto_results(year,month):
    base_url = 'https://www.lotteryextreme.com/newzealand/lotto-results'
    results = []
    d = {'tryb': 'rokmsc', '_month': str(month), '_year': str(year)}
    ret = requests.post(base_url, data=d)
    text=str(ret.content.decode('utf-8'))
    dateList = []
    NoList = []
    soup = BeautifulSoup(text,"html.parser")
    titles = soup.find_all('td', {'class': 'lotterygame2'})
    date_format = "%d %b %Y %A"

    # 使用datetime.strptime()方法解析日期字符串


    for title in titles:
        data0 = title.text
        # 正则表达式模式
        date_pattern = r'\d{2} \w{3} \d{4} \w+'
        sequence_pattern = r'\(Draw (\d+)\)'
        print(data0)
        # 提取日期
        date_match = re.search(date_pattern, data0)
        date = date_match.group() if date_match else None

        # 提取序号
        sequence_match = re.search(sequence_pattern, data0)
        sequence = sequence_match.group(1) if sequence_match else None
        if date is not None and sequence is not None:
            print("日期:", date)
            date_obj = datetime.datetime.strptime(date, date_format)
            dateList.append(date_obj)
            print("序号:", sequence)
            NoList.append(int(sequence))



    if len(dateList) == 0:
        return None
    containers = soup.find_all('td', {'class': 'res'})
    dataList = []
    index = 0
    for container in containers:
        results = container.find_all('table', {'class': 'results'})
        result0=results[0]
        # 提取所有<td>节点中的数字
        paragraph = get_tag_text_with_spaces(result0.find_all('td'))
        paragraphs=paragraph.split(" ")
        data0=paragraphs[:6]
        print("段落内容:", paragraph)
        numbers = [td.text for td in result0.find_all('td')]
        result1=results[1]
        data1 = result1.text
        result2=results[2]
        data2 = result2.text
        result3=results[3]
        paragraph = get_tag_text_with_spaces(result3.find_all('td'))
        paragraphs = paragraph.split(" ")
        data3 = paragraphs[:4]
        if len(data2) == 0:
            data2 = ['0']
        if len(data3) < 4:
            data3=['0','0','0','0']
        print(f'data0 {data0}  data1 {data1} data2 {data2} data3 {data3}')
        dataItem = {"date":dateList[index],"no":NoList[index],"data0": getNum(data0), "data1":[data1], "data2": [data2], "data3": getNum(data3)}
        dataList.append(dataItem)
        index=index+1

        print(data3)
    return dataList

def readDBConfig():
    config = configparser.ConfigParser()
    # 读取 INI 文件
    config.read("config.ini")
    # 获取配置值
    ip = config.get("database", "ip")
    port = int(config.get("database", "port"))
    user = config.get("database", "user")
    pwd = config.get("database", "password")
    database = config.get("database", "db")
    charset = config.get("database", "charset")
    dbconfig["ip"] = ip
    dbconfig["port"] = port
    dbconfig["user"] = user
    dbconfig["password"] = pwd
    dbconfig["db"] = database
    dbconfig["charset"] = charset
    return dbconfig
def openDB():
    # 创建配置解析器对象
    config = dbconfig
    if len(dbconfig) > 0:
        config =dbconfig
    else:
        config = readDBConfig()
    # 获取配置值
    ip = config["ip"]
    port = config["port"]
    user = config["user"]
    pwd = config["password"]
    database = config["db"]
    charset = config["charset"]
    db = None
    try:

        db = pymysql.connect(
            host=ip,
            port=port,
            user=user,
            password=pwd,
            database=database,
            charset=charset,
            connect_timeout=30,
        )
    except Exception as e:
        print(f"insertRecord An exception occurred: {e}")
    return db

def insertRecord(data):
        dataItem = (
            data['no'],
            data['date'],
            data['data0'][0],
            data['data0'][1],
            data['data0'][2],
            data['data0'][3],
            data['data0'][4],
            data['data0'][5],
            data['data1'][0],
            data['data2'][0],
            data['data3'][0],
            data['data3'][1],
            data['data3'][2],
            data['data3'][3],
        )
        connect = openDB()
        cursor = connect.cursor()
        try:
            sql = """insert into lotto(id,createtime,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
            cursor.execute(sql, dataItem)
            connect.commit()
        except Exception as e:
            print(f"insertRecord An exception occurred: {e}")
            connect.rollback()
        connect.close()
def queryDB(sql):
        connect = openDB()
        cursor = connect.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        connect.close()
        return result

def get_lotto_from_db(sql):
    resultDevice = queryDB(sql)
    return resultDevice


if __name__ == '__main__':
    allData=[]
    for year in range(2010,2025):
        for month in range(1,13):
            items=get_lotto_results(year,month)
            if items is not None:
                allData.extend(items)
    for data in allData:
        insertRecord(data)
    print(allData)

