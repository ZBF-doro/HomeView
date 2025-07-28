import csv
import json
import re
import os
import sys
import requests
from pymysql import *
from utils.query import querys

def init():
    if not os.path.exists("./hourseInfoData.csv"):
        with open("./hourseInfoData.csv","w",encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'title',
                'cover',
                'city',
                'region',
                'address',
                'room_desc',
                'area_range',
                'all_ready',
                'price',
                'hourseDecoration',
                'company',
                'hourseType',
                'on_time',
                'open_date',
                'tags',
                'totalPrice_range',
                'sale_status',
                'detail_url',
            ])

def writerRow(row):
    with open("./hourseInfoData.csv", "a", encoding="utf-8", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def get_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0'
    }
    response = requests.get(url, headers)
    if response.status_code == 200:
        return response.json()['data']['list']
    else:
        return None

def parse_data(hourseDataList,city,url):
    for hourseInfo in hourseDataList:
        print(hourseInfo)
        try:
            title = hourseInfo['title']
            cover = hourseInfo['cover_pic']
            region = hourseInfo['district']
            address = hourseInfo['address']
            room_desc = json.dumps(hourseInfo['frame_rooms_desc'].replace('居', '').split('/'))
            area_range = json.dumps(hourseInfo['resblock_frame_area_range'].replace('㎡', '').split('-'))
            all_ready = hourseInfo['permit_all_ready']
            price = hourseInfo['average_price']
            hourseDecoration = hourseInfo['decoration']
            company = hourseInfo['developer_company'][0]
            hourseType = hourseInfo['house_type']
            on_time = hourseInfo['on_time']
            open_date = hourseInfo['open_date']
            tags = json.dumps(hourseInfo['tags'])
            totalPrice_range = json.dumps(hourseInfo['reference_total_price'].split('-'))
            sale_status = hourseInfo['process_status']
            detail_url = url

            writerRow([
                title,
                cover,
                city,
                region,
                address,
                room_desc,
                area_range,
                all_ready,
                price,
                hourseDecoration,
                company,
                hourseType,
                on_time,
                open_date,
                tags,
                totalPrice_range,
                sale_status,
                detail_url,
            ])
        except:
            continue

def save_to_sql():
    with open('./hourseInfoData.csv', 'r', encoding='utf-8') as reader:
        readerCsv = csv.reader(reader)
        next(readerCsv)
        for h in readerCsv:
            querys('''
                  insert into hourse_info(title,cover,city,region,address,room_desc,area_range,all_ready,price,hourseDecoration,company,hourseType,on_time,open_date,tags,totalPrice_range,sale_status,detail_url)
                  values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ''',[
                h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7],h[8],h[9],h[10],h[11],h[12],h[13],h[14],h[15],
                h[16],h[17]
            ])

def main():
    init()
    with open("./cityData.csv", "r", encoding="utf-8") as readerFile:
        reader = csv.reader(readerFile)
        next(reader)
        for city in reader:
            try:
                for page in range(1,50):
                        url = 'https:' + re.sub('pg1','pg' + str(page), city[1])
                        print('正在爬取 %s 城市的房屋数据正在第 %s 页 路径为：%s' % (
                            city[0],
                            page,
                            url
                        ))
                        hourseDatailList = get_data(url)
                        parse_data(hourseDatailList,city[0],url)
            except:
                continue

if __name__ == '__main__':
    # main()
    save_to_sql()