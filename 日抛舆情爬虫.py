import re
import time

import requests
import execjs
import json
import csv

headers = {
    "authority": "edith.xiaohongshu.com",
    "accept": "application/json, text/plain, */*",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "content-type": "application/json;charset=UTF-8",
    "origin": "https://www.xiaohongshu.com",
    "referer": "https://www.xiaohongshu.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188"
}

cookies = {#cookies要根据具体情况更换，此处仅为示例。
     "sec_poison_id": "ee8f4d7d-fafd-402f-a17b-bc9b59749e22",
     "gid": "yYSKYWyfyW34yYSKYWySKI870KAK6uj7Y90KWKhCldW9E7286MdiWf888qW4W2j8qifSf884",
     "a1": "18c6871c6a6q6mz9u8ta6xoe7twmsef7in7ud0u6v50000374759",
     "websectiga": "82e85efc5500b609ac1166aaf086ff8aa4261153a448ef0be5b17417e4512f28",
     "webId": "a0f12a8433f12419df766d791ed3137e",
     "web_session": "0400698c10b7e9565466c8eeb3374b2105d31a",
     "xsecappid": "xhs-pc-web",
     "webBuild": "3.18.3"
}

js = execjs.compile(open(r'info.js', 'r', encoding='utf-8').read())

# 向csv文件写入表头  评论数据csv文件
header = ["用户ID", "用户名", "头像链接", "评论时间", "IP属地", "点赞数量", "评论内容"]
f = open(f"小红书评论.csv", "w", encoding="utf-8-sig", newline="")
writer = csv.DictWriter(f, header)
writer.writeheader()

comment_count = 0


# 发送爬取请求
def spider(url):
    response = requests.get(url, headers=headers, cookies=cookies)
    response.encoding = 'utf-8'
    return response.json()


# 时间戳转换成日期
def get_time(ctime):
    timeArray = time.localtime(int(ctime / 1000))
    otherStyleTime = time.strftime("%Y.%m.%d", timeArray)
    return str(otherStyleTime)


# 保存评论
def sava_data(comment):
    try:
        ip_location = comment['ip_location']
    except:
        ip_location = '未知'

    data_dict = {
        "用户ID": comment['user_info']['user_id'].strip(),
        "用户名": comment['user_info']['nickname'].strip(),
        "头像链接": comment['user_info']['image'].strip(),
        "评论时间": get_time(comment['create_time']),
        "IP属地": ip_location,
        "点赞数量": comment['like_count'],
        "评论内容": comment['content'].strip().replace('\n', '')
    }

    # 评论数量+1
    global comment_count
    comment_count += 1

    print(f"当前评论数: {comment_count}\n",
          f"用户ID：{data_dict['用户ID']}\n",
          f"用户名：{data_dict['用户名']}\n",
          f"头像链接：{data_dict['头像链接']}\n",
          f"评论时间：{data_dict['评论时间']}\n",
          f"IP属地：{data_dict['IP属地']}\n",
          f"点赞数量：{data_dict['点赞数量']}\n",
          f"评论内容：{data_dict['评论内容']}\n"
          )
    writer.writerow(data_dict)


# 爬取根评论
def get_comments(note_id):
    cursor = ''
    page = 0
    while True:
        url = f'https://edith.xiaohongshu.com/api/sns/web/v2/comment/page?note_id={note_id}&cursor={cursor}&top_comment_id=&image_scenes=FD_WM_WEBP,CRD_WM_WEBP'
        # 爬一次就睡1秒
        time.sleep(1)
        json_data = spider(url)

        for comment in json_data['data']['comments']:

            sava_data(comment)

        if not json_data['data']['has_more']:
            break

        # 下一页评论的地址
        cursor = json_data['data']['cursor']

        # 每爬完一页，页数加1
        page = page + 1
        print('================爬取Page{}完毕================'.format(page))


def keyword_search(keyword):
    api = '/api/sns/web/v1/search/notes'

    search_url = "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes"

    # 排序方式 general: 综合排序 popularity_descending: 热门排序 time_descending: 最新排序
    data = {
        "image_scenes": "FD_PRV_WEBP,FD_WM_WEBP",
        "keyword": "",
        "note_type": "0",
        "page": "",
        "page_size": "20",
        "search_id": "2c7hu5b3kzoivkh848hp0",
        "sort": "general"
    }

    data = json.dumps(data, separators=(',', ':'))
    data = re.sub(r'"keyword":".*?"', f'"keyword":"{keyword}"', data)

    page_count = 20  # 爬取的页数, 一页有 20 条笔记，这里设置爬20页仅为示例，实际要根据所需的评论数量来指定页数。
    for page in range(1, page_count):
        data = re.sub(r'"page":".*?"', f'"page":"{page}"', data)

        ret = js.call('get_xs', api, data, cookies['a1'])
        headers['x-s'], headers['x-t'] = ret['X-s'], str(ret['X-t'])

        response = requests.post(search_url, headers=headers, cookies=cookies, data=data.encode('utf-8'))
        json_data = response.json()
        try:
            notes = json_data['data']['items']
        except:
            print('================爬取完毕================'.format(page))
            break

        for note in notes:
            note_id = note['id']
            if len(note_id) != 24:
                continue

            try:
                note_title = note['note_card']['display_title']
            except:
                note_title = ''
            print(f"================正在爬取笔记标题：{note_title}的评论================\n")

            time.sleep(1)

            get_comments(note_id)


def main():
    keyword = '一次性'  # 搜索的关键词

    keyword_search(keyword)


if __name__ == "__main__":
    main()





