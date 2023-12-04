import pandas as pd

# 创建示例数据
data = {
    'android_res_id': ["1001", "1002", "1003"],
    'iOS stringid': ['ios_1', 'ios_2', 'ios_3'],
    '中文': ['你好', '这是一个示例', '多语言文案'],
    '繁体中文': ['你好', '這是一個示例', '多語言文案'],
    '英语': ['Hello', 'This is an example', 'Multilingual text'],
    '日语': ['こんにちは', 'これは例です', '多言語テキスト'],
    '韩语': ['안녕하세요', '이것은 예제입니다', '다국어 텍스트'],
    '泰语': ['สวัสดี', 'นี่คือตัวอย่าง', 'ข้อความหลากภาษา']
}

# 将数据转换为DataFrame
# df = pd.DataFrame(data)

# 将DataFrame保存为CSV文件
# csv_file_path = 'data_1.csv'
# df.to_csv(csv_file_path, index=False)

csv_file_path = 'data_example.csv'
df = pd.read_csv(csv_file_path)
# df.set_index("android_res_id", drop=True)
# print(df.columns)
# print(df)

df.columns = ['android_res_id', '中文', '繁体中文','英语raw', '英语', '日语', '韩语', '泰语']
# df.reindex(['android_res_id'])


# 生成Android多语言文件 strings.xml
android_strings = df[['android_res_id', '中文', '繁体中文', '英语', '日语', '韩语', '泰语']]
# print(df)
# android_strings.set_index('android_res_id', inplace=True)
android_strings.columns = ['android_res_id','zh', 'zh-rTW', 'en', 'ja', 'ko', 'th']

langs = ['zh', 'zh-rTW', 'en', 'ja', 'ko', 'th']
for lang in langs:
    
    df_part = android_strings[['android_res_id', lang]]
    print(df_part)

    # android_strings_xml = df_part.apply(lambda row:  '\n'.join([f'    <string name="{index}">{value}</string>' for index, value in row.items()]), axis=1)
    android_strings_xml = df_part.apply(lambda row:  f'    <string name="{row["android_res_id"]}">{row[lang]}</string>', axis=1)

    print(f'android_strings_xml {android_strings_xml}')

    android_strings_xml_file = f'android_strings_{lang}.xml'
    with open(android_strings_xml_file, 'w', encoding='utf-8') as f:
        f.write('<resources>\n')
        f.write(android_strings_xml.str.cat(sep='\n'))
        f.write('\n</resources>')

    print(f'CSV文件已创建：{csv_file_path}')
    print(f'Android多语言文件已创建：{android_strings_xml_file}')
