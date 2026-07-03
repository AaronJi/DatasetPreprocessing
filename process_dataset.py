import numpy as np
import pandas as pd
from datetime import datetime, timezone
import secrets
import string
import random

INDUSTRY_CODE  = 'DATA'

data_path = "D:/Data/low-resource_language"
data_name = "flickr30k_romanian.tsv"
output_data_name = "flickr30k_romanian.jsonl"

def read_tsv_pandas(file_path):
    """read tsv by pandas"""
    df = pd.read_csv(file_path, sep='\t')
    return df

def build_data_content(row):
    # media_type: text, image, video, audio, maybe others
    image_data = {
        'media_type': 'image',
        'content': row['image_id']
    }
    # text
    text_data = {
        'media_type': 'text',
        'instruction': 'Please translate the English caption into Romanian',
        'content': row['english_description']
    }
    return [text_data, image_data]

def build_annotation(row):
    # annotation_method: 人工标注, 自动标注, 半自动标注
    # annotator: 普通标注员, 专业标注员, 行业领域专家
    return {
            'label': row['romanian_description'],
            'annotation_method': '人工标注',
            'annotator': '行业领域专家'
        }


def generate_check_digit(original_number: str) -> int:
    """
    根据Luhn算法为原始数字生成1位校验码。

    Args:
        original_number: 不含校验码的原始数字字符串。

    Returns:
        计算出的校验码 (0-9)。
    """
    # 1. 从右向左遍历，从倒数第一位（索引-1）开始，步长为2
    #    (即从右边数第1, 3, 5...位)
    total = 0
    # 为了便于理解，先反转字符串，从左向右处理
    # 反转后，原数字的个位在索引0，十位在索引1...
    reversed_digits = original_number[::-1]

    for i, char in enumerate(reversed_digits):
        digit = int(char)
        # 2. 偶数索引（0, 2, 4...）对应原数字从右数的第1, 3, 5...位，需要翻倍
        if i % 2 == 0:
            doubled = digit * 2
            # 3. 如果翻倍结果 >= 10，则将其各位相加（即减9）
            if doubled >= 10:
                doubled = doubled - 9  # 等价于 1 + (doubled - 10)
            total += doubled
        else:
            total += digit

    # 4. 计算校验码: (10 - total % 10) % 10
    check_digit = (10 - (total % 10)) % 10
    return check_digit

def generate_ids(data_len):
    # 获取当前UTC时间，并格式化为14位字符串
    TIME_PART  = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    # 生成8位随机码（大小写字母+数字）
    #random_part = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    #check_number = generate_check_digit(time_part)
    #unique_id = f"{INDUSTRY_CODE}{time_part}{random_part}{check_number}"
    #print(unique_id)

    #random_digits = np.random.randint(0, 10 ** 8, size=data_len).astype(str).str.zfill(8)
    random_digits = [f"{random.randint(0, 10 ** 8 - 1):08d}" for _ in range(data_len)]

    # 构造每行的基础数字串：时间戳 + 行索引(补6位) + 随机8位数字
    # 行索引确保即使同一秒生成多条记录也不冲突（实际也可用随机数完全替代）
    row_indices = [f"{i:06d}" for i in range(data_len)]
    # 组合：时间戳(14) + 行索引(6) + 随机数字(8) = 28 位纯数字
    base_digits = [TIME_PART + row_idx + rand_digit for row_idx, rand_digit in zip(row_indices, random_digits)]

    # 计算校验码（apply 逐行）
    check_digits = [generate_check_digit(d) for d in base_digits]

    return [INDUSTRY_CODE + base + str(check) for base, check in zip(base_digits, check_digits)]

if __name__ == '__main__':
    data_file = data_path + "/" + data_name
    output_data_file = data_path + "/" + output_data_name

    data_df = read_tsv_pandas(data_file)
    #print(data_df.shape)
    #print(data_df.columns)
    #print(data_df.head())

    print('Processing!!!')

    #data_df['id'] = data_df.index
    data_df.insert(1, 'id', 0)
    data_df.insert(2, 'rid', np.nan)
    data_df.insert(3, 'data_content', 0)
    data_df.insert(4, 'annotation', 0)
    data_df.insert(5, 'original_time', "2023-3-3")
    data_df.insert(6, 'last_modified_time', "2023-3-29")
    data_df.insert(7, 'version', "1.0.0-alpha")
    data_df.insert(8, 'license', "开源")
    data_df.insert(9, 'source', "论文")
    data_df.insert(10, 'source_details', "来自于论文《Parameter Efficient Multimodal Instruction Tuning for Romanian Vision Language Models》，数据地址：https://github.com/dima331453/Flickr30k-Romanian/")
    data_df.insert(11, 'generated_data_indicator', 0)

    # ---------- 4. 为每行生成唯一 ID（含校验码） ----------
    # 用 numpy 生成每行不同的 8 位随机数字（可含前导零）
    np.random.seed(42)

    # 组合完整 ID：行业代码 + 基础数字串 + 校验码
    data_df['id'] = generate_ids(len(data_df))

    data_df['rid'] = data_df['rid'].apply(lambda x: [])

    data_df['data_content'] = data_df.apply(build_data_content, axis=1)
    data_df['annotation'] = data_df.apply(build_annotation, axis=1)

    data_df.drop('index', axis=1, inplace=True, errors='ignore')
    data_df.drop('image_id', axis=1, inplace=True, errors='ignore')
    data_df.drop('english_description', axis=1, inplace=True, errors='ignore')
    data_df.drop('romanian_description', axis=1, inplace=True, errors='ignore')

    print(data_df.shape)
    print(data_df.columns)
    print(data_df.head())
    print(data_df.iloc[0])
    print(data_df.iloc[0]['data_content'])
    print(data_df.iloc[0]['annotation'])

    data_df.to_json(output_data_file, orient='records', lines=True, force_ascii=False, index=False)
