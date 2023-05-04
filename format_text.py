# 打开文件
with open('./texts/jiadian_zhihu.txt', 'r') as f:
    # 读取文件内容
    lines = f.readlines()

print(lines[:4])

for line in lines:
    if line == "\n":
        # 写入文件，
        pass
    else:
        # 拼接
        new_line += line.strip()
