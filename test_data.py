import json

file_path = './data/psg/psg_all_test.json'

with open(file_path, encoding='utf-8') as f:
    result = json.load(f)


'''result_val['data'] = result_all['data']

temp = json.dumps(result_val)
f2 = open('psg_all_test.json', 'w')
f2.write(temp)
f2.close()'''

pass