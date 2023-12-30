import requests

url = 'http://localhost:6691'

data = {'key1': 'value1', 'key2': 'value2'}

print(requests.post(url=url+'/call', json=data).text)

data.update({'func_name': 'foo', 'key3': 'value3'})

print(requests.post(url=url+'/other_func', json=data).text)
