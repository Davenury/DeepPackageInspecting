import requests
import json
import threading
import time

f = open('data.json')
url = 'http://127.0.0.1:5000'
data_json = json.load(f)

def generate_traffic(name, data):
    for rpm in data:
        sleep_time = 60 / rpm
        for _ in range(rpm):
            res_data = {'Country': name}
            requests.post(url, json = res_data)
            time.sleep(sleep_time)


if __name__ == "__main__":
    fr_thread = threading.Thread(target=generate_traffic, args=('FR', data_json['FR']))
    fr_thread.start()
    sz_thread = threading.Thread(target=generate_traffic, args=('SZ', data_json['SZ']))
    sz_thread.start()
