
import http.client


conn = http.client.HTTPSConnection('http://localhost:32000')
conn.request('PUT', '/')

resp = conn.getresponse()
content = resp.read()

conn.close()

text = content.decode('utf-8')

print(text)


# curl -v -XPUT localhost:32000/Server/Start
# curl -v -XGET localhost:32000/Server/DeviceManager/Connected
# curl -v -XOPTIONS localhost:32000/Driver/UPBv2/Start?DriverUniqueKey=c445948e-1005-4f40-b1f4-eb971ab41363
# curl -v -XPUT localhost:32000/Driver/UPBv2/Power/2/On?DriverUniqueKey=c445948e-1005-4f40-b1f4-eb971ab41363
# curl -v -XPUT localhost:32000/Driver/UPBv2/Report/Environment?DriverUniqueKey=c445948e-1005-4f40-b1f4-eb971ab41363