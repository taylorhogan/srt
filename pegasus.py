
import http.client
import requests

url = 'http://localhost:32000/Server/Start'
response = requests.put(url)
print (response)



# curl -v -XPUT localhost:32000/Server/Start
# curl -v -XGET localhost:32000/Server/DeviceManager/Connected
# curl -v -XOPTIONS localhost:32000/Driver/UPBv2/Start?DriverUniqueKey=c445948e-1005-4f40-b1f4-eb971ab41363
# curl -v -XPUT localhost:32000/Driver/UPBv2/Power/2/On?DriverUniqueKey=c445948e-1005-4f40-b1f4-eb971ab41363
# curl -v -XPUT localhost:32000/Driver/UPBv2/Report/Environment?DriverUniqueKey=c445948e-1005-4f40-b1f4-eb971ab41363