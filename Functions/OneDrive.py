# import cloudsync
#
#
# prov = cloudsync.create_provider("onedrive")
# creds = prov.authenticate()
# prov.connect(creds)
# with open("Data/AllFormatTest.sav", "wb") as f:
#     prov.download_path("/Users/rlavelle-hill/OneDrive - UT Cloud/MunichPanelData", f)
#
# from office365.runtime.auth.authentication_context import AuthenticationContext
# from office365.sharepoint.client_context import ClientContext
# from office365.sharepoint.files.file import File
#

#
# url = OneDriveConfig.url
# password = OneDriveConfig.password
# username = OneDriveConfig.username
# relative_url = OneDriveConfig.relative_url
#
# filename = 'AllFormat.sav'
#
# ctx_auth = AuthenticationContext(url)
# if ctx_auth.acquire_token_for_user(username, password):
#   ctx = ClientContext(url, ctx_auth)
#   web = ctx.web
#   ctx.load(web)
#   ctx.execute_query()
#   print("Web title: {0}".format(web.properties['Title']))
#
#   with open(filename, 'wb') as output_file:
#     response = File.open_binary(ctx, relative_url)
#     output_file.write(response.content)
#
# else:
#   print(ctx_auth.get_last_error())

url = 'https://unitc-my.sharepoint.com/:u:/g/personal/sebrl01_cloud_uni-tuebingen_de/Documents/MunichPanelData/AllFormat.sav'
# url = 'https://unitc-my.sharepoint.com/:u:/g/personal/sebrl01_cloud_uni-tuebingen_de/Documents/Test.docx'

import getpass
import requests
import pyreadstat
import shutil

from requests_ntlm import HttpNtlmAuth
from urllib.parse import unquote
from pathlib import Path
from Credentials import OneDriveConfig

domain = 'unitc'
user = getpass.getuser()
pwd = getpass.getpass(prompt='Please enter your password:')
if pwd == OneDriveConfig.password:
    print('Password correct')
else:
    print('The password entered is incorrect')

filename = unquote(Path(url).name)

with requests.get(url, stream=True, auth=HttpNtlmAuth(f'{domain}\\{user}', pwd)) as r:
    with open(filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

# resp = requests.get(url, auth=HttpNtlmAuth(f'{domain}\\{user}', pwd))
# open(filename, 'wb').write(resp.content)

file_name = "Functions/AllFormat.sav"

df, meta = pyreadstat.read_sav(file_name, encoding="latin1")

print('done!')