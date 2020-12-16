import json
import requests
from requests_oauthlib import OAuth1


with open("twitter_secrets_dbf.json", "r") as f:
    secrets = json.load(f)