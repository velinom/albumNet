from PIL import Image
import requests
from io import BytesIO
import csv

with open('dataset_links.tsv') as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  for row in reader:
    link = row[0]
    response = requests.get(link)
    if response.status_code != 404:
      img = Image.open(BytesIO(response.content))
      img.save(row[1])