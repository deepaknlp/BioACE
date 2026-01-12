from urllib.parse import urlencode
import requests
import json
import xml.etree.ElementTree as ET
import sys

def get_abstracts(pmids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ','.join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        "api_key": "a5ad30960560a4b4b6a04d306ae24bee3308"
    }
    url = f"{base_url}?{urlencode(params)}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return(response.text)
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve abstract for PMID {e}")

def parse_XML(xml):
    root = ET.fromstring(xml)
    abstracts = []

    for pubmedArticle in root:
        if pubmedArticle.tag == "PubmedArticle":
            article = pubmedArticle[0].find("Article")
            title = article.find("ArticleTitle").text
            abstract = article.find("Abstract")[0].text

        else:
            title = pubmedArticle[0].find("ArticleTitle").text
            abstract = pubmedArticle[0].find("Abstract")[0].text
            
        abstracts.append((title, abstract))

    return abstracts

pmids = set()
abstracts = {}

with open("src/data/sentencewise_nuggets2.json") as file:
    data = json.load(file)

for value in data.values():
    for sentence in value:
        pmids.update(sentence["pmids"])

chunk_size = 20
chunks = [list(pmids)[i:i + chunk_size] for i in range(0, len(pmids), chunk_size)]

for chunk in chunks:
    xml = get_abstracts(chunks[0])

    if xml:
        try:
            parsed = parse_XML(xml)

        except Exception as e:
            print(f"Failed to parse XML with exception {e}")

    else:
        print(f"XML empty")
        sys.exit(0)


    for i in range(len(chunk)):
        abstracts[chunk[i]] = { "title": parsed[i][0], "abstract": parsed[i][1]}

with open('pubmed_data.json', 'w') as file:
    json.dump(abstracts, file, indent=4)