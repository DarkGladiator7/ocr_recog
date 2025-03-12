import requests
import json

url = "https://dl.localzoho.com/api/v2/nlp/translation/translate"

payload = json.dumps({
  "text": {
    "sentences": [
      "Paragraph :  Florian steht jeden Tag um sechs Uhr auf. Zuerst wäscht er sein Gesicht und putzt sich die Zähne. Dann geht er nach unten, um zu frühstücken. Nach dem Frühstück zieht er sich an und geht zur Schule. In der Schule hat Florian viel Spaß mit seinen Freunden. Auch im Unterricht lernt er Neues. Florian ist ein sehr guter Schüler! Nach der Schule geht er nach Hause und isst mit seiner Familie zu Abend. Dann macht er seine Hausaufgaben und bereitet sich auf den nächsten Tag vor. Florian geht normalerweise um neun Uhr ins Bett. Julia ist Florians beste Freundin. Sie ist auch ein sehr beschäftigtes Mädchen. Julia steht um sieben Uhr morgens auf. Sie wäscht sich das Gesicht und putzt sich die Zähne. Dann geht sie nach unten, um zu frühstücken. Nach dem Frühstück zieht sie sich an und geht zur Schule. In der Schule hat Julia viel Spaß mit ihren Freunden. Auch im Unterricht lernt sie Neues. Julia ist eine sehr gute Schülerin! Nach der Schule geht sie nach Hause und isst mit ihrer Familie zu Abend. Dann macht sie ihre Hausaufgaben und bereitet sich auf den nächsten Tag vor.爪puBeT"
    ]
  },
  "source_language": "de",
  "target_language": "en",
  "align": False
})
headers = {
  'Authorization': 'Zoho-oauthtoken 1000.6a94bae3464ab07d3af0c8f8edbd7f6f.05c2227c8697903fe78022856259aaaa',
  'Content-Type': 'application/json',
}

response = requests.request("POST", url, headers=headers, data=payload)

dic = json.loads(response.text)
print(json.dumps(dic, indent=5))