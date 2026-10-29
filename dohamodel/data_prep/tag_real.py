"""Tag the real Kabir dohe with the same tags the synthetic ones carry (PRD build step 4, "tag the real dohe too").

  theme   from the doha's अंग section (the book's own grouping); sections that match none of the 10 themes get none
  images  Devanagari word forms for each of the 30 images, matched as whole words (zero, one or more per doha)
  device  cue words: साधो/संतो/रे मन -> sadho, a question word -> prashn, कबीर/कबिरा -> kahekabir (in that order);
          paradox (ulatbansi) has no reliable cue word, so a real doha never gets it

The keyword lists in the classifier's config are English (for English requests), so these are separate.
Writes data/clean/kabir_real_tagged.jsonl (same rows plus theme, images, device) and prints coverage.

Run:  .venv/bin/python data_prep/tag_real.py
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent

SECTION_THEME = {
    "guru": ["गुरुदेव", "गुरु शिष्य खोज", "झूठे गुरु", "गुरुमुख", "निगुरा"],
    "maya": ["माया", "कनक और कामिनी", "मोह", "आशा", "तृष्णा", "लोभ"],
    "mrityu": ["चितावनी"],
    "ahankar": ["मान और हंगता"],
    "pakhand": ["भेष", "पंडित और संस्कृत", "तीर्थ ब्रत"],
    "bhakti": ["भक्ति", "सुमिरन", "नाम"],
    "vani": ["शब्द", "मौन"],
    "dhairya": ["धीरज"],
    "satsang": ["सतसंग", "साध", "कुसंग", "असाधु"],
    "prem": ["प्रेम", "बिरह", "लौ"],
}
THEME_OF = {s: t for t, ss in SECTION_THEME.items() for s in ss}

# whole-word forms (old spellings included); दिया and दीया are left out: in these texts they mostly mean "gave" (मन दीया)
IMAGE_WORDS = {
    "deepak": "दीपक दीप दीवा दिवला बाती",
    "kumhar": "कुम्हार कुम्हरा कुंभार कुम्भार कुम्हारा",
    "kuan": "कुआँ कुआं कुँआ कुवाँ कुवा कूप कूवा कूवे कुएँ",
    "machhli": "मछली मछरी मीन मच्छ मछिया",
    "mala": "माला मनका मनिया",
    "chandan": "चंदन चन्दन",
    "patang": "पतंग पतंगा पतंगे",
    "bazaar": "बाजार बाज़ार बजार हाट",
    "naav": "नाव नैया नौका नाँव बेड़ा बेड़ा",
    "vriksh": "वृक्ष बिरछ बृच्छ बिरिछ तरुवर तरवर रूख पेड़ बिरवा",
    "kargha": "करघा करघे ताना बाना जुलाहा जोलाहा",
    "charkha": "चरखा चरखे",
    "kapas": "कपास",
    "chadar": "चादर चदरिया चदरी चादरि",
    "dhaga": "धागा तागा सूत",
    "sui": "सुई सूई",
    "chakki": "चक्की चाकी",
    "kolhu": "कोल्हू",
    "bhatti": "भट्ठी भाठी भट्टी भठी",
    "kasauti": "कसौटी",
    "tarazu": "तराजू तराजु तुला",
    "rassi": "रस्सी रसरी जेवरी",
    "ghada": "घड़ा घड़ा घट गागर कुंभ कुम्भ",
    "koyal": "कोयल कोकिल",
    "hans": "हंस हंसा",
    "kauwa": "कौआ कौवा काग कागा",
    "kasturi": "कस्तूरी मृग मिरग",
    "bagula": "बगुला बगला",
    "chakor": "चकोर",
    "papiha": "पपीहा पपिहा चातक",
}
WORD_IMAGE = {w: i for i, ws in IMAGE_WORDS.items() for w in ws.split()}

SADHO = re.compile(r"(?:^|\s)(?:साधो|साधौ|संतो|सन्तो)(?:\s|,|$)|रे मन|मन रे|सुनो भाई|सुनु भाई")
PRASHN = {"क्यों", "क्यौं", "कैसे", "कैसें", "कौन", "काहे", "किमि", "कित", "कितने", "क्या"}
KABIR = re.compile(r"कबीर|कबिरा")


def tag(text):
    words = re.findall(r"[ऀ-ॿ]+", text)
    images = sorted({WORD_IMAGE[w] for w in words if w in WORD_IMAGE})
    if SADHO.search(text):
        device = "sadho"
    elif PRASHN & set(words):
        device = "prashn"
    elif KABIR.search(text):
        device = "kahekabir"
    else:
        device = None
    return images, device


if __name__ == "__main__":
    rows = [json.loads(l) for l in (ROOT / "data/clean/kabir_real.jsonl").open(encoding="utf-8")]
    out = []
    for r in rows:
        images, device = tag(r["text"])
        out.append({**r, "theme": THEME_OF.get(r["section"]), "images": images, "device": device})
    (ROOT / "data/clean/kabir_real_tagged.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    kept = [r for r in out if r["split"] != "drop"]
    n = len(kept)
    print(f"{n} real dohe (train + val)")
    print(f"  theme:  {sum(bool(r['theme']) for r in kept) / n:.0%} have one   "
          + "  ".join(f"{k} {v}" for k, v in sorted(Counter(r['theme'] for r in kept if r['theme']).items())))
    print(f"  images: {sum(bool(r['images']) for r in kept) / n:.0%} have one or more   "
          + "  ".join(f"{k} {v}" for k, v in Counter(i for r in kept for i in r['images']).most_common(12)))
    print(f"  device: {sum(bool(r['device']) for r in kept) / n:.0%} have one   "
          + "  ".join(f"{k} {v}" for k, v in Counter(r['device'] for r in kept if r['device']).items()))
    for r in [r for r in kept if r["images"]][:6]:
        print(f"    [{r['theme']}/{'+'.join(r['images'])}/{r['device']}] {r['text'].replace(chr(10), ' / ')}")
