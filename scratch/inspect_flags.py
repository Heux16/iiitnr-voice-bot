import json

with open("benchmark_200_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

missing_cit = [d for d in data if "missing_page_citation" in d["verdict"]]
print(f"Total missing_page_citation: {len(missing_cit)}")
for d in missing_cit[:8]:
    print("=" * 60)
    print(f"{d['id']} ({d['lang']}): {d['question']}")
    print("Refused at retrieval:", d.get("refused_at_retrieval"))
    print("Is refusal:", d.get("is_refusal"))
    print("Response:", repr(d["response_text"]))

