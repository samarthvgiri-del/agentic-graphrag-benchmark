import json

def run():
    found = set()
    with open('data/questions/eval_public.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            q = json.loads(line)
            # We just need 5 diverse questions
            print(f"{i}: {q['question']}")
            if i > 20: break

run()
