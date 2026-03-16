#!/usr/bin/env python3
import os, json, glob, re

def parse_fennix_logs(logs_dir, min_length=10):
    log_files = sorted(glob.glob(os.path.join(logs_dir, "*.md")))
    print(f"[Fennix] Found {len(log_files)} log files")
    all_pairs = []
    for filepath in log_files:
        try:
            content = open(filepath, encoding='utf-8').read()
            blocks = re.split(r'## \[', content)
            messages = []
            for block in blocks[1:]:
                lines = block.strip().split('\n')
                header = lines[0]
                body = '\n'.join(lines[1:]).strip()
                body = re.sub(r'\*\*.*?\*\*', '', body).strip()
                if not body or len(body) < min_length:
                    continue
                if 'User' in header or 'You' in header:
                    messages.append({'role': 'user', 'text': body})
                elif 'Fennix' in header or 'Assistant' in header or 'Bot' in header:
                    messages.append({'role': 'fennix', 'text': body})
            i = 0
            while i < len(messages) - 1:
                msg = messages[i]
                next_msg = messages[i+1]
                # Your message followed by Fennix — use YOUR message as assistant voice
                if msg['role'] == 'fennix' and next_msg['role'] == 'user':
                    if len(next_msg['text']) >= min_length:
                        all_pairs.append({"messages": [
                            {"role": "user", "content": msg['text'][:300]},
                            {"role": "assistant", "content": next_msg['text']}
                        ]})
                # Also capture your standalone longer messages
                elif msg['role'] == 'user' and len(msg['text']) > 80:
                    all_pairs.append({"messages": [
                        {"role": "user", "content": "What do you want to do?"},
                        {"role": "assistant", "content": msg['text']}
                    ]})
                i += 1
        except Exception as e:
            print(f"  [Skip] {filepath}: {e}")
    print(f"[Fennix] Extracted {len(all_pairs)} pairs from {len(log_files)} days")
    return all_pairs

def save(pairs, output_file):
    seen = set()
    unique = []
    for pair in pairs:
        key = pair['messages'][-1]['content'][:100]
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in unique:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"[Done] Saved {len(unique)} pairs to {output_file}")

if __name__ == "__main__":
    pairs = parse_fennix_logs("/root/jarvis/logs/daily/")
    if pairs:
        save(pairs, "/root/jarvis/training/arakkha_v4_fennix.jsonl")
    else:
        print("No pairs found — check log format")
