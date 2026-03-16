#!/usr/bin/env python3
import json, re

def parse_claude(input_file, min_length=10, min_words=3):
    data = json.load(open(input_file, encoding='utf-8'))
    print(f"[Claude] Found {len(data)} conversations")
    all_pairs = []
    for conv in data:
        messages = conv.get('chat_messages', [])
        for i in range(len(messages) - 1):
            msg = messages[i]
            next_msg = messages[i+1]
            sender = msg.get('sender', '')
            next_sender = next_msg.get('sender', '')
            def get_text(m):
                content = m.get('content', [])
                if isinstance(content, str):
                    return content.strip()
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get('type') == 'text':
                        parts.append(c.get('text', ''))
                return ' '.join(parts).strip()
            text = get_text(msg)
            next_text = get_text(next_msg)
            if not text or not next_text:
                continue
            # Your message (human) as assistant voice
            if sender == 'assistant' and next_sender == 'human':
                if len(next_text) >= min_length and len(next_text.split()) >= min_words:
                    all_pairs.append({"messages": [
                        {"role": "user", "content": text[:300]},
                        {"role": "assistant", "content": next_text}
                    ]})
            # Your standalone longer messages
            elif sender == 'human' and len(text.split()) > 15:
                all_pairs.append({"messages": [
                    {"role": "user", "content": "What do you think?"},
                    {"role": "assistant", "content": text}
                ]})
    print(f"[Claude] Extracted {len(all_pairs)} pairs")
    seen = set()
    unique = []
    for pair in all_pairs:
        key = pair['messages'][-1]['content'][:100]
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    output = '/root/jarvis/training/arakkha_v4_claude.jsonl'
    with open(output, 'w', encoding='utf-8') as f:
        for pair in unique:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"[Done] Saved {len(unique)} pairs to {output}")

parse_claude('/root/jarvis/training/claude_conversations.json')
