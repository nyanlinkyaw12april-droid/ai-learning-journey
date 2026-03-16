# ARAKKHA & Fennix — Personal AI Twin Project 🤖

Building a personal AI digital twin from scratch. No CS degree. After hours. Rayong, Thailand.

## About

I'm Nyan Lin Kyaw — HR Manager + CAD freelancer at TMA (519) Co., Ltd. in Rayong, Thailand.

In early 2026 I started building my own personal AI system with zero formal AI background.
This repository documents everything I built, learned, and figured out along the way.

## What I Built

### Fennix
A production AI assistant running 24/7 on a DigitalOcean server.
- Telegram bot interface
- Claude + Gemini fallback brain
- RAG memory system
- Google Drive, Gmail, Sheets, Calendar integration
- Auto conversation logging for training data
- Self-healing watchdog process

### ARAKKHA
A personal AI model fine-tuned on my own voice and personality.
- Base: Meta Llama 3.2 3B
- Fine-tuned with LoRA on my real conversations
- Training data: ChatGPT + Facebook + Fennix logs + Claude.ai exports
- 14,270 pairs — 100% my voice only
- Running locally via Ollama — zero API cost

## The Vision

A tiny personality core (~300MB) + internet as external brain.
No massive knowledge baked in. Just my personality, reasoning style, and voice.
When it needs facts — it searches. Just like a human brain.

## Scripts

- `parse_chatgpt_v4.py` — Extract your voice from ChatGPT exports
- `parse_facebook_v4.py` — Extract your voice from Facebook data
- `parse_fennix_logs.py` — Parse Fennix daily conversation logs
- `parse_claude_export.py` — Extract your voice from Claude.ai export
- `train_v4.py` — LoRA fine-tuning script for RTX 4060

## Stack

Python · Telegram Bot API · Anthropic Claude · Google Gemini · 
Meta Llama 3.2 · LoRA · Ollama · DigitalOcean · Google APIs

## Location
Rayong, Thailand 🇹🇭
