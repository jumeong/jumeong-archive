# ëª¨ë¸ ê°œìš”

- ?”ì¦˜ ?€ë¶€ë¶?MoEë§??°ëŠ”??Dense Model??ê°œë°œ
- E2B, E4B ëª¨ë¸?€ ?€ê²??˜ë“œ?¨ì–´ê°€ ?¨ë””ë°”ì´?¤ì´ë¯€ë¡?VRAM ?©ëŸ‰??ì¤„ì´ê¸??„í•¨
- 26B-A4B ëª¨ë¸?€ VRAM ?¬ìœ ê°€ ?ˆëŠ” ?Œí¬?¤í…Œ?´ì…˜?´ë‚˜ ?œë²„?ì„œ 31B Dense ëª¨ë¸??ê·¼ì ‘?˜ëŠ” ?±ëŠ¥??4B ëª¨ë¸ ?˜ì???ì§€???œê°„?¼ë¡œ ?»ê¸° ?„í•¨
- E4B ëª¨ë¸?€ ?Œë¼ë¯¸í„° ?˜ê? Gemma 3 27B??1/6??ë¶ˆê³¼?˜ì?ë§?ëª¨ë“  ë²¤ì¹˜ë§ˆí¬?ì„œ ?¥ê?

![alt text](images/image.png)
![alt text](images/image-1.png)
![alt text](images/image-2.png)

# Per-Layer Embedding (PLE)

- Reference: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
- ?¨ë””ë°”ì´?¤ì—???°ì‚°?‰ì„ ì¤„ì´ê¸??„í•´ ë£©ì—… ?Œì´ë¸?ê¸°ë°˜??PLEë¥??„ì…
- PLE ?´ì 
    - ì§€???€?¥ì†Œ??ë¶„ë¦¬: Attention?€ ë¬¸ë§¥ ?Œì•…??ì§‘ì¤‘?˜ê³ , PLE??ê³ ì •???¬ì‹¤?´ë‚˜ ì§€?ì„ ?€?¥í•˜??ë°©í–¥?¼ë¡œ ?™ìŠµ
    - Diversed Feature Extraction: ?™ì¼???…ë ¥???€???œë¡œ ?¤ë¥¸ ê´€?ì˜ ?¹ì§• ì¶”ì¶œ
    - Decoder Block Layer ?˜ë? ì¤„ì´???€??FFN Layer??Element-wise multiplication ?°ì‚°??ì¶”ê??´ì„œ Processing ë¶€?´ì„ ì¤„ì„
- ???¥ì ë§?ë´¤ì„ ?ŒëŠ” ??ëª¨ë¸?ì„œ???ìš©?˜ì? ?Šì„ ?´ìœ ê°€ ?†ì–´ë³´ì´?”ë°?
    - ??ëª¨ë¸?ì„œ??Decode Stage?ì„œ???´ë? IO Bound??ê±¸ë ¤?ˆìœ¼ë¯€ë¡?Processing Cycle??ì¤„ì´?”ê²Œ ?¬ê²Œ ?˜ë? ?†ìŒ.
    - ?€??ëª¨ë¸?ì„œ???´ë? ì¶©ë¶„?????Œë¼ë¯¸í„° ?˜ë? ê°€ì§€ê³??ˆìœ¼ë¯€ë¡?PLEë³´ë‹¤??MoEë¡??°ì‚°?‰ì? ì¤„ì´??Dense ëª¨ë¸??ê·¼ì ‘???±ëŠ¥???´ëŠ” ê²ƒì´ ?˜ìŒ.

![alt text](images/image-3.png)
![alt text](images/image-4.png)
![alt text](images/image-5.png)
![alt text](images/image-6.png)

# Speculative Decoding

## ?™ì‘?ë¦¬
- https://arxiv.org/abs/2211.17192
- ![alt text](images/image-7.png)
- Autoregressive ë°©ì‹?€ ??ê°?? í°??ì¶”ì¸¡?˜ëŠ” ë°©ì‹?´ë¼??Decode Stage?ì„œ Vector by Matrix ë¬¸ì œ?ì„œ Processing / IO ë°¸ëŸ°?¤ê? ë§ì? ?Šë˜ ê²ƒì„ Speculative Decoding???µí•´ Processing ë¹„ìœ¨???˜ë¦´ ???ˆê²Œ ??
- Draft Model (e.g., google/gemma-4-E4B-it-assistant) ?€ Target Model (e.g., google/gemma-4-E4B-it) ???ì„±??KV Cacheë¥?ê³µìœ ?????ˆìŒ.
    - ëª¨ë¸ ?Œë¼ë¯¸í„°ê°€ 78.8M params vs 8B paramsë¡??œì°¸ ì°¨ì´?˜ëŠ”???´ë–»ê²? ê·¸ë¦¬ê³??´ë–¤ ?ˆì´?´ì˜ KV Cacheë¥?ê³µìœ ?
    - Gemma 4 ships with a small "assistant" head that predicts several future tokens from the target model's last hidden state. 
    - Draft Model??config.json??ë³´ë©´ KV Cacheë¥?ê³µìœ ?˜ê¸° ?„í•œ ì¡°ê±´??ë§Œì¡±?˜ë„ë¡??¤ê³„?˜ì–´ ?ˆëŠ” ê²ƒìœ¼ë¡?ë³´ì„.
    - <details>
        <summary>config.json</summary>
        "backbone_hidden_size": 2560, <br>
        "global_head_dim": 512, <br>
        "head_dim": 256, <br>
        "num_hidden_layers": 4, <br> 
        "num_key_value_heads": 2, <br>
        "num_kv_shared_layers": 4, <br>
      </details>
- ??ëª¨ë¸??ë³‘ë ¬ë¡??Œë¦¬?¤ëŠ” ?œë„ (Staged Speculative Decoding)???ˆì?ë§? ê¸°ë³¸?ìœ¼ë¡?Sequential?˜ê²Œ ?™ì‘
    - Target Model??Accept ê²°ì •???˜ê¸° ?„ì— Draft Model???¤ìŒ ? í° ë¬¶ìŒ??ë¯¸ë¦¬ ?ˆì¸¡?˜ëŠ” ë°©ì‹?¸ë°, Accept, Reject ê²°ê³¼???°ë¼ ì»¨íŠ¸ë¡¤ì´ ?´ë ¤????

## Huggingface êµ¬í˜„
```python
model.generate()
  ?”â??€ GenerationMixin.generate()
        ?”â??€ _assisted_decoding()
              ?œâ??€ AssistedCandidateGenerator.get_candidates()
              ??    ?”â??€ assistant_model.forward()  # draft ? í° Nê°??ì„± (Në²ˆì˜ forward pass)
              ?œâ??€ target_model.forward()            # Nê°?? í° ë³‘ë ¬ ê²€ì¦? (1ë²ˆì˜ forward pass)
              ?”â??€ _speculative_sampling()           # reject sampling?¼ë¡œ ?˜ë½/ê±°ë?
```

## ?±ëŠ¥ (Decoding Tput)
### Light User ê´€??
- https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-on-off.gif

### Benchmark
- ![alt text](images/image-8.png)
- ì¶œì²˜: https://www.reddit.com/r/LocalLLaMA/comments/1sjct6a/speculative_decoding_works_great_for_gemma_4_31b/?solution=281a0ed603877bc8281a0ed603877bc8&js_challenge=1&token=7afd7253fec22262ff1c52b1703fe9ec7fe009426c8f7f6113d08be287d18ff1&jsc_orig_r=

### Self Experiment
- Tested on H100

#### Prompt-heavy: 8000 input / 1000 output
| ì§€??(Metric) | Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 | google/gemma-4-26B-A4B-it wo/ SD | google/gemma-4-26B-A4B-it w/ SD |
| :--- | :--- | :--- | :--- |
| **Successful requests** | 16 | 16 | 16 |
| **Failed requests** | 0 | 0 | 0 |
| **Maximum request concurrency** | 1 | 1 | 1 |
| **Benchmark duration (s)** | 95.41 | 106.69 | 121.42 |
| **Total input tokens** | 128,000 | 128,000 | 128,000 |
| **Total generated tokens** | 16,000 | 16,000 | 16,000 |
| **Request throughput (req/s)** | 0.17 | 0.15 | 0.13 |
| **Output token throughput (tok/s)** | 167.69 | 149.97 | 131.78 |
| **Peak output token throughput (tok/s)** | 175.00 | 163.00 | 50.00 |
| **Peak concurrent requests** | 2.00 | 2.00 | 2.00 |
| **Total token throughput (tok/s)** | 1509.20 | 1349.69 | 1185.98 |
| **Mean TTFT (ms)** | 219.48 | 437.15 | 490.21 |
| **Median TTFT (ms)** | 261.13 | 430.10 | 482.46 |
| **P99 TTFT (ms)** | 315.53 | 592.80 | 643.24 |
| **Mean TPOT (ms)** | 5.75 | 6.24 | 7.10 |
| **Median TPOT (ms)** | 5.76 | 6.23 | 5.94 |
| **P99 TPOT (ms)** | 5.77 | 6.27 | 19.86 |
| **Mean ITL (ms)** | 5.75 | 6.25 | 20.89 |
| **Median ITL (ms)** | 5.76 | 6.26 | 20.94 |
| **P99 ITL (ms)** | 6.18 | 6.45 | 22.19 |
| **SD Acceptance rate (%)** | - | - | 48.62 |
| **SD Acceptance length** | - | - | 2.94 |
| **SD Drafts** | - | - | 5435 |
| **SD Draft tokens** | - | - | 21740 |
| **SD Accepted tokens** | - | - | 10569 |
| **SD Position 0 acceptance (%)** | - | - | 62.85 |
| **SD Position 1 acceptance (%)** | - | - | 52.51 |
| **SD Position 2 acceptance (%)** | - | - | 42.32 |
| **SD Position 3 acceptance (%)** | - | - | 36.78 |

- SD ì¼ ê²Œ ???ˆì¢‹??WHY??????
    - Draft Model??Long Context ì²˜ë¦¬ ?¨ìœ¨???¨ì–´ì§€ë¯€ë¡?8k ?…ë ¥???€?´ì„œ??Acceptance Rate???¨ì–´ì§€??ê²ƒìœ¼ë¡?ë³´ì„.
    - ê·¸ëŸ¬ë¯€ë¡? SD ?†ì´ Autoregressive?˜ê²Œ ?™ì‘??ëª¨ë¸???˜ì? ?˜ì??¼ë¡œ TPê°€ ?€??
    - ì¶”ê??ìœ¼ë¡? Balance ?œë‚˜ë¦¬ì˜¤??ë¹„í•´ ?ˆì¸¡???€?¸ì„ ??ê°€ì§€??penaltyê°€ ????(input tokensê°€ ??ê¸¸ì–´??decoding ?¨ê³„?ì„œ ?Œìš”?˜ëŠ” ?œê°„????ê¸¸ê¸° ?Œë¬¸)

#### Decode-heavy: 1000 input / 8000 output
| ì§€??(Metric) | Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 | google/gemma-4-26B-A4B-it wo/ SD | google/gemma-4-26B-A4B-it w/ SD |
| :--- | :--- | :--- | :--- |
| **Successful requests** | 16 | 16 | 16 |
| **Failed requests** | 0 | 0 | 0 |
| **Maximum request concurrency** | 1 | 1 | 1 |
| **Benchmark duration (s)** | 727.30 | 742.57 | 515.01 |
| **Total input tokens** | 16,000 | 16,000 | 16,000 |
| **Total generated tokens** | 128,000 | 128,000 | 128,000 |
| **Request throughput (req/s)** | 0.02 | 0.02 | 0.03 |
| **Output token throughput (tok/s)** | 175.99 | 172.37 | 248.54 |
| **Peak output token throughput (tok/s)** | 181.00 | 190.00 | 86.00 |
| **Peak concurrent requests** | 2.00 | 2.00 | 2.00 |
| **Total token throughput (tok/s)** | 197.99 | 193.92 | 279.60 |
| **Mean TTFT (ms)** | 65.09 | 39.53 | 52.74 |
| **Median TTFT (ms)** | 66.81 | 39.99 | 51.68 |
| **P99 TTFT (ms)** | 73.15 | 43.30 | 72.59 |
| **Mean TPOT (ms)** | 5.67 | 5.80 | 4.02 |
| **Median TPOT (ms)** | 5.67 | 5.79 | 3.47 |
| **P99 TPOT (ms)** | 5.68 | 5.82 | 6.52 |
| **Mean ITL (ms)** | 5.67 | 5.80 | 16.31 |
| **Median ITL (ms)** | 5.68 | 5.80 | 16.21 |
| **P99 ITL (ms)** | 5.96 | 6.36 | 21.97 |
| **SD Acceptance rate (%)** | - | - | 76.55 |
| **SD Acceptance length** | - | - | 4.06 |
| **SD Drafts** | - | - | 31515 |
| **SD Draft tokens** | - | - | 126060 |
| **SD Accepted tokens** | - | - | 96493 |
| **SD Position 0 acceptance (%)** | - | - | 85.69 |
| **SD Position 1 acceptance (%)** | - | - | 78.32 |
| **SD Position 2 acceptance (%)** | - | - | 71.49 |
| **SD Position 3 acceptance (%)** | - | - | 70.69 |

- ?¤ë¥¸ ì¼€?´ìŠ¤??ë¹„í•´ Decode-heavy ì¼€?´ìŠ¤??Acceptance rate???‰ê· ?ìœ¼ë¡??’ê²Œ ?˜ì˜¤?”ë° ?œì¼ê¹?
    - ?¤ë¥¸ ??ì¼€?´ìŠ¤??Acceptance rate????50%?´ê³  ?´ë‹¹ ì¼€?´ìŠ¤????70~80%
    - ë¨¼ì?, Prompt-Heavy??ê²½ìš°, Draft Model?ê²Œ??ê¸?ë¬¸ë§¥?????ˆìŒ.
    - Decode-Heavy?ì„œ??input tokenê³?output token ?©ì´ 8kë¥??˜ì–´ê°€ë©??‘ê°™?€ ê²??„ë‹Œì§€? => ?¨ì´ ë§Œë“  ë¬¸ë§¥ vs. ?¤ìŠ¤ë¡?ë§Œë“  ì»¨í…?¤íŠ¸. ?„ìê°€ ???ˆì¸¡?˜ê¸° ?¬ìš¸ ê²?
    - Balanced??ê²½ìš°, ?…ì¶œ??Total 2ì²?? í°?€ ë¬¸ì¥ ?„í™˜?´ë‚˜ ?ˆë¡œ???•ë³´ê°€ ?˜ì˜¬ ?•ë¥ ???’ì•„ ?”íŠ¸ë¡œí”¼ê°€ ?ë??ìœ¼ë¡??’ì„ ???ˆìŒ.

#### Balanced: 1000 input / 1000 output
| ì§€??(Metric) | Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 | google/gemma-4-26B-A4B-it wo/ SD | google/gemma-4-26B-A4B-it w/ SD |
| :--- | :--- | :--- | :--- |
| **Successful requests** | 16 | 16 | 16 |
| **Failed requests** | 0 | 0 | 0 |
| **Maximum request concurrency** | 1 | 1 | 1 |
| **Benchmark duration (s)** | 91.14 | 89.48 | 70.38 |
| **Total input tokens** | 16,000 | 16,000 | 16,000 |
| **Total generated tokens** | 16,000 | 16,000 | 16,000 |
| **Request throughput (req/s)** | 0.18 | 0.18 | 0.23 |
| **Output token throughput (tok/s)** | 175.56 | 178.82 | 227.33 |
| **Peak output token throughput (tok/s)** | 181.00 | 190.00 | 83.00 |
| **Peak concurrent requests** | 2.00 | 2.00 | 2.00 |
| **Total token throughput (tok/s)** | 351.11 | 357.63 | 454.65 |
| **Mean TTFT (ms)** | 93.50 | 258.04 | 394.95 |
| **Median TTFT (ms)** | 114.15 | 69.64 | 85.01 |
| **P99 TTFT (ms)** | 134.28 | 2633.03 | 4295.53 |
| **Mean TPOT (ms)** | 5.61 | 5.34 | 4.01 |
| **Median TPOT (ms)** | 5.57 | 5.33 | 3.36 |
| **P99 TPOT (ms)** | 5.91 | 5.36 | 9.06 |
| **Mean ITL (ms)** | 5.61 | 5.34 | 12.60 |
| **Median ITL (ms)** | 5.57 | 5.34 | 12.55 |
| **P99 ITL (ms)** | 6.41 | 5.56 | 13.99 |
| **SD Acceptance rate (%)** | - | - | 53.72 |
| **SD Acceptance length** | - | - | 3.15 |
| **SD Drafts** | - | - | 5085 |
| **SD Draft tokens** | - | - | 20340 |
| **SD Accepted tokens** | - | - | 10927 |
| **SD Position 0 acceptance (%)** | - | - | 67.59 |
| **SD Position 1 acceptance (%)** | - | - | 57.01 |
| **SD Position 2 acceptance (%)** | - | - | 45.39 |
| **SD Position 3 acceptance (%)** | - | - | 44.90 |

- Peak output token throughput ê³„ì‚°??SD ?ìš©???œë?ë¡??˜ì? ?Šì? ????Peakê°€ ?‰ê· ì¹˜ë³´????œ¼ë¯€ë¡??´ìƒ.
- SDë¡??¸í•´, ?‰ê·  output token tp??gemma w/ SDê°€ ?°ì„¸.
- position ë³„ë¡œ ?¤ë¡œ ê°ˆìˆ˜ë¡?accpetanceê°€ ??? ê²ƒì? ?ì—°?¤ëŸ¬???„ìƒ??

## ?´ìŠˆ
https://news.hada.io/topic?id=29219

- ?”ì•½: Google??MTPë¡??™ìŠµ?œí‚¨ Gemma 4?ì„œ ?´ë‹¹ ê¸°ëŠ¥??ê³µê°œ ë°°í¬?ì—???œê±°?ˆë‹¤ê°€, ì»¤ë??ˆí‹°??ë¦¬ë²„???”ì??ˆì–´ë§ìœ¼ë¡??¤í†µ?????¸ë? ë³´ì¡° ëª¨ë¸ ?•íƒœë¡??¤ëŠ¦ê²?ì§€?ì„ ?œì‘
- ë°œë‹¨: HuggingFace??ê³µê°œ???œì? ëª¨ë¸ ê°€ì¤‘ì¹˜?ëŠ” ì¡´ì¬?˜ì? ?ŠëŠ” MTP(Multi-Token Prediction, ?¤ì¤‘ ? í° ?ˆì¸¡) ?„í‚¤?ì²˜ê°€ ?£ì???ì»´íŒŒ???Œì¼?ë§Œ ?¬í•¨
- Google??ë³€ëª? "MTP ê´€???ˆì¸¡ ?¤ë“œ??HuggingFace Transformers API?€???¸í™˜?±ì„ ?„í•´ ê³µê°œ ëª¨ë¸?ì„œ ?˜ë„?ìœ¼ë¡??œì™¸?ˆë‹¤. LiteRT ?°í??„ì—???¨ë””ë°”ì´???±ëŠ¥ ?¥ìƒ???„í•´ ë³´ì¡´?ˆë‹¤." 
- ë¯¸ë¬¸?œí™” ë¹„íŒ: MTPë¡??™ìŠµ?œì¼œ ?“ê³  ê³µê°œ ë°°í¬?ì—??ê³ ì˜ë¡??œê±°?˜ë©´???„ë¬´???¸ê¸‰???†ì—ˆ?¤ëŠ” ??
- ?ì—…???˜ë„ ?˜í˜¹: "ë¡œì»¬?ì„œ êµ¬ë™?˜ëŠ” ?¤í”ˆ?ŒìŠ¤ 31B ëª¨ë¸???ˆë¬´ ë¹¨ë¼ì§€ë©??ì‚¬ ?ìš© API(Flash Lite ????ê²½ìŸ?¥ì„ ?„í˜‘?˜ê¸° ?Œë¬¸???˜ë„?ìœ¼ë¡??ˆí”„?ˆë‹¤"??ì£¼ì¥.
