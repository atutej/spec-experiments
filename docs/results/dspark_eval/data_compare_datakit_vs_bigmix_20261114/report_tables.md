## 1. Structure (per row)
| measure | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
| rows scanned | 2,168,169 | 296,000 | 281,000 | 2,929,705 |
| tokens per row | 7,101 | 10,864 | 9,513 | 1,491 |
| supervised tokens per row | 3,760 | 10,344 | 6,193 | 902 |
| prompt-side (unsupervised) tokens per row | 3,341 | 520 | 3,321 | 588 |
| assistant turns per row | 2.63 | 1.43 | 2.57 | 1.53 |
| rows at the 32,768 window | 4.7% | 16.2% | 11.0% | 0.1% |
| supervised share of all tokens | 52.9% | 95.2% | 65.1% | 60.5% |

## 2. Reasoning mode and terminal-agent prompts
| share | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
| rows with `/think` header | 63.6% | 100.0% | 63.6% | 18.6% |
| supervised tokens in `/think` rows | 85.4% | 100.0% | 89.6% | 37.0% |
| rows with `/nothink` header | 36.4% | 0.0% | 36.4% | 81.4% |
| supervised tokens in `/nothink` rows | 14.6% | 0.0% | 10.4% | 63.0% |
| rows with `/none` header | 0.0% | 0.0% | 0.0% | 0.0% |
| supervised tokens in `/none` rows | 0.0% | 0.0% | 0.0% | 0.0% |
| rows with the Terminus-2 terminal prompt | 4.7% | 6.0% | 4.7% | 5.9% |
| supervised tokens in terminal rows | 3.2% | 9.5% | 2.7% | 26.2% |
| rows containing a think block | 63.6% | 98.8% | 63.7% | 18.6% |
| think-block tokens as share of supervised | 74.0% | 86.6% | 83.3% | 22.2% |
| think blocks never closed (truncated) | 3.8% of think rows | 18.2% of think rows | 17.4% of think rows | 0.1% of think rows |

## 3. Per-category: supervised tokens per row, window rows, context copying

**think / non-terminal**: row share | supervised/row | window rows | supervised tokens copied from context | repeated from earlier assistant text
|  | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
|  | 63.6% / 5,051 / 4.6% / 5.7% / 21.1% | 94.0% / 9,959 / 14.8% / 2.6% / 9.7% | 63.6% / 8,724 / 15.5% / 3.2% / 13.8% | 18.6% / 1,794 / 0.1% / 3.9% / 10.7% |

**think / terminal**: row share | supervised/row | window rows | supervised tokens copied from context | repeated from earlier assistant text
|  | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
|  | - | 6.0% / 16,404 / 36.7% / 4.6% / 12.2% | - | - |

**nothink / non-terminal**: row share | supervised/row | window rows | supervised tokens copied from context | repeated from earlier assistant text
|  | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
|  | 31.7% / 1,352 / 5.6% / 8.8% / 23.0% | - | 31.7% / 1,499 / 3.1% / 7.6% / 19.0% | 75.4% / 440 / 0.0% / 4.4% / 10.4% |

**nothink / terminal**: row share | supervised/row | window rows | supervised tokens copied from context | repeated from earlier assistant text
|  | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
|  | 4.7% / 2,529 / 0.5% / 5.8% / 35.6% | - | 4.7% / 3,556 / 2.0% / 6.9% / 23.8% | 5.9% / 3,976 / 1.0% / 4.3% / 35.2% |

## 4. Whole-dataset copy shares (8-gram seen earlier in the row)
| share of supervised tokens | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
| copied from the prompt or tool output | 6.0% | 2.8% | 3.7% | 4.2% |
| repeated from earlier assistant text | 21.8% | 9.9% | 14.5% | 17.0% |

## 5. Token classes of supervised text
| class | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
| special | 0.12% | 0.04% | 0.07% | 0.21% |
| CJK | 0.72% | 0.20% | 0.51% | 0.50% |
| other non-ASCII | 2.94% | 1.26% | 2.75% | 1.26% |
| digits | 8.76% | 6.30% | 6.71% | 4.80% |
| whitespace/newline | 4.93% | 3.06% | 3.26% | 4.52% |
| punctuation/operators | 22.40% | 20.30% | 20.34% | 19.21% |
| latex/code symbols | 5.25% | 2.40% | 3.91% | 6.48% |
| English word pieces | 54.88% | 66.42% | 62.45% | 63.02% |
| mixed other | 0.00% | 0.00% | 0.00% | 0.00% |

## 6. Draft-vocabulary coverage of each dataset's own supervised tokens (rows: drafter vocabulary)
| drafter | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
| Datakit orig 24h | 97.97% | 98.02% | 98.35% | 96.09% |
| NoFin orig 24h | 97.93% | 98.02% | 98.31% | 96.09% |
| BigMix orig 24h | 96.42% | 97.81% | 97.03% | 97.31% |
| BigMix TG 28h | 97.00% | 98.60% | 97.79% | 96.40% |
| Datakit TG 25h | 97.90% | 98.15% | 98.45% | 96.04% |

## 7. Supervised length per row: share of rows by size
| supervised tokens | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
| 0 to 256 | 23.4% | 6.4% | 21.0% | 54.8% |
| 256 to 1,024 | 37.6% | 24.5% | 31.2% | 36.4% |
| 1,024 to 4,096 | 26.7% | 31.1% | 26.6% | 8.1% |
| 4,096 to 16,384 | 12.4% | 38.0% | 21.3% | 0.7% |
| 16,384+ | 0.0% | 0.0% | 0.0% | 0.0% |

## 8. Text-sample features (sampled rows)
| feature | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
| rows in the text sample | 26,012 | 14,852 | 14,092 | 34,963 |
| zlib bytes / raw bytes of supervised text | 0.299 | 0.319 | 0.309 | 0.357 |
| `Wait` per 1k supervised tokens | 0.85 | 1.78 | 1.76 | 0.52 |
| `Hmm` per 1k supervised tokens | 0.12 | 0.06 | 0.15 | 0.07 |
| `Alternatively` per 1k supervised tokens | 0.35 | 0.12 | 0.32 | 0.24 |
| `But wait` per 1k supervised tokens | 0.11 | 0.11 | 0.11 | 0.03 |
| `Let me` per 1k supervised tokens | 0.51 | 0.16 | 0.39 | 0.51 |
| `Actually` per 1k supervised tokens | 0.82 | 2.52 | 2.19 | 0.07 |
| `Okay` per 1k supervised tokens | 0.06 | 0.08 | 0.06 | 0.22 |
| prompt asks for \boxed | 10.6% | 6.7% | 10.9% | 6.2% |
| prompt says step by step | 8.9% | 3.1% | 9.1% | 3.1% |
| prompt has a code fence | 8.1% | 1.5% | 7.9% | 1.5% |
| prompt has LaTeX | 27.7% | 21.3% | 27.8% | 20.9% |
| prompt mostly non-ASCII (>30%) | 5.2% | 0.1% | 5.3% | 0.2% |

## 9. Unigram closeness of eval replies to each dataset's supervised text (bits per token of the replies under the dataset's unigram model; lower = closer)
| eval set | Datakit orig | BigMix TG | Datakit TG | BigMix orig |
|---|---:|---:|---:|---:|
| GSM8K | 10.908 | 10.562 | 10.881 | 11.090 |
| MATH-500 | 9.409 | 9.486 | 9.449 | 10.736 |
| HumanEval | 10.588 | 10.400 | 10.535 | 11.286 |
| MBPP | 10.787 | 10.668 | 10.833 | 11.394 |
| MT-Bench | 11.444 | 11.176 | 11.455 | 11.485 |
| AIME | 9.768 | 9.650 | 9.632 | 10.958 |
| Datakit held-out | 10.864 | 10.933 | 10.836 | 11.477 |
| BigMix held-out | 11.558 | 11.327 | 11.563 | 11.604 |
| finance_v1 | 10.889 | 11.138 | 10.908 | 11.573 |
| Snowball replay | 12.233 | 12.474 | 12.446 | 13.039 |