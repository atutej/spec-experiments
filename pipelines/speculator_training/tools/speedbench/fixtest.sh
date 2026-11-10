#!/bin/bash
# CPU-only test of AIPerf's dispatch flags against the fake server: which prompts/turns does each variant send? (no GPU, no model)
SB=/scratch/09749/atutej/marin_speed_bench; M=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom
export HF_HOME=/scratch/09749/atutej/marin_speculator/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/scratch/09749/atutej/marin_speculator/envs/embed/bin/python
python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('/scratch/09749/atutej/marin_speed_bench/aiperf_data/qualitative.jsonl')]
for c in ('reasoning','roleplay','rag'):
    r=[x for x in rows if x['category']==c]; d={}
    for x in r: d[len(x['messages'])]=d.get(len(x['messages']),0)+1
    print('expected',c,'sessions',len(r),'turns',sum(len(x['messages']) for x in r),'sessions by #turns',dict(sorted(d.items())),flush=True)
PY
port=8780
trial() {  # trial <label> <category> <extra aiperf flags...>
  lab=$1; cat=$2; shift 2; port=$((port+1))
  $PY $SB/fake_server.py $port $M > $SB/fixtest/fake_$lab.log 2>&1 & FP=$!; sleep 3
  timeout 300 $SB/aiperf_venv/bin/aiperf profile --model $M --tokenizer $M --endpoint-type chat --streaming --url localhost:$port --custom-dataset-type speed_bench_$cat --input-file $SB/aiperf_data/qualitative.jsonl --server-metrics http://localhost:$port/metrics --osl 4096 --extra-inputs temperature:1 top_p:1.0 --concurrency 32 --no-gpu-telemetry --ui-type none --output-artifact-dir $SB/fixtest/art_$lab "$@" > $SB/fixtest/aiperf_$lab.log 2>&1
  echo "[$lab $cat $*] $(curl -s localhost:$port/stats | python3 -c 'import sys,json; d=json.load(sys.stdin); print({k:d[k] for k in ("requests","unique_first_prompts","first_turn_requests","repeated_first_prompts","by_depth")})')" ; kill $FP 2>/dev/null; wait $FP 2>/dev/null
}
for cat in reasoning roleplay; do
  trial base_$cat $cat --request-count $( [ $cat = reasoning ] && echo 190 || echo 158 )
  trial nowrap_$cat $cat --request-count $( [ $cat = reasoning ] && echo 190 || echo 158 ) --no-allow-dataset-wrap
  trial numconv_$cat $cat --num-conversations 80
  trial numconv_nowrap_$cat $cat --num-conversations 80 --no-allow-dataset-wrap
done
echo FIXTEST DONE
