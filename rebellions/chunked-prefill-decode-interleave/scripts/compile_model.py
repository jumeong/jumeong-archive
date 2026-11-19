import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from optimum.rbln import RBLNAutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
OUT = ROOT / 'models/compiled-b4-s4096'

if __name__ == '__main__':
    source = snapshot_download('Qwen/Qwen3-4B', revision=REVISION,
                               allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
    config = dict(rbln_create_runtimes=False, rbln_batch_size=4,
                  rbln_max_seq_len=4096, rbln_kvcache_block_size=4096,
                  rbln_kvcache_num_blocks=4, rbln_decoder_batch_sizes=[4],
                  rbln_prefill_chunk_size=128, rbln_attn_impl='eager',
                  rbln_tensor_parallel_size=1)
    (ROOT / 'metadata/compile_manifest.json').write_text(json.dumps(dict(
        model='Qwen/Qwen3-4B', revision=REVISION, source=source,
        torch_dtype='bfloat16', quantization=None, config=config), indent=2))
    print('Compiling original BF16 weights:', source, config, flush=True)
    model = RBLNAutoModelForCausalLM.from_pretrained(
        source, export=True, torch_dtype=torch.bfloat16, **config)
    model.save_pretrained(OUT)
    AutoTokenizer.from_pretrained(source).save_pretrained(OUT)
    print('Saved:', OUT, flush=True)
