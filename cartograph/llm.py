"""LLM generators behind one interface: ``generate(messages) -> str`` and ``generate_batch``.

- HFGenerator: Transformers, greedy, one request at a time (the sequential reference).
- VLLMGenerator: vLLM offline engine with the same chat template and sampling (continuous batching).
- OpenAIGenerator: any OpenAI-compatible server (vllm serve, Ray Serve LLM).
All use greedy decoding and the tokenizer's chat template, so prompts are identical token sequences.
"""

import time

MODELS = {
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b-instruct-awq": ("Qwen/Qwen2.5-1.5B-Instruct-AWQ", "3ecffa0ceb27851800f45519bab9c457a04405e1"),
    "qwen2.5-1.5b-instruct-gptq-int4": ("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "4f5a0e31008e2966e8f787964bcc4e1105e03dc3"),
}


def resolve(name: str) -> tuple[str, str | None]:
    return MODELS.get(name, (name, None))


class HFGenerator:
    def __init__(self, model: str, revision: str | None = None, adapter: str | None = None,
                 device: str = "cuda", max_new_tokens: int = 256, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=getattr(torch, dtype)).to(device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.device, self.max_new_tokens = device, max_new_tokens

    def generate(self, messages: list[dict]) -> str:
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def generate_batch(self, batch: list[list[dict]]) -> list[str]:
        return [self.generate(m) for m in batch]


class VLLMGenerator:
    def __init__(self, model: str, revision: str | None = None, adapter: str | None = None,
                 max_new_tokens: int = 256, max_model_len: int = 2048, gpu_memory_utilization: float = 0.8,
                 quantization: str | None = None, enable_prefix_caching: bool = True, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, revision=revision, max_model_len=max_model_len, enable_lora=bool(adapter),
                       gpu_memory_utilization=gpu_memory_utilization, quantization=quantization,
                       enable_prefix_caching=enable_prefix_caching, tensor_parallel_size=tensor_parallel_size,
                       seed=0)
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("adapter", 1, adapter)

    def generate_batch(self, batch: list[list[dict]]) -> list[str]:
        outputs = self.llm.chat(batch, self.params, lora_request=self.lora, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def generate(self, messages: list[dict]) -> str:
        return self.generate_batch([messages])[0]


class OpenAIGenerator:
    def __init__(self, base_url: str, model: str, max_new_tokens: int = 256, timeout: float = 120):
        self.url, self.model, self.max_new_tokens, self.timeout = base_url.rstrip("/"), model, max_new_tokens, timeout

    def generate(self, messages: list[dict]) -> str:
        import json
        import urllib.request
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.0,
                           "max_tokens": self.max_new_tokens}).encode()
        request = urllib.request.Request(f"{self.url}/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"]

    def generate_batch(self, batch):
        return [self.generate(m) for m in batch]


class SolverGenerator:
    """Deterministic baseline: reads the solver answer embedded in the task (via a lookup)."""

    def __init__(self, tasks_by_prompt: dict):
        self.tasks_by_prompt = tasks_by_prompt

    def generate(self, messages):
        from .assistant import solver_reply
        return solver_reply(self.tasks_by_prompt[messages[1]["content"]])

    def generate_batch(self, batch):
        return [self.generate(m) for m in batch]


def timed(generator):
    """Wrap generate() to record per-call wall time."""
    calls = []

    def generate(messages):
        start = time.perf_counter()
        text = generator.generate(messages)
        calls.append(time.perf_counter() - start)
        return text
    return generate, calls
