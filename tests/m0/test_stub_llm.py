from trace_bench.adapters.llm import StubLLMAdapter


def test_stub_llm_deterministic():
    messages = [
        {"role": "system", "content": "You are a test"},
        {"role": "user", "content": "Hello"},
    ]
    llm1 = StubLLMAdapter(seed=123)
    llm2 = StubLLMAdapter(seed=123)
    assert llm1.generate(messages) == llm2.generate(messages)
