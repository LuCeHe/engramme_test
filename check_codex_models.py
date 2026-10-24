
from openai_codex import Codex

with Codex() as codex:
    print("Codex runtime metadata:")
    print(codex.metadata)

    print("\nModels available to this Codex session:")
    try:
        result = codex.models()
        print(result)
    except Exception as exc:
        print(f"Could not list models: {exc}")
