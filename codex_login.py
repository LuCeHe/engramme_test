
from openai_codex import Codex

with Codex() as codex:
    login = codex.login_chatgpt()
    print("Open this URL in your browser:")
    print(login.auth_url)

    result = login.wait()
    print("Login successful:", result.success)
