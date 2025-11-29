import ollama
response = ollama.chat(model="llama3", messages=[{"role":"user","content":"こんにちわ"}])
print(response["message"]["content"])
