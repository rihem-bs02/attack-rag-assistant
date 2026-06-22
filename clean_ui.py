import re

with open("app6.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove the large CUSTOM_CSS block and replace it with an empty string
css_pattern = re.compile(r'CUSTOM_CSS = """[\s\S]*?"""', re.MULTILINE)
content = css_pattern.sub('CUSTOM_CSS = ""', content)

# 2. Fix the gr.Blocks warning
blocks_pattern = re.compile(r'with gr\.Blocks\(\s*title="MITRE ATT&CK RAG Chatbot",\s*css=CUSTOM_CSS,\s*theme=gr\.themes\.Base\([^)]+\),\s*\) as demo:', re.MULTILINE)
content = blocks_pattern.sub('with gr.Blocks(title="MITRE ATT&CK RAG Chatbot") as demo:', content)

# 3. Remove all the specific emojis
emojis_to_remove = [
    "🛡️ ", "💬 ", "🤖 ", "🧑 ", "🔴 ", "🔵 ", "⚡ ", "🔭 ", "🎯 ", 
    "🔑 ", "💉 ", "📜 ", "👁️ ", "🛡️", "🔓 ", "🚪 ", "⚙️ ", "📂 ", 
    "🔓", "🔽 ", "🎛️ ", "📋 ", "📄 ", "⚠️ ", "📚 ", "🔍  ", "🗑️  "
]

for emoji in emojis_to_remove:
    content = content.replace(emoji, "")

# Some emojis without trailing spaces
content = content.replace("🛡️", "")
content = content.replace("⚠️", "")
content = content.replace("🔗 ", "Link: ")
content = content.replace("💬", "")

with open("app6.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done cleaning UI.")
