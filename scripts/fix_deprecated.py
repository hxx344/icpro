"""Fix deprecated Streamlit API usage."""
import re

with open("trader/dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace use_container_width=True -> width="stretch"
content = content.replace("use_container_width=True", 'width="stretch"')
# Replace use_container_width=False -> width="content"
content = content.replace("use_container_width=False", 'width="content"')

with open("trader/dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done:", content.count('width="stretch"'), "replacements")
