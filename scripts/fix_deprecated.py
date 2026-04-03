"""Fix legacy Streamlit width arguments for current dashboard code."""

with open("trader/dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace legacy width arguments with broadly compatible use_container_width
content = content.replace('width="stretch"', "use_container_width=True")
content = content.replace("width='stretch'", "use_container_width=True")
content = content.replace('width="content"', "use_container_width=False")
content = content.replace("width='content'", "use_container_width=False")

with open("trader/dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done:", content.count("use_container_width=True"), "stretch replacements applied")
