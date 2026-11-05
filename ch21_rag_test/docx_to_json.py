"""
Word 文档转 JSON 工具
把 Word 格式的知识库解析成 list[dict]，存到 knowledge_docs.json
"""

import json
import docx  # pip install python-docx


def load_docx(filepath):
    """读取 Word 文档，按标题段落切分成文档块"""
    doc = docx.Document(filepath)

    docs = []
    current_title = ""
    current_content = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if para.style.name.startswith("Heading"):
            if current_title and current_content:
                docs.append({
                    "doc_id": f"DOC-{len(docs)+1:02d}",
                    "title": current_title,
                    "content": "\n".join(current_content),
                })
            current_title = text
            current_content = []
        else:
            current_content.append(text)

    if current_title and current_content:
        docs.append({
            "doc_id": f"DOC-{len(docs)+1:02d}",
            "title": current_title,
            "content": "\n".join(current_content),
        })

    return docs


docs = load_docx("员工手册.docx")
print(f"已加载 {len(docs)} 篇文档")

for doc in docs:
    print(f"  [{doc['doc_id']}] {doc['title']}（{len(doc['content'])} 字符）")

with open("knowledge_docs.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)
print(f"已保存到 knowledge_docs.json")