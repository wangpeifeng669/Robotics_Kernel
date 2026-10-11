# -*- coding: utf-8 -*-
"""
技术文档库站点生成器。
扫描仓库根目录下已登记分类的 Markdown，生成自包含单文件 HTML。
用法：python3.6 build_site.py <输出.html>
"""
import json
import os
import re
import sys
import tempfile

# 分类顺序与展示元数据（icon 对应模板内 SVG 精灵）
CATEGORY_META = [
    ("语音交互", "语音交互 Pipeline：ASR / VAD / TTS / 唤醒", "mic", "#0d9488"),
    ("端侧部署", "边缘设备模型部署：RK3588 / 量化 / 推理优化", "chip", "#0891b2"),
    ("模型原理", "模型架构与原理：Transformer / 注意力 / 量化", "brain", "#2563eb"),
    ("模型训练", "预训练、微调、数据准备与调参实验", "chart", "#7c3aed"),
    ("数据工程", "数据处理、特征工程、RAG 与向量数据库", "database", "#d97706"),
    ("通信架构", "板级通信、网络协议与服务架构", "network", "#dc2626"),
    ("多模态业务", "VLM、视觉问答与多模态业务集成", "layers", "#db2777"),
    ("智能体开发", "Agent 编排：LangGraph / CrewAI / AutoGen", "agent", "#9333ea"),
    ("前沿分析", "行业前沿、范式演进与工具动态", "trend", "#4f46e5"),
    ("AI工具思考", "AI 辅助开发的使用思考与方法论", "tool", "#65a30d"),
    ("日常运维", "运维自动化、CI/CD、镜像与部署脚本", "server", "#64748b"),
    ("个人哲思", "佛学、道德经、哲学与心理学的实践感悟", "book", "#b45309"),
]
META_BY_NAME = {item[0]: item for item in CATEGORY_META}


def site_dir():
    return os.path.dirname(os.path.abspath(__file__))


def kernel_root():
    return os.path.dirname(site_dir())


def output_path():
    if len(sys.argv) != 2:
        sys.stderr.write("用法: python3.6 build_site.py <输出.html>\n")
        sys.exit(1)
    return os.path.abspath(sys.argv[1])


def read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def strip_frontmatter(text):
    if text.startswith("---"):
        matched = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.S)
        if matched:
            return text[matched.end():]
    return text


def extract_title(text, fallback):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def extract_summary(text):
    """优先取核心摘要，否则取标题后第一个正文段落。"""
    lines = text.splitlines()
    for line in lines[:12]:
        if "核心摘要" not in line:
            continue
        summary = re.sub(r"^.*?核心摘要[*]*\s*[：:]\s*", "", line.strip())
        summary = summary.replace("**", "").strip()
        if summary:
            return summary
    return first_paragraph(lines)


def first_paragraph(lines):
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", ">", "---", "***", "```", "|", "- ", "* ", "![", "    ")):
            continue
        stripped = re.sub(r"[*_`]", "", stripped)
        if len(stripped) >= 8:
            return stripped
    return ""


def parse_doc(category, path, rel):
    text = strip_frontmatter(read_text(path))
    filename = os.path.splitext(os.path.basename(path))[0]
    matched = re.match(r"^(\d{4}-\d{2}-\d{2})_(.+)$", filename)
    if matched:
        date, slug = matched.group(1), matched.group(2)
    else:
        date, slug = "", filename
    return {
        "cat": category,
        "title": extract_title(text, slug),
        "date": date,
        "summary": extract_summary(text),
        "file": rel,
        "content": text,
    }


def scan_category(root, name, full, docs, intros):
    for current, _, files in os.walk(full):
        for filename in files:
            if not filename.endswith(".md"):
                continue
            path = os.path.join(current, filename)
            rel = os.path.relpath(path, root).replace("\\", "/")
            if filename.lower() == "readme.md":
                intros[name] = strip_frontmatter(read_text(path))
                continue
            docs.append(parse_doc(name, path, rel))


def scan_repo(root):
    """只收录分类元数据里登记过的目录，避免把工具缓存扫进网页。"""
    docs = []
    intros = {}
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full) or name.startswith("."):
            continue
        if name not in META_BY_NAME:
            continue
        scan_category(root, name, full, docs, intros)
    return docs, intros


def build_categories(intros, docs):
    counter = {}
    for doc in docs:
        name = doc["cat"]
        counter[name] = counter.get(name, 0) + 1
    categories = []
    for name, desc, icon, color in CATEGORY_META:
        count = counter.get(name, 0)
        if count == 0 and name not in intros:
            continue
        categories.append({
            "name": name,
            "desc": desc,
            "icon": icon,
            "color": color,
            "count": count,
            "intro": intros.get(name, ""),
        })
    return categories


def assign_ids(docs):
    docs.sort(key=lambda doc: (doc["date"] == "", doc["date"]), reverse=False)
    docs.sort(key=lambda doc: doc["date"] or "0000", reverse=True)
    for index, doc in enumerate(docs):
        doc["id"] = index


def render(categories, docs, about):
    marked_src = read_text(os.path.join(site_dir(), "marked.min.js"))
    template = read_text(os.path.join(site_dir(), "template.html"))
    payload = json.dumps(
        {"categories": categories, "docs": docs},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    about_js = json.dumps(about, ensure_ascii=False).replace("</", "<\\/")
    return (
        template
        .replace("/*__MARKED__*/", marked_src)
        .replace("\"__DATA__\"", payload)
        .replace("\"__ABOUT__\"", about_js)
    )


def write_atomic(path, html):
    """先写同目录临时文件再替换，中途失败时保留原来的线上页面。"""
    directory = os.path.dirname(path) or "."
    if not os.path.isdir(directory):
        os.makedirs(directory)
    fd, tmp = tempfile.mkstemp(prefix=".docs-", suffix=".html", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(html)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    # mkstemp 默认 600，httpd 以 apache 用户读页面，必须放开为 644
    os.chmod(path, 0o644)


def print_stats(categories, docs, out_html, html):
    print("分类数: %d" % len(categories))
    print("文档数: %d" % len(docs))
    for category in categories:
        print("  %-8s %3d 篇" % (category["name"], category["count"]))
    print("输出: %s" % out_html)
    print("大小: %.1f KB" % (len(html.encode("utf-8")) / 1024))


def main():
    root = kernel_root()
    docs, intros = scan_repo(root)
    categories = build_categories(intros, docs)
    assign_ids(docs)
    about = strip_frontmatter(read_text(os.path.join(root, "README.md")))
    html = render(categories, docs, about)
    out_html = output_path()
    write_atomic(out_html, html)
    print_stats(categories, docs, out_html, html)


if __name__ == "__main__":
    main()
