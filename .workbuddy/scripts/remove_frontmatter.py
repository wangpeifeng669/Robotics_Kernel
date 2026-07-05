#!/usr/bin/env python3
"""
批量移除知识库文章的 YAML frontmatter，并在顶部添加 H1 标题。
"""

import os
import re

ROOT = 'D:/wpf/study/Robotics_Kernel'
SKIP_DIRS = {'.workbuddy', 'node_modules', '.git'}

def extract_title_from_filename(filename):
    name = os.path.splitext(filename)[0]
    m = re.match(r'\d{4}-\d{2}-\d{2}_(.+)', name)
    return m.group(1) if m else name

def remove_frontmatter(content):
    lines = content.split('\n')
    if not lines or lines[0].strip() != '---':
        return content, False
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == '---':
            end_idx = i
            break
    if end_idx == -1:
        return content, False
    return '\n'.join(lines[end_idx + 1:]).strip(), True

def has_h1_title(content):
    for line in content.split('\n'):
        if line.startswith('# ') and not line.startswith('## '):
            return True
    return False

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    content, removed = remove_frontmatter(content)
    if not removed:
        return False, 'no_frontmatter'
    title = extract_title_from_filename(os.path.basename(filepath))
    if has_h1_title(content):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, 'kept_existing_h1'
    else:
        new_content = f'# {title}\n\n{content}'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True, 'added_h1'

def main():
    processed = []
    skipped = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if not filename.endswith('.md') or filename == 'README.md':
                continue
            filepath = os.path.join(dirpath, filename)
            try:
                ok, reason = process_file(filepath)
                if ok:
                    processed.append((filepath, reason))
                else:
                    skipped.append((filepath, reason))
            except Exception as e:
                skipped.append((filepath, str(e)))

    print(f'处理完成：{len(processed)} 个文件已更新，{len(skipped)} 个跳过')
    print('\n已更新：')
    for filepath, reason in processed:
        print(f'  {os.path.relpath(filepath, ROOT)} [{reason}]')
    if skipped:
        print('\n跳过：')
        for filepath, reason in skipped:
            print(f'  {os.path.relpath(filepath, ROOT)} [{reason}]')

if __name__ == '__main__':
    main()
