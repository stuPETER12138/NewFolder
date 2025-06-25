"""
This script generates a tree structure of a directory up to a specified depth and writes it to a markdown file.
"""

import os


def generate_tree(path: str, max_depth: int, depth=0) -> str:
    if depth >= max_depth:
        return ""
    structure_tree = ""
    if os.path.isdir(path):
        basename = os.path.basename(path)
        structure_tree += f"{'  ' * depth}- **{basename}/**\n"
        for name in sorted(os.listdir(path)):
            structure_tree += generate_tree(
                os.path.join(path, name), max_depth, depth + 1
            )
    else:
        basename = os.path.basename(path)
        structure_tree += f"{'  ' * depth}- {basename}\n"
    return structure_tree


if __name__ == "__main__":
    root_dir = "D:\Coding\stuPETER12138.github.io"  # 指定项目目录
    output_file = os.path.join(root_dir, "strct.md")  # 输出文件
    tree = generate_tree(root_dir, 3)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("### 文件结构\n\n")
        f.write(tree)
        f.write("\n")
