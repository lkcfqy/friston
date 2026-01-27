import os
from typing import List, Generator, Optional
import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_python

class CodeWalker:
    @staticmethod
    def walk(root_dir: str, extensions: List[str] = [".py"]) -> Generator[str, None, None]:
        for root, _, files in os.walk(root_dir):
            if "site-packages" in root or ".git" in root or "__pycache__" in root:
                continue
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    yield os.path.join(root, file)

class ASTParser:
    def __init__(self):
        try:
            # New tree-sitter API in recent versions
            self.PY_LANGUAGE = Language(tree_sitter_python.language())
        except Exception:
            # Fallback for older versions: Language(ptr, name)
            self.PY_LANGUAGE = Language(tree_sitter_python.language(), "python")
            
        self.parser = Parser()
        self.parser.set_language(self.PY_LANGUAGE)

    def parse(self, code: str) -> tree_sitter.Tree:
        return self.parser.parse(bytes(code, "utf8"))

    def get_functions(self, tree: tree_sitter.Tree) -> List[tree_sitter.Node]:
        """Retrieve all function definition nodes."""
        query_scm = """
        (function_definition) @function
        """
        query = self.PY_LANGUAGE.query(query_scm)
        captures = query.captures(tree.root_node)
        all_nodes = [node for node, _ in captures]
        return all_nodes

    def get_classes(self, tree: tree_sitter.Tree) -> List[tree_sitter.Node]:
        """Retrieve all class definition nodes."""
        query_scm = """
        (class_definition) @class
        """
        query = self.PY_LANGUAGE.query(query_scm)
        captures = query.captures(tree.root_node)
        all_nodes = [node for node, _ in captures]
        return all_nodes

    def get_code_from_node(self, node: tree_sitter.Node) -> str:
        return node.text.decode("utf8")
