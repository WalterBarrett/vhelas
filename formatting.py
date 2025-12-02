import re
from textwrap import indent
from typing import Any, Match, cast

import mistune
from mistune.block_parser import BlockParser
from mistune.core import BlockState
from mistune.directives import FencedDirective
from mistune.directives._base import BaseDirective, DirectivePlugin
from mistune.markdown import Markdown
from mistune.renderers._list import render_list
from mistune.renderers.html import HTMLRenderer
from mistune.renderers.markdown import MarkdownRenderer


class NamedTable(DirectivePlugin):
    def parse(self, block: BlockParser, m: Match[str], state: BlockState) -> dict[str, Any]:
        title = self.parse_title(m)
        if not title:
            title = "Table"

        tmp = list(self.parse_tokens(block, self.parse_content(m), state))

        for node in tmp:
            if node and node.get("type", None) == "table":
                node_params = node.get("params", None)
                if node_params is None:
                    node["params"] = {}
                    node_params = node["params"]
                node_params["title"] = title

        return {
            "type": "named_table",
            "children": [{
                "type": "named_table_label",
                "text": title,
            }] + tmp,
            "attrs": {},
        }

    def __call__(self, directive: BaseDirective, md: Markdown) -> None:
        directive.register("table", self.parse)

        assert md.renderer is not None
        if md.renderer.NAME == "html":
            def render_named_table(self, text: str, **attrs: Any) -> str:
                return f"<figure>{text}</figure>"

            def render_named_table_label(self, text: str, **attrs: Any) -> str:
                return f'<figcaption>{text}</figcaption>'

            md.renderer.register("named_table", render_named_table)
            md.renderer.register("named_table_label", render_named_table_label)


class PlainTextRenderer(MarkdownRenderer):
    """This is for removing Markdown formatting for parsers."""

    def link(self, token: dict[str, Any], state: BlockState) -> str:
        return self.render_children(token, state)

    def image(self, token: dict[str, Any], state: BlockState) -> str:
        return self.render_children(token, state)

    def list(self, token: dict[str, Any], state: BlockState):
        return render_list(self, token, state)

    def strikethrough(self, token: dict[str, Any], state: BlockState) -> str:
        return ""

    def emphasis(self, token: dict[str, Any], state: BlockState) -> str:
        return self.render_children(token, state)

    def strong(self, token: dict[str, Any], state: BlockState) -> str:
        return self.render_children(token, state)

    def codespan(self, token: dict[str, Any], state: BlockState) -> str:
        return cast(str, token["raw"])

    def linebreak(self, token: dict[str, Any], state: BlockState) -> str:
        return "\n"

    def paragraph(self, token: dict[str, Any], state: BlockState) -> str:
        return f"{self.render_children(token, state)}\n"

    def heading(self, token: dict[str, Any], state: BlockState) -> str:
        return f"{self.render_children(token, state)}:\n"

    def thematic_break(self, token: dict[str, Any], state: BlockState) -> str:
        return "\n"

    def block_code(self, token: dict[str, Any], state: BlockState) -> str:
        attrs = token.get("attrs", {})
        info = cast(str, attrs.get("info", ""))
        code = cast(str, token["raw"])
        match info:
            case "py":
                info = "Python"
        if info:
            return f"{info}:\n{code}\n"
        else:
            return f"{code}\n"

    def block_quote(self, token: dict[str, Any], state: BlockState) -> str:
        return ''.join(['"', indent(self.render_children(token, state), "", lambda _: True).strip("> \n"), '"\n'])

    def admonition(self, token: dict[str, Any], state) -> str:
        kind = self.render_children(token['children'][0], state).strip()
        body = self.render_children(token['children'][1], state).strip()
        if body:
            return f"{kind}: {body}\n"
        return f"{kind}:\n"

    def render_tokens(self, tokens, state):
        output = ''.join(self.render_token(token, state) for token in tokens).strip()
        output = re.sub(r'\n{2,}', '\n', output)
        output = re.sub(r' {2,}', ' ', output)
        output = re.sub(r'\n ', '\n', output)
        return output


def remove_markdown_formatting(markdown: str) -> str:
    if not markdown:
        return ""
    return cast(str, mistune.create_markdown(renderer=PlainTextRenderer(), plugins=['strikethrough'])(markdown))


def markdown_to_html(markdown: str) -> str:
    if not markdown:
        return ""
    return cast(str, mistune.create_markdown(renderer=HTMLRenderer(), plugins=['strikethrough', 'table', FencedDirective([NamedTable()])])(markdown))
