from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flask import Flask, render_template_string


@dataclass(frozen=True)
class MappingNode:
	node_id: str
	label: str


@dataclass(frozen=True)
class MappingEdge:
	source: str
	target: str


class MermaidFlowBuilder:
	"""Build Mermaid flowchart markup from nodes and directed edges."""

	def __init__(self, direction: str = "LR"):
		self.direction = direction

	def build(self, nodes: list[MappingNode], edges: list[MappingEdge]) -> str:
		lines: list[str] = [f"flowchart {self.direction}"]

		for node in nodes:
			safe_label = node.label.replace('"', "\\\"").replace("\n", " ")
			lines.append(f"    {node.node_id}[\"{safe_label}\"]")

		for edge in edges:
			lines.append(f"    {edge.source} --> {edge.target}")

		return "\n".join(lines)


class HtmlTemplateLoader:
	"""Load the html shell and inject the Mermaid graph definition."""

	def __init__(self, template_path: Path):
		self.template_path = template_path

	def render(self, mermaid_graph: str) -> str:
		template = self.template_path.read_text(encoding="utf-8")
		return template.replace("__MERMAID_GRAPH__", mermaid_graph)


class MidiMappingVisualizerApp:
	def __init__(self):
		self.app = Flask(__name__)
		self.template_loader = HtmlTemplateLoader(
			Path(__file__).with_name("midi_mapping_visualizer_app.html")
		)
		self.flow_builder = MermaidFlowBuilder(direction="LR")
		self._register_routes()

	def _graph_nodes(self) -> list[MappingNode]:
		return [
			MappingNode("gm_in", "Generic MIDI Mapping (Input)"),
			MappingNode("cgm", "Compressed Generic MIDI Mapping"),
			MappingNode("vae", "VAE Model"),
			MappingNode("gm_out", "Generic MIDI Mapping (Output)"),
			MappingNode("groove", "Groove MIDI Mapping"),
		]

	def _graph_edges(self) -> list[MappingEdge]:
		return [
			MappingEdge("gm_in", "cgm"),
			MappingEdge("groove", "cgm"),
			MappingEdge("cgm", "vae"),
			MappingEdge("vae", "gm_out"),
		]

	def _register_routes(self) -> None:
		@self.app.get("/")
		def index() -> str:
			graph = self.flow_builder.build(self._graph_nodes(), self._graph_edges())
			html = self.template_loader.render(graph)
			return render_template_string(html)

	def run(self, host: str = "127.0.0.1", port: int = 5050, debug: bool = False) -> None:
		self.app.run(host=host, port=port, debug=debug)


def main() -> None:
	MidiMappingVisualizerApp().run()


if __name__ == "__main__":
	main()
