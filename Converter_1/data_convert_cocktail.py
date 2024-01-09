from openai import OpenAI
import instructor
import json
from graphviz import Digraph
from typing import List, Optional
from pydantic import BaseModel, Field

client = instructor.patch(OpenAI())


class Node(BaseModel):
    """A node in the knowledge graph represents an entity or attribute. It has an ID, and a label. A label can only appear once. For example: Two cocktails (two nodes) both contain sugar (one node), they must use the same node. Each node has a color that represents the class. For example: cocktails have a red color, ingredients have a blue color, ingredients notes have a yellow color and tasting notes have a green color."""
    id: int
    label: str = Field(..., max_length=30)
    color: str
    type: str = Field(..., description="The type of node, e.g., 'cocktail', 'ingredient', etc.")

    def __hash__(self) -> int:
        return hash((self.id, self.label))


class Edge(BaseModel):
    """An edge in the knowledge graph represents a relationship between nodes. It has a source and a target, a label, and a static color (black). Because nodes cannot be duplicated multiple source nodes might connect to the same target node or vice versa."""
    source: int
    target: int
    label: str
    color: str = "black"

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.label))


class KnowledgeGraph(BaseModel):
    nodes: Optional[List[Node]] = Field(..., default_factory=list)
    edges: Optional[List[Edge]] = Field(..., default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with the other graph, deduplicating nodes and edges."""
        new_nodes = self.nodes if self.nodes is not None else []
        new_edges = self.edges if self.edges is not None else []

        if other.nodes is not None:
            new_nodes += other.nodes
        if other.edges is not None:
            new_edges += other.edges

        return KnowledgeGraph(
            nodes=list(set(new_nodes)),
            edges=list(set(new_edges))
        )

    def draw(self, prefix: str = None):
        dot = Digraph(comment="Knowledge Graph", engine='sfdp', format='png')
        dot.attr(dpi='300', ratio='fill', pad='0.5', nodesep='0.5', ranksep='2')
        dot.attr('node', style='filled', shape='box', fontname='Helvetica Neue', fontsize='11')
        dot.attr('edge', fontname='Helvetica Neue', fontsize='10', arrowsize='0.5')

        color_scheme = {
            'red': '#e6194B', 'green': '#3cb44b', 'yellow': '#ffe119',
            'blue': '#4363d8', 'orange': '#f58231', 'purple': '#911eb4',
            'cyan': '#42d4f4', 'magenta': '#f032e6', 'lime': '#bfef45',
            'pink': '#fabebe', 'teal': '#469990', 'lavender': '#e6beff',
            'brown': '#9A6324', 'beige': '#fffac8', 'maroon': '#800000',
            'mint': '#aaffc3', 'olive': '#808000', 'apricot': '#ffd8b1',
            'navy': '#000075', 'grey': '#a9a9a9', 'white': '#ffffff',
            'black': '#000000'
        }

        for node in self.nodes:
            color = color_scheme.get(node.color, '#FFFFFF')
            dot.node(str(node.id), node.label, fillcolor=color, fontcolor='black')

        for edge in self.edges:
            dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)

        self.create_clusters(dot)

        filename = f"{prefix}_graph"
        dot.render(filename, view=True)
        return filename

    def create_clusters(self, dot):
        types = set(node.type for node in self.nodes)
        for node_type in types:
            with dot.subgraph(name=f'cluster_{node_type}') as c:
                c.attr(color='blue', label=node_type.capitalize())
                for node in self.nodes:
                    if node.type == node_type:
                        c.node(str(node.id))


def generate_graph(input_files: List[str]) -> KnowledgeGraph:
    cur_state = KnowledgeGraph()
    num_iterations = len(input_files)
    for i, file_name in enumerate(input_files):
        with open(file_name, 'r') as file:
            json_data = json.load(file)

        new_updates = client.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are an advanced iterative knowledge graph builder, specialized in processing JSON data of cocktails. You are an expert mixologist with a deep understanding of flavor and gastrophysics. Your task involves:
                        1. Parsing the JSON data to extract cocktail-related information including: the name, ingredients, price, glass type, flavor notes of the ingredients and how they contribute to the overall tasting notes.
                        2. Identifying and extracting entities (i.e. cocktail names) and attributes (i.e. ingredients, tasting notes, glass type, price).
                        3. Summarize large bodies of text to enhance the graph's utility and clarity.
                        4. Checking for existing entities and atributes and relationships in the current state of the graph.
                        5. Adding new entities and attributes, avoiding duplicates. If an entity or attribute already exists, use the existing node, do not add a new one with a synonym. 
                        6. Ad new relationships between entities and attributes. Nodes may connect to multiple other nodes. 
                        7. Ensuring accuracy and relevance of the information.
                        8. Synthesizing this information to enhance the graph's utility and clarity.
                        Your role is critical in creating a comprehensive and coherent knowledge graph.
                        9. Design the graph so that it is easy to understand and use.

                        Take a deep breath and execute your task step by step. You will receive a generous tip if you complete your task successfully and in great detail with no errors and showing deep understanding of the the relationships between entities and attributes.

                        # Ensure that related concepts are visually grouped together by placing them close spatially.
                    """
                },
                {
                    "role": "user",
                    "content": f"""Extract nodes and edges from this JSON data using the following:
                        # Part {i+1}/{num_iterations} of the input:
                        {json.dumps(json_data, indent=2)}
                    """
                },
                {
                    "role": "user",
                    "content": f"""Current state of the graph:
                        {cur_state.model_dump_json(indent=2)}
                    """
                },
            ],
            response_model=KnowledgeGraph,
        )  

        cur_state = cur_state.update(new_updates)
        cur_state.draw(prefix=f"iteration_{i+1}")
    return cur_state


file_names = ['file_1.json', 'file_2.json', 'file_3.json']
graph: KnowledgeGraph = generate_graph(file_names)
graph.draw(prefix="final")
