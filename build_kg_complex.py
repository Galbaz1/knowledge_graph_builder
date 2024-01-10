from openai import OpenAI
import instructor
from graphviz import Digraph
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Patch OpenAI client with instructor
client = instructor.patch(OpenAI())

class Node(BaseModel):
    """
    A node in the knowledge graph representing a concept, entity, attribute, or key point from the text input.
    Each node should be visually distinct to reflect its importance and should maintain clear connections to related nodes, even across clusters.
    """
    id: int
    label: str = Field(..., description="A brief and clear description of the concept, entity, attribute, or key point from the text input.")
    color: str = Field(..., description="A color that represents the category or type of the node for visual distinction.")
    type: str = Field(..., description="A category that this node belongs to, which will determine its cluster in the graph. Common types include 'idea', 'concept', 'attribute', 'entity', etc.")
    related_nodes: List[int] = Field(..., description="IDs of nodes that have a direct relationship with this node, allowing for connections across clusters.")
    importance: float = Field(..., description="A value between 0 and 1 indicating the node's importance, which will affect its size in the graph.")

    def __hash__(self):
        return hash((self.id, self.label))

class Edge(BaseModel):
    """
    An edge in the knowledge graph represents the relationship between two nodes.
    It should visually indicate the nature and strength of the connection, including across clusters.
    """
    source: int = Field(..., description="The ID of the source node in the relationship.")
    target: int = Field(..., description="The ID of the target node in the relationship.")
    label: str = Field(..., description="A concise description of the relationship between the nodes.")
    color: str = Field("black", description="The color of the edge, which may vary to represent different types of relationships.")
    weight: float = Field(..., description="A value between 0 and 1 representing the strength of the connection, influencing the edge's thickness.")
    style: Literal["solid", "dashed", "dotted", "bold", "invis", "tapered"] = Field(
        "solid", description="The visual style of the edge, which can indicate different kinds of relationships.")

    def __hash__(self):
        return hash((self.source, self.target, self.label))

class KnowledgeGraph(BaseModel):
    nodes: Optional[List[Node]] = Field(default_factory=list)
    edges: Optional[List[Edge]] = Field(default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with another, deduplicating nodes and edges."""
        updated_nodes = {**{node.id: node for node in self.nodes}, **{node.id: node for node in other.nodes}}
        updated_edges = set(self.edges + other.edges)
        return KnowledgeGraph(nodes=list(updated_nodes.values()), edges=list(updated_edges))

    def draw(self, prefix: str = None):
        dot = Digraph(comment="Knowledge Graph")
        dot.attr(rankdir='TB', nodesep='0.6', ranksep='1.2')  # Adjust nodesep and ranksep as needed for better spacing

        # Lighten the fill colors for better readability
        fill_colors = {
            'blue': '#8ab4f8',   # Lighter blue
            'red': '#f87171',    # Lighter red
            'orange': '#fdba74', # Lighter orange
            'green': '#34d399',  # Lighter green
            'purple': '#a78bfa', # Lighter purple
            'grey': '#9ca3af',   # Lighter grey
            'yellow': '#fde047', # Lighter yellow
        }

        # Define attributes for clusters
        cluster_attributes = {
            'style': 'filled',
            'color': 'lightgrey',
            'labeljust': 'l',
            'fontcolor': 'black',
            'fontname': 'Helvetica',
            'fontsize': '12'
        }

        # Common sections in business plans, which will be our clusters
        sections = [
            'executive_summary',
            'market_analysis',
            'product_description',
            'marketing_strategy',
            'organizational_structure',
            'financial_projections',
            'sales_strategy',
            'competitive_analysis',
            'operational_plan',
            'appendix'
        ]

        # Creating clusters for each section
        for section in sections:
            with dot.subgraph(name=f'cluster_{section}') as c:
                c.attr(**cluster_attributes, label=section.replace('_', ' ').title())
                for node in filter(lambda n: n.type == section, self.nodes):
                    # Set fillcolor based on the type of node
                    node_color = fill_colors.get(node.color, 'white')  # Default to white if color not found
                    c.node(str(node.id), node.label, style='filled', fillcolor=node_color)

        # Node customization based on importance
        for node in self.nodes:
            if node.type not in sections:  # Only customize nodes not already in clusters
                importance_factor = node.importance * 0.5  # Adjusting size based on importance
                size = str(0.5 + importance_factor)  # Node size based on importance
                node_color = fill_colors.get(node.color, 'white')  # Default to white if color not found
                shape = 'box' if node.type in sections else 'ellipse'
                dot.node(str(node.id), node.label, shape=shape, style='filled', fillcolor=node_color, width=size, height=size)

        # Edge customization based on weight
        for edge in self.edges:
            weight_factor = edge.weight * 2  # Factor to amplify the visual impact of weight
            penwidth = str(1 + weight_factor) if edge.weight > 0 else '1'
            edge_color = fill_colors.get(edge.color, 'black')  # Use the fill colors mapping for edges too
            dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge_color, penwidth=penwidth, dir="forward")

        # Render the graph to a file (the prefix can include a directory path)
        dot.render(prefix, format='png', view=True)
        #output_path = f'{prefix if prefix else "graph"}.png'
        #dot.render(output_path, view=False, format='png')
        #print(f'Graph is saved as {output_path}')



def generate_graph(input: List[str]) -> KnowledgeGraph:
    """Generates the knowledge graph from the given input text chunks."""
    cur_state = KnowledgeGraph()
    for i, inp in enumerate(input):
        new_updates = client.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=0.5,
            messages=[
                {"role": "system", "content": """
You are an iterative knowledge graph builder. Your task is to analyze text inputs and organize the extracted information into a comprehensive knowledge graph. Here's how you will proceed:

1. Read the text to identify key concepts, entities, attributes, and their relationships.
2. Create nodes for each identified item, assigning them a unique ID, label, and type. The 'type' will correspond to a cluster in the graph.
3. Determine the importance of each node and assign a relative size in the graph. The more important a node, the larger it should be.
4. Create edges to represent the relationships between nodes. Specify the source, target, label, color, weight, and style for each edge.
5. Pay special attention to relationships that cross different clusters, as these are crucial for understanding the interrelatedness of concepts.
6. Translate this information into DOT language for Graphviz, organizing nodes into clusters but ensuring that cross-cluster relationships are visible and clear.
7. Render the knowledge graph, making sure it is easy to read and visually cohesive, with importance and relationships clearly depicted.

Your output should include DOT language statements defining nodes and edges, as well as cluster subgraphs that collectively form the entire knowledge graph.
"""
                
},
                {"role": "user", "content": f"Extract new nodes and edges from: # Part {i+1}/{len(input)} of the input: {inp}"},
                {"role": "user", "content": f"Here is the current state of the graph: {cur_state.model_dump_json(indent=2)}"},
            ],
            response_model=KnowledgeGraph,
        )
        cur_state = cur_state.update(new_updates)
        cur_state.draw(prefix=f"iteration_{i}")

    return cur_state

# Processing text chunks from business plan
file_contents = []
for part in ['Business_Plan_Part_1.txt', 'Business_Plan_Part_2.txt', 'Business_Plan_Part_3.txt']:
    with open(part, 'r') as file:
        file_contents.append(file.read())

graph = generate_graph(file_contents)
graph.draw(prefix="final")
