from openai import OpenAI
import instructor
from graphviz import Digraph
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# Patch OpenAI client with instructor
client = instructor.patch(OpenAI())

class Node(BaseModel):
    """
    A node in the knowledge graph representing a concept, or key point from the text input.
    Each node should be visually distinct to reflect its importance and should maintain clear connections to related nodes.
    """
    id: int
    label: str = Field(..., description="A brief and clear description of the concept, or key point.")
    color: str = Field(..., description="A color that represents the category or type of the node for visual distinction.")
    related_nodes: List[int] = Field(..., description="IDs of nodes that have a direct relationship with this node")
    importance: float = Field(..., description="A value between 0 and 1 indicating the node's importance, which will affect its size in the graph.")

    def __hash__(self):
        return hash((self.id, self.label))

class Edge(BaseModel):
    """
    An edge in the knowledge graph represents the relationship between two nodes.
    """
    source: int = Field(..., description="The ID of the source node in the relationship.")
    target: int = Field(..., description="The ID of the target node in the relationship.")
    label: str = Field(..., description="A concise description of the relationship between the nodes.")
    color: str = Field("black", description="The color of the edge")
    weight: float = Field(..., description="A value between 0 and 1 representing the strength of the connection, influencing the edge's thickness and style")
    style: Literal["dotted", "solid", "bold"] = Field(description="The visual style of the edge representing the weight of the connection.")

    def __hash__(self):
        return hash((self.source, self.target, self.label))

class KnowledgeGraph(BaseModel):
    nodes: Optional[List[Node]] = Field(default_factory=list)
    edges: Optional[List[Edge]] = Field(default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with the other graph, deduplicating nodes and edges."""
        return KnowledgeGraph(
            nodes=list(set(self.nodes + other.nodes)),
            edges=list(set(self.edges + other.edges)),
        )


    def draw(self, prefix: str = None):
        dot = Digraph(comment="Knowledge Graph")

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

        # Node customization based on importance
        for node in self.nodes:
            importance_factor = node.importance * 0.5  # Adjusting size based on importance
            size = str(0.5 + importance_factor)  # Node size based on importance
            node_color = fill_colors.get(node.color, 'white')  # Default to white if color not found
            shape = 'ellipse'  # Using ellipse as default shape
            dot.node(str(node.id), node.label, shape=shape, style='filled', fillcolor=node_color, width=size, height=size)

         # Edge customization based on weight and style
        for edge in self.edges:
            edge_color = fill_colors.get(edge.color, 'black')  # Use the fill colors mapping for edges
            style = edge.style if edge.style else "solid"  # Default style is solid
            dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge_color, style=style, dir="forward")

        dot.render(prefix, format='png', view=True)

    



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

1. Read the text to identify key concepts, and their relationships.
2. Create nodes for each identified item, assigning them a unique ID, label, and type. 
3. Determine the importance of each node and assign a relative size in the graph. The more important a node, the larger it should be, use a 0-1 scale to represent importance.
4. Create edges to represent the relationships between nodes. Specify the source, target, label, color, weight, and style for each edge. The style ranges from dotted to solid to bold, with dotted edges representing weaker relationships and bold edges representing stronger relationships.
5. Pay special attention to nodes that have multiple relationships with other nodes, as these are crucial for understanding the interrelatedness of concepts.
6. Translate this information into DOT language for Graphviz.
7. Render the knowledge graph, making sure it is easy to read and visually cohesive, with importance and relationships clearly depicted.

Do NOT duplicate notes, if you see a semantically similar node, use the existing node instead of creating a new one.
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
