from openai import OpenAI
import instructor
from graphviz import Digraph
from typing import List, Optional
from pydantic import BaseModel, Field

client = instructor.patch(OpenAI())


class Node(BaseModel):
    """A node in the knowledge graph represents a concept, entity, attribute or key point from the business plan. It has a label, a color for visualization, and additional attributes to provide more context about the nature of the node."""
    id: int
    chain_of_thought: str = Field(
        ..., 
        description="Step by step reasoning for the node's label and its relation to other nodes"
    )
    label: str = Field(
        ..., 
        description="Specific concept, entity, attribute or key point from the business plan, like 'SWOT Analysis', 'Funding Requirements', 'Company Name' etc."
    )
    color: str
    type: str = Field(
        ..., 
        description="The type of node, e.g., 'idea', 'concept', 'attribute', 'entity', etc..."
    )
    related_nodes: List[int] = Field(
        ..., 
        description="Correct and complete list of node IDs, representing relationships between nodes."
    )
    importance: float = Field(
        ..., 
        description="A numerical value indicating the importance of the node, determined based on analysis or relevance."
    )

    def __hash__(self) -> int:
        return hash((self.id, self.label))


class Edge(BaseModel):
    """
    An edge in the knowledge graph represents a relationship between nodes.
    It has a source and a target, a label to describe the relationship, a color for visualization, 
    and additional attributes to provide more context about the nature of the connection.
    """
    source: int = Field(
        ..., 
        description="The ID of the source node in the relationship."
    )
    target: int = Field(
        ..., 
        description="The ID of the target node in the relationship."
    )
    label: str = Field(
        ..., 
        description="A descriptive label for the relationship, such as 'supports', 'contradicts', 'expands upon', etc."
    )
    color: str = Field(
        "black", 
        description="The color of the edge in the graph, used for visualization purposes."
    )
    weight: float = Field(
        1.0,
        description="A numerical value indicating the strength or significance of the relationship."
    )
    style: str = Field(
        "solid",
        description="The style of the edge line in the graph, such as 'solid', 'dashed', 'dotted', etc."
    )

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.label))



class KnowledgeGraph(BaseModel):
    nodes: Optional[List[Node]] = Field(..., default_factory=list)
    edges: Optional[List[Edge]] = Field(..., default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with the other graph, deduplicating nodes and edges."""
        return KnowledgeGraph(
            nodes=list(set(self.nodes + other.nodes)),
            edges=list(set(self.edges + other.edges)),
        )

    def draw(self, prefix: str = None):
        dot = Digraph(comment="Knowledge Graph")
        dot.attr(rankdir='LR')  # Left to Right layout

        # Customize node attributes
        dot.attr('node', shape='circle', fontsize='12', fontname='Helvetica')
        
        # Add nodes with specific shapes based on type
        for node in self.nodes:
            if node.type == 'idea':
                shape = 'ellipse'
            elif node.type == 'concept':
                shape = 'box'
            else:  # default shape
                shape = 'circle'
            dot.node(str(node.id), node.label, color=node.color, shape=shape)

        # Customize edge attributes
        dot.attr('edge', fontsize='10', fontname='Helvetica')

        # Add edges with specific styles if needed
        for edge in self.edges:
            style = 'solid'  # default style
            if edge.label == 'special_relationship':
                style = 'dashed'
            dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color, style=style)

        dot.render(prefix, format="png", view=True)

def draw(self, prefix: str = None):
    dot = Digraph(comment="Knowledge Graph")
    dot.attr(rankdir='LR')  # Left to Right layout

    # Customize node attributes
    dot.attr('node', shape='circle', fontsize='12', fontname='Helvetica')
    
    # Add nodes with specific shapes based on type
    for node in self.nodes:
        if node.type == 'idea':
            shape = 'ellipse'
        elif node.type == 'concept':
            shape = 'box'
        else:  # default shape
            shape = 'circle'
        dot.node(str(node.id), node.label, color=node.color, shape=shape)

    # Customize edge attributes
    dot.attr('edge', fontsize='10', fontname='Helvetica')

    # Add edges with specific styles if needed
    for edge in self.edges:
        style = 'solid'  # default style
        if edge.label == 'special_relationship':
            style = 'dashed'
        dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color, style=style)

    dot.render(prefix, format="png", view=True)



def generate_graph(input: List[str]) -> KnowledgeGraph:
    cur_state = KnowledgeGraph()
    num_iterations = len(input)
    for i, inp in enumerate(input):
        new_updates = client.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=0.8,

            messages=[
                {
                    "role": "system",
                    "content": """You are an iterative knowledge graph builder.
                    You are given the current state of the graph, and you must append the nodes and edges to it. Do not procide any duplcates and try to reuse nodes as much as possible.""",
                },
                {
                    "role": "user",
                    "content": f"""Extract any new nodes and edges from the following:
                    # Part {i}/{num_iterations} of the input:

                    {inp}""",
                },
                {
                    "role": "user",
                    "content": f"""Here is the current state of the graph:
                    {cur_state.model_dump_json(indent=2)}""",
                },
            ],
            response_model=KnowledgeGraph,
        )  # type: ignore

        # Update the current state
        cur_state = cur_state.update(new_updates)
        cur_state.draw(prefix=f"iteration_{i}")
    return cur_state


# here we assume that we have to process the text in chunks
# one at a time since they may not fit in the prompt otherwise
with open('Business_Plan_Part_1.txt', 'r') as file:
    file_1_content = file.read()

with open('Business_Plan_Part_2.txt', 'r') as file:
    file_2_content = file.read()

with open('Business_Plan_Part_3.txt', 'r') as file:
    file_3_content = file.read()

text_chunks = [
    file_1_content,
    file_2_content,
    file_3_content
]

graph: KnowledgeGraph = generate_graph(text_chunks)

graph.draw(prefix="final")