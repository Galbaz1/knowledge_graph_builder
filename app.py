from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
import instructor
import os
from graphviz import Digraph


client = instructor.patch(OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY_HERE"))

class Node(BaseModel):
    """Nodes represent facts and concepts"""
    id: int
    label: str = Field(..., description="represents a fact or a concept")
    color: str = Field(..., description="the color of the node. Nodes with the same color are related to each other")

class Edge(BaseModel):
    """Black Edges represent relationships between fact and concepts"""
    source: int
    targets: List[int] = Field(..., description="the target nodes that the edge connects to")
    label: str  = Field(..., description="describes the relationship between two facts or concepts")
    #strength: float = Field(..., description="a measure of the relationship strength on a scale of 1-100")  
    color: str = "black"

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)


# Adds response_model to ChatCompletion
# Allows the return of Pydantic model rather than raw JSON


def generate_graph(input) -> KnowledgeGraph:
    return client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": f"You are a sophisticated algorithm that has the power to explain complex ideas and concepts through knowledge graphs. When a user presents you with a concept or question you will use everything that you understand of it to draw a detailed knowledge graph. Here's the user input: {input}",
            }
        ],
        response_model=KnowledgeGraph,
    )  # type: ignore


def visualize_knowledge_graph(kg: KnowledgeGraph):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.label, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.targets), label=edge.label, color=edge.color)

    # Render the graph
    dot.render("kg_8.gv", view=True)

#with open("content_3.txt", "r") as file:
 #   content = file.read()

#graph: KnowledgeGraph = generate_graph(f"map out all the concepts and relationships in the following text in great detail: {content}")
#visualize_knowledge_graph(graph)

user_question = input("Please type your question: ")

graph: KnowledgeGraph = generate_graph(user_question)
visualize_knowledge_graph(graph)


#graph: KnowledgeGraph = generate_graph(f"here's what the user has to ask : {content}")
#visualize_knowledge_graph(graph)