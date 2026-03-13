from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str

def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight_kg"]
    height = state["height_m"]
    bmi = weight / (height ** 2)
    state["bmi"] = bmi
    return state

def check_fitness(state: BMIState) -> BMIState:
    if state["bmi"] < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= state["bmi"] < 24.9:
        state["category"] = "Normal weight"
    elif 25 <= state["bmi"] < 29.9:
        state["category"] = "Overweight"    
    else:
        state["category"] = "Obesity"
    return state

graph = StateGraph(BMIState)

#define nodes
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("check_fitness", check_fitness)

#define edges
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "check_fitness")
graph.add_edge("check_fitness", END)

#compile the graph
workflow = graph.compile()

#execute the graph
initial_state = {
    "weight_kg": 70,
    "height_m": 1.75
}
final_state = workflow.invoke(initial_state)
print(final_state)
