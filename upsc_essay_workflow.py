from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

load_dotenv()

model = ChatOpenAI(temperature=0, model='gpt-4o-mini')

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

essay = """India in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), India stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how India navigates this landscape will shape its socio-economic and geopolitical future.

India's strengths in the AI domain are rooted in its vast pool of skilled engineers, a thriving IT industry, and a growing startup ecosystem. With over 5 million STEM graduates annually and a burgeoning base of AI researchers, India possesses the intellectual capital required to build cutting-edge AI systems. Institutions like IITs, IIITs, and IISc have begun fostering AI research, while private players such as TCS, Infosys, and Wipro are integrating AI into their global services. In 2020, the government launched the National AI Strategy (AI for All) with a focus on inclusive growth, aiming to leverage AI in healthcare, agriculture, education, and smart mobility.

One of the most promising applications of AI in India lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control. In healthcare, AI-powered diagnostics can help address India’s doctor-patient ratio crisis, particularly in rural areas. Educational platforms are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While metropolitan cities may embrace AI-driven solutions, rural India continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. India is still shaping its data protection laws, and in the absence of a strong regulatory framework, AI systems may risk misuse or bias.

To harness AI responsibly, India must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

India’s demographic dividend, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, India in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine India’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""

class UPSCState(StateGraph):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overhaul_feedback: str
    score: Annotated[list[int], operator.add]
    avg_score: int

graph = StateGraph(UPSCState)

#functions
def language_evaluation(state: UPSCState) -> UPSCState:
    prompt = f"Evaluate the language used in the following essay and provide feedback and a score out of 10. Essay: {state["essay"]}"
    output = structured_model.invoke(prompt)
    return {"language_feedback": output.feedback, "score": [output.score]}

def analysis_evaluation(state: UPSCState) -> UPSCState:
    prompt = f"Evaluate the analysis provided in the following essay and provide feedback and a score out of 10. Essay: {state["essay"]}"
    output = structured_model.invoke(prompt)
    return {"analysis_feedback": output.feedback, "score": [output.score]}

def clarity_evaluation(state: UPSCState) -> UPSCState:
    prompt = f"Evaluate the clarity of the arguments presented in the following essay and provide feedback and a score out of 10. Essay: {state["essay"]}"
    output = structured_model.invoke(prompt)
    return {"clarity_feedback": output.feedback, "score": [output.score]}

def overhaul_evaluation(state: UPSCState) -> UPSCState:
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}' 
    #summery
    final_feedback = model.invoke(prompt).content
    #average score
    average_score = sum(state['score'])/len(state['score'])
    state['overhaul_feedback'] = final_feedback
    state['avg_score'] = average_score
    return state

#create nodes
graph.add_node("language_evaluation", language_evaluation)
graph.add_node("analysis_evaluation", analysis_evaluation)
graph.add_node("clarity_evaluation", clarity_evaluation)
graph.add_node("overhaul_evaluation", overhaul_evaluation)


#create edges
graph.add_edge(START, "language_evaluation")
graph.add_edge(START, "analysis_evaluation")
graph.add_edge(START, "clarity_evaluation")
graph.add_edge("language_evaluation", "overhaul_evaluation")
graph.add_edge("analysis_evaluation", "overhaul_evaluation")
graph.add_edge("clarity_evaluation", "overhaul_evaluation")
graph.add_edge("overhaul_evaluation", END)

workflow = graph.compile()

result = workflow.invoke({"essay": essay})

print(result)