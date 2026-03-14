from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

generator_llm = ChatOpenAI(model='gpt-4o-mini')
evaluatator_llm = ChatOpenAI(model='gpt-4o-mini')
optimizer_llm = ChatOpenAI(model='gpt-4o-mini')

class EvaluationSchema(BaseModel):
    feedback: str
    evaluation: Literal["approved", "needs_improvement"] = Field(..., description="The evaluation result, either 'approved' or 'needs_improvement'.")


structured_evaluator_llm = evaluatator_llm.with_structured_output(EvaluationSchema)

class PostSchema(TypedDict):
    title: str
    content: str
    feedback: str
    evaluation: Literal["approved", "needs_improvement"]
    iteration: int
    max_iterations: int

graph = StateGraph(PostSchema)

#define functions
def generate_post(state: PostSchema) -> PostSchema:
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['title']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ]
    prompt = "Generate a social media post related to the following topic: " + state['title']
    output = generator_llm.invoke(prompt).content
    return {"content": output}

def evaluate_post(state: PostSchema) -> PostSchema:
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
    Evaluate the following tweet:

    Tweet: "{state['title']}"

    Use the criteria below to evaluate the tweet:

    1. Originality – Is this fresh, or have you seen it a hundred times before?  
    2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
    3. Punchiness – Is it short, sharp, and scroll-stopping?  
    4. Virality Potential – Would people retweet or share it?  
    5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

    Auto-reject if:
    - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
    - It exceeds 280 characters
    - It reads like a traditional setup-punchline joke
    - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

    ### Respond ONLY in structured format:
    - evaluation: "approved" or "needs_improvement"  
    - feedback: One paragraph explaining the strengths and weaknesses 
    """)
    ]
    
    res = structured_evaluator_llm.invoke(messages)
    state['feedback'] = res.feedback
    state['evaluation'] = res.evaluation
    return state


def optimize_post(state: PostSchema) -> PostSchema:
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
        Improve the tweet based on this feedback:
        "{state['feedback']}"

        Topic: "{state['title']}"
        Original Tweet:
        {state['content']}

        Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
        """)
    ]
    output = optimizer_llm.invoke(messages)
    state['content'] = output.content
    state['iteration'] += 1
    return state


def check_if_approved(state: PostSchema) -> bool:
    return state['evaluation'] == "approved" or state['iteration'] >= state['max_iterations']

#define nodes
graph.add_node('generate_post', generate_post)
graph.add_node('evaluate_post', evaluate_post),
graph.add_node('optimize_post', optimize_post)

#define edges
graph.add_edge(START, 'generate_post')
graph.add_edge('generate_post', 'evaluate_post')
graph.add_conditional_edges("evaluate_post", check_if_approved,  {True: END, False: 'optimize_post'})
graph.add_edge('optimize_post', 'evaluate_post') 

workflow = graph.compile()
initial_state = {"title": "The struggle of working from home", "iteration": 1, "max_iterations": 3}

output = workflow.invoke(initial_state)

print(output)