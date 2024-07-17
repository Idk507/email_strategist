import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent,Task,Crew,Process
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.llms import HuggingFaceHub

search_tool = DuckDuckGoSearchRun()
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model = "gemini-pro",
                            verbose = True,
                            temperature = 0.5 ,
                            google_api_key ="")

#define agents 

email_author = Agent(
    role = "Email Author",
    goal = "Write a compelling email about the meeting",
    backstory = """
    You are a content strategist known for making complex text topics interesting and easy to
    understand and analyze""",
    verbose = True,
    allow_delegation = False,
    tools = [search_tool],
    llm = llm,
    memory = True,
)


marketing_strategist = Agent(
    role = "Marketing Strategist",
    goal = "Lead the tea in creatnig effective cold emails",
    backstory = """"
    A seasoned chief marketin officder with a keen eye for standout marketing content creator 
    """,
    verbose = True,
    allow_delegation = True,
    #tools = [search_tool],
    llm = llm,
    memory = True,
)

content_specialist = Agent(
    role = "Content Specialist",
    goal = "Write a compelling content on tech advancements",
    backstory = """
    You are a content strategist known for making complex texh topics interesting and easy to 
    understand and analyze""",
    verbose = True,
    allow_delegation = False,
    llm = llm,
    memory = True,
)

#task 

email_task = Task(
    description = """
    Write a compelling email about the meeting,Evaluate the written emails for their effectiveness and engagement,
    Scrutinize the emails for grammatical correctness and clarity,Adjust the emails to align with best practices for cold outreach.
    Revise the emails based on all feedback ,creating two final versions
    """,
    agent = marketing_strategist,
      expected_output='Two final versions of the cold email that are effective, engaging, grammatically correct, and aligned with best practices for cold outreach.',
)


#create a single crew

email_crew = Crew(
    agents = [email_author,marketing_strategist,content_specialist],
    tasks = [email_task],
    verbose=True,
    process = Process.sequential
)

#Execution Flow 

print("Crew working on Emails")
email_output = email_crew.kickoff()

