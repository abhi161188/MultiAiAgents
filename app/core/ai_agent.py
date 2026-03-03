from langchain_groq import ChatGroq
#rom langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
#from langchain_classic.agents.react.agent import create_react_agent
from langchain import agents
from langchain_core.messages.ai import AIMessage
from app.config.settings import settings
from app.common.logger import get_logger

logger = get_logger(__name__)

def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):
    logger.info(llm_id)
    llm = ChatGroq(model=llm_id)
    tools = [TavilySearch(max_results = 2)] if allow_search else []
    try :
        agent = agents.create_agent(
            model = llm,
            tools = tools,
            system_prompt = system_prompt
        )
    except Exception as e:
        logger.info(f"Error {e}")
    state = {"messages" : query}
    
    response = agent.invoke(state)
    messages = response.get("messages")

    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]