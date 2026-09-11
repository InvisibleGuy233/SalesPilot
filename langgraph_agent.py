import milvus_config as config
from pymilvus import MilvusClient
from openai import OpenAI
import milvus_query
# from tool_set import tools
from ddgs import DDGS
from dotenv import load_dotenv
import os

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langfuse import observe, Langfuse

load_dotenv()

db_client = None
llm_client = None
chat_client = None

def get_db_client():
    global db_client
    if db_client is None:
        db_client = MilvusClient(
            uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}",
            token="root:Milvus"
        )
        print(f"✓ 已连接到 Milvus ({config.MILVUS_HOST}:{config.MILVUS_PORT})")
    return db_client

def get_llm_client():
    global llm_client
    if llm_client is None:
        llm_client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        print(f"✓ 已创建llm_client")
    return llm_client

def get_chat_client():
    global chat_client
    if chat_client is None:
        chat_client = ChatOpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model = "qwen-plus",
            streaming=True
        )
        print(f"✓ 已创建llm_client")
    return chat_client

@tool
def web_search(query): 
    '''
    Search the web for user query, and LLM could answer user query based on web pages.
    use when judged web search is needed.
    上网查找答案信息，当大模型判定需要上网才能查到的时候用

    Args:
        query: The search term to find on web search.

    Returns:
        LLM's answer for the query.
    '''
    
    print("进入web_search了\n")
    llm_client = get_llm_client()
    results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except:
        results = None
    
    search_result = ""
    if results:
        for i, r in enumerate(results):
            search_result += f"搜索出的第{i+1}个结果：标题 - {r['title']}\n" + f"\t 链接：{r['href']}\n" + f"\t摘要: {r['body']}\n\n"
    else:
        search_result += "搜索出错或者搜索出0个结果。" 
    
    prompt = f'''
        你是一个回答问题大模型，用户问了问题，给你网络搜索的答案，你根据问题和答案组织一下回答语言。如果搜索出错或者搜索出0个结果，请直接回复搜索出错或者搜索出0个结果。
        用户问题：{query}
        搜索出的结果：
            {search_result}
    '''
    try:
        content = llm_client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "system", "content": prompt}],
                temperature=1
            ).choices[0].message.content
        return {"messages": AIMessage(content=content)}
    except Exception as e:
        return f"错误，错误信息：{e}" 
    
@tool    
def rag(query, top_k=5):
    '''
    Search the database for documents matching a query.
    use when judged doc search in database is needed.
    本地文档检索，当需要再本地文档搜索时使用。

    Args:
        query: The search term to find in the database.
        top_k: recall top_k chunks as reference for LLM to answer. 5 by default.
        
    Returns:
        LLM's answer for the query
    '''
    print("进入rag了\n")
    db_client = get_db_client()
    llm_client = get_llm_client()
    content = milvus_query.generate_user_answer(llm_client, db_client, query, top_k=top_k)
    return {"messages": AIMessage(content=content)}

def chatbot(state: MessagesState):
    print("进入chatbot\n")
    print("看看state messages: ")
    print(state["messages"])
    chat_client = get_chat_client()
    response = chat_client.invoke(state["messages"])
    print("打印chatbot response.")
    print(response)
    
    return {"messages": [response]}

def goto_function_call(state: MessagesState):
    print("进入goto_function_call，判断要不要工具\n")
    last_msg = state['messages'][-1]
    print("打印last message:")
    print(last_msg)
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        print("return tools")
        return "tools"
    return 'final_answer'

def get_tool_node():
    global chat_client
    tools = [rag, web_search]
    chat_client = get_chat_client()
    chat_client = chat_client.bind_tools(tools)
    print("chat_client 已加载tools")
    # 创建 ToolNode
    return ToolNode(tools)
    

def final_answer(state: MessagesState):
    print("进入final_answer\n")
    print(state["messages"])
    state["messages"].append(AIMessage(content="请综合以上消息，给用户最终回答。"))
    chat_client = get_chat_client()
    response = chat_client.invoke(state["messages"])
    
    return {"messages": [response]}


# TOOL_MAP = {
#     "rag": rag,
#     "web_search": web_search,
# }

# 然后再学记忆再加上就好啦！就能实现最终的多轮对话啦！！
def run_agent_langgraph(user_query):
    graph = StateGraph(MessagesState)

    tool_node = get_tool_node()
    print("✅️ 已经拿到tool node")
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", tool_node)
    graph.add_node("final_answer", final_answer)
    
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges(
        "chatbot",
        goto_function_call,
        {
            "tools": "tools",
            "final_answer": END
        }
    )
    graph.add_edge("tools", "chatbot")
    
    system_prompt = '''
    你是一个智能销售Agent，请根据用户输入调用工具并做出回答。
    调用工具，则要求优先本地检索。若检测到相关信息，请标注出处（包括source, section等）。
    若查不到再联网检索，请标注出处（包括标题，网络链接等）。
    若本地检索与联网搜索均失败，请按照模型训练的结果回答用户，并明确告诉用户本地检索及联网检索均未搜索到答案。
    '''

    userState = {
        "messages" : [SystemMessage(content=system_prompt)]
    }
    userState["messages"].append(HumanMessage(content=user_query))
    
    app = graph.compile()
    # result = app.invoke(userState)
    # print(result["messages"][-1].content)
    
    # 流式输出
    full_response = ""
    for chunk in app.stream(
        {"messages": userState["messages"]},
        stream_mode="messages"
    ):
        # chunk 是一个元组 (message, metadata)
        if hasattr(chunk[0], 'content') and chunk[0].content:
            print(chunk[0].content, end="", flush=True)
            full_response += chunk[0].content
    
    print("\n")
    return full_response
    
    

# def run_agent(user_query):
#     llm_client = get_llm_client()
#     system_prompt = '''
#         你是一个智能销售Agent，请根据用户输入的要求调用工具并做出回答。
#         思考步骤参考：
#         1. 理解用户意图，如开放性提问、Agent做事要求、网络搜索等
#         2. 判断是否需要调用工具，要求优先rag检索，若查不到再联网检索，若联网搜索失败请按照模型训练的结果回答用户，并告诉用户本地检索及联网检索均未搜索到答案。
#         3. 若不需要判断工具，需要形成用户能懂的回答
#     '''
#     messages = [{"role": "system", "content": system_prompt}] #这里要用数据库做持久化记忆，暂时用这个记录
#     messages.append({"role": "user", "content": user_query})
#     while True:
#         print('='*10 + '\tloop start message\t' + '='*10)
#         print(str(messages))
#         result = llm_client.chat.completions.create(
#                 model="qwen-plus",
#                 messages=messages,
#                 tools=tools,
#                 tool_choice="auto"
#             ).choices[0]
#         print('='*10 + '\tLLM output result\t' + '='*10)
#         print(str(result))
        
#         if result.finish_reason == 'stop':
#             return result.message.content
        
#         role = result.message.role
        
#         for tool_call in result.message.tool_calls:
#             arguments = json.loads(tool_call.function.arguments)
#             name = tool_call.function.name
#             if name in TOOL_MAP:
#                 func = TOOL_MAP[name]
#                 res = func(arguments)
#             else:
#                 res = f"No function available, the function name is {name}."
            
#             print('='*10 + '\ttool_call output\t' + '='*10)
#             print("tool_id: ", tool_call.id)
#             print("tool_name: ", name)
#             print("function result: ", str(res))
        
#             messages.append({"role": role, "tool_id": tool_call.id, "content": str(res)})

