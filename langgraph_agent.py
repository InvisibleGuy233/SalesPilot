import milvus_query
from global_arg import *
# from tool_set import tools
from ddgs import DDGS
from dotenv import load_dotenv
from logs import Log
import os

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langfuse import observe, Langfuse
from pydantic import BaseModel

load_dotenv()

MAX_TOOL_CALL_ROUNDS = 3

class AgentState(MessagesState):
    reflect_num: int
    tool_call_count: int
    reflection: str
    should_retry: bool

class ReflectionDecision(BaseModel):
    should_retry: bool
    reason: str

@tool
def web_search(query): 
    '''
    Search the web for user query, and LLM could answer user query based on web pages.
    use when judged web search is needed.
    上网查找答案信息，当大模型判定需要上网才能查到的时候用

    Args:
        query: The search term to find on web search.

    Returns:
        Web search results returned for the query, including titles, links, and abstracts. type: string.
    '''
    print("进入web_search了\n")
    results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        Log("web_search 调用失败", level="ERROR", error=exc, context={"query": query})
        return f"用户的问题是：{query}，web search 调用失败：{type(exc).__name__}: {exc}"

    search_result = f"用户的问题是：{query}，以下是web search的结果\n"
    if results:
        for i, r in enumerate(results):
            search_result += f"搜索出的第{i+1}个结果：标题 - {r['title']}\n" + f"\t 链接：{r['href']}\n" + f"\t摘要: {r['body']}\n\n"
    else:
        search_result += "web search 调用成功，但没有搜索到结果。"
        
    Log(f"进入web search, 返回结果：\n" + search_result)
    
    return search_result
    
@tool    
def rag(query, top_k=5):
    '''
    Search the database for documents matching a query.
    Use when a document search in the database is needed.
    Local docs cover Huawei products only. If the query involves Huawei products (e.g., the MTS series, ESM series, etc.), use this tool.
    本地文档检索，当需要再本地文档搜索时使用，本地文档包括华为产品，如果涉及到华为产品比如MTS系列，ESM系列，用此工具。

    Args:
        query: The search term to find in the database.
        top_k: recall top_k chunks as reference for LLM to answer. 5 by default.
        
    Returns:
        Searched chunks by RAG returned for the query, including body, section, source and keywords. type: str
    '''
    print("进入rag了\n")
    db_client = get_db_client()
    rag_res = milvus_query.get_topk_ans(db_client, query, top_k=top_k)
    Log(f"进入rag, 返回{top_k}个片段")
    return rag_res

def get_tool_result(last_msg):
    if '工具调用失败' in last_msg: 
        return last_msg + "请根据错误信息，反思是否需要重新调用工具。"
    
    llm_client = get_llm_client()
    prompt = f'''你是一个回答问题的智能体，现在给你以及检索出来的片段，其中包括text, source, section等,要求根据检索出来的片段回答。
        输入：
        {last_msg}
        
        如果没有答案，则回答：暂未检索到答案。如果有，请严格按照以下格式回答：
                
        回答：你生成出来的答案。
        相关文档：source file - section in source file
    '''
    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[{"role": "system", "content": prompt}]
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"llm_client 调用失败。失败原因：{e}."

tool_enabled_chat_client = None
def get_tool_enabled_chat_client():
    global tool_enabled_chat_client
    if tool_enabled_chat_client is None:
        tool_enabled_chat_client = get_chat_client().bind_tools([rag, web_search])
    return tool_enabled_chat_client

def chatbot(state: AgentState):
    print("进入chatbot \n")
    invoke_messages = list(state["messages"])
    state_update_messages = []
    if state.get("reflection"):
        correction_message = HumanMessage(
            content=f"根据质量检查结果修正上一轮回答并重新执行：{state['reflection']}"
        )
        invoke_messages.append(correction_message)
        state_update_messages.append(correction_message)
    chat_client = get_tool_enabled_chat_client()
    response = chat_client.invoke(invoke_messages)
    state_update_messages.append(response)
    Log("chatbot 已生成响应", last_message=response)
    return {
        "messages": state_update_messages,
        "reflection": "",
        "should_retry": False,
    }

def goto_function_call(state: AgentState):
    print("进入goto_function_call，判断要不要工具\n")
    last_msg = state['messages'][-1]
    # print("打印last message:")
    # print(last_msg)
    Log("判断是否需要调用工具", last_message=last_msg)
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        if state["tool_call_count"] >= MAX_TOOL_CALL_ROUNDS:
            Log(
                "工具调用次数已达到上限",
                level="WARNING",
                last_message=last_msg,
                context={"tool_call_count": state["tool_call_count"]},
            )
            return "tool_limit"
        print("return tools")
        return "tools"
    return 'no_func'

def handle_tool_errors(e: Exception) -> str:
    return f"工具调用失败：{type(e).__name__}: {str(e)}。"

def run_tools(state: AgentState):
    tool_node = ToolNode([rag, web_search], handle_tool_errors=handle_tool_errors)
    result = tool_node.invoke(state)
    return {
        "messages": result["messages"],
        "tool_call_count": state["tool_call_count"] + 1,
    }

def should_go_back_after_reflect(state: AgentState):
    print("进入should_go_back_after_reflect，判断要不要回去\n")
    Log(
        "判断反思后是否重新执行",
        last_message=state['messages'][-1],
        context={"should_retry": state["should_retry"]},
    )
    if state["should_retry"]:
        return 'goback'
    return 'final_answer'

def reflect(state: AgentState):
    print("进入 reflect \n")
    Log("开始反思当前执行结果", state=state)
    if state["reflect_num"] >= 3:
        state["reflect_num"] = 0
        return {"messages": [AIMessage(content="<response>反思次数大于3次，基于之前所有对话，生成最终答案。显式标注出：反思大于3次，强制终止。并给出多轮反思的原因。</response>")], "reflect_num": state["reflect_num"]}
    reflect_prompt = """
        你是一个质量控制分析师。请审查刚才的执行结果，并判断是否必须重新执行。

        分析以下方面：
        1. 工具调用是否成功，返回的数据是否足以回答用户问题。
        2. 结果中是否存在错误、状态停滞或关键字段缺失。
        3. 当前答案是否已经正确、完整并且有充分依据。

        只有必须重新调用工具或重新生成回答时，should_retry 才能为 true。
        reason 必须简洁说明判断依据以及下一步应如何修正。
        """
    invoke_messages = [*state["messages"], HumanMessage(content=reflect_prompt)]
    reflect_client = get_chat_client().with_structured_output(ReflectionDecision,  method="function_calling")
    decision = reflect_client.invoke(invoke_messages)
    reflection_message = AIMessage(
        content=f"质量检查结论：{decision.reason}；是否需要重做：{decision.should_retry}"
    )

    return {
        "messages": [reflection_message],
        "reflect_num": state["reflect_num"] + 1,
        "reflection": decision.reason,
        "should_retry": decision.should_retry,
    }

def final_answer(state: AgentState):
    print("进入final_answer\n")
    Log("开始生成最终答案", state=state)
    invoke_messages = [*state["messages"],
       HumanMessage(content="请综合以上消息，给用户最终回答。"),
   ]
    chat_client = get_chat_client()
    response = chat_client.invoke(invoke_messages)
    
    return {"messages": [response]}

# 然后再学记忆再加上就好啦！就能实现最终的多轮对话啦！！
def run_agent_langgraph(user_query):
    graph = StateGraph(AgentState)

    print("✅️ 已经拿到tool node")
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", run_tools)
    graph.add_node("reflect", reflect)
    graph.add_node("final_answer", final_answer)
    
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges(
        "chatbot",
        goto_function_call,
        {
            "tools": "tools",
            "tool_limit": "final_answer",
            "no_func": "reflect"
        }
    )
    graph.add_edge("tools", "chatbot")
    graph.add_conditional_edges(
        "reflect",
        should_go_back_after_reflect,
        {
            "goback": "chatbot",
            "final_answer": "final_answer"
        }
    )
    graph.add_edge("final_answer", END)
    
    system_prompt = '''
    你是一个智能销售Agent，请根据用户输入调用工具并做出回答。
    要求专业，有凭有据，不允许出现幻觉。
    
    请按照以下互斥规则决定如何回答：
    1. 如果问题涉及本地知识库中的华为产品资料，先调用 rag；若检索到有效信息，依据结果回答并标注 source 和 section。
    2. 如果 rag 没有检索到有效信息，再调用 web_search；依据联网结果回答并标注标题和网络链接。
    3. 如果问题需要实时或外部信息，且不属于本地华为产品资料，直接调用 web_search，并标注标题和网络链接。
    4. 如果问题依靠通识知识即可可靠回答，不调用任何工具，直接回答。
    5. 如果本地检索和联网检索都失败，可以依据模型已有知识回答，但必须明确说明两种检索均未获得答案。
    '''

    userState = {
        "messages" : [SystemMessage(content=system_prompt)],
        "reflect_num": 0,
        "tool_call_count": 0,
        "reflection": "",
        "should_retry": False,
    }
    userState["messages"].append(HumanMessage(content=user_query))
    
    app = graph.compile()
    # result = app.invoke(userState)
    # print(result["messages"][-1].content)
    
    # 流式输出
    full_response = ""
    for chunk in app.stream(userState, stream_mode="messages"):
        message_chunk, metadata = chunk
        # chunk 是一个元组 (message, metadata)
        if metadata.get("langgraph_node") == "final_answer" and hasattr(message_chunk, 'content') and message_chunk.content:
            print(message_chunk.content, end="", flush=True)
            full_response += message_chunk.content
    
    print("\n")
    return full_response
    
# TOOL_MAP = {
#     "rag": rag,
#     "web_search": web_search,
# }

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

