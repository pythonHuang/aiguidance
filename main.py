# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import os

from Agent.AutoGPT import AutoGPT
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from Tools import *
from Tools.PythonTool import ExcelAnalyser
from langchain_community.chat_models import ChatZhipuAI 

import gradio as gr

agent=None
welcome_message="您有什么症状？"
def launch_agent(agent: AutoGPT):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    """
    while True:
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        reply = agent.run(task, verbose=True)
        print(f"{ai_icon}：{reply}\n")
    """
    global welcome_message
    # 初始化 gradio
    demo = gr.ChatInterface(chat, additional_inputs=[
        gr.Textbox("你是一个医院智能导诊系统，准确简明地回答导诊相关问题", label="System Message"),
        gr.Slider(minimum=0.0, maximum=2.0, step=0.1,
                  value=0.7, label="Temperature")
        ],
        textbox=gr.Textbox(placeholder='请输入您的症状...', container=False, scale=8),
        title='XX医院智能导诊系统',
        description=welcome_message,
        #examples=[[welcome_message], ['您的症状是什么？']], 
        ##cache_examples=[[welcome_message], ['您的症状是什么？']], 
        undo_btn=None,
        clear_btn=None,
        retry_btn=None,
        submit_btn='发送',
        ##welcome_message=welcome_message
    )

    # 启动 gradio
    demo.queue().launch()

    
# 每次点击 submit 按钮，都会调用这个函数
# prompt 是用户输入的文本
# history 是用户和机器人的对话历史
# system_message 是自定义的系统消息
# temperature 是温度参数
async def chat(prompt, history, system_message, temperature):

    """
    # 构造对话历史
    messages = [{"role": "system", "content": system_message}]
    for human_message, ai_message in history:
        messages.append({"role": "user", "content": human_message})
        messages.append({"role": "assistant", "content": ai_message})
    messages.append({"role": "user", "content": prompt})

    # 流式调用 LLM
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        messages=messages,
        stream=True,
    )

    response = []
    for part in stream:
        response.append(part.choices[0].delta.content or "")
        yield "".join(response).strip()
    """
    global agent
    task = "根据患者的症状，导诊需要挂的科室;AI:您有什么症状？"
    for human_message, ai_message in history:
        task+="患者:"+human_message+";"
        task+="AI:"+ai_message+"?"
    
    task+="患者:"+prompt+";"
    reply = agent.run(task, verbose=True)
    return reply

def main():

    # 语言模型
    
    llm = ChatOpenAI(
        model="gpt-4-1106-preview",
        temperature=0,
        model_kwargs={
            "seed": 42
        },
    )
    """
    llm = ChatZhipuAI(
        model="chatglm_turbo",
        api_key=os.getenv("ZHIPU_API_KEY"),
    )"""
    # 存储长时记忆的向量数据库
    db = Chroma.from_documents([Document(page_content="")], OpenAIEmbeddings(model="text-embedding-ada-002"))
    retriever = db.as_retriever(
        search_kwargs={"k": 1}
    )

    # 自定义工具集
    tools = [
        #document_qa_tool,
        #document_generation_tool,
        #email_tool,
        #excel_inspection_tool,
        #directory_inspection_tool,
        ask_placeholder,
        finish_placeholder,
        # ExcelAnalyser(
            # prompt_file="./prompts/tools/excel_analyser.txt",
            # verbose=True
        # ).as_tool()
    ]
    global agent
    # 定义智能体
    agent = AutoGPT(
        llm=llm,
        tools=tools,
        work_dir="./data",
        main_prompt_file="./prompts/main/main.txt",
        final_prompt_file="./prompts/main/final_step.txt",
        max_thought_steps=20,
        memery_retriever=retriever
    )

    # 运行智能体
    launch_agent(agent)


if __name__ == "__main__":
    main()
