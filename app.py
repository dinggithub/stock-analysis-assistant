import streamlit as st
import openai
import json
from datetime import datetime
import pandas as pd
import logging
import time
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志系统
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('stock_analysis')

# 设置页面配置
st.set_page_config(page_title="A股新闻分析助手", layout="wide")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_first_message" not in st.session_state:
    st.session_state.is_first_message = True

if "history" not in st.session_state:
    st.session_state.history = []

if "should_stop" not in st.session_state:
    st.session_state.should_stop = False

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# 内置提示词
prompt_a = """
你是一位专业的金融分析师和A股市场专家。你的任务是分析新闻对A股市场的影响。

当用户提供新闻内容时，请给出以下分析结论：

1. 利好：
- 利好原因
- 受益行业
- 相关A股上市公司

2. 利空：
- 利空原因
- 受损行业

注意，如果新闻对A股影响不明显，请直接说明"影响有限"。

新闻内容：

{input}
"""

prompt_b = """
你是一位资深的风险分析师。现在需要你基于刚才的新闻对前期分析的利好行业和公司进行风险提示。

请从行业、公司及时间（短期和中长期）维度进行风险分析：

注意，保持客观专业，避免过度悲观；风险分析要有针对性，避免泛泛而谈；如果认为某项风险特别重要，请用"⚠️"标注

如果前期分析显示"影响有限"，则直接回复"无需进行风险分析"。
"""

# 结构化日志记录函数
def log_generation(input_context: str, output: str, reasoning: str, model: str, duration: float, status: str, step: str):
    """记录生成过程的日志"""
    try:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "input": input_context[:500] + "..." if len(input_context) > 500 else input_context,
            "output": output[:500] + "..." if len(output) > 500 else output,
            "reasoning": reasoning[:500] + "..." if reasoning and len(reasoning) > 500 else reasoning,
            "model_parameters": {
                "model": model,
            },
            "duration_seconds": duration,
            "status": status
        }
        logger.info(json.dumps(log_data, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Failed to log generation: {str(e)}")

# 将思维链转换为引用格式的函数
def format_reasoning_as_quote(reasoning_text):
    """将思维链转换为Markdown引用格式"""
    if not reasoning_text:
        return ""
    lines = reasoning_text.split('\n')
    quoted_lines = [f"> {line}" for line in lines]
    return '\n'.join(quoted_lines)

# 创建布局函数
def create_layout():
    msg_container = st.container()
    input_container = st.container()
    return msg_container, input_container

# 显示消息函数
def display_messages(container):
    with container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# 首次消息处理（新闻分析）
def handle_first_message(user_input, msg_container):
    st.session_state.should_stop = False
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with msg_container.chat_message("assistant"):
        st.markdown("### 📊 利好分析")
        reasoning1_placeholder = st.empty()
        impact_placeholder = st.empty()
        
        col1, col2 = st.columns([5,1])
        with col2:
            stop_button = st.button("停止生成", key="stop_button")
            if stop_button:
                st.session_state.should_stop = True
                return

        start_time_impact = time.time()
        formatted_prompt = prompt_a.format(input=user_input)
        first_reasoning = ""
        first_result = ""

        first_analysis_stream = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "你是一位专业的金融分析师和A股市场专家。"},
                {"role": "user", "content": formatted_prompt}
            ],
            stream=True
        )

        for chunk in first_analysis_stream:
            if st.session_state.should_stop:
                break
            if chunk.choices[0].delta.reasoning_content:
                content = chunk.choices[0].delta.reasoning_content
                first_reasoning += content
                reasoning1_placeholder.markdown(format_reasoning_as_quote(first_reasoning))
            elif chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                first_result += content
                impact_placeholder.markdown(first_result)

        impact_duration = time.time() - start_time_impact
        log_generation(
            input_context=formatted_prompt,
            output=first_result,
            reasoning=first_reasoning,
            model="deepseek-reasoner",
            duration=impact_duration,
            status="success",
            step="impact_analysis"
        )

        if not st.session_state.should_stop:
            st.markdown("### ⚠️ 风险提示")
            reasoning2_placeholder = st.empty()
            risk_placeholder = st.empty()
            start_time_risk = time.time()
            second_reasoning = ""
            second_result = ""

            second_analysis_stream = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "你是一位资深的风险分析师。"},
                    {"role": "user", "content": formatted_prompt},
                    {"role": "assistant", "content": first_result},
                    {"role": "user", "content": prompt_b}
                ],
                stream=True
            )

            for chunk in second_analysis_stream:
                if st.session_state.should_stop:
                    break
                if chunk.choices[0].delta.reasoning_content:
                    content = chunk.choices[0].delta.reasoning_content
                    second_reasoning += content
                    reasoning2_placeholder.markdown(format_reasoning_as_quote(second_reasoning))
                elif chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    second_result += content
                    risk_placeholder.markdown(second_result)

            risk_duration = time.time() - start_time_risk
            log_generation(
                input_context=f"{formatted_prompt}\n\n{first_result}\n\n{prompt_b}",
                output=second_result,
                reasoning=second_reasoning,
                model="deepseek-reasoner",
                duration=risk_duration,
                status="success",
                step="risk_analysis"
            )

            complete_response = f"### 📊 利好分析\n\n{first_result}\n\n### ⚠️ 风险提示\n\n{second_result}"
            st.session_state.messages.append({"role": "assistant", "content": complete_response})

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.append({
                "timestamp": timestamp,
                "news": user_input,
                "impact_analysis": first_result,
                "impact_reasoning": first_reasoning,
                "risk_analysis": second_result,
                "risk_reasoning": second_reasoning
            })

        st.session_state.is_first_message = False
        return first_result, second_result

# 后续消息处理（常规对话）
def handle_regular_message(user_input, msg_container):
    st.session_state.should_stop = False
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    messages = []
    for msg in st.session_state.messages[:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    with msg_container.chat_message("assistant"):
        reasoning_placeholder = st.empty()
        message_placeholder = st.empty()
        
        col1, col2 = st.columns([5,1])
        with col2:
            stop_button = st.button("停止生成", key="stop_button")
            if stop_button:
                st.session_state.should_stop = True
                return

        start_time = time.time()
        reasoning = ""
        response = ""

        response_stream = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=True
        )

        for chunk in response_stream:
            if st.session_state.should_stop:
                break
            if chunk.choices[0].delta.reasoning_content:
                content = chunk.choices[0].delta.reasoning_content
                reasoning += content
                reasoning_placeholder.markdown(format_reasoning_as_quote(reasoning))
            elif chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response += content
                message_placeholder.markdown(response)

        duration = time.time() - start_time
        log_generation(
            input_context=user_input,
            output=response,
            reasoning=reasoning,
            model="deepseek-reasoner",
            duration=duration,
            status="success",
            step="conversation"
        )

        st.session_state.messages.append({"role": "assistant", "content": response})
        return response

# 显示历史记录
def show_history():
    st.subheader("历史分析记录")
    if not st.session_state.history:
        st.info("暂无历史记录")
        return

    for i, record in enumerate(reversed(st.session_state.history)):
        with st.expander(f"#{i+1} - {record['timestamp']}"):
            st.markdown("#### 新闻内容")
            try:
                news_text = record["news"][:200] + "..." if len(record["news"]) > 200 else record["news"]
                st.text(news_text)
            except Exception as e:
                st.text("(新闻内容显示错误)")

            if st.button("查看完整分析", key=f"view_history_{i}"):
                st.markdown("#### 📊 股市分析")
                st.markdown(format_reasoning_as_quote(record["impact_reasoning"]))
                st.markdown(record["impact_analysis"])
                st.markdown("#### ⚠️ 风险提示")
                st.markdown(format_reasoning_as_quote(record["risk_reasoning"]))
                st.markdown(record["risk_analysis"])

# 添加CSS样式
def inject_css():
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
        }
        .fixed-bottom {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            border-top: 1px solid #ddd;
        }
        .message-container {
            margin-bottom: 100px;
        }
        </style>
    """, unsafe_allow_html=True)

# 主函数
def main():
    st.title("A股新闻分析助手")
    st.write("基于 DeepSeek R1 模型的新闻分析工具，帮助您了解新闻可能对 A 股市场的影响。注意，R1模型有较强的幻觉，可能会输出不准确的数据信息，请核实后使用")

    msg_container, input_container = create_layout()
    
    tab1, tab2 = st.tabs(["对话", "历史记录"])
    
    with tab1:
        display_messages(msg_container)
        
        with input_container:
            col1, col2 = st.columns([6,1])
            with col1:
                user_input = st.text_input("输入新闻或您的问题...", key="user_input")
            with col2:
                if st.button("发送", key="send_button"):
                    if user_input:
                        try:
                            if st.session_state.is_first_message:
                                handle_first_message(user_input, msg_container)
                            else:
                                handle_regular_message(user_input, msg_container)
                            # 添加以下行来清空输入框
                            st.session_state.user_input = ""
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"处理失败: {str(e)}")
            
            if st.button("开始新对话", key="new_conversation"):
                st.session_state.messages = []
                st.session_state.is_first_message = True
                st.experimental_rerun()
    
    with tab2:
        show_history()

if __name__ == "__main__":
    try:
        inject_css()
        main()
    except Exception as e:
        log_generation(
            input_context="应用启动",
            output=str(e),
            reasoning="",
            model="deepseek-reasoner",
            duration=0.0,
            status="critical_error",
            step="app_startup"
        )
        st.error(f"应用运行错误: {str(e)}")