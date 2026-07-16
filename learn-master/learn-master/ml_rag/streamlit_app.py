import streamlit as st
import time
from rag_indexer import RAGIndexer
from agent.ml_rag_graph import MLRagGraph

st.set_page_config(
    page_title="ML RAG 智能体",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def get_ml_rag_graph():
    indexer = RAGIndexer()
    vector_stores = indexer.load_index()
    
    if not vector_stores:
        st.warning("未找到向量索引，正在构建...")
        with st.spinner("构建知识库索引中..."):
            vector_stores = indexer.build_index()
        st.success("索引构建完成！")
    
    return MLRagGraph(vector_stores)

ml_rag_graph = get_ml_rag_graph()

st.title("🧠 机器学习知识库 RAG 智能体")

st.markdown("""
基于精选机器学习教材、经典论文和官方文档的智能问答系统。
支持查询重写、多知识库联合检索、公式渲染和代码示例。
""")

with st.sidebar:
    st.header("知识库选择")
    
    kb_options = {
        'textbooks': '📚 ML 教材',
        'papers': '📄 经典论文',
        'docs': '📖 官方文档',
        'competitions': '🏆 竞赛方案'
    }
    
    selected_kbs = st.multiselect(
        "选择知识库",
        options=list(kb_options.keys()),
        default=['textbooks', 'papers'],
        format_func=lambda x: kb_options[x]
    )
    
    st.header("检索参数")
    top_k = st.slider("检索结果数量", min_value=3, max_value=15, value=5, step=1)
    
    st.header("难度级别")
    difficulty = st.radio(
        "选择难度",
        options=['auto', 'beginner', 'intermediate', 'advanced'],
        index=0
    )

query = st.text_input("请输入您的机器学习问题：", placeholder="例如：什么是梯度下降？")

if query:
    with st.spinner("分析查询并检索知识库..."):
        start_time = time.time()
        
        result = ml_rag_graph.run(
            query=query,
            user_id="streamlit_user",
            selected_kb_names=selected_kbs
        )
        
        elapsed_time = time.time() - start_time
    
    st.subheader("⏱️ 检索耗时")
    st.info(f"总耗时: {elapsed_time:.2f} 秒")
    
    st.subheader("📊 查询分析")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("难度级别", result.get("difficulty", "unknown"))
    with col2:
        st.metric("目标领域", ", ".join(result.get("domains", [])))
    with col3:
        st.metric("检索轮次", result.get("retrieval_iterations", 0))
    
    st.subheader("📚 检索到的知识库原文")
    
    sources = result.get("sources", [])
    if sources:
        for i, source in enumerate(sources):
            with st.expander(f"来源 {i+1}: {source['document_title']} (相关性: {source['score']:.2%})", expanded=True):
                st.markdown(f"**Chunk {source['chunk_index'] + 1}**")
                st.markdown(source['content'])
                
                meta = source.get('metadata', {})
                if meta:
                    with st.expander("元数据"):
                        st.json(meta)
    else:
        st.warning("未检索到相关知识库内容")
    
    st.subheader("🤖 AI 回答")
    
    answer = result.get("answer", "")
    if answer:
        st.markdown(answer)
    else:
        st.warning("无法生成回答")
    
    citations = result.get("citations", [])
    if citations:
        st.subheader("📖 引用来源")
        for citation in citations:
            status = "✅" if citation.get("verified", False) else "❌"
            st.markdown(f"{status} [{citation['index']}] {citation['document_title']}, Chunk {citation['chunk_index']}")
    
    quality_score = result.get("quality_score", 0)
    st.subheader("⭐ 回答质量评估")
    
    quality_color = "green" if quality_score >= 7 else "orange" if quality_score >= 4 else "red"
    st.markdown(f"<span style='color:{quality_color}; font-size:24px;'>质量评分: {quality_score}/10</span>", unsafe_allow_html=True)
    
    issues = result.get("quality_issues", [])
    if issues:
        st.warning("发现问题:")
        for issue in issues:
            st.markdown(f"- {issue}")

st.markdown("---")
st.markdown("**提示**: 尝试提问如 '什么是反向传播？'、'解释 Transformer 架构'、'如何使用 PyTorch 构建神经网络？'")
