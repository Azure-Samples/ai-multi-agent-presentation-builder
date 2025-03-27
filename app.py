import asyncio
import streamlit as st
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from src.mas import Orchestrator, MultiAgent

st.set_page_config(
    page_title="AI Multi-Agent Presentation Builder",
    page_icon=":robot_face:",
    layout="wide"
)

# Enhanced CSS for a more vibrant design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(-45deg, #f9d423, #ff4e50, #ff6f61, #ffa69e);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        margin: 0;
        padding: 0;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .reportview-container, .main .block-container {
        background: transparent;
        color: #333;
    }
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 1em;
    }
    .stButton>button {
        background-color: #ff4e50;
        color: #fff;
        border: none;
        border-radius: 10px;
        font-size: 1em;
        padding: 0.6em 1em;
        margin-top: 1em;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(255, 78, 80, 0.3);
    }
    .stButton>button:hover {
        background-color: #f9d423;
        color: #333;
        box-shadow: 0 6px 20px rgba(249, 212, 35, 0.3);
    }
    h1, h2, h3 {
        color: #fff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

async def run(user_input, num_agents):
    orchestrator = Orchestrator(user_input, num_agents)
    dynamic_agents = orchestrator.run()

    mas = MultiAgent()
    expert_agents = mas.create_agents(dynamic_agents)
    expert_agents_names = [agent.name for agent in expert_agents]

    return expert_agents, expert_agents_names, mas

async def main(user_input, num_agents, max_interactions=10):
    expert_agents, expert_agents_names, mas = await run(user_input, num_agents)

    with st.sidebar:
        st.title("Expert Agents")
        agent_placeholders = {name: st.empty() for name in expert_agents_names}
        for agent_name in expert_agents_names:
            agent_placeholders[agent_name].info(agent_name)
            await asyncio.sleep(2)

    selection_function = mas.create_selection_function(expert_agents_names)
    termination_keyword = 'yes'
    termination_function = mas.create_termination_function(termination_keyword)

    group = mas.create_chat_group(
        expert_agents,
        selection_function,
        termination_function,
        termination_keyword
    )

    interactions: int = 0
    is_complete: bool = False
    with st.spinner("Generating presentation..."):
        while not is_complete and interactions < max_interactions:
            await group.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
            interactions += 1

            async for response in group.invoke():
                agent_name = response.name
                if agent_name in agent_placeholders:
                    agent_placeholders[agent_name].warning(agent_name)
                    await asyncio.sleep(3)  # Highlight duration
                    agent_placeholders[agent_name].info(agent_name)

                st.markdown(f"**{response.role} - {response.name or '*'}**")
                st.info(response.content)

            if group.is_complete:
                st.success("Conversation completed!")
                st.download_button(
                    "Download Presentation",
                    data=open("presentation.pptx", "rb").read(),
                    file_name="presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )

                break

def app():
    st.title(":robot_face: AI Multi-Agent Draft Presentation Builder :robot_face:")
    st.subheader("Create draft presentations with AI experts")
    col = st.columns([1, 1, 1])[0]
    user_input = col.text_input(
        "Enter the theme:",
        max_chars=150,
        help="Provide the main theme for your presentation.",
        placeholder="e.g., Artificial Intelligence in Healthcare"
    )
    if user_input:
        st.success("Theme received!")

    col = st.columns([1, 2, 1])[0]
    with col:
        num_agents = st.slider("Number of agents", min_value=2, max_value=5, value=4)

    if 'run_button' in st.session_state and st.session_state.run_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False

    if st.button('Create a draft Presentation', disabled=st.session_state.running, key='run_button'):
        if user_input:
            asyncio.run(main(user_input, num_agents))
        else:
            st.warning("Please enter a theme!")

if __name__ == "__main__":
    app()