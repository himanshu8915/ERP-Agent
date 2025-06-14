import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool

# Set API keys (using yours provided)
os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "vly-dev-hVz3FK8LnSQrwXoRVEFqYukRj8BapSOTt"

class ERPAssistant:
    def __init__(self):
        self.llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.search = TavilySearchResults()

    def process_files(self, website_url: str, pdf_files):
        """Process both website and PDFs into a knowledge base"""
        documents = []
        
        # Load website content
        if website_url:
            try:
                loader = WebBaseLoader(website_url)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Couldn't load website: {str(e)}")

        # Load PDFs
        for pdf_file in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getbuffer())
                tmp_path = tmp.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Couldn't process {pdf_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
        
        if not documents:
            raise ValueError("No valid content found")
        
        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        return FAISS.from_documents(documents=splits, embedding=self.embeddings)

    @tool
    def web_search(self, query: str) -> str:
        """Search the web for current information"""
        try:
            results = self.search.invoke({"query": query})
            return "\n".join([f"Source: {r['url']}\nContent: {r['content'][:200]}..." for r in results][:3]) if results else ""
        except Exception:
            return ""

    def create_agent(self, vectorstore, company_name: str, module_name: str):
        """Create the ERP agent"""
        retriever = vectorstore.as_retriever()
        
        @tool
        def knowledge_lookup(query: str) -> str:
            """Search company knowledge base"""
            try:
                docs = retriever.invoke(query)
                return "\n\n".join([doc.page_content for doc in docs][:3]) if docs else ""
            except Exception:
                return ""

        tools = [
            Tool(
                name="company_knowledge",
                func=knowledge_lookup,
                description=f"Search {company_name}'s internal documents"
            ),
            Tool(
                name="web_search",
                func=self.web_search,
                description="Search the web when company docs don't have answers"
            )
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are {company_name}'s {module_name} ERP assistant. Follow these rules:
            
            1. First check company knowledge
            2. For greetings/simple questions, respond directly
            3. For unknown queries, say "I couldn't find this in our docs. Would you like me to web search?"
            4. Never show raw tool responses
            5. Be friendly but professional
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors="Check your input and try again."
        )

def main():
    st.set_page_config(page_title="ERP Chatbot", page_icon="🤖")
    
    # Initialize session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None

    assistant = ERPAssistant()

    # Setup sidebar
    with st.sidebar:
        st.title("Setup")
        company_name = st.text_input("Company Name", "Acme Inc")
        company_website = st.text_input("Website URL", "https://www.acme.com")
        module_name = st.text_input("Module Name", "Finance")
        pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Initialize Assistant"):
            if company_website or pdf_files:
                with st.spinner("Building knowledge base..."):
                    try:
                        vectorstore = assistant.process_files(company_website, pdf_files or [])
                        st.session_state.agent = assistant.create_agent(
                            vectorstore,
                            company_name,
                            module_name
                        )
                        st.success("Ready to chat!")
                    except Exception as e:
                        st.error(f"Setup failed: {str(e)}")
            else:
                st.error("Please provide at least a website or PDFs")

    # Chat interface
    st.title(f"{company_name} {module_name} Assistant")
    
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.write(msg.content)

    if prompt := st.chat_input("Ask about the module..."):
        if not st.session_state.agent:
            st.error("Please initialize the assistant first")
            return
        
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        with st.spinner("Thinking..."):
            try:
                # Handle greetings directly
                if prompt.lower().strip() in ["hello", "hi", "hey"]:
                    response = f"Hello! I'm {company_name}'s {module_name} assistant. How can I help?"
                else:
                    response = st.session_state.agent.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })["output"]
                
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.session_state.chat_history.append(AIMessage(
                    content="Sorry, I had trouble processing that. Please try again."
                ))
                st.error(f"Error: {str(e)}")
        
        st.rerun()

if __name__ == "__main__":
    main()