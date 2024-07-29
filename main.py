from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory

import warnings
import time 
start_time=time.time()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*`resume_download` is deprecated.*")
warnings.filterwarnings("ignore", message=".*The class `LLMChain` was deprecated.*")




# Load and split the PDF
loader = PyPDFLoader('Fixed.pdf')
pages = loader.load_and_split()
print(f"Loaded {len(pages)} pages from PDF")

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

# Create HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Created HuggingFace embeddings")

# Create FAISS vector store
vector_db = FAISS.from_documents(chunks, embeddings)
print("Created FAISS vector store")

# Set up the language model
local_model = "mistral"
llm = ChatOllama(model=local_model)

# Set up the retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Create custom prompt
def create_custom_prompt():
    system_template = """You are an AI assistant that provides solutions for tech support issues. Your knowledge comes from a database containing information about products, issue types, issues, and their corresponding solutions.

    Rules:
    - Given a product, issue type, and specific issue, provide the most relevant solution(s) from the database.
    - Give only one solution from the file.
    - If no exact match is found,say Error
    - Always aim for accuracy and relevance in your solutions.
    - Keep your responses concise and focused on the solution.
    - Do not create a new solution on your own your are not allowed to do this.
    -I want you to give the answer in just one line thats it nothing else just give me the solution string exactly same from the file.
    

    
    Chat History:
    {chat_history}
    """
    
    human_template = """Context: {context}

    User Input:
    Product: {product}
    Issue Type: {issue_type}
    Issue: {issue}

    Task: Provide the most relevant solution(s) for the given issue.
    """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

prompt = create_custom_prompt()

# Set up the chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chat_history = ChatMessageHistory()

def get_chat_history():
    return "\n".join([f"{message.type}: {message.content}" for message in chat_history.messages])

def get_response(input_dict):
    context = format_docs(retriever.get_relevant_documents(f"{input_dict['product']} {input_dict['issue_type']} {input_dict['issue']}"))
    chat_history_str = get_chat_history()
    
    messages = prompt.format_messages(
        context=context,
        product=input_dict['product'],
        issue_type=input_dict['issue_type'],
        issue=input_dict['issue'],
        chat_history=chat_history_str
    )
    
    response = llm(messages)
    return response.content

chain = RunnablePassthrough() | get_response

# Run the chain
def chat():
    print("AI: Hello! I'm here to help with tech support issues. Please provide the following information:")
    
    while True:
        product = input("Enter the product: ")
        if product.lower() in ['quit', 'exit', 'bye']:
            print("AI: Thank you for using our service. Have a great day!")
            break

        issue_type = input("Enter the issue type: ")
        if issue_type.lower() in ['quit', 'exit', 'bye']:
            print("AI: Thank you for using our service. Have a great day!")
            break

        issue = input("Enter the specific issue: ")
        if issue.lower() in ['quit', 'exit', 'bye']:
            print("AI: Thank you for using our service. Have a great day!")
            break

        try:
            response = chain.invoke({"product": product, "issue_type": issue_type, "issue": issue})
            print("\nAI:", response)

            # Update chat history
            chat_history.add_user_message(f"Product: {product}, Issue Type: {issue_type}, Issue: {issue}")
            chat_history.add_ai_message(response)

        except Exception as e:
            print(f"An error occurred: {e}")

        print("\nIs there anything else you'd like help with?")

if __name__ == "__main__":
    chat()