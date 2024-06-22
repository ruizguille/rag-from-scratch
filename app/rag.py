from nomic import embed
from groq import Groq
from app.config import settings
from app.vector_store import VectorStore

SYSTEM_PROMPT = """You are an assistant that answers user questions about a collection of movie screenplays."""

USER_PROMPT = """
Use the following pieces of context from movie screenplays to answer the user question.
You must only use the facts from the context to answer. If the answer cannot be found in the context, say that you don't have enough information to answer the question and provide any facts from the context that could be relevant to the answer.
Don't mention the context explicitly in your answer (e.g. don't say "according to the context..."), just answer the question like it's your own knowledge.

Context:
{context}

User Question:
{question}
"""

groq_client = Groq(api_key=settings.GROQ_API_KEY)

def answer_question(question, vector_store):
    query_embed = embed.text(
        texts=[question],
        model='nomic-embed-text-v1.5',
        task_type='search_query',
        inference_mode=settings.NOMIC_INFERENCE_MODE
    )
    query_vector = query_embed['embeddings'][0]
    chunks = vector_store.query(query_vector)

    context = '\n\n---\n\n'.join([chunk['text'] for chunk in chunks]) + '\n\n---'
    user_message =  USER_PROMPT.format(context=context, question=question)
    messages=[
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_message}
    ]
    chat_completion = groq_client.chat.completions.create(
        messages=messages, model=settings.GROQ_MODEL
    )
    return chat_completion.choices[0].message.content

def main():
    vector_store = VectorStore()
    vector_store.load()

    print("Ask a question about Christopher Nolan's Inception movie:\n")
    while True:
        question = input()
        answer = answer_question(question, vector_store)
        print(answer, '\n')
