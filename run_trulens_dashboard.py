import streamlit as st
from utils import create_sidebar
from trulens_eval import TruChain, Feedback, Tru, feedback
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from langchain.chains import RetrievalQA
from trulens.apps.custom import TruApp
from trulens.core import TruSession
from trulens.apps.custom import instrument
from trulens.dashboard import run_dashboard

uw_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
"""

wsu_info = """
Washington State University, commonly known as WSU, founded in 1890, is a public research university in Pullman, Washington.
With multiple campuses across the state, it is the state's second largest institution of higher education.
WSU is known for its programs in veterinary medicine, agriculture, engineering, architecture, and pharmacy.
"""

seattle_info = """
Seattle, a city on Puget Sound in the Pacific Northwest, is surrounded by water, mountains and evergreen forests, and contains thousands of acres of parkland.
It's home to a large tech industry, with Microsoft and Amazon headquartered in its metropolitan area.
The futuristic Space Needle, a legacy of the 1962 World's Fair, is its most iconic landmark.
"""

starbucks_info = """
Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves headquartered in Seattle, Washington.
As the world's largest coffeehouse chain, Starbucks is seen to be the main representation of the United States' second wave of coffee culture.
"""

newzealand_info = """
New Zealand is an island country located in the southwestern Pacific Ocean. It comprises two main landmasses—the North Island and the South Island—and over 700 smaller islands.
The country is known for its stunning landscapes, ranging from lush forests and mountains to beaches and lakes. New Zealand has a rich cultural heritage, with influences from 
both the indigenous Māori people and European settlers. The capital city is Wellington, while the largest city is Auckland. New Zealand is also famous for its adventure tourism,
including activities like bungee jumping, skiing, and hiking.
"""

create_sidebar()

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get(st.secrets["OPENAI_API_KEY"]),
    model_name="text-embedding-ada-002",
)
chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection(
    name="Washington", embedding_function=embedding_function
)
vector_store.add("uw_info", documents=uw_info)
vector_store.add("wsu_info", documents=wsu_info)
vector_store.add("seattle_info", documents=seattle_info)
vector_store.add("starbucks_info", documents=starbucks_info)
vector_store.add("newzealand_info", documents=newzealand_info)

from trulens.apps.app import instrument
from trulens.core import TruSession
session = TruSession()
session.reset_database()

from openai import OpenAI
oai_client = OpenAI()
class RAG:
    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(query_texts=query, n_results=4)
        # Flatten the list of lists into a single list
        return [doc for sublist in results["documents"] for doc in sublist]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        if len(context_str) == 0:
            return "Sorry, I couldn't find an answer to your question."

        completion = (
            oai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"We have provided context information below. \n"
                        f"---------------------\n"
                        f"{context_str}"
                        f"\n---------------------\n"
                        f"First, say hello and that you're happy to help. \n"
                        f"\n---------------------\n"
                        f"Then, given this information, please answer the question: {query}",
                    }
                ],
            )
            .choices[0]
            .message.content
        )
        if completion:
            return completion
        else:
            return "Did not find an answer."

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query=query)
        completion = self.generate_completion(
            query=query, context_str=context_str
        )
        return completion
rag = RAG()


import numpy as np
from trulens.core import Feedback
from trulens.core import Select
from trulens.providers.openai import OpenAI
provider = OpenAI(model_engine="gpt-4")
# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    .on_input()
    .on(Select.RecordCalls.retrieve.rets[:])
    .aggregate(np.mean)  # choose a different aggregation method if you wish
)

from trulens.apps.app import TruApp
tru_rag = TruApp(
    rag,
    app_name="RAG",
    app_version="base",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)
with tru_rag as recording:
    rag.query(
        "What wave of coffee culture is Starbucks seen to represent in the United States?"
    )
    rag.query(
        "What wave of coffee culture is Starbucks seen to represent in the New Zealand?"
    )
    rag.query("Does Washington State have Starbucks on campus?")
session.get_leaderboard()

from trulens.core.guardrails.base import context_filter
# note: feedback function used for guardrail must only return a score, not also reasons
f_context_relevance_score = Feedback(
    provider.context_relevance, name="Context Relevance"
)


class FilteredRAG(RAG):
    @instrument
    @context_filter(
        feedback=f_context_relevance_score,
        threshold=0.75,
        keyword_for_prompt="query",
    )
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(query_texts=query, n_results=4)
        if "documents" in results and results["documents"]:
            return [doc for sublist in results["documents"] for doc in sublist]
        else:
            return []

filtered_rag = FilteredRAG()

from trulens.apps.custom import TruCustomApp
filtered_tru_rag = TruCustomApp(
    filtered_rag,
    app_name="RAG",
    app_version="filtered",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)
with filtered_tru_rag as recording:
    filtered_rag.query(
        query="What wave of coffee culture is Starbucks seen to represent in the United States?"
    )
    filtered_rag.query(
        "What wave of coffee culture is Starbucks seen to represent in the New Zealand?"
    )
    filtered_rag.query("Does Washington State have Starbucks on campus?")

session.get_leaderboard()

run_dashboard(session)
