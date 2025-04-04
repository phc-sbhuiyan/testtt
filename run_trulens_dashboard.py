# from trulens_eval import Tru
# from trulens.core import TruSession

# #database_file="default.sqlite"

# tru = Tru()

# #session = TruSession()
# #session.migrate_database(prior_prefix="")

# #TruSession().migrate_database(prior_prefix="")
# #tru .reset_database()

# tru.run_dashboard()


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

create_sidebar()

tru = Tru()
# session = TruSession()
# session.reset_database()

# OpenAI as feedback provider
llm = feedback.OpenAI()

# Question/answer relevance between overall question and answer.
qa_relevance = Feedback(llm.relevance).on_input_output()

api_key = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)
assistant = pc.assistant.Assistant(
    assistant_name="pineapple-employee-assistant-bot", 
)

# wrap with TruLens
truchain = TruApp(
    assistant,
    app_id='Chat_QA101_PHQ',
    feedbacks=[qa_relevance])

user_query = "Do you have any information about walk-in guests?"
msg = Message(role="user", content=user_query)
with truchain as recording:
  rag_assistant = assistant.chat(messages=[msg])

tru.get_leaderboard(app_ids=["Chat_QA101_PHQ"])

tru.run.run_dashboard()
