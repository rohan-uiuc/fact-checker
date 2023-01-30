import streamlit as st
import sys
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

def fact_check(question):
    llm = OpenAI(temperature=0.7)
    template = """{question}\n\n"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
    prompt_template = PromptTemplate(input_variables=["statement"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """In light of the above facts, how would you answer the question '{}'""".format(question)
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain], verbose=True)

    return overall_chain.run(question)

if __name__=="__main__":
    st.text_input("Your question", key="question")
    st.text_input("OpenAI api key", key="api_key", placeholder= "Optional")
    if len(sys.argv) > 1:
        question = sys.argv[1]
        api_key = os.environ['OPENAI_API_KEY']
    else:
        question = st.session_state.question
        if st.session_state.api_key:
            api_key = st.session_state.api_key
        else:
            api_key = os.environ['OPENAI_API_KEY']
        print(question)
    if question and api_key:
        answer = fact_check(question)
        st.write(answer)