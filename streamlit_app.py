import streamlit as st
from openai import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_experimental.synthetic_data import (
    DatasetGenerator,
    create_data_generation_chain,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model
from datetime import datetime
from typing import List, Literal,Optional
from uuid import UUID, uuid4
import yaml
import json

def uploadFile(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('json'):
            return json.load(uploaded_file)
        elif uploaded_file.name.endswith('yaml'):
            return yaml.safe_load(uploaded_file)
        else:
            st.error("Unsupported file format. Upload YAML or JSON file")
    return {}

def create_output_schema(schema_structure):
    input_data = {}
    for field, field_type in schema_structure.items():
        if field_type == "str":
            input_data[field] = (str, ...)
        elif field_type == "int":
            input_data[field] = (int, ...)
        elif field_type == "bool":
            input_data[field] = (bool, ...)
        elif field_type == "datetime":
            input_data[field] = (datetime, ...)
        else:
            st.warning('Unsupported field type')
    return create_model('DynamicOuputSchema', **input_data)

# Show title and description.
st.title("💬 Synthethic Data Generator with File Upload")

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)
    model = ChatOpenAI(api_key=openai_api_key,model="gpt-3.5-turbo", temperature=0.7)
    # Upload schema and example files
    uploaded_schema = st.file_uploader("Upload YAML/JSON file with initial data schema")
    schema_file = uploadFile(uploaded_schema)
    uploaded_example = st.file_uploader("Upload YAML/JSON file with sample data given the schema")
    examples_file = uploadFile(uploaded_example)

    if schema_file and examples_file:
        examples = examples_file['examples']
        schema_type = st.selectbox("Choose schema type:", list(schema_file.keys()))
        if schema_type:
            schema_structure = schema_file[schema_type]
            schema = create_output_schema(schema_file[schema_type])
            
        OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")
    
        prompt_template = FewShotPromptTemplate(
            prefix=SYNTHETIC_FEW_SHOT_PREFIX,
            examples=examples,
            suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
            input_variables=["subject",   "extra"],
            example_prompt=OPENAI_TEMPLATE
        )
        extra = st.text_input('How you want to generate the data?')

        if extra and st.button('Generate synthetic data'):
            synthetic_data_generator = create_openai_data_generator(
            output_schema=schema,
            llm=model,
            prompt=prompt_template)
            
            synthetic_results = synthetic_data_generator.generate(
            subject="account_identity",
            extra=extra,
            runs=3,
            )

            if synthetic_results:
                st.write('Generated synthetic data')
                st.json(synthetic_results)
    else:
        st.info("Upload both a schema definition file and a sample data JSON file")
