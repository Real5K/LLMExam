import warnings
warnings.filterwarnings("ignore")

import os
import re

import PyPDF2

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType

from mistralai import Mistral

class Database:
    def __init__(self):
        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url="https://pgwkmycmtbor78al14g5ya.c0.asia-southeast1.gcp.weaviate.cloud",  
            auth_credentials=Auth.api_key("1TZymK1LSn6FnBTnvou7T5ghjJAXx2EItPgJ")  
        )

        if self.weaviate_client.is_ready():
            print("✅ Successfully connected to Weaviate Cloud!")
        else:
            print("❌ Failed to connect. Check your URL and API Key.")

        self.weaviate_client.collections.delete("Llmexam")

        self.mistral_client = Mistral(api_key="HCOHSJGVjzxKYjhBjO4u8rGmZFKObnLf")

        self.collection_name = "Llmexam"

        if self.collection_name not in self.weaviate_client.collections.list_all():
            self.weaviate_client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="question_text", data_type=DataType.TEXT),
                    Property(name="question_embedding", data_type=DataType.NUMBER_ARRAY),
                ],
                vectorizer_config=None 
            )
            print(f"✅ Collection '{self.collection_name}' created.")
        else:
            print(f"⚡ Collection '{self.collection_name}' already exists.")

    def get_embeddings(self, questions):
        response = self.mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=questions
        )
        return response.data 

    def store_questions(self, questions, embeddings):
        collection = self.weaviate_client.collections.get(self.collection_name)
        for question, emb in zip(questions, embeddings):
            vector = emb.embedding if hasattr(emb, "embedding") else emb
            collection.data.insert(
                properties={
                    "question_text": question,
                    "question_embedding": vector
                },
                vector=vector  
            )
        print(f"✅ {len(questions)} questions stored successfully!")

    def retrieve_context(self, query_text, class_name="Llmexam", top_k=3):
        query_vector = self.mistral_client.embeddings.create(
            model=embedding_model,
            inputs=query_text
        ).data[0].embedding  # Generate embedding for the query
        questions = self.weaviate_client.collections.get(class_name)
        result = (
            questions.query.near_vector(near_vector=query_vector, limit=top_k)
        )
        relevant_questions = result["data"]["Get"][class_name]
        context = "\n".join([q["questionText"] for q in relevant_questions])
        return context

    def retrieve_all(self, class_name="Llmexam"):
        questions = self.weaviate_client.collections.get(class_name)
        all_questions = []
        for item in questions.iterator():
            # print(item.properties['question_text'])
            all_questions.append(item.properties['question_text'])
        context = "\n".join(all_questions)
        return context