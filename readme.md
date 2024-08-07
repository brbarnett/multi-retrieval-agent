# Multi-retrieval agent
Most conversational AI agent samples are based on a very simple retrieval scenario where the user is querying over a single dataset, which typically lives in a vector DB for similarity search.

In real world scenarios, it is more common to have multiple sets of structured and unstructured data that the user wants to query. This repository is an example of how to set up a simple multi-retrieval agent that can make its own decisions on which datasets to query, which can then be reasoned over by an LLM.
