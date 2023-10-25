# SurrealDB Docs Retrieval Pipeline

## About:

I wrote a small tool for SurrealDB over the weekend, just for fun, to learn more about SurrealDB and LangChain agents.

## Problem:

I could not locate much information about SurrealDB  on StackOverflow and ChatGPT. I found  information on the SurrealDB documentation, available at -> https://surrealdb.com/docs. I wanted to create a tool that would allow me to ask questions about how to perform certain queries in SurrealDB, and receive precise answers with references.


## Flow:


1. Use SurrealDB documentation from GitHub
2. Convert ember.js hbs templates and code snippets into markdown
3. Built a search index on top of the markdown with code
4. Enable queries about Surreal DB features.

## How to Query:

- Simple Question and Answer (QA) retrieval chain
- Conv retrieval chain
- Agent retrieval

Since SurrealDB is an active project, I've set up the pipeline to re-run once every day!

## Results:

None of the approaches provided me with satisfactory results. I had high hopes for the Agent - and while it performs well for some questions, it struggles with some of the basic ones.

## Dagster flow:

![DAG](./docs/Asset_Group_default.svg)

## Run code yourself

```
dagit -f dosc_retrieval_pipeline.py
```


# References

- https://dagster.io/blog/chatgpt-langchain
- https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa
- https://python.langchain.com/docs/use_cases/question_answering/code_understanding
- https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents
