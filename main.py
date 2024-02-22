from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOllama

loader = UnstructuredEPubLoader("book.epub", mode="elements")
data = loader.load()

data = data[127:]  # specific to this book, a lot of frontmatter
titles = [idx for idx, x in enumerate(data) if x.metadata["category"] == "Title"]
partitions = list(zip(titles, titles[1:]))
chapters = [data[start:end] for start, end in partitions][:50]

llm = ChatOllama(model="zephyr", temperature=0.5, num_ctx=12768)

PROMPT_TEMPLATE = """Instructions:
Below is a section of a book. Your task is to summarize that section.
Follow these rules strictly:
 - Write a response that appropriately completes the request, brings out the major events and themes present in the page.
 - Major events are clearly retained.
 - The plot is clearly explained and concise.
 - The new summary covers the main events.
 - Do not invent things, to do assume or create names, use only the names given in the section.
 - If a part of the summary is not in the section, do not include it.

Page: 
{page_text}
Summary:
"""

previous_sum = ""
summaries = []
# for page in pages:
for page in chapters:
    page_text = " ".join([x.page_content for x in page])
    summary = llm.invoke(PROMPT_TEMPLATE.format(page_text=page_text))
    print(summary)
    print("----------")
    summaries.append(summary)

# From here combine the summaries

__import__("ipdb").set_trace()
