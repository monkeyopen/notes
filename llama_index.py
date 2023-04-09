###LlamaIndex Usage Pattern
###https://github.com/jerryjliu/llama_index/blob/main/docs/guides/primer/usage_pattern.md
###不确定对应哪个版本

from llama_index import SimpleDirectoryReader,GPTSimpleVectorIndex
from llama_index.node_parser import SimpleNodeParser
from gpt_index.docstore import DocumentStore



def construct_index(directory_path):

	documents = SimpleDirectoryReader(directory_path).load_data()

	# from llama_index import Document
	# text_list = [text1, text2, ...]
	# documents = [Document(t) for t in text_list]

	if True:
		#Feed the Document object directly into the index (see section 3).
		index = GPTSimpleVectorIndex.from_documents(documents)
	else:
		#First convert the Document into Node objects (see section 2).
		parser = SimpleNodeParser()
		nodes = parser.get_nodes_from_documents(documents)
		index = GPTSimpleVectorIndex(nodes)

		# from llama_index.data_structs.node_v2 import Node, DocumentRelationship
		# node1 = Node(text="<text_chunk>", doc_id="<node_id>")
		# node2 = Node(text="<text_chunk>", doc_id="<node_id>")
		# # set relationships
		# node1.relationships[DocumentRelationship.NEXT] = node2.get_doc_id()
		# node2.relationships[DocumentRelationship.PREVIOUS] = node1.get_doc_id()

		# # share these Node objects across multiple index structures
		# docstore = DocumentStore()
		# docstore.add_documents(nodes)

		# index1 = GPTSimpleVectorIndex(nodes, docstore=docstore)
		# index2 = GPTListIndex(nodes, docstore=docstore)



from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
from langchain import OpenAI



# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)



# save to disk
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json', service_context=service_context)



response = index.query("What did the author do growing up?")
print(response)


index = GPTListIndex.from_documents(documents)
# mode="default"
response = index.query("What did the author do growing up?", mode="default")
# mode="embedding"
response = index.query("What did the author do growing up?", mode="embedding")


# mode="default"
response = index.query("What did the author do growing up?", response_mode="default")
# mode="compact"
response = index.query("What did the author do growing up?", response_mode="compact")
# mode="tree_summarize"
response = index.query("What did the author do growing up?", response_mode="tree_summarize")


index.query(
    "What did the author do after Y Combinator?", required_keywords=["Combinator"], 
    exclude_keywords=["Italy"]
)


response = index.query("<query_str>")

# get response
# response.response
str(response)

# get sources
response.source_nodes
# formatted sources
response.get_formatted_sources()






