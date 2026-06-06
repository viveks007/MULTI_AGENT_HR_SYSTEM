from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

def get_embedding(text):

    return model.encode(text)