import openai
import pinecone
from datasets import Dataset

openai.api_key = "KEY"

openai.Engine.list()  # check we have authenticated

MODEL = "text-embedding-ada-002"

pinecone.init(
    api_key="KEY",
    environment="us-east4-gcp"  # find next to API key in console
)


if 'test' not in pinecone.list_indexes():
    pinecone.create_index('test', dimension=1536)
index = pinecone.Index('test')

def gen():
    yield {"text": "Deportes: Penarol le gano a Nacional"}
    yield {"text": "Inconvenientes entre el gobierno y la oposicion"}
    yield {"text": "Presidente de la republica se reune con el ministro de salud"}
    yield {"text": "Deportes: defensor le gano a racing"}
    yield {"text": "Alerta: se espera una tormenta"}



dataset = Dataset.from_generator(gen)

from tqdm.auto import tqdm  # this is our progress bar

batch_size = 2  # process everything in batches of 32
for i in tqdm(range(0, len(dataset['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(dataset['text']))
    # get batch of lines and IDs
    lines_batch = dataset['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))


query = "Politica"
xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(match)
