import time
import os
import json
from llama_index.node_parser import SimpleNodeParser
from sentence_transformers import SentenceTransformer
import faiss
import argparse
import yaml
from glob import glob
from itertools import chain

with open("../../../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["bge-large-en-v1.5"],
    default="bge-large-en-v1.5",
    help="Model to use",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="2wikimultihopqa",
    choices=["bioasq","2wikimultihopqa","hotpotqa","musique"],
    help="Dataset to use",
)
parser.add_argument("--chunk_size", type=int, default=512, help="chunk size")
parser.add_argument("--chunk_overlap", type=int, default=0, help="chunk overlap")
parser.add_argument("--device", type=str, default="cuda:3", help="Device to use")
args = parser.parse_args()




import json
from llama_index.node_parser import SimpleNodeParser

import json
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from tqdm import tqdm
def split_text(data):
    documents = []
    for record in data:
        if record["title"]:
            combined_text = record["title"] + "\n" + record["content"]
        else:
            combined_text = record["content"]
        documents.append(Document(text=combined_text))
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    contents = [node.text for node in nodes]
    return contents

def create_index(embeddings, vectorstore_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, vectorstore_path)


if __name__ == "__main__":

    model = SentenceTransformer(config["model"][args.model], device=args.device)
    vectorstore_path = f"../../../data/corpus/{args.dataset}/{args.dataset}.index"
    print("loading document ...")
    create_data = []
    start = time.time()

    if args.dataset == "bioasq":
        jsonl_files = glob.glob(os.path.join("../../../corpus/bioasq/chunk", "*.jsonl"))

        
        for filepath in tqdm(jsonl_files):
            with open(filepath, "r") as f:
                for line in f:
                    create_data.append(json.loads(line.strip()))

    elif args.dataset == "2wikimultihopqa":
        train = json.load(open("../../../data/corpus/2wikimultihopqa/train.json", "r"))
        dev = json.load(open("../../../data/corpus/2wikimultihopqa/dev.json", "r"))
        test = json.load(open("../../../data/corpus/2wikimultihopqa/test.json", "r"))

        data = {}
        for item in tqdm(chain(train, dev, test)):
            for title, sentences in item["context"]:
                para = " ".join(sentences)
                data[para] = title
        create_data = [
            {"id": i, "content": text, "title": title}
            for i, (text, title) in enumerate(data.items())
        ]
    elif args.dataset == "hotpotqa":
        import bz2
        from multiprocessing import Pool
        def process_line(line):
            data = json.loads(line)
            item = {
                "id": data["id"],
                "title": data["title"],
                "content": "".join(data["text"]),
            }
            return item
        def generate_indexing_queries_from_bz2(bz2file, dry=False):
            if dry:
                return

            with bz2.open(bz2file, "rt") as f:
                body = [process_line(line) for line in f]

            return body
        filelist = glob("../../../data/corpus/hotpotqa/*/wiki_*.bz2")

        print("Making indexing queries...")
        pool = Pool()

        for result in tqdm(pool.imap(generate_indexing_queries_from_bz2, filelist), total=len(filelist)):
            create_data.extend(result)
    elif args.dataset == "musique":
        train = [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_ans_v1.0_train.jsonl")
        ] + [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_full_v1.0_train.jsonl")
        ]
        dev = [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_ans_v1.0_dev.jsonl")
        ] + [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_full_v1.0_dev.jsonl")
        ]
        test = [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_ans_v1.0_test.jsonl")
        ] + [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_full_v1.0_test.jsonl")
        ]

        tot = 0
        hist = set()
        for item in tqdm(chain(train, dev, test)):
            for p in item["paragraphs"]:
                stamp = p["title"] + " " + p["paragraph_text"]
                if not stamp in hist:
                    create_data.append(
                        {"id": tot, "content": p["paragraph_text"], "title": p["title"]}
                    )
                    hist.add(stamp)
                    tot += 1

        print("Num:",len(create_data))
        contents=split_text(create_data)
        contents = list(set(contents))
        print("Generating embeddings ...",len(contents))
        embeddings = model.encode(contents, batch_size=1200,show_progress_bar=True)
        with open(f"../../../data/corpus/{args.dataset}/chunk.json", "w", encoding="utf-8") as fout:
            json.dump(contents, fout, ensure_ascii=False)
    print("Creating index ...",len(embeddings))
    create_index(embeddings, vectorstore_path)
    end = time.time()
    print("speed time",end-start)
