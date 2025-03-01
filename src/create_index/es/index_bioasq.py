from argparse import ArgumentParser
from elasticsearch import Elasticsearch
import html
import json
import os
import glob
from tqdm import tqdm


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def process_line(data):
    item = {
        "id": data["id"],
        "title": data["title"],
        "text": data["contents"],
        "original_json": json.dumps(data),
    }
    return "{}\n{}".format(
        json.dumps({"index": {"_id": "{}-{}".format(INDEX_NAME,data["id"])}}), json.dumps(item)
    )


es = Elasticsearch(hosts="http://localhost:9200", timeout=100)


def index_chunk(chunk):
    res = es.bulk(index=INDEX_NAME, body="\n".join(chunk), timeout="100s")
    assert not res["errors"], res


def main(args):
    datadir = "../../../data/corpus/bioasq/chunk"
    jsonl_files = glob.glob(os.path.join(datadir, "*.jsonl"))

    create_data = []
    for filepath in jsonl_files:
        with open(filepath, "r") as f:
            for line in f:
                create_data.append(json.loads(line.strip()))
    if not args.dry:
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME, ignore=[400, 403])
        es.indices.create(
            index=INDEX_NAME,
            ignore=400,
            body=json.dumps(
                {
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "title": {
                                "type": "text",
                                "analyzer": "simple",
                            },
                            "text": {
                                "type": "text",
                                "analyzer": "my_english_analyzer",
                            },
                            "original_json": {"type": "text"},
                        }
                    },
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "my_english_analyzer": {
                                    "type": "standard",
                                    "stopwords": "_english_",
                                },
                            }
                        },
                    },
                }
            ),
        )

    print("Making indexing queries...")
    all_queries = []
    for item in tqdm(create_data):
        all_queries.append(process_line(item))

    count = sum(len(queries.split("\n")) for queries in all_queries) // 2

    if not args.dry:
        print("Indexing...")
        chunksize = 100
        for chunk in tqdm(
            chunks(all_queries, chunksize),
            total=(len(all_queries) + chunksize - 1) // chunksize,
        ):
            res = es.bulk(index=INDEX_NAME, body="\n".join(chunk), timeout="100s")
            assert not res["errors"], res

    print(f"{count} documents indexed in total")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--reindex", action="store_true", help="Reindex everything")
    parser.add_argument("--dry", action="store_true", help="Dry run")

    args = parser.parse_args()
    INDEX_NAME = "bioasq"
    main(args)