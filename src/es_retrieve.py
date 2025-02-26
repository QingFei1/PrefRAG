import json
from elasticsearch import Elasticsearch
import re
import yaml

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

core_title_matcher = re.compile("([^()]+[^\s()])(?:\s*\(.+\))?")
core_title_filter = lambda x: (
    core_title_matcher.match(x).group(1) if core_title_matcher.match(x) else x
)


class ElasticSearch:
    def __init__(self, index_name):
        self.index_name = index_name
        self.client = Elasticsearch(config["es"]["url"])

    def _extract_one(self, item, lazy=False, index_type=1):
        if index_type == 1:
            res = {
                k: item["_source"][k]
                for k in ["id", "url", "title", "text", "title_unescape"]
            }
        else:
            res = {
                k: item["_source"][k]
                for k in ["id", "title", "text"]
            }
        res["_score"] = item["_score"]
        return res

    def rerank_with_query(self, query, results, index_type=1):
        def score_boost(item, query):
            score = item["_score"]
            core_title = core_title_filter(item["title_unescape"] if index_type == 1 else item["title"])
            if query.startswith("The ") or query.startswith("the "):
                query1 = query[4:]
            else:
                query1 = query
            if query == (item["title_unescape"] if index_type == 1 else item["title"]) or query1 == (item["title_unescape"] if index_type == 1 else item["title"]):
                score *= 1.5
            elif (
                query.lower() == (item["title_unescape"] if index_type == 1 else item["title"]).lower()
                or query1.lower() == (item["title_unescape"] if index_type == 1 else item["title"]).lower()
            ):
                score *= 1.2
            elif item["title"].lower() in query:
                score *= 1.1
            elif query == core_title or query1 == core_title:
                score *= 1.2
            elif (
                query.lower() == core_title.lower()
                or query1.lower() == core_title.lower()
            ):
                score *= 1.1
            elif core_title.lower() in query.lower():
                score *= 1.05

            item["_score"] = score
            return item

        return list(
            sorted(
                [score_boost(item, query) for item in results],
                key=lambda item: -item["_score"],
            )
        )

    def single_text_query(self, query, topn=10, lazy=False, rerank_topn=50, index_type=1):

        if index_type == 1:
            fields = [
                "title^1.25",
                "title_unescape^1.25",
                "text",
                "title_bigram^1.25",
                "title_unescape_bigram^1.25",
                "text_bigram",
            ]
        else:
            fields = [
                "title^1.25",
                "text",
            ]

        constructed_query = {
            "multi_match": {
                "query": query,
                "fields": fields,
            }
        }
        res = self.client.search(
            index=self.index_name,
            body={"query": constructed_query, "timeout": "100s"},
            size=max(topn, rerank_topn),
            request_timeout=100,
        )

        res = [self._extract_one(x, lazy=lazy, index_type=index_type) for x in res["hits"]["hits"]]
        res = self.rerank_with_query(query, res, index_type=index_type)[:topn]
        res = [{"title": _["title"], "paragraph_text": _["text"]} for _ in res]
        return res

    def search(self, question, k=10, index_type=1):
        try:
            res = self.single_text_query(query=question, topn=k, index_type=index_type)
            return json.dumps(res, ensure_ascii=False)
        except Exception as err:
            print(Exception, err)
            raise


def retrieve(index_name, query, topk):

    if index_name =="bioasq":
        index_type = 2
    else:
        index_type = 1
    ES = ElasticSearch(index_name)
    result = ES.search(query, topk, index_type=index_type)
    result = json.loads(result)
    result = [f"""title:{d["title"]}+\ncontent:{d["paragraph_text"]}""" for d in result]
    return result


if __name__ == "__main__":
    print(retrieve("musique", "What does the acronym of the organization Danish Football Union is part of stand for?", 5))