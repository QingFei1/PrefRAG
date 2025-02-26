from jinja2 import Template
import re
import time
import json
import os
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import logging
import faiss
from sentence_transformers import SentenceTransformer
import random
import yaml
import argparse
from duckduckgo_search import DDGS 
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from es_retrieve import retrieve
from utils import acc_score, F1_scorer, compute_exact, acc_score_yn,seed_everything
from vllm import LLM, SamplingParams
from retry import retry

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=["gpt-4o-mini-2024-07-18","glm-4-plus","llama3.1-8b-instruct","glm4-9b-chat","glm4-9b-chat-dpo","llama-3.1-70b-instruct"], default='glm4', help="Specify the model to use for inference")
parser.add_argument('--MaxClients', type=int, default=1, help="Maximum number of concurrent clients")
parser.add_argument('--retrieve_method', type=str, default="es", help="Retrieval method to use: elasticsearch (es) or embedding-based (emb)")
parser.add_argument('--retrieve_top_k', type=int, default=5, help="Number of top documents to retrieve for each query")
parser.add_argument('--max_step', type=int, default=3, help="Maximum number of reasoning steps")
parser.add_argument('--dataset', type=str, choices=["2wikimultihopqa", "hotpotqa", "musique","bioasq"], default='hotpotqa', help="Dataset to evaluate on")
parser.add_argument('--method', type=str, default="prefrag", choices=["prefrag","base_local","base_web","base_local_web","base_wo_retri"], help="Method for question answering")
parser.add_argument('--resume_path', type=str, default="", help="Path to checkpoint file to resume generation from")
parser.add_argument('--temperature', type=float, default=0.1, help="Sampling temperature for generation")
parser.add_argument('--top_p', type=float, default=0.9, help="Top-p sampling parameter for generation")
parser.add_argument('--device', type=str, default='cuda:6', help="Device for model inference (e.g., cuda:0, cpu)")
parser.add_argument('--gpu_memory_utilization', type=float, default=0.45, help="Fraction of GPU memory to use")


with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

from datetime import datetime
system_prompt = f'''Current date: {datetime.now().strftime('%Y-%m-%d')}'''

@retry(((Exception)), delay=1, backoff=2, max_delay=20,jitter=(1, 15))
def call_api(prompt,stop=None):
    messages=[{"role": "user", "content": prompt}]
    res = api_gen(args.model, messages,temperature=args.temperature,top_p=args.top_p,stop=stop)
    assert res is not None
    return res



def call_local(prompt,stop=None):

    if "llama" in args.model:
        model_template = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "glm" in args.model:
        model_template=f"<|user|>\n{prompt}<|assistant|>\n"
    sampling_params = SamplingParams(max_tokens=4096,temperature=args.temperature,top_p=args.top_p,stop=stop,include_stop_str_in_output=True)
    response = llm.generate(model_template, sampling_params)[0].outputs[0].text
    return response


def save_log_to_file(logger, log_file="my_log", log_folder="logs"):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    current_date = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = f"{log_file}-{current_date}.log"
    file_handler = logging.FileHandler(os.path.join(log_folder, log_file_name))
    logger.addHandler(file_handler)

def base_local(question):
    refs=Search_Prefer_Local(args.dataset,args.retrieve_top_k).call(question)
    prompt=refs+base_format.format(question=question)
    answer=call_llm(prompt)
    return answer

def base_web(question):
    refs=Search_Engine(args.dataset,args.retrieve_top_k).call(question)

    prompt=refs+base_format.format(question=question)
    answer=call_llm(prompt)
    return answer

def base_local_web(question):
    refs=Search_Prefer_Local(args.dataset,args.retrieve_top_k).call(question)+"\n"+Search_Engine(args.dataset,args.retrieve_top_k).call(question)

    prompt=refs+base_format.format(question=question)
    answer=call_llm(prompt)
    return answer

def base_wo_retri(question):
    prompt=f'Answer this question :\n{question}\nGive me only the answer without including any other words.\n\nAnswer:'
    if args.dataset=="bioasq":
        prompt=f'ANSWER "yes" OR "no" ONLY (You have to choose the most likely option).\nAnswer:'
    answer=call_llm(prompt)
    return answer



class Search_Engine:
    def __init__(self, dataset,topk):
        self.dataset = dataset
        self.topk = topk

    @staticmethod
    def introduction():
        return """{"name":"Search_Engine", "description":"This is a knowledge base general search engine that can be used to query external knowledge, learn facts, etc.", "input":"The phrase or question to be searched."}"""
    
    @retry((Exception), delay=1, backoff=2, max_delay=10,jitter=(1, 10))  
    def call(self,query):
        query=str(query).strip('"')
        try:
            with DDGS(timeout=60, proxy=None, verify=False) as ddgs:
                result=[f"""title:{r["title"]}\ncontent:{r["body"]}""" for r in ddgs.text(query, region='wt-wt', safesearch='off',max_results=self.topk)]
        except DuckDuckGoSearchException as e:
            if "return None" in str(e):
                print(f"Caught specific DuckDuckGoSearchException: {e}")
                result = "No related information was found."
                return result
            else:
                raise e
        return "\n\n".join(result)

class Search_Prefer_Local:
    def __init__(self, dataset,topk):
        self.dataset = dataset
        self.topk = topk

    @staticmethod
    def introduction():
        return """{"name":"Search_Prefer_Local", "description":"This is a local knowledge base general search engine that can be used to query external knowledge, learn facts, etc.", "input":"The phrase or question to be searched."}"""

    def call(self,query):
        query=str(query).strip('"')
        refs = retrieve(self.dataset, query=query, topk=self.topk)
        return "\n\n".join(refs)



class Pref:
    def __init__(self, max_step=3, tools=[]):

        self.tools = tools
        self.max_step = max_step

    def call(self,question):
            
        return self.gen_answer(question)

    def gen_answer(self,question):

        prompt_template = Template(config["prompt"]["prefrag"])
        answer_try_search = False
        output_process, observations_logs, evaluation_process = [], [], []
        i = 0
        while i <= self.max_step:
            prompt = prompt_template.render(
                answer_format=answer_format, max_step=self.max_step,
                question=question, tools=self.tools, thought="\n".join(output_process)
            ).strip()
            output=call_llm(prompt,stop=["Observation:","Observation:\n"])
            answer = self.extract_final_answer(output)
            action, action_input = self.extract_action_info(output)
            
            if answer:
                self_evaluation_match=self.extract_self_evaluation(output)
                if self_evaluation_match and ('PARTIALLY CORRECT' in self_evaluation_match or 'INCORRECT' in self_evaluation_match):
                    if not answer_try_search:
                        
                            evaluation_process.append(output)
                            i-=1 
                            search_info_local = Search_Prefer_Local(args.dataset,args.retrieve_top_k).call(question)
                            search_info = Search_Engine(args.dataset,args.retrieve_top_k).call(question)
                            search_info=search_info_local+search_info
                            observations_logs.append({"info":search_info,"type":"web"})
                            output_process.extend([output + "\nObservation:", search_info])
                            answer_try_search = True
                            continue
                output_process.append(output)
                return (answer.strip(),'\n'.join(output_process))
            if action and action_input:
                tool = next((t for t in self.tools if t.__name__.lower() in action.lower()), None)
                if tool:
                    observation = self.prefer_retrieval(question,action_input,observations_logs)
                    observations_logs.append({"info":observation})
                    if "Observation" not in output:
                        output+="Observation:"   
                    output_process.extend([output, observation])
            elif "Observation" not in output or "Action" not in output:
                break
            i += 1
        thoughts_str='\n'.join(output_process)
        prompt=thoughts_str+base_format.format(question=question)
        output=call_llm(prompt)
        answer=self.extract_final_answer("Final Answer:"+output)
        if not answer:
            answer=output.strip(":").strip()
        thoughts_str+=f"Final Answer:{answer}"
        return (answer,thoughts_str)
    
    def prefer_retrieval(self,question,new_q,obser_logs):
        if not obser_logs:
            observation=Search_Prefer_Local(args.dataset,args.retrieve_top_k).call(new_q)
            return observation
        observation=Search_Prefer_Local(args.dataset,args.retrieve_top_k).call(new_q)
        existed_info="\n".join([d["info"] for d in obser_logs])
        template = config["prompt"]["prefer_retrieval"]
        prompt = template.format(question=question, existed_info=existed_info, observation=observation).strip()

        response=call_llm(prompt)
        try:    
            if 'json' in response:
                response = response[response.index('{'):response.rindex('}')+1]

            result=json.loads(response)
            res=result["status"] 
           
            if res.lower()=="true":
                return observation
        except:
            match=re.search("True|true",response)
            if match:
                return observation
        observation=Search_Engine(args.dataset,args.retrieve_top_k).call(new_q)
        return observation

    @staticmethod
    def extract_self_evaluation(output):
        match = re.search(r"Self-Evaluation\s*:\s*(.+?)(?:\s|$)Explanation", output, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()

        return output
    
    @staticmethod
    def extract_final_answer(output):
        matches = re.findall(r"Final Answer\s*:\s*(.*?)(?:\n\s*Self-Evaluation|Explanation:|Observation:|Thought:|$)", output, re.IGNORECASE | re.DOTALL)
    
        if matches:
            return matches[-1].strip(":").strip()
        elif "Self-Evaluation:" in output:
            match = re.search(r"(.*?)(?:\n\s*Self-Evaluation|$)", output, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip(":").strip()
        return None

    @staticmethod
    def extract_action_info(output):
        action_match = re.search(r"Action\s*:\s*(.*)", output, re.IGNORECASE)
        action_input_match = re.search(r"Action Input\s*:\s*(\".*?\"|.*?)\s*(?=Observation:|Thought:|Final Answer:|Self-Evaluation:|$)", output, re.IGNORECASE | re.DOTALL)

        if action_match and action_input_match:
            action=action_match.group(1).strip()
            action_input=action_input_match.group(1).strip()
            if "none" not in [action.lower(),action_input.lower()]:
                return action, action_input
        return None, None


if __name__ == "__main__":
    args = parser.parse_args()
    seed_everything(44)
    
    answer_format=config["prompt"]["answer_format"]
    base_format=config["prompt"]["base_answer_format"]

    if args.dataset=="bioasq":
        answer_format=config["prompt"]["bio_answer_format"]
        base_format=config["prompt"]["bio_base_answer_format"]

    if args.retrieve_method=="emb":

        vector = faiss.read_index(f"../data/corpus/{args.dataset}/{args.dataset}.index")   
        emb_model = SentenceTransformer(
            config["model"]["bge-large-en-v1.5"], device=args.device
        )
        with open(f"../data/corpus/{args.dataset}/chunk.json", encoding="utf-8") as f:
            raw_data = json.load(f)

        def retrieve(_, query, topk):
            feature = emb_model.encode([query])
            _, match_id = vector.search(feature, topk)
            return [raw_data[i] for i in match_id[0]]


    if "gpt" in args.model or args.model in ["glm-4-plus"]:
        from utils import api_gen
        call_llm = call_api
    else:
        llm = LLM(model=config["model"][args.model], tensor_parallel_size=1, trust_remote_code=True, dtype='bfloat16', gpu_memory_utilization=args.gpu_memory_utilization)

        call_llm = call_local

    prefrag=Pref(tools=[Search_Engine],max_step=args.max_step)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    save_log_to_file(
        logger,
        log_file=f"{args.dataset}_{args.method}_{args.model}",
        log_folder="../log",
    )
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)

    formatted_time = time.strftime("%Y%m%d-%H%M%S")

    with open(f"../data/eval/{args.dataset}/test.json", encoding="utf-8") as f:
        qa_data = json.load(f)

    save_path = f"../output/{args.dataset}/{args.method}/{args.model}/{args.retrieve_method}"
    os.makedirs(save_path, exist_ok=True)

    all_result = []

    if args.resume_path:
        with open(args.resume_path, "r", encoding="utf-8") as fin:
            resume_data = [json.loads(i) for i in fin.readlines()]
            all_result = resume_data
            filepath = args.resume_path
    else:
        resume_data = []
        if args.method=="prefrag":
            filepath = (
                f"{save_path}/topk-{args.retrieve_top_k}_max_step-{args.max_step}_{formatted_time}.jsonl"
            )
        else:
            filepath = (
                f"{save_path}/topk-{args.retrieve_top_k}_{formatted_time}.jsonl"
            )
    logger.info(f"The predicted results will be saved in '{filepath}'.")
    last_id = len(resume_data)
    batch_size = args.MaxClients
    for bid in tqdm(range(last_id, len(qa_data), batch_size)):
        pool = ThreadPool(processes=args.MaxClients)
        current_batch = qa_data[bid : bid + batch_size]
        tasks = [
            (cb["question"],) for cb in current_batch
        ]
        if args.method=="prefrag":
            outputs = pool.starmap(prefrag.call, tasks)
        elif args.method=="base_local":
            outputs = pool.starmap(base_local, tasks)
        elif args.method=="base_web":
            outputs = pool.starmap(base_web, tasks)
        elif args.method=="base_local_web":
            outputs = pool.starmap(base_local_web, tasks)
        elif args.method=="base_wo_retri":
            outputs = pool.starmap(base_wo_retri, tasks)
        pool.close()
        pool.join()

        for id,output in enumerate(outputs):
            if output:
                if args.method=="prefrag":
                    result={"id": bid+id, "question": current_batch[id]["question"], "answer": current_batch[id]["answer"], "output": output[0],"thoughts": output[1]}
                else:
                    result={"id": bid+id, "question": current_batch[id]["question"], "answer": current_batch[id]["answer"], "output": output}
                all_result.append(result)
                with open(filepath, "a", buffering=1) as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    predictions = [data["output"] for data in all_result]
    answers = [data["answer"] for data in all_result]
    if args.dataset == "bioasq":
        eval_result = {"Acc": acc_score_yn(predictions, answers)}
    else:
        eval_result = {"Acc": acc_score(predictions, answers), "F1": F1_scorer(predictions, answers), "EM": compute_exact(predictions, answers)}
    if eval_result:
        with open(filepath, "a", buffering=1) as fout:
            fout.write(json.dumps(eval_result, ensure_ascii=False) + "\n")

    logger.info(f"eval result: {eval_result}")
