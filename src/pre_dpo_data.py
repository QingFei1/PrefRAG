import os
import re
import json
import random
import datetime
import argparse
import multiprocessing as mp
from tqdm import tqdm
from retry import retry
import yaml
from jinja2 import Template
from vllm import LLM, SamplingParams
from utils import  api_gen
from main import Pref, Search_Engine, Search_Prefer_Local


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["2wikimultihopqa"], default="2wikimultihopqa", help="Dataset to use for data generation")
parser.add_argument("--input_data_path", type=str, default="../data/corpus/2wikimultihopqa/train.json", help="Path to the input training data file")
parser.add_argument("--output_path", type=str, default="../data/dpo_data", help="Directory to save the generated DPO training data")
parser.add_argument("--num_samples", type=int, default=15000, help="Total number of samples to generate")
parser.add_argument("--per_data_gen_num", type=int, default=9, help="Number of different outputs to generate for each input sample")
parser.add_argument("--model", type=str, choices=["glm4-9b-chat"], default="glm4-9b-chat", help="Model used for generating answers")
parser.add_argument("--score_model", type=str, choices=["gpt-4o-mini-2024-07-18", "glm4-plus"], default="glm4-plus", help="Model used to select chosen/rejected pairs for DPO training")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.5, help="Fraction of GPU memory to be utilized")
parser.add_argument("--device", type=str, default="0,1,2,3,4,5,6,7", help="GPU device IDs for parallel processing")
args = parser.parse_args()
prefrag = Pref(tools=[Search_Engine], max_step=3)

@retry(Exception, delay=1, backoff=2, max_delay=60, jitter=(1, 10))
def call_api(prompt):
    messages=[{"role": "user", "content": prompt}]
    result = api_gen(args.score_model, messages, temperature=0.1, top_p=0.9)
    assert result is not None
    return result

def parse_prefer_flag(response, label=None):
    """
    Extract preference flag and analysis from the API response.

    Parameters:
        response (str): The API response.
        label (bool): Optional label for resolving ambiguous responses.

    Returns:
        tuple: Analysis (str) and flag (bool).
    """
    flag = True
    analysis = ""
    try:
        if 'json' in response:
            response = response[response.index('{'):response.rindex('}') + 1]
        result = json.loads(response)
        res, analysis = result["status"], result["analysis"]
        if res.lower() == "true":
            flag = False
    except:
        if re.search("True|true", response) and re.search("False|false", response):
            flag = label
        elif re.search("True|true", response):
            flag = False
        analysis = response
    return analysis, flag

def LLM_score_prefer_retrieval(prompt, responses):
    """
    Evaluate responses and determine the best and worst answers.

    Parameters:
        prompt (str): Evaluation prompt.
        responses (list): A list of model responses.

    Returns:
        tuple: Best response, worst response, and final switch flag.
    """
    switches, best_ids, worst_ids = [], [], []
    label_response = call_api(prompt)
    label_analysis, label_prefer_flag = parse_prefer_flag(label_response)

    for idx, response in enumerate(responses):
        switches.append(parse_prefer_flag(response, label_prefer_flag))

    for idx, switch in enumerate(switches):
        (best_ids if switch[1] == label_prefer_flag else worst_ids).append(idx)

    if best_ids and worst_ids:
        analysis = [{"entry_id": idx, "analysis": switches[response_id][0]} for idx, response_id in enumerate(best_ids)]
        prompt=f"I will provide you with a standard answer analysis. Compare the standard answer analysis with the results in the list below to determine which one is the most similar. Output the result as a dictionary in the following JSON format:\n```json\n{{\n    \"id\": \"<entry_id of the most similar analysis>\"\n}}\n``` Standard answer analysis: {label_analysis}. List to compare: {analysis}."
        score = call_api(prompt)
        try:
            if 'json' in score:
                best_id = best_ids[json.loads(score[score.index('{'):score.rindex('}') + 1])["id"]]
                return responses[best_id], responses[random.choice(worst_ids)], switches[best_id][1]
        except:
            pass
        return responses[random.choice(best_ids)], responses[random.choice(worst_ids)], switches[best_ids[0]][1]
    return "", "", switches[0][1]

def load_and_sample_data(file_path, sample_size):
    """
    Load and sample data from a JSON or JSONL file.

    Parameters:
        file_path (str): Path to the input file.
        sample_size (int): Number of samples.

    Returns:
        list: A random sample of data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f] if "jsonl" in file_path else json.load(f)
    data=[{"question":d["question"],"answer":d["answer"]} for d in data]
    return random.sample(data, sample_size)


    

def call_partition(partition, device_id):
    """
    Execute a worker partition pipeline.

    Parameters:
        partition (list): Partitioned data.
        device_id (str): GPU device ID.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    llm = LLM(model=config["model"][args.model], tensor_parallel_size=1, trust_remote_code=True, dtype='bfloat16', gpu_memory_utilization=args.gpu_memory_utilization)
    sampling_params = SamplingParams(max_tokens=1000, temperature=1, top_p=0.9)
    def call_llm(prompts, params=[sampling_params]):
        if "llama" in args.model:
            template = "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "glm" in args.model:
            template="<|user|>\n{prompt}<|assistant|>\n"
        prompts = [template.format(prompt=p) for p in prompts]
        return [response.outputs[0].text for response in llm.generate(prompts, params, use_tqdm=False)]

    temp_top_p_combinations = [(t, p) for t in [0.1, 0.5, 0.9] for p in [0.1, 0.5, 0.9]]
    param_combinations = (temp_top_p_combinations[:args.per_data_gen_num] if args.per_data_gen_num <= len(temp_top_p_combinations) else temp_top_p_combinations + random.choices(temp_top_p_combinations, k=args.per_data_gen_num - len(temp_top_p_combinations)))
    
    params_dict = {"max_tokens": 4096, "stop": ["Observation:", "Observation:\n"], "include_stop_str_in_output": True}
    sampling_params = [SamplingParams(**params_dict, temperature=t, top_p=p) for t, p in param_combinations]

    for id,qa_data in enumerate(tqdm(partition)):
        save_list = []
        question=qa_data["question"]
        top_k = random.randint(2, 5)
        observations_logs = [] 
        output_process=[] 
        i = 0
        action_input=""

        
        sampling_params=[SamplingParams(**params_dict, temperature=temp, top_p=top_p) for temp, top_p in param_combinations]
        while i <= prefrag.max_step:
            template = Template(config["prompt"]["prefrag"]) 
            prompt = template.render(answer_format=answer_format,max_step=prefrag.max_step,question=question, tools=prefrag.tools, thought="\n".join(output_process)).strip()
            output = call_llm([prompt], [SamplingParams(**params_dict, temperature = 0.9, top_p = 0.9)])[0]
            # Extract the final answer, action and action input from the output
            answer = prefrag.extract_final_answer(output)
            action, action_input = prefrag.extract_action_info(output)
            
            if answer:
                output_process.append(output)
                break

            if action and action_input:

                tool = next((t for t in prefrag.tools if t.__name__.lower() in action.lower()), None)
                
                if tool:
                    observation=Search_Prefer_Local(args.dataset,top_k).call(action_input)
                
                    if observations_logs:
                        existed_info="\n".join([d["info"] for d in observations_logs])
                        template = config["prompt"]["prefer_retrieval"]
                        prompt = template.format(question=question, existed_info=existed_info, observation=observation).strip()
                        
                        prefer_retrieval_generated = call_llm([prompt for i in range(len(param_combinations))], sampling_params)
                        best_prefer_retrieval, worst_prefer_retrieval,switch = LLM_score_prefer_retrieval(prompt, prefer_retrieval_generated)
                            
                        if best_prefer_retrieval and worst_prefer_retrieval:
                            save_list.append({
                                'id': f"prefer_retrieval-{device_id}-{id}",
                                'raw_question': question,
                                'prompt': prompt,
                                'chosen': best_prefer_retrieval,
                                'rejected': worst_prefer_retrieval,
                            })
                                            
                        if switch:
                            observation = Search_Engine(args.dataset,top_k).call(action_input)
                    observations_logs.append({"info":observation})
                    if "Observation" not in output:
                        output+="Observation:"                       
                    output_process.append(output)
                    output_process.append(observation)
            elif "Observation" not in output or "Action" not in output:
                break
            i += 1
        output_process.append(output)

        if save_list:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(json.dumps(data, ensure_ascii=False) for data in save_list) + '\n')


def main():

    data = load_and_sample_data(args.input_data_path, args.num_samples)

    partitions = [data[i::len(args.device.split(","))] for i in range(len(args.device.split(",")))]
    processes = []

    for rank, device_id in enumerate(args.device.split(",")):
        process = mp.Process(target=call_partition, args=(partitions[rank], device_id))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    current_date = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    answer_format = config["prompt"]["answer_format"]
    output_file = os.path.join(args.output_path, f"generated_data-_num-{args.num_samples}_para-{args.per_data_gen_num}-{args.model}-max_step{prefrag.max_step}-{args.score_model}-{current_date}.jsonl")
    main()