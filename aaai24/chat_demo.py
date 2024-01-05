from uuid import uuid4
from aaai24.retrieval import WebRetriever, WikiRetriever, GenerationRetriever
from aaai24.knowledge import SummarizeConstructor, ContriverConstructor
from aaai24.response import StandardGenerator
from aaai24.voting import StandardVoter
from aaai24.scorer import RewardScorer
from aaai24.utils import LLM, truncate_en_doc, truncate_zh_doc
from torch.multiprocessing import Process
from torch.multiprocessing import Manager
import gradio as gr
import traceback
from loguru import logger
import multiprocessing
from typing import List, Optional, Tuple
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

# openai.api_key = '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'
multiprocessing.set_start_method('spawn', force=True)

model_name2url = {
    "llama2-chat": "https://ai.meta.com/llama/",
    "xverse-chat": "https://github.com/xverse-ai/XVERSE-13B",
    "chatglm2": "https://github.com/THUDM/ChatGLM2-6B",
}

# def chat_with_gpt(prompt):
#     response = openai.Completion.create(
#         engine="text-davinci-003",  
#         prompt=prompt,
#         max_tokens=150,  
#         temperature=0.7,  
#         stop=None  
#     )

#     generated_text = response.choices[0].text.strip()
#     return generated_text

def chat_with_gpt(prompt):
    client = OpenAI(api_key = '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')
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    generated_text = response.choices[0].message.content
    return generated_text

# for multiprocessing
def _update(task_map, gpu, uuid, task_item):
    temp = task_map[gpu]
    temp.update({uuid: task_item})
    task_map[gpu] = temp


def _remove(task_map, gpu, uuid):
    temp = task_map[gpu]
    ret = temp.pop(uuid)
    task_map[gpu] = temp
    return ret


def run(rank, config, task_map, wiki_retriever, gen_retriever, summarizer, contriver, generator, voter, scorer):
    try:
        gpu = f"gpu{rank}"

        LLM.get_llm_config(config["LLMConfig"])
        LLM.gpu_ids = config["gpu"]
        LLM.ddp = config["ddp"]

        LLM.initial_all(gpu, config["LLMMap"][gpu])

        # logger.info(f"\nllm:\n{LLM.llms}")

        if rank == 0:
            wiki_retriever.initialize()

        # gpu: {uuid: (task, input, output)}
        task_map[gpu] = {}

        while True:
            uuid_list = list(task_map[gpu].keys())

            for uuid in uuid_list:
                task_item = task_map[gpu].get(uuid, None)

                if task_item is not None and task_item[2] is None:
                    task, kwargs, _ = task_item
                    if task == "wiki":
                        result = wiki_retriever.retrieve(**kwargs)
                    elif task == "gendoc":
                        result = gen_retriever.retrieve(**kwargs)
                    elif task == "summarize":
                        result = summarizer.batch_construct(**kwargs)
                    elif task == "contrive":
                        result = contriver.construct(**kwargs)
                    elif task == "response":
                        result = generator.batch_response(**kwargs)
                    elif task == "vote":
                        result = voter.voting(**kwargs)
                    elif task == "score":
                        result = scorer.batch_score(**kwargs)
                    else:
                        assert False
                    _update(task_map, gpu, uuid, (task, kwargs, result))
    except:
        print(f"{rank}报错")
        print(traceback.format_exc())
        exit(-1)


def build_demo(config, debug=False):
    # initial module and language model
    task_map = Manager().dict()

    web_retriever = WebRetriever(config["ModuleConfig"]["Web"])
    wiki_retriever = WikiRetriever(config["ModuleConfig"]["Wiki"])
    gen_retriever = GenerationRetriever(config["ModuleConfig"]["Gendoc"])

    summarizer = SummarizeConstructor(config["ModuleConfig"]["Summarizer"])
    contriver = ContriverConstructor(config["ModuleConfig"]["Contriver"])

    generator = StandardGenerator(config["ModuleConfig"]["Generator"])
    voter = StandardVoter(config["ModuleConfig"]["Voter"])
    scorer = RewardScorer(config["ModuleConfig"]["Scorer"])

    process_list = []

    for i in range(len(config["LLMMap"])):
        process = Process(target=run, args=(
        i, config, task_map, wiki_retriever, gen_retriever, summarizer, contriver, generator, voter, scorer))
        process_list.append(process)
        process.start()

    while (len(task_map) < len(config["LLMMap"])):
        continue

    # demo executor
    def demo_query(query, language, history: Optional[List[Tuple[str, str]]] = None):
        raw_query = query
        if history is None:
            history = []
        try:
            #log query
            logger.info(f"\nquery:\n{query}")

            if history != []:
                sentence = ''
                for q, response in history:
                    sentence = sentence + '用户：' + q + '\n' + '助手：' + response 
                # query = sentence + '用户：' + query + '\n' + '请根据以上对话重写最后一个问题用于检索，回复的开头不要有助手：或用户：'
                # print('--------------------------------------------query for rewriter-----------------------------')
                # print(query)
                # try:
                #     query = chat_with_gpt(query)
                #     print(query)
                # except Exception as e:
                #     print(f"An error occurred when rewrite: {e}")
                query = "如下是一段示例：\n\n\n用户：今天几号\n助手：今天是2023年12月19日。\n用户：明天呢\n 请根据以上对话将最后一个问题重写为清晰、简洁的搜索查询格式\n回答：明天是几号?\n\n\n" + sentence + '用户：' + query + '\n' + '请根据以上对话将最后一个问题重写为清晰、简洁的搜索查询格式\n回答：'
                print('--------------------------------------------query for rewriter-----------------------------')
                print(query)
                try:
                    # model, tokenizer=LLM.llms["gpu1:chatglm2"]
                    # logger.info(f"\nllm:\n{LLM.llms}")
                    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
                    model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
                    model = model.eval()
                    query, history0 = model.chat(tokenizer, query, history=[])
                    #query = chat_with_gpt(query)
                    print(query)
                except Exception as e:
                    print(f"An error occurred when rewrite: {e}")
                
           

            _dict = {}
            # retrieval
            kwargs = {"query": query, "language": language, "gpu": "gpu1"}
            task_item = ("gendoc", kwargs, None)
            uuid_1 = str(uuid4())
            _update(task_map, "gpu1", uuid_1, task_item)

            kwargs = {"query": query, "language": language, "gpu": "gpu0"}
            task_item = ("wiki", kwargs, None)
            uuid_2 = str(uuid4())
            _update(task_map, "gpu0", uuid_2, task_item)

            web_docs = web_retriever.retrieve(query, language)
            #处理web检索为空的情况
            if web_docs == []:
                uuid_6 = str(uuid4())
                kwargs = {"queries": [raw_query], "language_list": [language], "knowledge_list": [' '], "query_only": True, "history": history,
                          "gpu": "gpu1"}
                task_item = ("response", kwargs, None)
                _update(task_map, "gpu1", uuid_6, task_item)
                while task_map["gpu1"][uuid_6][2] is None:
                    continue
                response_list = _remove(task_map, "gpu1", uuid_6)[2]
                answer = response_list[0]
                logger.info(f"\nquery_only answer:\n{answer}")
                #直接返回query_only
                history = history + [(query, answer)]
                return answer, history


            kwargs = {"query": query, "language": language, "doc": "\n".join(web_docs), "gpu": "gpu0", "top_k": 10}
            task_item = ("contrive", kwargs, None)
            uuid_3 = str(uuid4())
            _update(task_map, "gpu0", uuid_3, task_item)

            while task_map["gpu1"][uuid_1][2] is None or task_map["gpu0"][uuid_2][2] is None or task_map["gpu0"][uuid_3][2] is None:
                continue

            gen_doc = _remove(task_map, "gpu1", uuid_1)[2]
            wiki_docs = _remove(task_map, "gpu0", uuid_2)[2]
            contrive_google_10 = _remove(task_map, "gpu0", uuid_3)[2]

            #all_sources
            all_sources = "\n".join(web_docs) + "\n" + "\n".join(wiki_docs)
            # all_sources = "\n".join(web_docs) + "\n" + "\n".join(wiki_docs) + "\n" + gen_doc

            _dict["docs"] = []
            info_dict = web_retriever.info_dict[query]
            url_item_list, url2doc_dict = info_dict["search"]["organic"], info_dict["fetch"]
            url2item_dict = {item["link"]: item for item in url_item_list}
            doc2url_dict = {doc: url for url, doc in url2doc_dict.items()}
            for doc in web_docs:
                url = doc2url_dict[doc]
                item = url2item_dict[url]
                _dict["docs"].append({"title": item["title"], "url": url, "text": doc})

            for doc in wiki_docs[:2]:
                _dict["docs"].append({"title": "Wikipedia Item",
                                      "url": "https://en.wikipedia.org/wiki/Main_Page" if language == "en" else
                                      "https://zh.wikipedia.org/wiki/Wikipedia:%E9%A6%96%E9%A1%B5",
                                      "text": doc})

            _dict["docs"].append({"title": "LLM Generate Doc",
                                  "url": model_name2url[config["ModuleConfig"]["Gendoc"][f"{language}_model_name"]],
                                  "text": gen_doc})
            

            # knowledge
            # select_num = min(len(web_docs), 3)
            # truncate_doc = truncate_zh_doc if language == "zh" else truncate_en_doc
            # _temp_doc_0 = "\n".join([truncate_doc(web_docs[i], 1200 // select_num) for i in range(select_num)]
            #                       + [truncate_doc(wiki_docs[i], 150) for i in range(2)])
            # _temp_doc_1 = web_docs[0] if len(web_docs) > 0 else ""
            temp_doc_0 = contrive_google_10
            select_num = min(len(web_docs), 2)
            truncate_doc = truncate_zh_doc if language == "zh" else truncate_en_doc
            #gm_w
            google_merge_plus_wiki = "\n".join([truncate_doc(web_docs[i], 250) for i in range(select_num)]
                                  + [wiki_docs[i] for i in range(1)])


            kwargs = {"queries": query, "language_list": [language], "doc_list": [temp_doc_0], "gpu": "gpu1"}
            task_item = ("summarize", kwargs, None)
            uuid_1 = str(uuid4())
            _update(task_map, "gpu1", uuid_1, task_item)

            kwargs = {"query": query, "language": language, "doc": all_sources, "gpu": "gpu0", "top_k": 10}
            task_item = ("contrive", kwargs, None)
            uuid_2 = str(uuid4())
            _update(task_map, "gpu0", uuid_2, task_item)

            kwargs = {"query": query, "language": language, "doc": "\n".join(web_docs), "gpu": "gpu0", "top_k": 5}
            task_item = ("contrive", kwargs, None)
            uuid_3 = str(uuid4())
            _update(task_map, "gpu0", uuid_3, task_item)

            while task_map["gpu1"][uuid_1][2] is None or task_map["gpu0"][uuid_2][2] is None or task_map["gpu0"][uuid_3][2] is None:
                continue

            #gc_sum
            summarize_google_contrive_list = _remove(task_map, "gpu1", uuid_1)[2]
            summarize_google_contrive = "\n".join(summarize_google_contrive_list)
            #c_all
            contrieve_all_sources = _remove(task_map, "gpu0", uuid_2)[2]
            #gc_w 
            contrive_google_5 = _remove(task_map, "gpu0", uuid_3)[2]
            google_contrieve_plus_wiki = "\n".join([contrive_google_5] + wiki_docs[:1])
            #wiki
            wiki = "\n".join(wiki_docs[:5])
            

            _dict["knowledge"] = [google_contrieve_plus_wiki, google_merge_plus_wiki, wiki, summarize_google_contrive, contrieve_all_sources]
            

            # response
            from datetime import datetime
            
            if language == "en":
                time = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S")
                time_prompt = f"The current time is {time}.\n"
            else:
                time = datetime.now().strftime("%Y年%m月%d日 %A %H点%M分")
                time_prompt = f"当前时间是{time}。\n"
                weekday = {"Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三", "Thursday": "星期四", "Friday": "星期五", "Saturday": "星期六", "Sunday": "星期日"}

                for k,v in weekday.items():
                    time_prompt = time_prompt.replace(k,v)
            
            doc_list = [google_contrieve_plus_wiki, google_merge_plus_wiki, wiki, summarize_google_contrive, contrieve_all_sources]

            for i in range(len(doc_list)):
                doc_list[i] = time_prompt + doc_list[i]

            #log knowlegdes
            logger.info(f"\ngoogle_contrieve_plus_wiki: \n{doc_list[0]}")
            logger.info(f"\ngoogle_merge_plus_wiki: \n{doc_list[1]}")
            logger.info(f"\nwiki: \n{doc_list[2]}")
            logger.info(f"\nsummarize_google_contrive: \n{doc_list[3]}")
            logger.info(f"\ncontrieve_all_sources: \n{doc_list[4]}")
            


            uuid_1 = str(uuid4())
            kwargs = {"queries": [raw_query] * 3, "language_list": [language] * 3, "knowledge_list": doc_list[:3], "query_only": False, "history": history,
                      "gpu": "gpu0"}
            task_item = ("response", kwargs, None)
            _update(task_map, "gpu0", uuid_1, task_item)

            uuid_2 = str(uuid4())
            kwargs = {"queries": [raw_query] * 2, "language_list": [language] * 2, "knowledge_list": doc_list[3:], "query_only": False, "history": history,
                      "gpu": "gpu1"}
            task_item = ("response", kwargs, None)
            _update(task_map, "gpu1", uuid_2, task_item)

            # uuid_3 = str(uuid4())
            # kwargs = {"queries": [query], "language_list": [language], "knowledge_list": [' '], "query_only": True, 
            #           "gpu": "gpu1"}
            # task_item = ("response", kwargs, None)
            # _update(task_map, "gpu1", uuid_3, task_item)
            
            while task_map["gpu0"][uuid_1][2] is None or task_map["gpu1"][uuid_2][2] is None:
                continue

            response_list = _remove(task_map, "gpu0", uuid_1)[2] + _remove(task_map, "gpu1", uuid_2)[2]

            for i in range(len(response_list)):
                if len(response_list[i].replace(" ", "")) == 0 or response_list[i] == query:
                    response_list[i] = "Sorry, I don't know about this." if language == "en" else "很抱歉，我不知道。"

            _dict["responses"] = response_list

            

            # voting
            kwargs = {"query": query, "language": language, "responses": response_list, "gpu": "gpu1"}
            task_item = ("vote", kwargs, None)
            uuid = str(uuid4())
            _update(task_map, "gpu1", uuid, task_item)

            while task_map["gpu1"][uuid][2] is None:
                continue

            vote_score_list = _remove(task_map, "gpu1", uuid)[2]

            #log voting scores
            logger.info(f"\nVoting score:\n{vote_score_list}")

            threshold = 0.75

            chosen_status_list = ["Candidate" if vote_score_list[i] > threshold else "Reject" for i in range(5)]
            select_response_list = [response_list[i] for i in range(5) if vote_score_list[i] > threshold]
            select_response_indexes = [i for i in range(5) if vote_score_list[i] > threshold]

            answer = None
            if len(select_response_list) == 0:
                chosen_status_list = ["Candidate" for _ in range(5)]
                select_response_list = response_list
                select_response_indexes = [i for i in range(5)]
            elif len(select_response_list) == 1:
                chosen_status_list[chosen_status_list.index("Candidate")] = "Answer"
                answer = select_response_list[0]

            response_list_with_status = [chosen_status_list[i] + "\n" + response_list[i] for i in range(5)]
            _dict["responses"] = response_list_with_status
            

            if answer is not None:
                _dict["responses"] = response_list_with_status
                
                #log responses
                logger.info(f"\nrseponse1:\n{response_list_with_status[0]}")
                logger.info(f"\nrseponse2:\n{response_list_with_status[1]}")
                logger.info(f"\nrseponse3:\n{response_list_with_status[2]}")
                logger.info(f"\nrseponse4:\n{response_list_with_status[3]}")
                logger.info(f"\nrseponse5:\n{response_list_with_status[4]}")
    

                _dict["answer"] = answer

                #log final answer
                logger.info(f"\nfinal answer:\n{answer}")

                
                history = history + [(query, answer)]
                return answer, history

            # score
            # gpu = "gpu0" if language == "zh" else "gpu1"
            gpu = "gpu0"
            kwargs = {"queries": [query] * len(select_response_list),
                      "language_list": [language] * len(select_response_list), "response_list": select_response_list,
                      "gpu": gpu}
            task_item = ("score", kwargs, None)
            uuid = str(uuid4())
            _update(task_map, gpu, uuid, task_item)

            while task_map[gpu][uuid][2] is None:
                continue

            score_list = _remove(task_map, gpu, uuid)[2]

            #log scoring scores
            logger.info(f"\nScoring score:\n{score_list}")

            answer = select_response_list[score_list.index(max(score_list))]
            chosen_status_list[select_response_indexes[score_list.index(max(score_list))]] = "Answer"

            response_list_with_status = [chosen_status_list[i] + "\n" + response_list[i] for i in range(5)]
            _dict["responses"] = response_list_with_status

            #log responses
            logger.info(f"\nrseponse1:\n{response_list_with_status[0]}")
            logger.info(f"\nrseponse2:\n{response_list_with_status[1]}")
            logger.info(f"\nrseponse3:\n{response_list_with_status[2]}")
            logger.info(f"\nrseponse4:\n{response_list_with_status[3]}")
            logger.info(f"\nrseponse5:\n{response_list_with_status[4]}")
            
            _dict["answer"] = answer

            #log final answer
            logger.info(f"\nfinal answer:\n{answer}")

            logger.info(f"\nquery:\n{query}")
            

            history = history + [(query, answer)]

            logger.info(f"\nhistory:\n{history}")
            return answer, history
            
        except:
            print("error!!!!!!")

    def Config_Chat(query, history):
        language = "en"
        for _char in query:
            if '\u4e00' <= _char <= '\u9fa5':
                language = "zh"
                break
        return demo_query(query, language, history)[0]

    def save_feedback(feedback):
        logger.info(f"收到的评价:{feedback}")

    # demo web framework based on gradio 
    with gr.Blocks() as demo:
        gr.ChatInterface(
            fn=Config_Chat,
            title="活字-日新",
            description="在SCIR实验室研制的huozi-7b-rlhf大模型基础上进行了检索增强，具有更好的使用体验",
            # 每一个都需要放在一个list内
            examples=[
                # ["介绍一下哈工大赛尔实验室"],
                ["2023年杭州亚运会中国金牌数是多少？"],
                ["今天是几号？"],
                ["when was the last time anyone was on the moon?"]
            ],
            submit_btn="提交",
            retry_btn="重新生成",
            undo_btn="撤销",
            clear_btn="清空",
        )
        with gr.Row():
            with gr.Column(scale=2):  
                feedback = gr.Radio(label="评价本次回复", choices=["满意", "不满意"])
            with gr.Column(scale=1):  
                submit_feedback = gr.Button("提交评价")
        submit_feedback.click(fn=save_feedback, inputs=feedback, outputs=[])

    return demo
