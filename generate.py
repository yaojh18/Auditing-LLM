import openai
import time
import json
import argparse
import pandas
import os

# set your openai api key here
# openai.api_key = "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"
openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_res_multi_round(msg):
    """
    :param msg: List of message to LLM (ChatGPT)
    :return: Output from LLM (ChatGPT)
    """
    history_message = []
    collected_message = []
    for message in msg:
        history_message.append({"role": "user", "content": message})
        while True:
            try:
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=history_message,
                    temperature=1,
                    max_tokens=256,
                    top_p=1
                )
                break
            except openai.error.RateLimitError:
                print('openai.error.RateLimitError\nRetrying...')
                time.sleep(60)
            except openai.error.ServiceUnavailableError:
                print('openai.error.ServiceUnavailableError\nRetrying...')
                time.sleep(20)
            except openai.error.Timeout:
                print('openai.error.Timeout\nRetrying...')
                time.sleep(20)
            except openai.error.APIError:
                print('openai.error.APIError\nRetrying...')
                time.sleep(20)
            except openai.error.APIConnectionError:
                print('openai.error.APIConnectionError\nRetrying...')
                time.sleep(20)
        history_message.append({"role": "assistant", "content": res['choices'][0]['message']['content']})
        collected_message.append(res['choices'][0]['message']['content'])

    print(history_message)
    return collected_message


def get_res_one_round(msg):
    """
    :param msg: List of message to LLM (ChatGPT)
    :return: Output from LLM (ChatGPT)
    """
    # TODO: finish single round conversation
    pass


def prompt_augmentation_by_context(seed_data, multi_round, selection_strategy):
    """
    :param seed_data: A Dataframe in the format of {"Question": "", "Best Answer": "", "Correct Answers": "", "Question": ""}
    :return: A dict in the format of {"context": "", "questions": [], "correct answers": []}
    """

    # read instructions
    context_gen_file = open(
        "./instruction/by-context_context_{}_{}.txt".format(multi_round, selection_strategy), 'r', encoding="utf-8"
    )
    context_gen_instruction = ''.join(context_gen_file.readlines())
    context_gen_file.close()
    question_gen_file = open(
        "./instruction/by-context_question_{}_{}.txt".format(multi_round, selection_strategy), 'r', encoding="utf-8"
    )
    question_gen_instruction = ''.join(question_gen_file.readlines())
    question_gen_file.close()
    answer_gen_file = open(
        "./instruction/by-context_answer_{}_{}.txt".format(multi_round, selection_strategy), 'r', encoding="utf-8"
    )
    answer_gen_instruction = ''.join(answer_gen_file.readlines())
    answer_gen_file.close()

    collected_res = []
    for idx, seed_sample in seed_data.iterrows():
        if multi_round:
            msg = context_gen_instruction + \
                  "\n#Question#: " + seed_sample["Question"] + \
                  "\n#Best Correct Answer#: " + seed_sample["Best Answer"] + \
                  "\n#Correct Answers#: " + seed_sample["Correct Answers"] + \
                  "\n#Incorrect Answers#: " + seed_sample["Incorrect Answers"]
            res = get_res_multi_round([msg, question_gen_instruction, answer_gen_instruction])
            context = res[0]
            questions = res[1].split('\n')
            answers = res[2].split('\n')
        else:
            # TODO
            raise NotImplementedError
        if len(questions) == len(answers):
            collected_res.append({"context": context, "questions": questions, "answers": answers})

    return collected_res


def prompt_augmentation_direct(seed_data, multi_round, selection_strategy):
    """
    :param seed_data: A Dataframe in the format of {"Question": "", "Best Answer": "", "Correct Answers": "", "Question": ""}
    :return: A dict in the format of {"context": "", "questions": [], "correct answers": []}
    """
    # TODO: Add function and instruction of direct prompt augmentation
    return {}


def prompt_selection(augmented_data, selection_strategy):
    """
    :param augmented_data: A list that contends dicts like {"context": "", "questions": [], "answers": []}
    :return: A list that contends dicts like {"context": "", "res":[{"question": "", "correct answer": "", "answer": "", "val question": "", "val answer": ""}]}
    The "correct answer" here is just the output of prompt augmentation without validation, we'll not reply on it to select prompts.
    """
    # read instructions
    question_ask_file = open(
        "./instruction/{}_ask.txt".format(selection_strategy), 'r', encoding="utf-8"
    )
    question_ask_instruction = ''.join(question_ask_file.readlines())
    question_ask_file.close()
    question_val_file = open(
        "./instruction/{}_val.txt".format(selection_strategy), 'r', encoding="utf-8"
    )
    question_val_instruction = ''.join(question_val_file.readlines())
    question_val_file.close()

    collected_res = []
    for augmented_sample in augmented_data:
        context = augmented_sample["context"]
        res_list = []
        for q, a in zip(augmented_sample["questions"], augmented_sample["answers"]):
            if selection_strategy == "snowball":
                msg = question_ask_instruction + "\n#Question#: " + q
                res = get_res_multi_round([msg])[0].split('\n')
                answer = res[0]
                explain = ''.join(res[1:])
                msg = question_val_instruction + "\n#Statement#: " + explain
                val_answer = get_res_multi_round([msg])[0]
                res_list.append({
                    "question": q,
                    "correct answer": a,
                    "answer": answer + explain,
                    "val question": question_val_instruction + "\n#Statement#: " + explain,
                    "val answer": val_answer
                })
            else:
                # TODO: finish other types of truthfulness test
                raise NotImplementedError
        collected_res.append({"context": context, "res": res_list})

    return collected_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prompt generation")
    parser.add_argument(
        "--augmentation_strategy",
        default="by-context",
        help="strategy for prompt augmentation: by-context or direct"
    )
    parser.add_argument(
        "--selection_strategy",
        default="snowball",
        help="strategy for prompt selection: snowball, sycophancy, knowledge-conflict, lack-knowledge"
    )
    parser.add_argument(
        "--seed_data",
        default="TruthfulQA",
        help="the file name of raw dataset"
    )
    parser.add_argument(
        "--trail_num",
        default=1,
        help="number of samples used in original dataset"
    )
    parser.add_argument(
        "--prompt_type",
        default="multi-round",
        help="If multi-round, jobs are done in pipeline; if one-round, jobs are done in"
    )
    args = parser.parse_args()

    # read original dataset, instructions
    # TODO: Unify the format of original dataset
    seed_data = pandas.read_csv("./dataset/{}.csv".format(args.seed_data))
    assert 1 <= args.trail_num <= len(seed_data)
    seed_data = seed_data[: args.trail_num]

    # prompt augmentation
    if args.augmentation_strategy == "by-context":
        augmented_res = prompt_augmentation_by_context(seed_data, args.prompt_type, args.selection_strategy)
    elif args.augmentation_strategy == "direct":
        augmented_res = prompt_augmentation_direct(seed_data, args.prompt_type, args.selection_strategy)
    else:
        raise ValueError("The strategy must be by-context or direct!")

    # prompt selection
    collected_res = prompt_selection(augmented_res, args.selection_strategy)

    output_file = open("./output/test.txt", 'a', encoding="utf-8")
    json.dump(collected_res, output_file)
    output_file.write('\n')
    output_file.close()
