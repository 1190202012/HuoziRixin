from openai import OpenAI

client = OpenAI(api_key = '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')
user_prompt = "用户：2023年杭州亚运会中国金牌数是多少？\n助手：根据文档，2023年杭州亚运会中国代表团共获得201金111银71铜，共计383枚奖牌。因此，中国代表团在金牌和奖牌双榜上遥遥领先，其亚运金牌总数已超1500枚。\n用户：今天是几号？\n助手：今天是2023年12月16日。\n用户：明天呢\n请根据以上对话重写最后一个问题用于检索，回复的开头不要有助手：或用户："
response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_prompt}
  ]
)

print(response)
print(response.choices[0].message.content)
# import openai

# # 设置你的OpenAI API密钥
# openai.api_key = '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'


# # 定义一个函数，用于与ChatGPT进行交互
# def chat_with_gpt(prompt):
#     # 调用OpenAI的Chat API
#     response = openai.Completion.create(
#         engine="text-davinci-003",  # 使用ChatGPT模型
#         prompt=prompt,
#         max_tokens=150,  # 控制生成的最大令牌数
#         temperature=0.7,  # 控制生成的创造性程度，值越高越创造性
#         stop=None  # 可以定义一个停止词列表，用于限制生成的文本
#     )

#     # 提取生成的文本
#     generated_text = response.choices[0].text.strip()
#     return generated_text


# # 与ChatGPT进行对话
# user_prompt = "用户：2023年杭州亚运会中国金牌数是多少？\n助手：根据文档，2023年杭州亚运会中国代表团共获得201金111银71铜，共计383枚奖牌。因此，中国代表团在金牌和奖牌双榜上遥遥领先，其亚运金牌总数已超1500枚。\n用户：今天是几号？\n助手：今天是2023年12月16日。\n用户：明天呢\n请根据以上对话重写最后一个问题用于检索，回复里不需要有助手："
# chat_reply = chat_with_gpt(user_prompt)
# print(chat_reply)

