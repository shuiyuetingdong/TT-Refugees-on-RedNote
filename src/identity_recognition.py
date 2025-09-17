import dspy
import csv
import pandas as pd
from typing import Literal



# set your own api key and api url
llm = dspy.LM(
    api_key='',
    api_base='',
    model='openai/gpt-4o',
    temperature=0,
    stop=None,
    cache=False
)
dspy.configure(lm=llm)


class identity_recognition_dspy(dspy.Signature):
    """你需要对来自小红书的用户评论进行身份识别，将这些用户分为两类，"中国"和"外国"。
    对"中国"用户的定义为：有着中国文化背景的用户（包括海外华侨华人）
    对"外国"用户的定义为：非中国文化背景的用户（包括旅居中国的外国人）
    请你综合考虑以下信息，推断出用户的身份。
    """
    用户昵称: str = dspy.InputField() 
    用户IP属地: str = dspy.InputField() 
    用户标签: str = dspy.InputField() 
    用户简介: str = dspy.InputField() 
    用户历史发帖的标题: str = dspy.InputField(desc='可能包含多篇帖子的标题') 

    # 分析过程: str = dspy.OutputField() 
    用户身份: Literal['中国', '外国'] = dspy.OutputField() 



def identity_recognition(input_file_path, output_file_path):
    testset = []
    user_url_dict = {}  # Used to store user URLs and corresponding indices
    with open(input_file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        # Use a dictionary to temporarily store each user's information
        user_dict = {}
        for row in reader:
            user_url = row['个人主页网址']
            if user_url not in user_dict:
                # First time encountering this user, initialize basic information
                user_dict[user_url] = {
                    '用户昵称': row['博主名称'],
                    '用户IP属地': row['IP属地'],
                    '用户标签': row['标签'],
                    '用户简介': row['简介'],
                    '用户历史发帖的标题': [row['笔记标题']],
                }
            else:
                # Existing user, only add post titles
                user_dict[user_url]['用户历史发帖的标题'].append(row['笔记标题'])
        
        # Create Example object for each unique user
        for user_url, info in user_dict.items():
            titles = '\n'.join(info['用户历史发帖的标题'])
            example = dspy.Example(
                用户昵称=info['用户昵称'],
                用户IP属地=info['用户IP属地'],
                用户标签=info['用户标签'],
                用户简介=info['用户简介'],
                用户历史发帖的标题=titles
            ).with_inputs("用户昵称", "用户IP属地", "用户标签", "用户简介", "用户历史发帖的标题")
            testset.append(example)
            user_url_dict[user_url] = len(testset) - 1  # Record the mapping between user URLs and indices in testset


        classify = dspy.Predict(identity_recognition_dspy)
        results = classify.batch(testset)

        print("\nNumber of results:", len(results))

        # Create a dictionary to store user URLs and corresponding identity recognition results
        identity_results = {}

        for user_url, idx in user_url_dict.items():
            identity_results[user_url] = results[idx]['用户身份']

    # Read the original CSV file as a DataFrame
    original_df = pd.read_csv(input_file_path, encoding='utf-8-sig')

    # Add identity column to the original DataFrame
    original_df['identity'] = original_df['个人主页网址'].map(identity_results)

    # Save results to a new CSV file
    original_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')


input_file_path = '../data/commentor_notes.csv'
output_file_path = '../data/commentor_notes_with_identity.csv'
identity_recognition(input_file_path, output_file_path)