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


class comment_analysis_dspy(dspy.Signature):
    """
    你需要对小红书上的帖子下的评论进行分析，判断评论针对所谈论的事件的情感倾向，并给出评论的立场。
    请你结合实际情况，综合考虑上下文语境（如帖子内容，该评论所回复的评论等），以及评论之间的关系，按照下面的评分选项，对用户评论的情感（蔑视、嫉妒、厌恶、恐惧、愤怒、惊奇、赞扬、自豪、快乐、尊敬）进行评分。
    评分分为以下5个等级，请回复1-5的数字以表示等级：
    1: 强烈反对
    2: 不同意
    3: 既不同意也不反对
    4: 同意
    5: 强烈同意
    此外，请综合分析背景信息，评论及其情感倾向，给出评论的立场（偏向中国、偏向外国、中立）。
    """

    帖子标题: str = dspy.InputField() 
    帖子内容: str = dspy.InputField() 
    用户昵称: str = dspy.InputField() 
    该评论所回复的内容: str = dspy.InputField(desc='可能包含一系列对话') 
    评论: str = dspy.InputField() 

    # 分析过程: str = dspy.OutputField() 
    蔑视: int = dspy.OutputField() # Contempt
    嫉妒: int = dspy.OutputField() # Jealousy
    厌恶: int = dspy.OutputField() # Disgust
    恐惧: int = dspy.OutputField() # Fear
    愤怒: int = dspy.OutputField() # Anger
    惊奇: int = dspy.OutputField() # Surprise
    赞扬: int = dspy.OutputField() # Praise
    自豪: int = dspy.OutputField() # Pride
    快乐: int = dspy.OutputField() # Joy
    尊敬: int = dspy.OutputField() # Respect
    立场: Literal['偏向中国', '偏向外国', '中立'] = dspy.OutputField() # Pro_China, Pro_Foreign, Neutral



def comment_analysis(input_file_path, output_file_path, max_recursion_depth=10):
    testset = []


    
    # First read the entire CSV file into a DataFrame for subsequent row lookups
    df = pd.read_csv(input_file_path, encoding='utf-8-sig', dtype=str)
    
    # Clean up secondary comments that have no corresponding primary comment
    # Find all secondary comments (rows where Primary_Key starts with 'R')
    secondary_comments = df[df['Primary_Key'].str[0] == 'R'].copy()
    # Get all primary comment Primary_Keys
    primary_keys = set(df[df['Primary_Key'].str[0] == 'C']['Primary_Key'])
    # Find indices of secondary comments without corresponding primary comments
    invalid_indices = secondary_comments[~secondary_comments['Foreign_Key'].isin(primary_keys)].index
    # Delete these rows from the DataFrame
    if len(invalid_indices) > 0:
        df.drop(invalid_indices, inplace=True)
    
    def get_reply(row, all_data):
        """Get reply chain"""
        if row['Primary_Key'][0] == 'C':  # If it's a primary comment, no need to get reply content
            return ""
            
        reply_content = []
        # Find the directly replied comment
        parent_row = all_data[all_data['Primary_Key'] == row['Foreign_Key']].iloc[0]
        reply_content.append(f"{parent_row['昵称']}：“{parent_row['评论']}”")
        reply_content = get_reply_chain(row, all_data, reply_content, current_depth=1, max_depth=max_recursion_depth)

        return reply_content

    def get_reply_chain(row, all_data, reply_content, current_depth, max_depth):
        """If there is a reply_user, continue searching the conversation chain"""
        if pd.notna(row['reply_user']) and current_depth < max_depth:
            current_foreign_key = row['Foreign_Key']
            current_primary_key_num = int(row['Primary_Key'][1:])  # Get the numeric part of Primary_Key
            
            # Only search within secondary comments with the same Foreign_Key
            same_thread_replies = all_data[all_data['Foreign_Key'] == current_foreign_key]
            
            # Find historical conversations that meet the criteria in the same thread comments and create a copy
            filtered_replies = same_thread_replies[
                (same_thread_replies['昵称'] == row['reply_user']) &
                (same_thread_replies['Primary_Key'].str[0] == 'R') &
                (same_thread_replies['Primary_Key'].str[1:].astype(int) < current_primary_key_num)
            ].copy()
            
            if not filtered_replies.empty:
                # Calculate the distance between each reply and the current comment
                filtered_replies['distance'] = current_primary_key_num - filtered_replies['Primary_Key'].str[1:].astype(int)
                # Get the record with the smallest distance
                closest_reply = filtered_replies.loc[filtered_replies['distance'].idxmin()]
                
                # If the closest reply also has a reply_user, recursively get its reply chain
                if pd.notna(closest_reply['reply_user']):
                    get_reply_chain(closest_reply, same_thread_replies, reply_content, current_depth + 1, max_depth)

                reply_content.append(f"\n{closest_reply['昵称']}：“{closest_reply['评论']}”")
                
        return "".join(reply_content)
    
    # Process each row of data
    for _, row in df.iterrows():
        # Get the content that this comment replies to
        reply_content = get_reply(row, df)
        
        # Create Example object
        example = dspy.Example(
            帖子标题=row['帖子标题'],
            帖子内容=row['笔记内容'],
            用户昵称=row['昵称'],
            该评论所回复的内容=reply_content,
            评论=row['评论']
        ).with_inputs("帖子标题", "帖子内容", "用户昵称", "该评论所回复的内容", "评论")
        testset.append(example)
        


    # Use dspy for classification
    classify = dspy.Predict(comment_analysis_dspy)
    results = classify.batch(testset, num_threads=200)

    '''    
    batch函数参数说明：
    def batch(
        self,
        examples,
        num_threads: int = 32,
        max_errors: int = 10,
        return_failed_examples: bool = False,
        provide_traceback: bool = False,
        disable_progress_bar: bool = False,
    ):'''

    # Ensure the number of results matches the DataFrame row count
    assert len(results) == len(df), f"结果数量 ({len(results)}) 与原始数据行数 ({len(df)}) 不匹配"

    # Add results to the original DataFrame
    # Define the column names to be added
    result_columns = ['蔑视', '嫉妒', '厌恶', '恐惧', '愤怒', '惊奇', '赞扬', '自豪', '快乐', '尊敬', '立场']
    
    # Create data for each new column
    for col in result_columns:
        if col == '立场':
            # Stance is string type
            df[col] = [result[col] for result in results]
        else:
            # Emotion values are integer type
            df[col] = [result[col] for result in results]

    # Save results
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig')



input_file_path = '../data/input_file_path.csv'
output_file_path = '../data/output_file_path.csv'


comment_analysis(input_file_path, output_file_path, max_recursion_depth=10)