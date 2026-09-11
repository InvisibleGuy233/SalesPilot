import re

'''
description: 将其他格式的文件提取并
'''
def extract_text_into_md(client, file): 
    result = client.parsing.parse(
        upload_file=file,
        tier="agentic_plus",
        version="latest",
        expand=['Markdown']
    )
    all_text = ""
    for page in result.markdown.pages:
        all_text += page.markdown
    
    file_name = file.split('\\')[-1]
    md_file_list = file_name.split('.')
    md_file_list[-1] = 'md'
    md_file = '.'.join(md_file_list)
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(all_text)
    return md_file
        
def split_big_line(line, max_size=256):
    sentences = re.split('。', line)
    res = []
    con = ''
    for sen in sentences:
        if len(con) + len(sen) > max_size:
            res.append(con)
            con = sen
            continue
        con += sen
    return res
             

def split_big_chunk(chunk, max_size=256):
    lines = re.split('\n', chunk)
    res = []
    content = ''
    inTable = False
    for line in lines:
        if not line:
            continue
        if len(line) > max_size:
            res.extend(split_big_line(line, max_size))
            continue
        if inTable:
            content += line
            if '</table>' in line:
                res.append(content)
                inTable = False
                content = ''
            continue
        if '<table>' in line:
            if content:
                res.append(content)
            content = line
            inTable = True
            continue
        if len(content) + len(line) > max_size:
            res.append(content)
            content = line
            continue
        content += line
    return res

def get_heading(text, level):
    heading = text.split("\n", 1)[0]
    return '#' * level + heading
        
def split_chunks(md_file, max_size=256):
    '''
    按照标题先分类，先分一级标题，内容里有二级标题按二级标题分，然后判断每块是否过长,过长再按max_size 分，但保留表格完整度
    chunks append字典，包括text, source, 以及heading
    '''
    with open(md_file, 'r', encoding='utf-8') as f:
        all_text = f.read()
    
    all_text = '\n' + all_text
    heading1_split = []
    for heading1_content in re.split("\n# ", all_text):
        heading1_split.append({"text": heading1_content, "heading": get_heading(heading1_content, 1), "source":md_file})
    heading2_split = []
    
    
    for heading1_content in heading1_split:
        if not heading1_content["text"]:
            continue
        if '\n## ' in heading1_content["text"]:
            for heading2_content in re.split("\n## ", heading1_content["text"]):
                heading2_split.append({"text": heading2_content, "heading": heading1_content["heading"] + ' ' + get_heading(heading2_content, 2), "source":md_file})
        else:
            heading2_split.append(heading1_content)
    chunks = []
            
    for heading2_content in heading2_split:
        if not heading2_content["text"]:
            continue
        if len(heading2_content["text"]) > max_size:
            for chunk_content in split_big_chunk(heading2_content["text"], max_size):
                chunks.append({"text": chunk_content, "heading": heading2_content["heading"], "source": md_file})
        else:
            chunks.append(heading2_content)
    
    return chunks

# if __name__ == '__main__':
#     client = LlamaCloud(api_key=config.LLAMA_CLOUD_API_KEY)
    
#     file = "./星辰电动ES9·未来旗舰电动SUV产品介绍.pdf"
#     file = "./汽车车身轻量化技术研究现状——基于材料、结构与制造工艺的综述.pptx"
#     extract_text_into_md(client, file)
        
#     md_file = './星辰电动ES9·未来旗舰电动SUV产品介绍.md'
#     chunks = split_chunks(md_file)
#     with open('rec.txt', 'w', encoding='utf-8') as f:
#         for i, chunk in enumerate(chunks):
#             f.write(f'\n====================Chunk No.{i}=================================\n')
#             f.write(str(chunk))
    
    