import re

'''
description: 将其他格式的文件提取并
'''
def extract_text_into_md(client, file): 
    result = client.parsing.parse(
        upload_file=file,
        tier="agentic",
        agentic_options={
            "custom_prompt": "Ignore all page headers and footers. Do not extract any page numbers, document titles, dates, or other information that repeatedly appears at the top or bottom of pages."
        },
        version="latest",
        expand=['Markdown']
    )
    all_text = ""
    for page in result.markdown.pages:
        all_text += page.markdown
    
    md_file = file.parent / (file.stem + ".md")
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

def get_heading_split(higher_level_heading, lower_level_heading, level, source):
    for heading_content in higher_level_heading:
        heading_text = heading_content["text"]
        if not heading_text:
            continue
        # 把所有# 前都加上\n， 避开其他标题
        pattern = r'(?<![#\n])' + re.escape('#' * level + ' ')
        judge = '\n' + '#' * level + ' '
        heading_text = re.sub(pattern, judge, heading_text)
        if judge in heading_text:
            for content in re.split(judge, heading_text):
                if level == 1: 
                    heading = get_heading(content, level)
                else:
                    heading = heading_content["heading"] + ' ' + get_heading(content, level)
                lower_level_heading.append({"text": content, "heading": heading, "source": source})
        else:
            lower_level_heading.append(heading_content)
    

def split_chunks(md_file, max_size=512):
    '''
    按照标题先分类，先分一级标题，内容里有二级标题按二级标题分，然后判断每块是否过长,过长再按max_size 分，但保留表格完整度
    chunks append字典，包括text, source, 以及heading
    '''
    with open(md_file, 'r', encoding='utf-8') as f:
        all_text = f.read()
    
    # 去掉# 2 Product Overview 之前的所有内容，这些内容都是重复的没用的内容
    manual_start = '# 2 Product Overview'
    if 'User Manual' in md_file.name:
        while manual_start in all_text:
            all_text = all_text.split(manual_start, 1)[1]
        all_text = manual_start + all_text
    
    init_split = [{'text': all_text, 'heading': '', 'source': ''}]
    
    heading1_split = []
    get_heading_split(init_split, heading1_split, 1, md_file.name)
    
    heading2_split = []
    get_heading_split(heading1_split, heading2_split, 2, md_file.name)
    
    heading3_split = []
    get_heading_split(heading2_split, heading3_split, 3, md_file.name)
    
    chunks = []
    for heading3_content in heading3_split:
        if not heading3_content["text"]:
            continue
        if len(heading3_content["text"]) > max_size:
            for chunk_content in split_big_chunk(heading3_content["text"], max_size):
                chunks.append({"text": chunk_content, "heading": heading3_content["heading"], "source": md_file.name})
        else:
            chunks.append(heading3_content)
    
    return chunks
    