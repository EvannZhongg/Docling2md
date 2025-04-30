#  TEXT_TYPE_PROMPT = "以下是一段文档中的文字，请判断它是标题（Heading）还是正文段落（Paragraph）。仅回答 'Heading' 或 'Paragraph'。\n\n文字内容："
TEXT_TYPE_PROMPT = """
以下是一段文档中的文字，请判断它是“标题”（Heading）还是“正文段落”（Paragraph）。
请仅返回 'Heading' 或 'Paragraph'。

判断规则如下：
1. 若该文字是一个段落、列表项（如以“·”、“-”、“•”等开头），或者陈述性的句子，应判断为 'Paragraph'。
2. 若该文字具有以下特征，则判断为 'Heading'：
   - 字符数较短，通常不超过 15 个词；
   - 不以句号、逗号或冒号结尾；
   - 没有完整的主谓结构；
   - 通常不以 bullet 符号或数字编号（如 '·'、'1.'）开头；

举例：
- “Features” → Heading  
- “·Low capacitance designs” → Paragraph  
- “Figure 2. Output Power vs Input Voltage” → Heading  
- “The device supports high reliability in harsh conditions.” → Paragraph  

请根据上述规则判断以下内容：

文字内容：
"""
