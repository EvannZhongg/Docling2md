VLM_PROMPT = """
图片是电子元器件数据手册中的插图，请根据以下分类对图片进行分类，选择最符合的类别：
- Device Package
- Functional Block Diagram
- Application Circuit
- Timing Diagram
- Test Waveform
- Characteristic Graphs
- PCB Layout Guidelines
- Mechanical Characteristics
- Company_Logo
- 其他

类别和图片描述在一行输出，不要分行，严格按照输出的格式：“类别名称：图片描述,ocr文字内容”
图片描述尽量简短，用一句话概括，如果图片中涉及数字或文字，详细输出里面的文字。
如果图片中含有多个电子器件的型号，请严格输出每个型号的名称，不允许进行任何省略。
如果图片内容无明显的意义：例如只有文字信息，或是不符合上述任何类别，种类直接输出“其他”。
请严格确认"Device Package"是否符合要求，图片中的内容必须是电子器件的封装图片才可以选择，不能出现纯文本的内容(例如图片中只显示字母CMOS而没有其他图像信息)。
"""
#  VLM_PROMPT = "请描述这张图片的内容（以一行为限），输出简洁中文描述。"

