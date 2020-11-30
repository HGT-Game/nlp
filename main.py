from flask import Flask, request
import hanlp
import json

app = Flask(__name__)

# 加载分词模型
tokenizer = hanlp.load('LARGE_ALBERT_BASE')
# 词性标注
tagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ALBERT_BASE)
# 依存句法分析
syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
# 语义依存分析
semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL16_NEWS_BIAFFINE_ZH)
# 流水线
pipeline = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(tokenizer, output_key='tokens') \
    .append(tagger, output_key='part_of_speech_tags') \
    .append(syntactic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='syntactic_dependencies') \
    .append(semantic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='semantic_dependencies')

@app.route("/nlp", methods=["POST"])
def check():
    # 默认返回内容

    return_dict = {'code': 200,
                   'message': 'success', 'result': False}
    # 获取传入的参数
    get_Data=request.get_data()
    # 传入的参数为bytes类型，需要转化成json
    get_Data=json.loads(get_Data)
    content = get_Data.get('content')
    res = pipeline(content)
    # 对参数进行操作
    return_dict['result'] = res

    return json.dumps(return_dict, ensure_ascii=False)


if __name__ == "__main__":
    app.run(port=8898, debug=False)
