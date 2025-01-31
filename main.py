from transformers import EncoderDecoderModel, PreTrainedTokenizerFast, BertJapaneseTokenizer
from flask import Flask, abort, request

# Set downloaded models here
encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"
model = EncoderDecoderModel.from_pretrained("sappho192/aihub-ja-ko-translator")

src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)
trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)

app = Flask(__name__)

def translate_from_model(text_src):
    embeddings = src_tokenizer(text_src, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    embeddings = {k: v for k, v in embeddings.items()}
    output = model.generate(**embeddings, max_length=500)[0, 1:-1]
    text_trg = trg_tokenizer.decode(output.cpu())
    return text_trg

@app.route('/')
def index():
    return 'Use /translate POST call with JSON "{q: \"text to translate\"}" - there is no web interface rigyht now'

@app.route('/translate', methods=['POST'])
def translate():
    req = request.get_json()
    if not isinstance(req, dict):
        abort(400, "Invalid JSON format")

    q = req.get("q")

    print("received translation request with query: ", q)

    if not q:
        abort(400, "Invalid request: missing param 'q'")

    try:
        translated_text = translate_from_model(q)
    except Exception as e:
        print("Error: ", e)
        abort(500, e)

    result = {"translatedText": translated_text}
    return result

if __name__ == '__main__':
    app.run()