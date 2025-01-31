### Simple Webserver Example for [Encoder-Decoder-Model](https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder) Translation

This is a simple example of a web server you can run on any environment that simply wraps around translation models.

There are many open source models available at [Hugging Face](https://huggingface.co).

Flask is a super simple, easy to use micro-framework you can use to quickly create the backend.

Besides Flask, all the other modules used are runtime modules regularly used for running safetensor based models like torch and transformers.

In this example, I used the following open source models to translate Japanese into Korean.

- Transaltion: [sappho192/aihub-ja-ko-translator](https://huggingface.co/sappho192/aihub-ja-ko-translator)
- Encoder [skt/kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2)
- Decoder [tohoku-nlp/bert-base-japanese](https://huggingface.co/tohoku-nlp/bert-base-japanese)

For Windows, you can run the following commands in Power Shell

```
python -m venv ./venv
./venv/Scripts/activate
pip install -r requirements.txt
python main.py
```

Models will be downloaded the first time you run (as of this writing, transformers downloads to `~/.cache/huggingface/hub/`).

You can curl like this to get the translated string:
```
curl --request POST \
  --url http://127.0.0.1:5000/translate \
  --header 'Content-Type: application/json' \
  --data '{"q":"この先生は本当に良い人だ"}'
```

And the server responds with
```
{
	"translatedText": "이 선생님은 정말 좋은 사람이야."
}
```
